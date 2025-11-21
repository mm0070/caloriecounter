import os
import json
import base64
from datetime import datetime, timedelta, timezone, date

from fastapi import FastAPI, HTTPException, Request, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
from sqlalchemy import (
    create_engine, Column, Integer, Float, Text, DateTime, func
)
from sqlalchemy.orm import sessionmaker, declarative_base, Session

from openai import OpenAI

# ---------- Config ----------

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("Set OPENAI_API_KEY in your environment.")

client = OpenAI(api_key=OPENAI_API_KEY)

DB_URL = "sqlite:///./calories.db"

engine = create_engine(DB_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")


# ---------- DB Model ----------

class Entry(Base):
    __tablename__ = "entries"

    id = Column(Integer, primary_key=True, index=True)
    ts = Column(DateTime(timezone=True), nullable=False, index=True)
    description = Column(Text, nullable=False)
    calories = Column(Float, nullable=False)
    protein = Column(Float, nullable=False)
    carbs = Column(Float, nullable=False)
    fat = Column(Float, nullable=False)
    raw_response = Column(Text, nullable=True)


class Note(Base):
    __tablename__ = "notes"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(Text, nullable=False, unique=True)
    content = Column(Text, nullable=False)
    created_at = Column(DateTime(timezone=True), nullable=False, default=datetime.now(timezone.utc))


Base.metadata.create_all(bind=engine)


# ---------- Schemas ----------

class AddEntryRequest(BaseModel):
    text: str


class AddNoteRequest(BaseModel):
    title: str
    content: str


class EntryOut(BaseModel):
    id: int
    ts: datetime
    description: str
    calories: float
    protein: float
    carbs: float
    fat: float


class NoteOut(BaseModel):
    id: int
    title: str
    content: str
    created_at: datetime


class StatsOut(BaseModel):
    total_calories: float
    total_protein: float
    total_carbs: float
    total_fat: float


class DashboardOut(BaseModel):
    today_stats: StatsOut
    last7_stats: StatsOut
    today_entries: list[EntryOut]


class DayOut(BaseModel):
    date: date
    stats: StatsOut
    entries: list[EntryOut]


# ---------- Helpers ----------

def get_db() -> Session:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_today_range():
    # Use local time on the server
    today = date.today()
    start = datetime.combine(today, datetime.min.time())
    end = start + timedelta(days=1)
    # store in UTC
    start_utc = start.astimezone(timezone.utc) if start.tzinfo else start.replace(tzinfo=timezone.utc)
    end_utc = end.astimezone(timezone.utc) if end.tzinfo else end.replace(tzinfo=timezone.utc)
    return start_utc, end_utc


def get_date_range(day: date):
    start = datetime.combine(day, datetime.min.time())
    end = start + timedelta(days=1)
    start_utc = start.astimezone(timezone.utc) if start.tzinfo else start.replace(tzinfo=timezone.utc)
    end_utc = end.astimezone(timezone.utc) if end.tzinfo else end.replace(tzinfo=timezone.utc)
    return start_utc, end_utc


def get_last7_range():
    today = date.today()
    start = datetime.combine(today - timedelta(days=6), datetime.min.time())
    end = datetime.combine(today + timedelta(days=1), datetime.min.time())
    start_utc = start.astimezone(timezone.utc) if start.tzinfo else start.replace(tzinfo=timezone.utc)
    end_utc = end.astimezone(timezone.utc) if end.tzinfo else end.replace(tzinfo=timezone.utc)
    return start_utc, end_utc


def empty_stats() -> StatsOut:
    return StatsOut(
        total_calories=0.0,
        total_protein=0.0,
        total_carbs=0.0,
        total_fat=0.0,
    )


def aggregate_stats(db: Session, start: datetime, end: datetime) -> StatsOut:
    result = (
        db.query(
            func.coalesce(func.sum(Entry.calories), 0),
            func.coalesce(func.sum(Entry.protein), 0),
            func.coalesce(func.sum(Entry.carbs), 0),
            func.coalesce(func.sum(Entry.fat), 0),
        )
        .filter(Entry.ts >= start, Entry.ts < end)
        .one()
    )
    return StatsOut(
        total_calories=float(result[0] or 0),
        total_protein=float(result[1] or 0),
        total_carbs=float(result[2] or 0),
        total_fat=float(result[3] or 0),
    )


def average_stats_on_logged_days(db: Session, start: datetime, end: datetime) -> StatsOut:
    daily_totals = (
        db.query(
            func.date(Entry.ts),
            func.coalesce(func.sum(Entry.calories), 0),
            func.coalesce(func.sum(Entry.protein), 0),
            func.coalesce(func.sum(Entry.carbs), 0),
            func.coalesce(func.sum(Entry.fat), 0),
        )
        .filter(Entry.ts >= start, Entry.ts < end)
        .group_by(func.date(Entry.ts))
        .all()
    )

    logged_days = [day for day in daily_totals if float(day[1]) > 0]
    if not logged_days:
        return empty_stats() 

    day_count = len(logged_days)
    total_calories = sum(float(day[1]) for day in logged_days)
    total_protein = sum(float(day[2]) for day in logged_days)
    total_carbs = sum(float(day[3]) for day in logged_days)
    total_fat = sum(float(day[4]) for day in logged_days)

    return StatsOut(
        total_calories=total_calories / day_count,
        total_protein=total_protein / day_count,
        total_carbs=total_carbs / day_count,
        total_fat=total_fat / day_count,
    )


def note_to_out(note: Note) -> NoteOut:
    return NoteOut(
        id=note.id,
        title=note.title,
        content=note.content,
        created_at=note.created_at,
    )


def notes_to_block(notes: list[Note]) -> str:
    note_lines = []
    for note in notes:
        title = note.title.strip()
        content = note.content.strip()
        if not title or not content:
            continue
        note_lines.append(f"- {title}: {content}")

    return "\n".join(note_lines) if note_lines else "None provided."


def call_nutrition_model(text: str, notes: list[Note]) -> dict:
    note_block = notes_to_block(notes)

    system_prompt = f"""
You are a nutrition assistant.

Given a description of everything a person ate or drank, estimate:
- total calories (kcal)
- total protein (g)
- total carbohydrates (g)
- total fat (g)

Return ONLY a JSON object with these keys:
- "calories"
- "protein_g"
- "carbs_g"
- "fat_g"

All values must be numbers, no units in the values.
If something is unclear, make a reasonable estimate.
If the user mentions items that look like these shorthands (even with minor typos), use the provided detail:
{note_block}
""".strip()

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text},
        ],
    )

    content = response.choices[0].message.content
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="Model returned invalid JSON")

    for key in ["calories", "protein_g", "carbs_g", "fat_g"]:
        if key not in data:
            raise HTTPException(status_code=500, detail=f"Missing key in model response: {key}")

    return data


def call_nutrition_model_image(image_bytes: bytes, content_type: str, notes: list[Note]) -> dict:
    if not content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    base64_image = base64.b64encode(image_bytes).decode("utf-8")
    data_url = f"data:{content_type};base64,{base64_image}"

    note_block = notes_to_block(notes)

    system_prompt = f"""
You are a nutrition assistant.

Given a photo of everything a person ate or drank, estimate:
- total calories (kcal)
- total protein (g)
- total carbohydrates (g)
- total fat (g)

Also provide a short human-readable description of what you see.

Return ONLY a JSON object with these keys:
- "description"
- "calories"
- "protein_g"
- "carbs_g"
- "fat_g"

All values must be numbers for macros, no units in the values. Description is free text.
If something is unclear, make a reasonable estimate.
If the items resemble these shorthands (even with minor typos), use the provided detail:
{note_block}
""".strip()

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Estimate nutrition for this meal photo. If multiple foods, sum totals."},
                    {
                        "type": "image_url",
                        "image_url": {"url": data_url, "detail": "high"},
                    },
                ],
            },
        ],
    )

    content = response.choices[0].message.content
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="Model returned invalid JSON")

    for key in ["calories", "protein_g", "carbs_g", "fat_g"]:
        if key not in data:
            raise HTTPException(status_code=500, detail=f"Missing key in model response: {key}")

    return data


def get_dashboard(db: Session) -> DashboardOut:
    today_start, today_end = get_today_range()
    last7_start, last7_end = get_last7_range()

    today_stats = aggregate_stats(db, today_start, today_end)
    last7_stats = average_stats_on_logged_days(db, last7_start, last7_end)

    today_entries = (
        db.query(Entry)
        .filter(Entry.ts >= today_start, Entry.ts < today_end)
        .order_by(Entry.ts.desc())
        .all()
    )

    entries_out = [
        EntryOut(
            id=e.id,
            ts=e.ts,
            description=e.description,
            calories=e.calories,
            protein=e.protein,
            carbs=e.carbs,
            fat=e.fat,
        )
        for e in today_entries
    ]

    return DashboardOut(
        today_stats=today_stats,
        last7_stats=last7_stats,
        today_entries=entries_out,
    )


# ---------- Routes ----------

@app.get("/", response_class=HTMLResponse)
async def index():
    with open("static/index.html", "r", encoding="utf-8") as f:
        return f.read()


@app.post("/api/entries", response_model=DashboardOut)
async def add_entry(payload: AddEntryRequest, request: Request):
    if not payload.text.strip():
        raise HTTPException(status_code=400, detail="Text is empty")

    db = next(get_db())
    notes = db.query(Note).all()
    original_text = payload.text.strip()
    model_data = call_nutrition_model(original_text, notes)

    now_utc = datetime.now(timezone.utc)
    entry = Entry(
        ts=now_utc,
        description=original_text,
        calories=float(model_data["calories"]),
        protein=float(model_data["protein_g"]),
        carbs=float(model_data["carbs_g"]),
        fat=float(model_data["fat_g"]),
        raw_response=json.dumps(model_data),
    )

    db.add(entry)
    db.commit()

    return get_dashboard(db)


@app.post("/api/entries/photo", response_model=DashboardOut)
async def add_entry_photo(file: UploadFile = File(...)):
    if not file:
        raise HTTPException(status_code=400, detail="No file uploaded")
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty image")

    db = next(get_db())
    notes = db.query(Note).all()
    model_data = call_nutrition_model_image(image_bytes, file.content_type, notes)

    now_utc = datetime.now(timezone.utc)
    description = model_data.get("description") or "Photo entry"
    entry = Entry(
        ts=now_utc,
        description=description,
        calories=float(model_data["calories"]),
        protein=float(model_data["protein_g"]),
        carbs=float(model_data["carbs_g"]),
        fat=float(model_data["fat_g"]),
        raw_response=json.dumps(model_data),
    )

    db.add(entry)
    db.commit()

    return get_dashboard(db)


@app.delete("/api/entries/{entry_id}", response_model=DashboardOut)
async def delete_entry(entry_id: int):
    db = next(get_db())
    entry = db.query(Entry).filter(Entry.id == entry_id).first()
    if not entry:
        raise HTTPException(status_code=404, detail="Entry not found")

    db.delete(entry)
    db.commit()

    return get_dashboard(db)


@app.get("/api/dashboard", response_model=DashboardOut)
async def dashboard():
    db = next(get_db())
    return get_dashboard(db)


@app.get("/api/day", response_model=DayOut)
async def day_view(day: str | None = None):
    db = next(get_db())
    try:
        target_date = date.fromisoformat(day) if day else date.today()
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD.")

    start, end = get_date_range(target_date)
    stats = aggregate_stats(db, start, end)
    entries = (
        db.query(Entry)
        .filter(Entry.ts >= start, Entry.ts < end)
        .order_by(Entry.ts.desc())
        .all()
    )
    entries_out = [
        EntryOut(
            id=e.id,
            ts=e.ts,
            description=e.description,
            calories=e.calories,
            protein=e.protein,
            carbs=e.carbs,
            fat=e.fat,
        )
        for e in entries
    ]

    return DayOut(date=target_date, stats=stats, entries=entries_out)


@app.get("/api/notes", response_model=list[NoteOut])
async def list_notes():
    db = next(get_db())
    notes = db.query(Note).order_by(Note.title.asc()).all()
    return [note_to_out(n) for n in notes]


@app.post("/api/notes", response_model=list[NoteOut])
async def add_note(payload: AddNoteRequest):
    title = payload.title.strip()
    content = payload.content.strip()
    if not title or not content:
        raise HTTPException(status_code=400, detail="Title and content are required")

    db = next(get_db())
    existing = db.query(Note).filter(func.lower(Note.title) == title.lower()).first()
    if existing:
        existing.content = content
        existing.created_at = datetime.now(timezone.utc)
    else:
        note = Note(title=title, content=content, created_at=datetime.now(timezone.utc))
        db.add(note)

    db.commit()
    return await list_notes()


@app.put("/api/notes/{note_id}", response_model=list[NoteOut])
async def update_note(note_id: int, payload: AddNoteRequest):
    title = payload.title.strip()
    content = payload.content.strip()
    if not title or not content:
        raise HTTPException(status_code=400, detail="Title and content are required")

    db = next(get_db())
    note = db.query(Note).filter(Note.id == note_id).first()
    if not note:
        raise HTTPException(status_code=404, detail="Note not found")

    duplicate = (
        db.query(Note)
        .filter(func.lower(Note.title) == title.lower(), Note.id != note_id)
        .first()
    )
    if duplicate:
        raise HTTPException(status_code=400, detail="Another note with this title exists")

    note.title = title
    note.content = content
    note.created_at = datetime.now(timezone.utc)
    db.commit()
    return await list_notes()


@app.delete("/api/notes/{note_id}", response_model=list[NoteOut])
async def delete_note(note_id: int):
    db = next(get_db())
    note = db.query(Note).filter(Note.id == note_id).first()
    if not note:
        raise HTTPException(status_code=404, detail="Note not found")

    db.delete(note)
    db.commit()
    return await list_notes()
