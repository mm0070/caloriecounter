import os
import json
import base64
from datetime import datetime, timedelta, timezone, date

from fastapi import FastAPI, HTTPException, Request, UploadFile, File, Form, Response
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
from sqlalchemy import (
    create_engine, Column, Integer, Float, Text, DateTime, Date, func, text
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
# Resolve static directory relative to this file so tests work regardless of cwd
STATIC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "static"))
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


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
    alcohol_units = Column(Float, nullable=False, default=0)
    raw_response = Column(Text, nullable=True)


class Note(Base):
    __tablename__ = "notes"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(Text, nullable=False, unique=True)
    content = Column(Text, nullable=False)
    created_at = Column(DateTime(timezone=True), nullable=False, default=datetime.now(timezone.utc))


class Mood(Base):
    __tablename__ = "moods"

    id = Column(Integer, primary_key=True, index=True)
    date = Column(Date, nullable=False, unique=True, index=True)
    mood_score = Column(Integer, nullable=False)
    mood_note = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, default=datetime.now(timezone.utc))


Base.metadata.create_all(bind=engine)


def ensure_alcohol_units_column():
    with engine.connect() as conn:
        cols = [row[1] for row in conn.execute(text("PRAGMA table_info(entries)"))]
        if "alcohol_units" not in cols:
            conn.execute(text("ALTER TABLE entries ADD COLUMN alcohol_units FLOAT NOT NULL DEFAULT 0"))


ensure_alcohol_units_column()


# ---------- Schemas ----------

class AddEntryRequest(BaseModel):
    text: str
    ts: datetime | None = None


class UpdateEntryRequest(BaseModel):
    description: str | None = None
    calories: float | None = None
    protein: float | None = None
    carbs: float | None = None
    fat: float | None = None
    alcohol_units: float | None = None
    ts: datetime | None = None


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
    alcohol_units: float


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
    total_alcohol_units: float


class MoodRequest(BaseModel):
    date: date
    mood_score: int
    mood_note: str | None = None


class MoodOut(BaseModel):
    date: date
    mood_score: int
    mood_note: str | None = None


class DashboardOut(BaseModel):
    today_stats: StatsOut
    last7_stats: StatsOut
    last30_alcohol_units: float
    days_since_last_alcohol: int | None = None
    today_entries: list[EntryOut]


class DayOut(BaseModel):
    date: date
    stats: StatsOut
    entries: list[EntryOut]
    mood: MoodOut | None = None


class CalendarDayOut(BaseModel):
    date: date
    total_calories: float
    total_protein: float
    total_carbs: float
    total_fat: float
    total_alcohol_units: float
    mood_score: int | None = None
    mood_note: str | None = None


class SeriesPoint(BaseModel):
    date: date
    calories: float
    protein: float
    mood_score: int | None = None


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
    # Only include fully completed days: last 7 days ending yesterday
    start = datetime.combine(today - timedelta(days=7), datetime.min.time())
    end = datetime.combine(today, datetime.min.time())
    start_utc = start.astimezone(timezone.utc) if start.tzinfo else start.replace(tzinfo=timezone.utc)
    end_utc = end.astimezone(timezone.utc) if end.tzinfo else end.replace(tzinfo=timezone.utc)
    return start_utc, end_utc


def get_last30_range():
    today = date.today()
    start = datetime.combine(today - timedelta(days=29), datetime.min.time())
    end = datetime.combine(today + timedelta(days=1), datetime.min.time())
    start_utc = start.astimezone(timezone.utc) if start.tzinfo else start.replace(tzinfo=timezone.utc)
    end_utc = end.astimezone(timezone.utc) if end.tzinfo else end.replace(tzinfo=timezone.utc)
    return start_utc, end_utc


def get_days_since_last_alcohol(db: Session) -> int | None:
    last_alcohol_entry = (
        db.query(Entry)
        .filter(Entry.alcohol_units > 0)
        .order_by(Entry.ts.desc())
        .first()
    )
    if not last_alcohol_entry:
        return None

    last_ts = last_alcohol_entry.ts
    if last_ts.tzinfo is None:
        last_ts = last_ts.replace(tzinfo=timezone.utc)
    last_date = last_ts.astimezone(timezone.utc).date()
    today = datetime.now(timezone.utc).date()
    return max((today - last_date).days, 0)


def empty_stats() -> StatsOut:
    return StatsOut(
        total_calories=0.0,
        total_protein=0.0,
        total_carbs=0.0,
        total_fat=0.0,
        total_alcohol_units=0.0,
    )


def aggregate_stats(db: Session, start: datetime, end: datetime) -> StatsOut:
    result = (
        db.query(
            func.coalesce(func.sum(Entry.calories), 0),
            func.coalesce(func.sum(Entry.protein), 0),
            func.coalesce(func.sum(Entry.carbs), 0),
            func.coalesce(func.sum(Entry.fat), 0),
            func.coalesce(func.sum(Entry.alcohol_units), 0),
        )
        .filter(Entry.ts >= start, Entry.ts < end)
        .one()
    )
    return StatsOut(
        total_calories=float(result[0] or 0),
        total_protein=float(result[1] or 0),
        total_carbs=float(result[2] or 0),
        total_fat=float(result[3] or 0),
        total_alcohol_units=float(result[4] or 0),
    )


def average_stats_on_logged_days(db: Session, start: datetime, end: datetime) -> StatsOut:
    daily_totals = (
        db.query(
            func.date(Entry.ts),
            func.coalesce(func.sum(Entry.calories), 0),
            func.coalesce(func.sum(Entry.protein), 0),
            func.coalesce(func.sum(Entry.carbs), 0),
            func.coalesce(func.sum(Entry.fat), 0),
            func.coalesce(func.sum(Entry.alcohol_units), 0),
        )
        .filter(Entry.ts >= start, Entry.ts < end)
        .group_by(func.date(Entry.ts))
        .all()
    )

    # Days appear in daily_totals only when at least one entry exists
    if not daily_totals:
        return empty_stats()

    day_count = len(daily_totals)
    total_calories = sum(float(day[1]) for day in daily_totals)
    total_protein = sum(float(day[2]) for day in daily_totals)
    total_carbs = sum(float(day[3]) for day in daily_totals)
    total_fat = sum(float(day[4]) for day in daily_totals)
    total_alcohol = sum(float(day[5]) for day in daily_totals)

    return StatsOut(
        total_calories=total_calories / day_count,
        total_protein=total_protein / day_count,
        total_carbs=total_carbs / day_count,
        total_fat=total_fat / day_count,
        total_alcohol_units=total_alcohol / day_count,
    )


def note_to_out(note: Note) -> NoteOut:
    return NoteOut(
        id=note.id,
        title=note.title,
        content=note.content,
        created_at=note.created_at,
    )


def mood_to_out(mood: Mood) -> MoodOut:
    return MoodOut(
        date=mood.date,
        mood_score=mood.mood_score,
        mood_note=mood.mood_note,
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
- total alcohol units (UK units)

If no alcohol is present, return 0 for alcohol_units.

Return ONLY a JSON object with these keys:
- "calories"
- "protein_g"
- "carbs_g"
- "fat_g"
- "alcohol_units"

All values must be numbers, no units in the values.
If something is unclear, make a reasonable estimate.
If the user mentions items that look like these shorthands (even with minor typos), use the provided detail:
{note_block}
""".strip()

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        response_format={"type": "json_object"},
        temperature=0,
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

    for key in ["calories", "protein_g", "carbs_g", "fat_g", "alcohol_units"]:
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
- total alcohol units (UK units)

Also provide a short human-readable description of what you see.

If no alcohol is present, return 0 for alcohol_units.

Return ONLY a JSON object with these keys:
- "description"
- "calories"
- "protein_g"
- "carbs_g"
- "fat_g"
- "alcohol_units"

All values must be numbers for macros, no units in the values. Description is free text.
If something is unclear, make a reasonable estimate.
If the items resemble these shorthands (even with minor typos), use the provided detail:
{note_block}
""".strip()

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        response_format={"type": "json_object"},
        temperature=0,
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

    for key in ["calories", "protein_g", "carbs_g", "fat_g", "alcohol_units"]:
        if key not in data:
            raise HTTPException(status_code=500, detail=f"Missing key in model response: {key}")

    return data


def get_dashboard(db: Session) -> DashboardOut:
    today_start, today_end = get_today_range()
    last7_start, last7_end = get_last7_range()
    last30_start, last30_end = get_last30_range()

    today_stats = aggregate_stats(db, today_start, today_end)
    last7_avg = average_stats_on_logged_days(db, last7_start, last7_end)
    last7_total = aggregate_stats(db, last7_start, last7_end)
    last7_stats = StatsOut(
        total_calories=last7_avg.total_calories,
        total_protein=last7_avg.total_protein,
        total_carbs=last7_avg.total_carbs,
        total_fat=last7_avg.total_fat,
        total_alcohol_units=last7_total.total_alcohol_units,
    )
    last30_stats = aggregate_stats(db, last30_start, last30_end)
    days_since_last_alcohol = get_days_since_last_alcohol(db)

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
            alcohol_units=e.alcohol_units,
        )
        for e in today_entries
    ]

    return DashboardOut(
        today_stats=today_stats,
        last7_stats=last7_stats,
        last30_alcohol_units=last30_stats.total_alcohol_units,
        days_since_last_alcohol=days_since_last_alcohol,
        today_entries=entries_out,
    )


def get_day_data(db: Session, target_date: date) -> DayOut:
    start, end = get_date_range(target_date)
    stats = aggregate_stats(db, start, end)
    mood = db.query(Mood).filter(Mood.date == target_date).first()
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
            alcohol_units=e.alcohol_units,
        )
        for e in entries
    ]

    return DayOut(
        date=target_date,
        stats=stats,
        entries=entries_out,
        mood=mood_to_out(mood) if mood else None,
    )


def get_month_range(target_month: str | None = None):
    today = date.today()
    if target_month:
        try:
            year, month = map(int, target_month.split("-"))
            start_date = date(year, month, 1)
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid month format. Use YYYY-MM.")
    else:
        start_date = date(today.year, today.month, 1)
    # compute first of next month
    if start_date.month == 12:
        next_month = date(start_date.year + 1, 1, 1)
    else:
        next_month = date(start_date.year, start_date.month + 1, 1)

    start_dt = datetime.combine(start_date, datetime.min.time()).replace(tzinfo=timezone.utc)
    end_dt = datetime.combine(next_month, datetime.min.time()).replace(tzinfo=timezone.utc)
    return start_date, start_dt, end_dt


# ---------- Routes ----------

@app.get("/", response_class=HTMLResponse)
async def index():
    index_path = os.path.join(STATIC_DIR, "index.html")
    with open(index_path, "r", encoding="utf-8") as f:
        return f.read()


@app.post("/api/entries", response_model=DashboardOut)
async def add_entry(payload: AddEntryRequest, request: Request):
    if not payload.text.strip():
        raise HTTPException(status_code=400, detail="Text is empty")

    db = next(get_db())
    notes = db.query(Note).all()
    original_text = payload.text.strip()
    model_data = call_nutrition_model(original_text, notes)

    if payload.ts:
        entry_ts = payload.ts
        if entry_ts.tzinfo is None:
            entry_ts = entry_ts.replace(tzinfo=timezone.utc)
    else:
        entry_ts = datetime.now(timezone.utc)

    entry = Entry(
        ts=entry_ts,
        description=original_text,
        calories=float(model_data["calories"]),
        protein=float(model_data["protein_g"]),
        carbs=float(model_data["carbs_g"]),
        fat=float(model_data["fat_g"]),
        alcohol_units=float(model_data.get("alcohol_units", 0)),
        raw_response=json.dumps(model_data),
    )

    db.add(entry)
    db.commit()

    return get_dashboard(db)


@app.post("/api/entries/photo", response_model=DashboardOut)
async def add_entry_photo(file: UploadFile = File(...), ts: datetime | None = Form(None)):
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

    if ts:
        entry_ts = ts
        if entry_ts.tzinfo is None:
            entry_ts = entry_ts.replace(tzinfo=timezone.utc)
    else:
        entry_ts = datetime.now(timezone.utc)

    description = model_data.get("description") or "Photo entry"
    entry = Entry(
        ts=entry_ts,
        description=description,
        calories=float(model_data["calories"]),
        protein=float(model_data["protein_g"]),
        carbs=float(model_data["carbs_g"]),
        fat=float(model_data["fat_g"]),
        alcohol_units=float(model_data.get("alcohol_units", 0)),
        raw_response=json.dumps(model_data),
    )

    db.add(entry)
    db.commit()

    return get_dashboard(db)


@app.put("/api/entries/{entry_id}", response_model=EntryOut)
async def update_entry(entry_id: int, payload: UpdateEntryRequest):
    if all(
        value is None
        for value in [
            payload.description,
            payload.calories,
            payload.protein,
            payload.carbs,
            payload.fat,
            payload.alcohol_units,
            payload.ts,
        ]
    ):
        raise HTTPException(status_code=400, detail="Provide at least one field to update")

    db = next(get_db())
    entry = db.query(Entry).filter(Entry.id == entry_id).first()
    if not entry:
        raise HTTPException(status_code=404, detail="Entry not found")

    if payload.description is not None:
        desc = payload.description.strip()
        if not desc:
            raise HTTPException(status_code=400, detail="Description cannot be empty")
        entry.description = desc

    def set_float(value: float | None, attr: str):
        if value is None:
            return
        try:
            number = float(value)
        except (TypeError, ValueError):
            raise HTTPException(status_code=400, detail=f"{attr.capitalize()} must be a number")
        setattr(entry, attr, number)

    set_float(payload.calories, "calories")
    set_float(payload.protein, "protein")
    set_float(payload.carbs, "carbs")
    set_float(payload.fat, "fat")
    set_float(payload.alcohol_units, "alcohol_units")

    if payload.ts is not None:
        new_ts = payload.ts
        if new_ts.tzinfo is None:
            new_ts = new_ts.replace(tzinfo=timezone.utc)
        entry.ts = new_ts

    db.commit()
    db.refresh(entry)

    return EntryOut(
        id=entry.id,
        ts=entry.ts,
        description=entry.description,
        calories=entry.calories,
        protein=entry.protein,
        carbs=entry.carbs,
        fat=entry.fat,
        alcohol_units=entry.alcohol_units,
    )


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

    return get_day_data(db, target_date)


@app.put("/api/mood", response_model=MoodOut)
async def upsert_mood(payload: MoodRequest):
    db = next(get_db())
    if payload.mood_score < 1 or payload.mood_score > 10:
        raise HTTPException(status_code=400, detail="mood_score must be between 1 and 10")

    mood = db.query(Mood).filter(Mood.date == payload.date).first()
    note = payload.mood_note.strip() if payload.mood_note else None

    if mood:
        mood.mood_score = payload.mood_score
        mood.mood_note = note
    else:
        mood = Mood(date=payload.date, mood_score=payload.mood_score, mood_note=note)
        db.add(mood)

    db.commit()
    db.refresh(mood)
    return mood_to_out(mood)


@app.get("/api/mood", response_model=MoodOut | None)
async def get_mood(day: str | None = None):
    db = next(get_db())
    try:
        target_date = date.fromisoformat(day) if day else date.today()
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD.")

    mood = db.query(Mood).filter(Mood.date == target_date).first()
    return mood_to_out(mood) if mood else None


@app.delete("/api/mood", status_code=204)
async def delete_mood(day: str):
    db = next(get_db())
    try:
        target_date = date.fromisoformat(day)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD.")

    mood = db.query(Mood).filter(Mood.date == target_date).first()
    if mood:
        db.delete(mood)
        db.commit()
    return Response(status_code=204)


@app.get("/api/calendar", response_model=list[CalendarDayOut])
async def calendar(month: str | None = None):
    db = next(get_db())
    first_day, start_dt, end_dt = get_month_range(month)
    moods = db.query(Mood).filter(Mood.date >= first_day, Mood.date < end_dt.date()).all()
    mood_map = {m.date: m for m in moods}
    entry_rows = (
        db.query(
            func.date(Entry.ts),
            func.coalesce(func.sum(Entry.calories), 0),
            func.coalesce(func.sum(Entry.protein), 0),
            func.coalesce(func.sum(Entry.carbs), 0),
            func.coalesce(func.sum(Entry.fat), 0),
            func.coalesce(func.sum(Entry.alcohol_units), 0),
        )
        .filter(Entry.ts >= start_dt, Entry.ts < end_dt)
        .group_by(func.date(Entry.ts))
        .all()
    )

    stats_map = {
        date.fromisoformat(row[0]): {
            "total_calories": float(row[1]),
            "total_protein": float(row[2]),
            "total_carbs": float(row[3]),
            "total_fat": float(row[4]),
            "total_alcohol_units": float(row[5]),
        }
        for row in entry_rows
    }

    all_dates = sorted(set(stats_map.keys()) | set(mood_map.keys()))

    return [
        CalendarDayOut(
            date=d,
            total_calories=stats_map.get(d, {}).get("total_calories", 0.0),
            total_protein=stats_map.get(d, {}).get("total_protein", 0.0),
            total_carbs=stats_map.get(d, {}).get("total_carbs", 0.0),
            total_fat=stats_map.get(d, {}).get("total_fat", 0.0),
            total_alcohol_units=stats_map.get(d, {}).get("total_alcohol_units", 0.0),
            mood_score=mood_map.get(d).mood_score if mood_map.get(d) else None,
            mood_note=mood_map.get(d).mood_note if mood_map.get(d) else None,
        )
        for d in all_dates
    ]


@app.get("/api/series/last7", response_model=list[SeriesPoint])
async def last7_series():
    db = next(get_db())
    start, end = get_last7_range()  # start = today-7, end = today (excludes current day)
    # Stats per day
    entry_rows = (
        db.query(
            func.date(Entry.ts),
            func.coalesce(func.sum(Entry.calories), 0),
            func.coalesce(func.sum(Entry.protein), 0),
        )
        .filter(Entry.ts >= start, Entry.ts < end)
        .group_by(func.date(Entry.ts))
        .all()
    )
    stats_map = {
        date.fromisoformat(row[0]): {
            "calories": float(row[1]),
            "protein": float(row[2]),
        }
        for row in entry_rows
    }

    # Moods per day
    mood_rows = db.query(Mood).filter(Mood.date >= start.date(), Mood.date < end.date()).all()
    mood_map = {m.date: m.mood_score for m in mood_rows}

    days = [start.date() + timedelta(days=i) for i in range((end - start).days)]

    series = []
    for d in days:
        series.append(
            SeriesPoint(
                date=d,
                calories=stats_map.get(d, {}).get("calories", 0.0),
                protein=stats_map.get(d, {}).get("protein", 0.0),
                mood_score=mood_map.get(d),
            )
        )
    return series


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
