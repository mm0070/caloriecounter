import os
import json
import base64
from datetime import datetime, timedelta, timezone, date

from fastapi import FastAPI, HTTPException, Request, UploadFile, File, Form, Response
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
from sqlalchemy import (
    create_engine, Column, Integer, Float, Text, DateTime, Date, func, text, UniqueConstraint
)
from sqlalchemy.orm import sessionmaker, declarative_base, Session

from openai import OpenAI
from app.prompts import (
    nutrition_image_prompt,
    nutrition_text_prompt,
    weekly_review_system_prompt,
    weekly_review_user_prompt,
)

# ---------- Config ----------

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("Set OPENAI_API_KEY in your environment.")

client = OpenAI(api_key=OPENAI_API_KEY)

DB_URL = "sqlite:///./calories.db"
DEFAULT_USER_NAME = "mihau"

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
    user_id = Column(Integer, index=True, nullable=False, default=1)
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
    __table_args__ = (UniqueConstraint("user_id", "title", name="uix_notes_user_title"),)

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, index=True, nullable=False, default=1)
    title = Column(Text, nullable=False)
    content = Column(Text, nullable=False)
    created_at = Column(DateTime(timezone=True), nullable=False, default=datetime.now(timezone.utc))


class Mood(Base):
    __tablename__ = "moods"
    __table_args__ = (UniqueConstraint("user_id", "date", name="uix_moods_user_date"),)

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, index=True, nullable=False, default=1)
    date = Column(Date, nullable=False, index=True)
    mood_score = Column(Integer, nullable=False)
    mood_note = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, default=datetime.now(timezone.utc))


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(Text, nullable=False, unique=True)


Base.metadata.create_all(bind=engine)


def ensure_alcohol_units_column():
    with engine.connect() as conn:
        cols = [row[1] for row in conn.execute(text("PRAGMA table_info(entries)"))]
        if "alcohol_units" not in cols:
            conn.execute(text("ALTER TABLE entries ADD COLUMN alcohol_units FLOAT NOT NULL DEFAULT 0"))


ensure_alcohol_units_column()


# ---------- Multi-user schema helpers ----------


def ensure_users_table_and_default():
    Base.metadata.create_all(bind=engine, tables=[User.__table__])
    db = SessionLocal()
    try:
        existing_target = db.query(User).filter(func.lower(User.name) == DEFAULT_USER_NAME.lower()).first()
        if existing_target:
            return

        legacy = db.query(User).filter(func.lower(User.name) == "you").first()
        if legacy:
            legacy.name = DEFAULT_USER_NAME
            db.commit()
            return

        any_user = db.query(User).first()
        if not any_user:
            user = User(name=DEFAULT_USER_NAME)
            db.add(user)
            db.commit()
    finally:
        db.close()


def get_default_user_id() -> int:
    db = SessionLocal()
    try:
        user = db.query(User).filter(func.lower(User.name) == DEFAULT_USER_NAME.lower()).first()
        if user:
            return user.id

        legacy = db.query(User).filter(func.lower(User.name) == "you").first()
        if legacy:
            legacy.name = DEFAULT_USER_NAME
            db.commit()
            return legacy.id

        user = User(name=DEFAULT_USER_NAME)
        db.add(user)
        db.commit()
        db.refresh(user)
        return user.id
    finally:
        db.close()


def ensure_user_id_column(table_name: str, default_user_id: int):
    with engine.connect() as conn:
        cols = [row[1] for row in conn.execute(text(f"PRAGMA table_info({table_name})"))]
        if "user_id" not in cols:
            conn.execute(text(f"ALTER TABLE {table_name} ADD COLUMN user_id INTEGER NOT NULL DEFAULT {default_user_id}"))


def migrate_table_with_user_and_unique(table_name: str, columns_sql: str, unique_sql: str, default_user_id: int):
    with engine.connect() as conn:
        cols = [row[1] for row in conn.execute(text(f"PRAGMA table_info({table_name})"))]
        if "user_id" in cols:
            return

        conn.execute(text("PRAGMA foreign_keys=off"))
        temp_table = f"{table_name}_new"
        conn.execute(text(f"CREATE TABLE {temp_table} ({columns_sql})"))
        conn.execute(
            text(
                f"INSERT INTO {temp_table} SELECT *, {default_user_id} as user_id FROM {table_name}"
            )
        )
        conn.execute(text(f"DROP TABLE {table_name}"))
        conn.execute(text(f"ALTER TABLE {temp_table} RENAME TO {table_name}"))
        if unique_sql:
            conn.execute(text(unique_sql))
        conn.execute(text("PRAGMA foreign_keys=on"))


def ensure_multiuser_schema():
    ensure_users_table_and_default()
    default_user_id = get_default_user_id()
    # entries: simple add column
    ensure_user_id_column("entries", default_user_id)
    ensure_alcohol_units_column()
    with engine.connect() as conn:
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_entries_user_id ON entries(user_id)"))

    # moods: add user_id if missing, then ensure unique/index
    ensure_user_id_column("moods", default_user_id)
    with engine.connect() as conn:
        conn.execute(text("CREATE UNIQUE INDEX IF NOT EXISTS uix_moods_user_date ON moods(user_id, date)"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_moods_user_id ON moods(user_id)"))

    # notes: add user_id if missing, then ensure unique/index
    ensure_user_id_column("notes", default_user_id)
    with engine.connect() as conn:
        conn.execute(text("CREATE UNIQUE INDEX IF NOT EXISTS uix_notes_user_title ON notes(user_id, lower(title))"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_notes_user_id ON notes(user_id)"))


ensure_multiuser_schema()


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


class UserOut(BaseModel):
    id: int
    name: str


class AddUserRequest(BaseModel):
    name: str


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


class WeeklyReviewOut(BaseModel):
    review: str
    generated_at: datetime
    range_start: date
    range_end: date
    days_with_entries: int
    total_entries: int


# ---------- Helpers ----------

def get_db() -> Session:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_user_id_from_request(request: Request, db: Session) -> int:
    header_val = request.headers.get("X-User-Id")
    if header_val is None:
        # default user fallback
        return get_default_user_id()
    try:
        user_id = int(header_val)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid X-User-Id header")

    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user.id


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


def get_days_since_last_alcohol(db: Session, user_id: int) -> int | None:
    last_alcohol_entry = (
        db.query(Entry)
        .filter(Entry.user_id == user_id)
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


def aggregate_stats(db: Session, user_id: int, start: datetime, end: datetime) -> StatsOut:
    result = (
        db.query(
            func.coalesce(func.sum(Entry.calories), 0),
            func.coalesce(func.sum(Entry.protein), 0),
            func.coalesce(func.sum(Entry.carbs), 0),
            func.coalesce(func.sum(Entry.fat), 0),
            func.coalesce(func.sum(Entry.alcohol_units), 0),
        )
        .filter(Entry.ts >= start, Entry.ts < end, Entry.user_id == user_id)
        .one()
    )
    return StatsOut(
        total_calories=float(result[0] or 0),
        total_protein=float(result[1] or 0),
        total_carbs=float(result[2] or 0),
        total_fat=float(result[3] or 0),
        total_alcohol_units=float(result[4] or 0),
    )


def average_stats_on_logged_days(db: Session, user_id: int, start: datetime, end: datetime) -> StatsOut:
    daily_totals = (
        db.query(
            func.date(Entry.ts),
            func.coalesce(func.sum(Entry.calories), 0),
            func.coalesce(func.sum(Entry.protein), 0),
            func.coalesce(func.sum(Entry.carbs), 0),
            func.coalesce(func.sum(Entry.fat), 0),
            func.coalesce(func.sum(Entry.alcohol_units), 0),
        )
        .filter(Entry.ts >= start, Entry.ts < end, Entry.user_id == user_id)
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


def build_last7_review_context(db: Session, user_id: int):
    start, end = get_last7_range()
    # Stats per day
    entry_rows = (
        db.query(
            func.date(Entry.ts),
            func.coalesce(func.sum(Entry.calories), 0),
            func.coalesce(func.sum(Entry.protein), 0),
            func.coalesce(func.sum(Entry.carbs), 0),
            func.coalesce(func.sum(Entry.fat), 0),
            func.coalesce(func.sum(Entry.alcohol_units), 0),
        )
        .filter(Entry.ts >= start, Entry.ts < end, Entry.user_id == user_id)
        .group_by(func.date(Entry.ts))
        .all()
    )
    stats_map = {
        date.fromisoformat(row[0]): {
            "calories": float(row[1]),
            "protein": float(row[2]),
            "carbs": float(row[3]),
            "fat": float(row[4]),
            "alcohol": float(row[5]),
        }
        for row in entry_rows
    }

    total_entries = int(
        db.query(func.count(Entry.id))
        .filter(Entry.ts >= start, Entry.ts < end, Entry.user_id == user_id)
        .scalar()
        or 0
    )
    days_with_entries = len(stats_map)

    mood_rows = (
        db.query(Mood)
        .filter(Mood.date >= start.date(), Mood.date < end.date(), Mood.user_id == user_id)
        .all()
    )
    mood_map = {m.date: m for m in mood_rows}

    days = [start.date() + timedelta(days=i) for i in range((end - start).days)]
    daily_lines = []
    for d in days:
        stats = stats_map.get(d)
        calories = stats["calories"] if stats else 0.0
        protein = stats["protein"] if stats else 0.0
        carbs = stats["carbs"] if stats else 0.0
        fat = stats["fat"] if stats else 0.0
        alcohol = stats["alcohol"] if stats else 0.0
        mood = mood_map.get(d)
        mood_part = f", Mood {mood.mood_score}/10" if mood else ""
        suffix = "" if stats else " (no entries)"
        daily_lines.append(
            f"{d.isoformat()}: {calories:.0f} kcal, P {protein:.1f}g, C {carbs:.1f}g, F {fat:.1f}g, Alc {alcohol:.1f}u{mood_part}{suffix}"
        )

    avg_stats = average_stats_on_logged_days(db, user_id, start, end)
    total_stats = aggregate_stats(db, user_id, start, end)

    return {
        "start": start,
        "end": end,
        "days": days,
        "daily_lines": daily_lines,
        "avg_stats": avg_stats,
        "total_alcohol_units": total_stats.total_alcohol_units,
        "days_with_entries": days_with_entries,
        "total_entries": total_entries,
    }


def generate_last7_review(db: Session, user_id: int) -> WeeklyReviewOut:
    context = build_last7_review_context(db, user_id)
    start = context["start"]
    end = context["end"]
    range_start = context["days"][0] if context["days"] else start.date()
    range_end = context["days"][-1] if context["days"] else (end - timedelta(days=1)).date()
    if context["total_entries"] == 0:
        return WeeklyReviewOut(
            review="No entries in the last 7 completed days, so there is nothing to review yet.",
            generated_at=datetime.now(timezone.utc),
            range_start=range_start,
            range_end=range_end,
            days_with_entries=0,
            total_entries=0,
        )

    daily_block = "\n".join(context["daily_lines"]) or "No daily data."
    avg = context["avg_stats"]

    user_prompt = weekly_review_user_prompt(
        daily_block,
        avg.total_calories,
        avg.total_protein,
        avg.total_carbs,
        avg.total_fat,
        context["total_alcohol_units"],
        context["days_with_entries"],
        context["total_entries"],
    )

    try:
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            temperature=0.2,
            messages=[
                {
                    "role": "system",
                    "content": weekly_review_system_prompt(),
                },
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ],
        )
    except Exception:
        raise HTTPException(status_code=500, detail="Failed to generate weekly review")

    review_text = (response.choices[0].message.content or "").strip()
    if not review_text:
        raise HTTPException(status_code=500, detail="Model returned empty review")

    return WeeklyReviewOut(
        review=review_text,
        generated_at=datetime.now(timezone.utc),
        range_start=range_start,
        range_end=range_end,
        days_with_entries=context["days_with_entries"],
        total_entries=context["total_entries"],
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

    system_prompt = nutrition_text_prompt(note_block)

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

    system_prompt = nutrition_image_prompt(note_block)

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


def get_dashboard(db: Session, user_id: int) -> DashboardOut:
    today_start, today_end = get_today_range()
    last7_start, last7_end = get_last7_range()
    last30_start, last30_end = get_last30_range()

    today_stats = aggregate_stats(db, user_id, today_start, today_end)
    last7_avg = average_stats_on_logged_days(db, user_id, last7_start, last7_end)
    last7_total = aggregate_stats(db, user_id, last7_start, last7_end)
    last7_stats = StatsOut(
        total_calories=last7_avg.total_calories,
        total_protein=last7_avg.total_protein,
        total_carbs=last7_avg.total_carbs,
        total_fat=last7_avg.total_fat,
        total_alcohol_units=last7_total.total_alcohol_units,
    )
    last30_stats = aggregate_stats(db, user_id, last30_start, last30_end)
    days_since_last_alcohol = get_days_since_last_alcohol(db, user_id)

    today_entries = (
        db.query(Entry)
        .filter(Entry.ts >= today_start, Entry.ts < today_end, Entry.user_id == user_id)
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


def get_day_data(db: Session, user_id: int, target_date: date) -> DayOut:
    start, end = get_date_range(target_date)
    stats = aggregate_stats(db, user_id, start, end)
    mood = db.query(Mood).filter(Mood.date == target_date, Mood.user_id == user_id).first()
    entries = (
        db.query(Entry)
        .filter(Entry.ts >= start, Entry.ts < end, Entry.user_id == user_id)
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
    user_id = get_user_id_from_request(request, db)
    notes = db.query(Note).filter(Note.user_id == user_id).all()
    original_text = payload.text.strip()
    model_data = call_nutrition_model(original_text, notes)

    if payload.ts:
        entry_ts = payload.ts
        if entry_ts.tzinfo is None:
            entry_ts = entry_ts.replace(tzinfo=timezone.utc)
    else:
        entry_ts = datetime.now(timezone.utc)

    entry = Entry(
        user_id=user_id,
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

    return get_dashboard(db, user_id)


@app.post("/api/entries/photo", response_model=DashboardOut)
async def add_entry_photo(request: Request, file: UploadFile = File(...), ts: datetime | None = Form(None)):
    if not file:
        raise HTTPException(status_code=400, detail="No file uploaded")
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty image")

    db = next(get_db())
    user_id = get_user_id_from_request(request, db)
    notes = db.query(Note).filter(Note.user_id == user_id).all()
    model_data = call_nutrition_model_image(image_bytes, file.content_type, notes)

    if ts:
        entry_ts = ts
        if entry_ts.tzinfo is None:
            entry_ts = entry_ts.replace(tzinfo=timezone.utc)
    else:
        entry_ts = datetime.now(timezone.utc)

    description = model_data.get("description") or "Photo entry"
    entry = Entry(
        user_id=user_id,
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

    return get_dashboard(db, user_id)


@app.put("/api/entries/{entry_id}", response_model=EntryOut)
async def update_entry(entry_id: int, payload: UpdateEntryRequest, request: Request):
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
    user_id = get_user_id_from_request(request, db)
    entry = db.query(Entry).filter(Entry.id == entry_id, Entry.user_id == user_id).first()
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
async def delete_entry(entry_id: int, request: Request):
    db = next(get_db())
    user_id = get_user_id_from_request(request, db)
    entry = db.query(Entry).filter(Entry.id == entry_id, Entry.user_id == user_id).first()
    if not entry:
        raise HTTPException(status_code=404, detail="Entry not found")

    db.delete(entry)
    db.commit()

    return get_dashboard(db, user_id)


@app.get("/api/dashboard", response_model=DashboardOut)
async def dashboard(request: Request):
    db = next(get_db())
    user_id = get_user_id_from_request(request, db)
    return get_dashboard(db, user_id)


@app.get("/api/review/last7", response_model=WeeklyReviewOut)
async def last7_review(request: Request):
    db = next(get_db())
    user_id = get_user_id_from_request(request, db)
    return generate_last7_review(db, user_id)


@app.get("/api/day", response_model=DayOut)
async def day_view(day: str | None = None, request: Request = None):
    db = next(get_db())
    user_id = get_user_id_from_request(request, db)
    try:
        target_date = date.fromisoformat(day) if day else date.today()
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD.")

    return get_day_data(db, user_id, target_date)


@app.put("/api/mood", response_model=MoodOut)
async def upsert_mood(payload: MoodRequest, request: Request):
    db = next(get_db())
    user_id = get_user_id_from_request(request, db)
    if payload.mood_score < 1 or payload.mood_score > 10:
        raise HTTPException(status_code=400, detail="mood_score must be between 1 and 10")

    mood = db.query(Mood).filter(Mood.date == payload.date, Mood.user_id == user_id).first()
    note = payload.mood_note.strip() if payload.mood_note else None

    if mood:
        mood.mood_score = payload.mood_score
        mood.mood_note = note
    else:
        mood = Mood(date=payload.date, mood_score=payload.mood_score, mood_note=note, user_id=user_id)
        db.add(mood)

    db.commit()
    db.refresh(mood)
    return mood_to_out(mood)


@app.get("/api/mood", response_model=MoodOut | None)
async def get_mood(day: str | None = None, request: Request = None):
    db = next(get_db())
    user_id = get_user_id_from_request(request, db)
    try:
        target_date = date.fromisoformat(day) if day else date.today()
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD.")

    mood = db.query(Mood).filter(Mood.date == target_date, Mood.user_id == user_id).first()
    return mood_to_out(mood) if mood else None


@app.delete("/api/mood", status_code=204)
async def delete_mood(day: str, request: Request):
    db = next(get_db())
    user_id = get_user_id_from_request(request, db)
    try:
        target_date = date.fromisoformat(day)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD.")

    mood = db.query(Mood).filter(Mood.date == target_date, Mood.user_id == user_id).first()
    if mood:
        db.delete(mood)
        db.commit()
    return Response(status_code=204)


@app.get("/api/calendar", response_model=list[CalendarDayOut])
async def calendar(month: str | None = None, request: Request = None):
    db = next(get_db())
    user_id = get_user_id_from_request(request, db)
    first_day, start_dt, end_dt = get_month_range(month)
    moods = (
        db.query(Mood)
        .filter(Mood.date >= first_day, Mood.date < end_dt.date(), Mood.user_id == user_id)
        .all()
    )
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
        .filter(Entry.ts >= start_dt, Entry.ts < end_dt, Entry.user_id == user_id)
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
async def last7_series(request: Request):
    db = next(get_db())
    user_id = get_user_id_from_request(request, db)
    start, end = get_last7_range()  # start = today-7, end = today (excludes current day)
    # Stats per day
    entry_rows = (
        db.query(
            func.date(Entry.ts),
            func.coalesce(func.sum(Entry.calories), 0),
            func.coalesce(func.sum(Entry.protein), 0),
        )
        .filter(Entry.ts >= start, Entry.ts < end, Entry.user_id == user_id)
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
    mood_rows = (
        db.query(Mood)
        .filter(Mood.date >= start.date(), Mood.date < end.date(), Mood.user_id == user_id)
        .all()
    )
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
async def list_notes(request: Request):
    db = next(get_db())
    user_id = get_user_id_from_request(request, db)
    notes = db.query(Note).filter(Note.user_id == user_id).order_by(Note.title.asc()).all()
    return [note_to_out(n) for n in notes]


@app.post("/api/notes", response_model=list[NoteOut])
async def add_note(payload: AddNoteRequest, request: Request):
    title = payload.title.strip()
    content = payload.content.strip()
    if not title or not content:
        raise HTTPException(status_code=400, detail="Title and content are required")

    db = next(get_db())
    user_id = get_user_id_from_request(request, db)
    existing = (
        db.query(Note)
        .filter(Note.user_id == user_id, func.lower(Note.title) == title.lower())
        .first()
    )
    if existing:
        existing.content = content
        existing.created_at = datetime.now(timezone.utc)
    else:
        note = Note(
            title=title,
            content=content,
            created_at=datetime.now(timezone.utc),
            user_id=user_id,
        )
        db.add(note)

    db.commit()
    return await list_notes(request)


@app.put("/api/notes/{note_id}", response_model=list[NoteOut])
async def update_note(note_id: int, payload: AddNoteRequest, request: Request):
    title = payload.title.strip()
    content = payload.content.strip()
    if not title or not content:
        raise HTTPException(status_code=400, detail="Title and content are required")

    db = next(get_db())
    user_id = get_user_id_from_request(request, db)
    note = db.query(Note).filter(Note.id == note_id, Note.user_id == user_id).first()
    if not note:
        raise HTTPException(status_code=404, detail="Note not found")

    duplicate = (
        db.query(Note)
        .filter(func.lower(Note.title) == title.lower(), Note.id != note_id, Note.user_id == user_id)
        .first()
    )
    if duplicate:
        raise HTTPException(status_code=400, detail="Another note with this title exists")

    note.title = title
    note.content = content
    note.created_at = datetime.now(timezone.utc)
    db.commit()
    return await list_notes(request)


@app.delete("/api/notes/{note_id}", response_model=list[NoteOut])
async def delete_note(note_id: int, request: Request):
    db = next(get_db())
    user_id = get_user_id_from_request(request, db)
    note = db.query(Note).filter(Note.id == note_id, Note.user_id == user_id).first()
    if not note:
        raise HTTPException(status_code=404, detail="Note not found")

    db.delete(note)
    db.commit()
    return await list_notes(request)


# ---------- Users ----------


def user_to_out(user: User) -> UserOut:
    return UserOut(id=user.id, name=user.name)


@app.get("/api/users", response_model=list[UserOut])
async def list_users():
    db = next(get_db())
    users = db.query(User).order_by(User.id.asc()).all()
    return [user_to_out(u) for u in users]


@app.post("/api/users", response_model=list[UserOut])
async def create_user(payload: AddUserRequest):
    name = payload.name.strip()
    if not name:
        raise HTTPException(status_code=400, detail="Name is required")
    db = next(get_db())
    existing = db.query(User).filter(func.lower(User.name) == name.lower()).first()
    if existing:
        raise HTTPException(status_code=400, detail="User with this name already exists")
    user = User(name=name)
    db.add(user)
    db.commit()
    users = db.query(User).order_by(User.id.asc()).all()
    return [user_to_out(u) for u in users]
