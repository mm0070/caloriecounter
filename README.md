# Calorie Tracker

FastAPI + simple HTML front-end to log daily food, get macro estimates with OpenAI, view history by date, and manage shorthand notes that inform the model.
Now also tracks alcohol units and shows totals for specific days and the last 7 days.

## Requirements
- Python 3.11+
- `OPENAI_API_KEY` in environment (see `.env.example` if present)

## Setup
```bash
pip install -r requirements.txt
```

## Run
```bash
cd app
uvicorn main:app --reload --port 8000 --env-file ../.env  # adjust --env-file path as needed
```
Then open `http://localhost:8000/`.

## Seed fake data (for testing)
```bash
cd app
python seed_fake_data.py --days 14 --reset  # populates today minus 14 days; omit --reset to append
```

## Mood tracking
- Set a daily mood (1–10 with optional note) on the Today page or History page; mood shows up in the calendar view for that day.

## Endpoints (high level)
- `POST /api/entries` — add entry (uses notes as context for the model); optional `ts` ISO timestamp to backdate
- `POST /api/entries/photo` — add entry from an uploaded image (supports optional `ts` form field to backdate)
- `GET /api/dashboard` — today + last-7-day averages (alcohol is total), 30-day alcohol total
- `GET /api/day?day=YYYY-MM-DD` — stats/entries for a specific date
- `GET /api/calendar?month=YYYY-MM` — per-day totals for a month (for calendar view)
- `PUT /api/entries/{id}` — update an entry’s description/macros/timestamp
- `DELETE /api/entries/{id}` — remove an entry
- `GET/POST/PUT/DELETE /api/notes` — manage shorthand notes

## Pages
- `static/index.html` — today view; add entry via text or photo upload
- `static/history.html` — select a date to review entries and totals
- `static/notes.html` — manage shorthand notes sent to the model
