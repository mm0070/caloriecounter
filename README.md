# Calorie Tracker

FastAPI + simple HTML front-end to log daily food, get macro estimates with OpenAI, view history by date, and manage shorthand notes that inform the model.

## Requirements
- Python 3.11+
- `OPENAI_API_KEY` in environment (see `.env.example` if present)

## Setup
```bash
pip install -r requirements.txt
```

## Run
```bash
uvicorn app.main:app --reload
```
Then open `http://localhost:8000/static/index.html`.

## Endpoints (high level)
- `POST /api/entries` — add entry (uses notes as context for the model)
- `POST /api/entries/photo` — add entry from an uploaded image
- `GET /api/dashboard` — today + last-7-day averages
- `GET /api/day?day=YYYY-MM-DD` — stats/entries for a specific date
- `DELETE /api/entries/{id}` — remove an entry
- `GET/POST/PUT/DELETE /api/notes` — manage shorthand notes

## Pages
- `static/index.html` — today view; add entry via text or photo upload
- `static/history.html` — select a date to review entries and totals
- `static/notes.html` — manage shorthand notes sent to the model
