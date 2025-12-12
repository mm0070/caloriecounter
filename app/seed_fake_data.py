import argparse
import os
import random
from datetime import date, datetime, timezone, timedelta

# Avoid import-time failure from OPENAI_API_KEY requirement
os.environ.setdefault("OPENAI_API_KEY", "test-key")

import main  # noqa: E402


def seed(days: int, reset: bool) -> None:
    session = next(main.get_db())

    if reset:
        session.query(main.Entry).delete()
        session.commit()

    today = date.today()
    # Oldest to newest for deterministic assignment
    day_dates = [today - timedelta(days=offset) for offset in reversed(range(days))]

    high_cal_day = day_dates[0] if day_dates else None
    low_protein_day = day_dates[1] if len(day_dates) > 1 else None

    entries = []
    for day in day_dates:
        # Ensure at least one high-calorie day
        if day == high_cal_day:
            calories = random.randint(3000, 3400)
            protein = random.randint(160, 190)
        # Ensure at least one lower-protein day
        elif day == low_protein_day:
            calories = random.randint(1900, 2400)
            protein = random.randint(90, 140)
        else:
            calories = random.randint(1800, 3300)
            protein = random.randint(110, 180)

        carbs = random.randint(180, 380)
        fat = random.randint(60, 140)
        alcohol_units = random.choice([0, 0, 0.5, 1, 2, 3])

        # Drop entries at random times during the day
        hour = random.randint(8, 20)
        minute = random.randint(0, 59)
        ts = datetime.combine(day, datetime.min.time()) + timedelta(hours=hour, minutes=minute)
        ts = ts.replace(tzinfo=timezone.utc)

        entries.append(
            main.Entry(
                ts=ts,
                description=f"Seeded day {day.isoformat()}",
                calories=float(calories),
                protein=float(protein),
                carbs=float(carbs),
                fat=float(fat),
                alcohol_units=float(alcohol_units),
                raw_response="",
            )
        )

    session.add_all(entries)
    session.commit()
    print(f"Inserted {len(entries)} entries spanning {days} day(s) ending {today.isoformat()}.")


def parse_args():
    parser = argparse.ArgumentParser(description="Seed calories.db with fake data.")
    parser.add_argument("--days", type=int, default=14, help="Number of days to seed (ending today).")
    parser.add_argument("--reset", action="store_true", help="Delete existing entries before seeding.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.days < 1:
        raise SystemExit("Days must be >= 1")
    seed(args.days, args.reset)
