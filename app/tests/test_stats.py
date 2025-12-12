import os
import unittest
from datetime import datetime, timezone, date
from unittest.mock import patch

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Prevent import-time failure when OPENAI_API_KEY is not set in test envs
os.environ.setdefault("OPENAI_API_KEY", "test-key")

import main  # noqa: E402


class StatsTests(unittest.TestCase):
    def setUp(self):
        # Use an isolated in-memory database for each test
        self.engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
        main.Base.metadata.create_all(bind=self.engine)
        TestingSessionLocal = sessionmaker(bind=self.engine, autoflush=False, autocommit=False)
        self.db = TestingSessionLocal()

    def tearDown(self):
        self.db.close()
        self.engine.dispose()

    def test_get_last7_range_excludes_today(self):
        class FakeDate(date):
            @classmethod
            def today(cls):
                return date(2025, 12, 12)

        with patch("main.date", FakeDate):
            start, end = main.get_last7_range()

        self.assertEqual(start, datetime(2025, 12, 5, tzinfo=timezone.utc))
        self.assertEqual(end, datetime(2025, 12, 12, tzinfo=timezone.utc))

    def test_average_stats_only_logged_days_are_counted(self):
        # Two logged days inside range; one gap day should be skipped
        entries = [
            main.Entry(
                ts=datetime(2025, 12, 1, 12, tzinfo=timezone.utc),
                description="Day1",
                calories=100,
                protein=10,
                carbs=20,
                fat=5,
                alcohol_units=0.5,
                raw_response="",
            ),
            main.Entry(
                ts=datetime(2025, 12, 3, 8, tzinfo=timezone.utc),
                description="Day3",
                calories=300,
                protein=30,
                carbs=40,
                fat=15,
                alcohol_units=1.5,
                raw_response="",
            ),
        ]
        self.db.add_all(entries)
        self.db.commit()

        start = datetime(2025, 11, 30, tzinfo=timezone.utc)
        end = datetime(2025, 12, 5, tzinfo=timezone.utc)

        stats = main.average_stats_on_logged_days(self.db, start, end)

        self.assertAlmostEqual(stats.total_calories, 200.0)
        self.assertAlmostEqual(stats.total_protein, 20.0)
        self.assertAlmostEqual(stats.total_carbs, 30.0)
        self.assertAlmostEqual(stats.total_fat, 10.0)
        self.assertAlmostEqual(stats.total_alcohol_units, 1.0)

    def test_average_stats_no_entries_returns_empty_stats(self):
        start = datetime(2025, 12, 1, tzinfo=timezone.utc)
        end = datetime(2025, 12, 2, tzinfo=timezone.utc)

        stats = main.average_stats_on_logged_days(self.db, start, end)

        self.assertEqual(stats.total_calories, 0.0)
        self.assertEqual(stats.total_protein, 0.0)
        self.assertEqual(stats.total_carbs, 0.0)
        self.assertEqual(stats.total_fat, 0.0)
        self.assertEqual(stats.total_alcohol_units, 0.0)


if __name__ == "__main__":
    unittest.main()
