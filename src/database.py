"""PostgreSQL helpers.

The database is a convenience, not a hard dependency. If PostgreSQL is not
running, every function degrades to the CSV files in data/processed/ so the
rest of the project still works.
"""

from __future__ import annotations

import pandas as pd

try:
    from src import config
except ImportError:  # running from inside src/
    import config


def _engine():
    from sqlalchemy import create_engine

    d = config.DB
    url = f"postgresql+psycopg2://{d['user']}:{d['password']}@{d['host']}:{d['port']}/{d['name']}"
    return create_engine(url)


def available() -> bool:
    """True if we can actually reach the database."""
    try:
        from sqlalchemy import text

        with _engine().connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception:
        return False


def save_table(df: pd.DataFrame, table: str) -> bool:
    """Write a dataframe to PostgreSQL. Returns False if the DB is unavailable."""
    try:
        df.to_sql(table, _engine(), if_exists="replace", index=False)
        print(f"  -> PostgreSQL: wrote {len(df):,} rows to '{table}'")
        return True
    except Exception as exc:
        print(f"  -> PostgreSQL unavailable ({type(exc).__name__}); skipping '{table}'")
        return False


def load_table(table: str, fallback_csv=None) -> pd.DataFrame:
    """Read a table from PostgreSQL, falling back to a CSV file."""
    try:
        return pd.read_sql_table(table, _engine())
    except Exception:
        if fallback_csv is not None:
            return pd.read_csv(fallback_csv)
        raise


def load_sensor_data() -> pd.DataFrame:
    return load_table("sensor_readings", config.CLEAN_CSV)


def load_segments() -> pd.DataFrame:
    return load_table("machine_segments", config.SEGMENTS_CSV)
