from __future__ import annotations

import os
from pathlib import Path
from sqlmodel import SQLModel, create_engine, Session
from dotenv import load_dotenv

load_dotenv()

DEFAULT_DB = Path(__file__).resolve().parents[2] / "data" / "app.db"
DB_PATH = os.getenv("SQLITE_PATH", str(DEFAULT_DB))
DB_URL = f"sqlite:///{DB_PATH}"

engine = create_engine(DB_URL, connect_args={"check_same_thread": False})


def init_db() -> None:
    Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    SQLModel.metadata.create_all(engine)


def get_session() -> Session:
    return Session(engine)
