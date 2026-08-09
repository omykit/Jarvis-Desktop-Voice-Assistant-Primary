from __future__ import annotations

import json
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable


Logger = Callable[[str], None]
MEMORY_PATH = Path(__file__).resolve().parent / "memory.json"
STORE_LOCK = threading.Lock()


def _default_store() -> dict:
    return {"memories": {}, "reminders": []}


def _safe_log(logger: Logger | None, message: str) -> None:
    if logger is not None:
        logger(message)


def _load_store_unlocked() -> dict:
    if not MEMORY_PATH.exists():
        store = _default_store()
        MEMORY_PATH.write_text(json.dumps(store, indent=2), encoding="utf-8")
        return store
    try:
        raw = json.loads(MEMORY_PATH.read_text(encoding="utf-8"))
    except Exception:
        raw = _default_store()
        MEMORY_PATH.write_text(json.dumps(raw, indent=2), encoding="utf-8")
        return raw
    if not isinstance(raw, dict):
        raw = _default_store()
    raw.setdefault("memories", {})
    raw.setdefault("reminders", [])
    return raw


def _save_store_unlocked(data: dict) -> None:
    MEMORY_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")


def load_memory() -> dict:
    with STORE_LOCK:
        return _load_store_unlocked()


def save_memory(data: dict) -> None:
    with STORE_LOCK:
        _save_store_unlocked(data)


def add_memory(key: str, value) -> None:
    with STORE_LOCK:
        data = _load_store_unlocked()
        data["memories"][key] = value
        _save_store_unlocked(data)


def get_memory(key: str):
    return load_memory().get("memories", {}).get(key)


def list_memories() -> dict:
    memories = load_memory().get("memories", {})
    return memories if isinstance(memories, dict) else {}


def parse_reminder_time(value: str, *, now: datetime | None = None) -> datetime | None:
    reference = now or datetime.now()
    text = " ".join(value.strip().split()).lower()
    if not text:
        return None

    day_offset = 0
    if text.startswith("tomorrow "):
        day_offset = 1
        text = text.removeprefix("tomorrow ").strip()
    if text.startswith("today "):
        text = text.removeprefix("today ").strip()
    if text.startswith("at "):
        text = text.removeprefix("at ").strip()

    formats = ("%I %p", "%I:%M %p", "%H:%M")
    parsed_time = None
    for fmt in formats:
        try:
            parsed_time = datetime.strptime(text.upper(), fmt)
            break
        except ValueError:
            continue
    if parsed_time is None:
        return None

    scheduled = reference.replace(
        hour=parsed_time.hour,
        minute=parsed_time.minute,
        second=0,
        microsecond=0,
    ) + timedelta(days=day_offset)
    if day_offset == 0 and scheduled <= reference:
        scheduled += timedelta(days=1)
    return scheduled


def add_reminder(text: str, time, logger: Logger | None = None) -> dict | None:
    if isinstance(time, str):
        scheduled_for = parse_reminder_time(time)
    else:
        scheduled_for = time
    if scheduled_for is None:
        return None

    reminder_text = " ".join(text.strip().split())
    reminder = {
        "id": f"reminder-{int(datetime.now().timestamp() * 1000)}",
        "text": reminder_text,
        "time": scheduled_for.isoformat(),
        "triggered": False,
        "created_at": datetime.now().isoformat(),
    }
    with STORE_LOCK:
        data = _load_store_unlocked()
        data.setdefault("reminders", []).append(reminder)
        _save_store_unlocked(data)
    _safe_log(logger, f"[MEMORY] Stored: reminder={reminder_text}")
    return reminder


def list_reminders(*, include_triggered: bool = False) -> list[dict]:
    reminders = load_memory().get("reminders", [])
    if not isinstance(reminders, list):
        return []
    filtered = [
        reminder
        for reminder in reminders
        if isinstance(reminder, dict) and (include_triggered or not reminder.get("triggered"))
    ]
    return sorted(filtered, key=lambda item: item.get("time", ""))


def check_reminders(*, now: datetime | None = None, logger: Logger | None = None) -> list[dict]:
    reference = now or datetime.now()
    with STORE_LOCK:
        data = _load_store_unlocked()
        reminders = data.get("reminders", [])
        if not isinstance(reminders, list):
            return []

        triggered: list[dict] = []
        changed = False
        for reminder in reminders:
            if not isinstance(reminder, dict) or reminder.get("triggered"):
                continue
            try:
                scheduled = datetime.fromisoformat(str(reminder.get("time")))
            except Exception:
                continue
            if scheduled <= reference:
                reminder["triggered"] = True
                changed = True
                triggered.append(dict(reminder))
                _safe_log(logger, f"[REMINDER] Triggered: {reminder.get('text', '')}")

        if changed:
            _save_store_unlocked(data)
        return triggered
