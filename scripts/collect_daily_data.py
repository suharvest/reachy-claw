#!/usr/bin/env python3
"""Collect and aggregate daily interaction logs into structured JSON.

Reads JSONL files from ~/.reachy-claw/daily-logs/YYYY-MM-DD/ and outputs
a structured JSON summary suitable for diary generation.

Usage:
    python collect_daily_data.py [--date YYYY-MM-DD] [--log-dir PATH]
"""

import argparse
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path


def read_jsonl(filepath: Path) -> list[dict]:
    """Read a JSONL file, returning list of parsed entries."""
    if not filepath.exists():
        return []
    entries = []
    with open(filepath, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return entries


def aggregate_emotions(entries: list[dict]) -> list[dict]:
    """Build mood curve from emotion log entries."""
    curve = []
    for e in entries:
        ts = e.get("ts", "")
        emotion = e.get("emotion", "neutral")
        # Map emotions to a 0-100 value
        emotion_values = {
            "happy": 85, "excited": 90, "laugh": 95, "curious": 70,
            "surprised": 65, "neutral": 50, "listening": 55,
            "thinking": 60, "confused": 40, "sad": 25, "angry": 20,
            "fear": 15,
        }
        value = emotion_values.get(emotion.lower(), 50)
        time_str = ts[11:16] if len(ts) >= 16 else ts  # Extract HH:MM
        curve.append({"t": time_str, "v": value, "emotion": emotion})
    return curve


def aggregate_conversations(entries: list[dict]) -> dict:
    """Aggregate conversation entries into summary + highlights."""
    total = len(entries)
    highlights = []
    for e in entries:
        user_text = e.get("user", "")
        reply_text = e.get("reply", "")
        if not user_text and not reply_text:
            continue
        ts = e.get("ts", "")
        time_str = ts[11:16] if len(ts) >= 16 else ts
        highlights.append({
            "time": time_str,
            "user": user_text,
            "reply": reply_text,
            "emotion": e.get("emotion", ""),
        })
    return {"total": total, "highlights": highlights}


def aggregate_faces(entries: list[dict]) -> dict:
    """Aggregate face detection and smile data."""
    face_counts = []
    smile_total = 0
    all_names = []

    for e in entries:
        if e.get("event") == "smile_capture":
            smile_total = max(smile_total, e.get("total", 0))
        else:
            count = e.get("count", 0)
            face_counts.append(count)
            names = e.get("names", [])
            all_names.extend(names)

    # Find peak hour
    peak_hour = ""
    hourly: dict[str, list[int]] = {}
    for e in entries:
        if e.get("event") == "smile_capture":
            continue
        ts = e.get("ts", "")
        hour = ts[11:13] if len(ts) >= 13 else ""
        if hour:
            hourly.setdefault(hour, []).append(e.get("count", 0))
    if hourly:
        peak_hour = max(hourly, key=lambda h: sum(hourly[h]) / len(hourly[h]))
        peak_hour = f"{peak_hour}:00"

    # Unique names seen
    name_counts = Counter(n for n in all_names if n)

    return {
        "faces_seen": max(face_counts) if face_counts else 0,
        "avg_faces": round(sum(face_counts) / len(face_counts), 1) if face_counts else 0,
        "smiles_captured": smile_total,
        "peak_hour": peak_hour,
        "known_people": dict(name_counts),
    }


def aggregate_thoughts(entries: list[dict]) -> list[dict]:
    """Extract thought/observation highlights."""
    highlights = []
    for e in entries:
        text = e.get("text", "")
        if not text:
            continue
        ts = e.get("ts", "")
        time_str = ts[11:16] if len(ts) >= 16 else ts
        highlights.append({
            "time": time_str,
            "text": text,
            "emotion": e.get("emotion", ""),
        })
    return highlights


def collect(date_str: str, log_dir: Path) -> dict:
    """Collect and aggregate all daily data for the given date."""
    day_dir = log_dir / date_str

    emotions = read_jsonl(day_dir / "emotions.jsonl")
    conversations = read_jsonl(day_dir / "conversations.jsonl")
    faces = read_jsonl(day_dir / "faces.jsonl")
    thoughts = read_jsonl(day_dir / "thoughts.jsonl")

    mood_curve = aggregate_emotions(emotions)
    conv_summary = aggregate_conversations(conversations)
    face_summary = aggregate_faces(faces)
    thought_highlights = aggregate_thoughts(thoughts)

    return {
        "date": date_str,
        "collected_at": datetime.now().isoformat(timespec="seconds"),
        "has_data": bool(emotions or conversations or faces or thoughts),
        "mood_curve": mood_curve,
        "conversations": conv_summary,
        "faces": face_summary,
        "thoughts": thought_highlights,
        "raw_counts": {
            "emotion_samples": len(emotions),
            "conversation_turns": len(conversations),
            "face_samples": len(faces),
            "thought_entries": len(thoughts),
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Collect daily interaction data")
    parser.add_argument(
        "--date", default=datetime.now().strftime("%Y-%m-%d"),
        help="Date to collect (YYYY-MM-DD, default: today)",
    )
    parser.add_argument(
        "--log-dir", type=Path,
        default=Path.home() / ".reachy-claw" / "daily-logs",
        help="Daily logs directory",
    )
    args = parser.parse_args()

    result = collect(args.date, args.log_dir)
    json.dump(result, sys.stdout, ensure_ascii=False, indent=2)
    print()  # trailing newline


if __name__ == "__main__":
    main()
