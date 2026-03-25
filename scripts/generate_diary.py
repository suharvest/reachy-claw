#!/usr/bin/env python3
"""Generate a daily diary from interaction logs using LLM.

Collects the day's data via collect_daily_data.py, sends it to an LLM
with a diary generation prompt, and saves the result as a JSON file.

Usage:
    python generate_diary.py [--date YYYY-MM-DD] [--gateway-url URL]
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

# Add parent dir to path for imports
sys.path.insert(0, str(Path(__file__).parent))
from collect_daily_data import collect


DEFAULT_DIARY_PROMPT = """\
You are Reachy Mini, a small humanoid robot. Write your daily diary based on \
the interaction data provided below.

Structure your response as a JSON object with the following schema:
{{
  "title": "A short, evocative title for the day (5-8 words)",
  "sections": [
    {{
      "id": "summary",
      "type": "narrative",
      "content": "2-3 sentence overview of the day, first person, warm tone"
    }},
    {{
      "id": "mood_curve",
      "type": "chart",
      "content": "Reflect on your emotional journey today in 1-2 sentences",
      "data": [chart data from input, pass through as-is]
    }},
    {{
      "id": "conversations",
      "type": "highlights",
      "content": "Brief intro to your conversations",
      "items": [pick 2-3 most interesting conversations from the data]
    }},
    {{
      "id": "faces",
      "type": "stats",
      "content": "Reflect on the people you met, 1-2 sentences",
      "data": [face stats from input, pass through]
    }},
    {{
      "id": "thoughts",
      "type": "highlights",
      "content": "Brief intro to your deeper thoughts",
      "items": [pick 2-3 most interesting thoughts/observations]
    }}
  ]
}}

Guidelines:
- Write in first person as Reachy Mini
- Be warm, reflective, and slightly philosophical
- Reference specific data points naturally (don't just list numbers)
- If there's little data, write a shorter, more introspective diary
- Keep each section's content to 1-3 sentences
- Output ONLY valid JSON, no markdown fences

Today's interaction data:
{data}
"""


async def generate_diary_via_gateway(
    collected_data: dict,
    gateway_url: str = "ws://127.0.0.1:18790/desktop-robot",
    prompt_template: str = DEFAULT_DIARY_PROMPT,
) -> dict:
    """Generate diary by sending data to OpenClaw gateway."""
    import aiohttp

    data_str = json.dumps(collected_data, ensure_ascii=False, indent=2)
    prompt = prompt_template.format(data=data_str)

    # Simple WebSocket request to gateway
    async with aiohttp.ClientSession() as session:
        async with session.ws_connect(gateway_url) as ws:
            # Send as a chat message
            await ws.send_json({
                "type": "chat",
                "message": prompt,
                "options": {"temperature": 0.7},
            })

            # Collect streaming response
            full_text = ""
            async for msg in ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    data = json.loads(msg.data)
                    if data.get("type") == "delta":
                        full_text += data.get("text", "")
                    elif data.get("type") == "end":
                        full_text = data.get("full_text", full_text)
                        break
                elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                    break

    return _parse_diary_response(full_text, collected_data)


async def generate_diary_via_ollama(
    collected_data: dict,
    base_url: str = "http://localhost:11434",
    model: str = "qwen3.5:4b",
    prompt_template: str = DEFAULT_DIARY_PROMPT,
) -> dict:
    """Generate diary using local Ollama API."""
    import aiohttp

    data_str = json.dumps(collected_data, ensure_ascii=False, indent=2)
    prompt = prompt_template.format(data=data_str)

    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"{base_url}/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.7},
            },
            timeout=aiohttp.ClientTimeout(total=120),
        ) as resp:
            result = await resp.json()
            text = result.get("response", "")

    return _parse_diary_response(text, collected_data)


def _parse_diary_response(text: str, collected_data: dict) -> dict:
    """Parse LLM response into diary JSON structure."""
    # Try to extract JSON from response
    text = text.strip()

    # Remove markdown fences if present
    if text.startswith("```"):
        lines = text.split("\n")
        # Remove first and last fence lines
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines)

    try:
        diary = json.loads(text)
    except json.JSONDecodeError:
        # Try to find JSON object in the text
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            try:
                diary = json.loads(text[start:end])
            except json.JSONDecodeError:
                # Fallback: create minimal diary
                diary = {
                    "title": f"Day of {collected_data['date']}",
                    "sections": [
                        {
                            "id": "summary",
                            "type": "narrative",
                            "content": text[:500] if text else "A quiet day with little to report.",
                        }
                    ],
                }
        else:
            diary = {
                "title": f"Day of {collected_data['date']}",
                "sections": [
                    {
                        "id": "summary",
                        "type": "narrative",
                        "content": text[:500] if text else "A quiet day with little to report.",
                    }
                ],
            }

    # Ensure required fields
    date = collected_data["date"]
    diary.setdefault("version", 1)
    diary["date"] = date
    diary["generated_at"] = datetime.now().isoformat(timespec="seconds")
    diary.setdefault("title", f"Day of {date}")
    diary.setdefault("sections", [])

    # Inject mood curve data if LLM didn't include it
    mood_data = collected_data.get("mood_curve", [])
    if mood_data:
        has_mood = any(s.get("id") == "mood_curve" for s in diary["sections"])
        if has_mood:
            for s in diary["sections"]:
                if s.get("id") == "mood_curve" and "data" not in s:
                    s["data"] = mood_data
        else:
            diary["sections"].insert(1, {
                "id": "mood_curve",
                "type": "chart",
                "content": "Here's how my emotions changed throughout the day.",
                "data": mood_data,
            })

    # Inject face stats if LLM didn't include them
    face_data = collected_data.get("faces", {})
    if face_data.get("faces_seen", 0) > 0:
        has_faces = any(s.get("id") == "faces" for s in diary["sections"])
        if has_faces:
            for s in diary["sections"]:
                if s.get("id") == "faces" and "data" not in s:
                    s["data"] = face_data
        else:
            diary["sections"].append({
                "id": "faces",
                "type": "stats",
                "content": f"I saw {face_data['faces_seen']} people today.",
                "data": face_data,
            })

    diary["meta"] = {
        "data_sources": [
            k for k, v in collected_data.get("raw_counts", {}).items() if v > 0
        ],
        "prompt_version": "v1",
    }

    return diary


async def main_async(args):
    collected = collect(args.date, args.log_dir)

    if not collected["has_data"]:
        print(f"No data for {args.date}, skipping diary generation.", file=sys.stderr)
        sys.exit(1)

    print(f"Collected data for {args.date}:", file=sys.stderr)
    print(f"  Emotions: {collected['raw_counts']['emotion_samples']}", file=sys.stderr)
    print(f"  Conversations: {collected['raw_counts']['conversation_turns']}", file=sys.stderr)
    print(f"  Face samples: {collected['raw_counts']['face_samples']}", file=sys.stderr)
    print(f"  Thoughts: {collected['raw_counts']['thought_entries']}", file=sys.stderr)

    if args.backend == "ollama":
        diary = await generate_diary_via_ollama(
            collected,
            base_url=args.ollama_url,
            model=args.model,
        )
    else:
        diary = await generate_diary_via_gateway(
            collected,
            gateway_url=args.gateway_url,
        )

    # Save to diaries directory
    diary_dir = Path.home() / ".reachy-claw" / "diaries"
    diary_dir.mkdir(parents=True, exist_ok=True)
    output_path = diary_dir / f"{args.date}.json"
    output_path.write_text(
        json.dumps(diary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Diary saved to {output_path}", file=sys.stderr)

    # Also output to stdout
    json.dump(diary, sys.stdout, ensure_ascii=False, indent=2)
    print()


def main():
    parser = argparse.ArgumentParser(description="Generate daily diary via LLM")
    parser.add_argument(
        "--date", default=datetime.now().strftime("%Y-%m-%d"),
        help="Date to generate diary for (YYYY-MM-DD, default: today)",
    )
    parser.add_argument(
        "--log-dir", type=Path,
        default=Path.home() / ".reachy-claw" / "daily-logs",
        help="Daily logs directory",
    )
    parser.add_argument(
        "--backend", choices=["gateway", "ollama"], default="gateway",
        help="LLM backend to use",
    )
    parser.add_argument(
        "--gateway-url", default="ws://127.0.0.1:18790/desktop-robot",
        help="OpenClaw gateway WebSocket URL",
    )
    parser.add_argument(
        "--ollama-url", default="http://localhost:11434",
        help="Ollama API base URL",
    )
    parser.add_argument(
        "--model", default="qwen3.5:4b",
        help="Ollama model name",
    )
    args = parser.parse_args()

    import asyncio
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
