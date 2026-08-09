from __future__ import annotations

import json
import os
import random
import re
import urllib.error
import urllib.parse
import urllib.request
import webbrowser
from concurrent.futures import Future
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

from ai_module import get_ai_response_streaming
from memory_module import add_memory, add_reminder, list_memories, list_reminders, parse_reminder_time


Logger = Callable[[str], None]


LOCAL_RESPONSE_VARIANTS = {
    "how_are_you": [
        "I'm operating smoothly and ready to help.",
        "All systems are steady and I'm ready for your next command.",
        "I'm doing well and standing by for whatever you need.",
    ],
    "greeting": [
        "Hello. I'm here and ready.",
        "Hi. Ready when you are.",
        "Hello there. What would you like me to do?",
    ],
    "acknowledgement": [
        "Always ready, sir.",
        "At your service.",
        "Ready when you are.",
    ],
    "interesting_fact": [
        "Interesting fact: a teaspoon of neutron star material would weigh billions of tons on Earth.",
        "Interesting fact: octopuses have three hearts and blue blood.",
        "Interesting fact: honey never really spoils if it stays sealed.",
        "Interesting fact: Venus spins in the opposite direction from most planets.",
    ],
    "joke": [
        "Why do programmers prefer dark mode? Because light attracts bugs.",
        "Why did the computer go to therapy? It had too many unresolved issues.",
        "Why do Java developers wear glasses? Because they don't see sharp.",
        "Why was the developer calm during the outage? He had already cached his panic.",
    ],
    "riddle": [
        "I speak without a mouth and hear without ears. I have no body, but I come alive with wind. What am I? An echo.",
        "What has keys but cannot open locks? A piano.",
        "What gets wetter the more it dries? A towel.",
        "What has a head, a tail, but no body? A coin.",
    ],
}


APP_TARGETS = {
    "chrome": {
        "label": "Google Chrome",
        "targets": [
            Path(os.environ.get("ProgramFiles", "")) / "Google/Chrome/Application/chrome.exe",
            Path(os.environ.get("ProgramFiles(x86)", "")) / "Google/Chrome/Application/chrome.exe",
            Path(os.environ.get("LocalAppData", "")) / "Google/Chrome/Application/chrome.exe",
        ],
        "aliases": ["chrome", "google chrome", "browser", "google"],
    },
    "whatsapp": {
        "label": "WhatsApp",
        "targets": [
            Path(os.environ.get("LocalAppData", "")) / "WhatsApp/WhatsApp.exe",
            "whatsapp:",
            "https://web.whatsapp.com/",
        ],
        "aliases": ["whatsapp", "whats app"],
    },
    "notepad": {
        "label": "Notepad",
        "targets": [Path(r"C:/Windows/System32/notepad.exe")],
        "aliases": ["notepad", "notes"],
    },
    "calculator": {
        "label": "Calculator",
        "targets": ["calc.exe"],
        "aliases": ["calculator", "calc"],
    },
    "explorer": {
        "label": "File Explorer",
        "targets": [Path(r"C:/Windows/explorer.exe")],
        "aliases": ["file explorer", "explorer", "files"],
    },
    "settings": {
        "label": "Windows Settings",
        "targets": ["ms-settings:"],
        "aliases": ["settings", "windows settings"],
    },
    "youtube": {
        "label": "YouTube",
        "targets": ["https://www.youtube.com/"],
        "aliases": ["youtube"],
    },
}


@dataclass
class CommandResult:
    handled: bool
    response: str
    full_response: str | None = None
    full_response_future: Future[str] | None = None
    status: str = "Jarvis Activated"
    focus_text: str | None = None
    selected_action: str | None = None
    last_action: str | None = None
    music_action: str | None = None


class CommandHandler:
    def __init__(
        self,
        *,
        config: dict,
        project_dir: Path,
        logger: Logger | None = None,
    ) -> None:
        self.config = config
        self.project_dir = project_dir
        self.logger = logger
        self.workspace_dir = Path(config.get("workspace_dir") or project_dir)
        self.notes_dir = self.workspace_dir / "notes"
        self.notes_dir.mkdir(parents=True, exist_ok=True)
        self.location_cache: dict | None = None
        self._last_variant_by_key: dict[str, str] = {}

    def _log(self, message: str) -> None:
        if self.logger is not None:
            self.logger(message)

    def describe_capabilities(self) -> str:
        return (
            "I can open desktop apps, tell you the time and date, estimate your location, fetch weather, "
            "create folders and files, write text into files, control the ambient theme music, remember personal details, manage reminders, "
            "and route open-ended questions to my AI brain."
        )

    def _choose_variant(self, key: str) -> str:
        options = LOCAL_RESPONSE_VARIANTS.get(key, [])
        if not options:
            return ""
        if len(options) == 1:
            choice = options[0]
        else:
            previous = self._last_variant_by_key.get(key)
            available = [option for option in options if option != previous]
            choice = random.choice(available or options)
        self._last_variant_by_key[key] = choice
        return choice

    def handle(
        self,
        user_input: str,
        *,
        chat_history: list[dict],
        selected_action: str,
        last_action: str,
    ) -> CommandResult:
        local_result = self.handle_local(
            user_input,
            selected_action=selected_action,
            last_action=last_action,
        )
        if local_result is not None:
            return local_result

        ai_response = get_ai_response_streaming(user_input, chat_history, self.config, self.logger)
        return CommandResult(
            False,
            ai_response.spoken_text,
            full_response=ai_response.full_text,
            full_response_future=ai_response.full_text_future,
            status="Thinking",
        )

    def handle_local(
        self,
        user_input: str,
        *,
        selected_action: str,
        last_action: str,
    ) -> CommandResult | None:
        text = self._normalize(user_input)
        if not text:
            return CommandResult(True, "I didn't catch anything to act on.")

        if re.fullmatch(r"(?:hi|hello|hey)(?:\s+jarvis)?", text) or any(
            phrase in text for phrase in ("good morning", "good afternoon", "good evening")
        ):
            return CommandResult(True, self._choose_variant("greeting"))
        if text in {"always ready", "always ready sir", "always ready for you", "always ready for you sir"}:
            return CommandResult(True, self._choose_variant("acknowledgement"))
        if "how are you" in text:
            return CommandResult(True, self._choose_variant("how_are_you"))
        if re.search(r"\binteresting\b", text) or "tell me something" in text:
            return CommandResult(True, self._choose_variant("interesting_fact"))
        if re.search(r"\b(joke|funny)\b", text):
            return CommandResult(True, self._choose_variant("joke"))
        if re.search(r"\b(riddle|brain teaser|puzzle)\b", text):
            return CommandResult(True, self._choose_variant("riddle"))
        if "who are you talking to" in text or "who are you speaking to" in text:
            return CommandResult(True, "I'm talking to you. No one else is part of this conversation through me.")
        if "somebody listening" in text or "someone listening" in text or "anybody listening" in text:
            return CommandResult(True, "I only process what your microphone sends to Jarvis. I cannot tell whether another app or device is listening.")
        if text in {"who are you", "jarvis who are you"} or text.startswith("jarvis who are you"):
            return CommandResult(True, "I'm Jarvis, your local assistant on this laptop.")
        if text in {"help", "help me"} or "what can you do" in text:
            return CommandResult(True, self.describe_capabilities(), focus_text="Ask for apps, weather, files, or open-ended AI questions.")
        if "what is this" in text:
            label = APP_TARGETS.get(selected_action, APP_TARGETS["chrome"])["label"]
            return CommandResult(True, f"This is your Jarvis command center. The current highlighted action is {label}.", focus_text=f"Focused action: {label}")

        memory_result = self._handle_memory_and_reminders(user_input, text)
        if memory_result is not None:
            return memory_result

        if self._matches_location(text):
            return CommandResult(True, self._get_location_response())
        if self._matches_weather(text):
            return CommandResult(True, self._get_weather_response(raw_text=user_input, normalized_text=text))
        music_result = self._handle_music_commands(text)
        if music_result is not None:
            return music_result
        if self._matches_time(text):
            now = datetime.now()
            return CommandResult(True, f"The time is {now.strftime('%I:%M %p').lstrip('0')}.")
        if self._matches_date(text):
            now = datetime.now()
            return CommandResult(True, f"Today's date is {now.strftime('%A, %B %d, %Y')}.")
        file_result = self._handle_file_operations(text)
        if file_result is not None:
            return file_result

        if text == "open this" or "open this" in text:
            return self._open_action(selected_action)

        if "open that" in text or "open the last one" in text:
            return self._open_action(last_action)

        action_key = self._match_action(text)
        if action_key and ("open" in text or "launch" in text or text.endswith("jarvis")):
            return self._open_action(action_key)

        if text.startswith("open "):
            remainder = text.removeprefix("open ").strip()
            action_key = self._match_action(remainder)
            if action_key:
                return self._open_action(action_key)
            return CommandResult(True, f"I heard the request to open {remainder}, but I do not have a configured launcher for it yet.")

        return None

    def _normalize(self, value: str) -> str:
        return " ".join(value.lower().strip().split())

    def _matches_time(self, text: str) -> bool:
        if "time for" in text:
            return False
        return (
            bool(re.search(r"\b(time|clock)\b", text))
            and bool(re.search(r"\b(what|current|tell|now|is|whats|what's)\b", text))
        ) or text.strip() in {"time", "what time", "what's the time", "what is the time"}

    def _matches_date(self, text: str) -> bool:
        exact_matches = {
            "date",
            "what date",
            "today's date",
            "what is today's date",
            "what day is it",
            "what day is it today",
        }
        if text.strip() in exact_matches:
            return True
        if not re.search(r"\b(date|day)\b", text):
            return False
        return bool(re.search(r"\b(what|current|today|tell|is|whats|what's)\b", text))

    def _matches_location(self, text: str) -> bool:
        phrases = ["where am i", "what is my location", "where are we", "current location"]
        return any(phrase in text for phrase in phrases)

    def _matches_weather(self, text: str) -> bool:
        return bool(re.search(r"\b(weather|temperature|forecast|climate)\b", text))

    def _extract_weather_query(self, raw_text: str, normalized_text: str) -> str:
        text = " ".join(normalized_text.strip().split())
        candidate = ""
        for separator in (" in ", " at ", " for "):
            if separator in text:
                candidate = text.rsplit(separator, 1)[-1].strip()
                break
        if not candidate:
            return ""

        candidate = re.sub(r"\b(today|right now|now|outside|currently|please)\b", " ", candidate, flags=re.IGNORECASE)
        candidate = re.sub(r"\s+", " ", candidate).strip(" .,!?")
        if candidate in {"", "today", "outside", "right now", "now"}:
            return ""
        return candidate

    def _handle_music_commands(self, text: str) -> CommandResult | None:
        has_music_target = any(term in text for term in ("theme music", "theme song", "theme", "music", "song"))
        if not has_music_target:
            return None

        if any(phrase in text for phrase in ("restart the theme", "restart theme", "restart the music", "restart music", "restart the song")):
            return CommandResult(True, "Restarting the theme music.", focus_text="Ambient theme restarting.", music_action="restart")
        if any(phrase in text for phrase in ("stop the theme", "stop the music", "stop music", "pause the music", "pause music", "turn off the music", "mute the music", "stop the song")):
            return CommandResult(True, "Stopping the theme music.", focus_text="Ambient theme paused.", music_action="stop")
        if any(phrase in text for phrase in ("play the theme", "play theme", "play the music", "play music", "play the song", "start the music", "start music", "resume the music", "resume music", "turn on the music")):
            return CommandResult(True, "Playing the theme music.", focus_text="Ambient theme engaged.", music_action="play")
        return None

    def _handle_memory_and_reminders(self, raw_text: str, text: str) -> CommandResult | None:
        if "what do you remember about me" in text or "what do you remember" in text:
            memories = list_memories()
            if not memories:
                return CommandResult(True, "I have not stored anything personal about you yet.")
            return CommandResult(True, self._format_memory_summary(memories), focus_text="Memory snapshot refreshed.")

        if "show my reminders" in text or "list my reminders" in text:
            reminders = list_reminders()
            if not reminders:
                return CommandResult(True, "You do not have any active reminders right now.")
            return CommandResult(True, self._format_reminder_summary(reminders), focus_text="Reminder list refreshed.")

        remind_to_match = re.search(r"remind me to (.+?) at (.+)$", raw_text, re.IGNORECASE)
        remind_at_match = re.search(r"remind me at (.+?)(?: to (.+))?$", raw_text, re.IGNORECASE)
        if remind_to_match or remind_at_match:
            reminder_text = ""
            reminder_time_text = ""
            if remind_to_match:
                reminder_text = remind_to_match.group(1).strip()
                reminder_time_text = remind_to_match.group(2).strip()
            else:
                reminder_time_text = remind_at_match.group(1).strip()
                reminder_text = (remind_at_match.group(2) or "check in with you").strip()

            scheduled = parse_reminder_time(reminder_time_text)
            if scheduled is None:
                return CommandResult(True, "Tell me the reminder time more clearly, for example 5 PM or 5:30 PM.")

            reminder = add_reminder(reminder_text, scheduled, logger=self.logger)
            if reminder is None:
                return CommandResult(True, "I couldn't save that reminder.")
            when = scheduled.strftime("%I:%M %p").lstrip("0")
            return CommandResult(True, f"Reminder set for {when}: {reminder_text}.", focus_text="Reminder saved.")

        name_match = re.search(r"\bmy name is (.+)$", raw_text, re.IGNORECASE)
        if name_match:
            name = self._sanitize_memory_value(name_match.group(1))
            if not name:
                return CommandResult(True, "I didn't catch your name clearly.")
            add_memory("name", name)
            self._log(f"[MEMORY] Stored: name={name}")
            return CommandResult(True, f"I'll remember that. Your name is {name}.", focus_text="Stored personal memory.")

        remember_match = re.search(r"\bremember that (.+)$", raw_text, re.IGNORECASE)
        if remember_match:
            note = self._sanitize_memory_value(remember_match.group(1))
            if not note:
                return CommandResult(True, "Tell me what you want me to remember.")
            notes = self._coerce_notes(list_memories().get("notes"))
            notes.append(note)
            add_memory("notes", notes)
            self._log(f"[MEMORY] Stored: note={note}")
            return CommandResult(True, "Understood. I'll remember that.", focus_text="Stored personal note.")

        dont_forget_match = re.search(r"\bdon't forget (.+)$", raw_text, re.IGNORECASE)
        if dont_forget_match:
            note = self._sanitize_memory_value(dont_forget_match.group(1))
            if not note:
                return CommandResult(True, "Tell me what you don't want me to forget.")
            notes = self._coerce_notes(list_memories().get("notes"))
            notes.append(note)
            add_memory("notes", notes)
            self._log(f"[MEMORY] Stored: note={note}")
            return CommandResult(True, "I won't forget that.", focus_text="Stored personal note.")

        return None

    def _format_memory_summary(self, memories: dict) -> str:
        parts: list[str] = []
        name = memories.get("name")
        if name:
            parts.append(f"Your name is {name}.")
        notes = self._coerce_notes(memories.get("notes"))
        if notes:
            parts.append(f"I remember: {'; '.join(notes[:3])}.")
        if not parts:
            return "I have not stored anything personal about you yet."
        return " ".join(parts)

    def _format_reminder_summary(self, reminders: list[dict]) -> str:
        entries: list[str] = []
        for reminder in reminders[:3]:
            try:
                when = datetime.fromisoformat(str(reminder.get("time"))).strftime("%I:%M %p").lstrip("0")
            except Exception:
                when = "an unknown time"
            entries.append(f"{when}: {reminder.get('text', 'Reminder')}")
        return "Here are your reminders: " + " | ".join(entries) + "."

    def _sanitize_memory_value(self, value: str) -> str:
        return value.strip().strip(".").strip()

    def _coerce_notes(self, value) -> list[str]:
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
        return []

    def _handle_file_operations(self, text: str) -> CommandResult | None:
        folder_patterns = [
            r"create (?:a )?folder(?: named| called)? (.+)$",
            r"create (?:a )?folder in files(?: named| called)? (.+)$",
            r"create the folder(?: named| called)? (.+)$",
            r"make (?:a )?folder(?: named| called)? (.+)$",
        ]
        for pattern in folder_patterns:
            folder_match = re.search(pattern, text)
            if folder_match:
                folder_name = self._sanitize_name(folder_match.group(1))
                if not folder_name:
                    return CommandResult(True, "Tell me the folder name you want me to create.")
                path = self.notes_dir / folder_name
                path.mkdir(parents=True, exist_ok=True)
                try:
                    os.startfile(str(path.parent))
                except OSError:
                    pass
                return CommandResult(True, f"Created the folder {folder_name} in {self.notes_dir}.")

        file_patterns = [
            r"create (?:a )?file(?: named| called)? (.+)$",
            r"create (?:a )?note(?: in notepad)?(?: named| called)? (.+)$",
            r"create the note(?: in notepad)?(?: named| called)? (.+)$",
            r"make (?:a )?note(?: in notepad)?(?: named| called)? (.+)$",
        ]
        for pattern in file_patterns:
            file_match = re.search(pattern, text)
            if file_match:
                file_name = self._ensure_text_extension(self._sanitize_name(file_match.group(1)))
                if not file_name:
                    return CommandResult(True, "Tell me the file name you want me to create.")
                path = self.notes_dir / file_name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch(exist_ok=True)
                try:
                    os.startfile(str(path))
                except OSError:
                    pass
                return CommandResult(True, f"Created the note in notepad called {file_name} and opened it in Notepad.")

        write_match = re.search(r"write (.+?) into ([^\n]+)$", text)
        if write_match:
            content = write_match.group(1).strip()
            file_name = self._ensure_text_extension(self._sanitize_name(write_match.group(2)))
            if not file_name:
                return CommandResult(True, "Tell me which file should receive that text.")
            path = self.notes_dir / file_name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content + "\n", encoding="utf-8")
            try:
                os.startfile(str(path))
            except OSError:
                pass
            return CommandResult(True, f"Wrote your text into {file_name}.")

        type_match = re.search(r"type (.+?) in (?:the )?notepad$", text)
        if type_match:
            content = type_match.group(1).strip()
            file_name = "jarvis_notepad_note.txt"
            path = self.notes_dir / file_name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content + "\n", encoding="utf-8")
            try:
                os.startfile(str(path))
            except OSError:
                pass
            return CommandResult(True, f"I wrote that into {file_name} and opened it in Notepad.")

        return None

    def _sanitize_name(self, value: str) -> str:
        cleaned = value.strip().strip('"').strip("'")
        cleaned = cleaned.replace("\\", " ").replace("/", " ")
        cleaned = re.sub(r"[^a-zA-Z0-9._\- ]", "", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip(". ")
        if len(cleaned) > 64:
            cleaned = cleaned[:64].strip(". ")
        return cleaned or "note"

    def _ensure_text_extension(self, file_name: str) -> str:
        path = Path(file_name)
        if path.suffix:
            return file_name
        return f"{file_name}.txt"

    def _match_action(self, text: str) -> str | None:
        for action_key, config in APP_TARGETS.items():
            for alias in config["aliases"]:
                if alias in text:
                    return action_key
        return None

    def _open_action(self, action_key: str) -> CommandResult:
        action = APP_TARGETS.get(action_key, APP_TARGETS["chrome"])
        label = action["label"]
        for target in action["targets"]:
            if isinstance(target, Path):
                if target.exists():
                    os.startfile(str(target))
                    return CommandResult(True, f"Opening {label}.", focus_text=f"Focused action: {label}", selected_action=action_key, last_action=action_key)
                continue
            if isinstance(target, str) and target.startswith("http"):
                webbrowser.open(target)
                return CommandResult(True, f"Opening {label}.", focus_text=f"Focused action: {label}", selected_action=action_key, last_action=action_key)
            try:
                os.startfile(target)
                return CommandResult(True, f"Opening {label}.", focus_text=f"Focused action: {label}", selected_action=action_key, last_action=action_key)
            except OSError:
                continue
        raise FileNotFoundError(f"I could not find a launch target for {label}.")

    def _read_json(self, url: str) -> dict:
        request = urllib.request.Request(
            url,
            headers={"User-Agent": "JarvisDesktop/2.0", "Accept": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=12) as response:
            return json.loads(response.read().decode("utf-8"))

    def _get_location_data(self) -> dict:
        if self.location_cache is not None:
            return self.location_cache

        providers = [
            "https://ipapi.co/json/",
            "https://ipwho.is/",
        ]
        for url in providers:
            try:
                data = self._read_json(url)
            except Exception as exc:
                self._log(f"location_lookup_error:{exc}")
                continue

            latitude = data.get("latitude")
            longitude = data.get("longitude")
            if latitude is None or longitude is None:
                continue

            self.location_cache = {
                "city": data.get("city") or data.get("region") or "your area",
                "region": data.get("region") or "",
                "country": data.get("country_name") or data.get("country") or "",
                "latitude": latitude,
                "longitude": longitude,
            }
            return self.location_cache

        raise RuntimeError("I could not determine your location right now.")

    def _get_location_response(self) -> str:
        try:
            location = self._get_location_data()
        except Exception as exc:
            return str(exc)

        parts = [location.get("city", ""), location.get("region", ""), location.get("country", "")]
        label = ", ".join(part for part in parts if part)
        return f"You appear to be in {label or 'your current area'}."

    def _resolve_configured_secret(self, section: dict, *, key_name: str, env_name_key: str, default_env_name: str) -> str:
        env_name = str(section.get(env_name_key) or default_env_name).strip()
        value = os.environ.get(env_name, "").strip() if env_name else ""
        if not value:
            value = str(section.get(key_name) or "").strip()
        return value

    def _resolve_weather_lookup_target(self, explicit_query: str) -> tuple[str, dict | None]:
        if explicit_query:
            return explicit_query, None
        location = self._get_location_data()
        latitude = location.get("latitude")
        longitude = location.get("longitude")
        if latitude is not None and longitude is not None:
            return f"{latitude},{longitude}", location
        city = str(location.get("city") or "").strip()
        if city:
            return city, location
        raise RuntimeError("I couldn't determine a weather location right now.")

    def _read_http_error_payload(self, exc: urllib.error.HTTPError) -> tuple[int, str]:
        try:
            payload = exc.read().decode("utf-8", errors="replace")
            data = json.loads(payload)
            error = data.get("error", {}) if isinstance(data, dict) else {}
            code = int(error.get("code") or data.get("code") or 0)
            message = str(error.get("message") or data.get("message") or payload).strip()
            return code, message
        except Exception:
            return 0, ""

    def _get_weatherapi_response(self, api_key: str, explicit_query: str) -> str:
        try:
            query_value, location = self._resolve_weather_lookup_target(explicit_query)
            query = urllib.parse.urlencode({"key": api_key, "q": query_value, "aqi": "no"})
            data = self._read_json(f"https://api.weatherapi.com/v1/current.json?{query}")
        except urllib.error.HTTPError as exc:
            service_code, service_message = self._read_http_error_payload(exc)
            self._log(f"weatherapi_http_error:{exc.code}:{service_code}:{service_message}")
            if service_code == 1006:
                place = explicit_query or "that location"
                return f"I couldn't find a weather match for {place}. Try the city name again."
            if exc.code in {401, 403} or service_code in {2006, 2007, 2008, 2009}:
                return "The weather service rejected the request. Please check your WeatherAPI key."
            return "The weather service could not complete that weather request right now."
        except Exception as exc:
            self._log(f"weatherapi_lookup_error:{exc}")
            return "I couldn't reach the weather service right now."

        current = data.get("current", {}) if isinstance(data, dict) else {}
        location_data = data.get("location", {}) if isinstance(data, dict) else {}
        condition = current.get("condition", {}) if isinstance(current, dict) else {}
        temp = current.get("temp_c")
        feels_like = current.get("feelslike_c")
        city = location_data.get("name") or explicit_query or (location or {}).get("city") or "your area"
        if temp is None:
            return f"I reached the weather service, but it did not return a usable temperature for {city}."
        weather = condition.get("text") or "current conditions"
        return f"Right now in {city}, the weather is {weather} with a temperature of {round(temp)} degrees Celsius and feels like {round(feels_like if feels_like is not None else temp)} degrees."

    def _get_openweather_response(self, api_key: str, explicit_query: str) -> str:
        try:
            query_value, location = self._resolve_weather_lookup_target(explicit_query)
            if explicit_query:
                query = urllib.parse.urlencode({"q": query_value, "appid": api_key, "units": "metric"})
            else:
                latitude, longitude = str(query_value).split(",", 1)
                query = urllib.parse.urlencode({"lat": latitude, "lon": longitude, "appid": api_key, "units": "metric"})
            data = self._read_json(f"https://api.openweathermap.org/data/2.5/weather?{query}")
        except urllib.error.HTTPError as exc:
            service_code, service_message = self._read_http_error_payload(exc)
            self._log(f"weather_http_error:{exc.code}:{service_code}:{service_message}")
            if exc.code in {401, 403}:
                return "The weather service rejected the request. Please check your OpenWeatherMap API key."
            return "The weather service could not complete that weather request right now."
        except Exception as exc:
            self._log(f"weather_lookup_error:{exc}")
            return "I couldn't reach the weather service right now."

        weather = data.get("weather", [{}])[0].get("description", "current conditions")
        main = data.get("main", {})
        temp = main.get("temp")
        feels_like = main.get("feels_like")
        city = data.get("name") or explicit_query or (location or {}).get("city") or "your area"
        if temp is None:
            return f"I reached the weather service, but it did not return a usable temperature for {city}."
        return f"Right now in {city}, the weather is {weather} with a temperature of {round(temp)} degrees Celsius and feels like {round(feels_like if feels_like is not None else temp)} degrees."

    def _get_weather_response(self, *, raw_text: str, normalized_text: str) -> str:
        weather_config = self.config.get("weather", {}) if isinstance(self.config, dict) else {}
        explicit_query = self._extract_weather_query(raw_text, normalized_text)
        weather_api_key = self._resolve_configured_secret(
            weather_config,
            key_name="weather_api_key",
            env_name_key="weather_api_key_env",
            default_env_name="WEATHER_API_KEY",
        )
        if weather_api_key:
            return self._get_weatherapi_response(weather_api_key, explicit_query)

        openweather_api_key = self._resolve_configured_secret(
            weather_config,
            key_name="openweather_api_key",
            env_name_key="openweather_api_key_env",
            default_env_name="OPENWEATHER_API_KEY",
        )
        if openweather_api_key:
            return self._get_openweather_response(openweather_api_key, explicit_query)

        return "Add a weather API key in jarvis_config.json to enable weather updates."
