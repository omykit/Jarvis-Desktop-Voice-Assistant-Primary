from __future__ import annotations

import json
import queue
import random
import threading
import time
from datetime import datetime
from pathlib import Path

import pygame
import tkinter as tk
from tkinter.scrolledtext import ScrolledText

from ai_module import initialize_ai_health
from command_handler import APP_TARGETS, CommandHandler
from memory_module import check_reminders, list_memories, list_reminders
from voice_engine import VoiceEngine

APP_DIR = Path(__file__).resolve().parent
RUNTIME_LOG = APP_DIR / "jarvis_runtime.log"
CONFIG_PATH = APP_DIR / "jarvis_config.json"
WAKE_PHRASES = [
    "are you up jarvis",
    "jarvis are you up",
    "hey jarvis are you up",
    "hey jarvis",
    "hi jarvis",
]
EXIT_PHRASE = "thank you jarvis"
MUSIC_VOLUME = 0.3
MUSIC_BASE_NAME = "theme"
AMBIENT_THEME_VOLUME = 0.18
MUSIC_EXTENSIONS = [".mp3", ".wav", ".ogg"]

BG = "#06131f"
PANEL = "#0d2233"
PANEL_ALT = "#102b40"
CARD = "#133a52"
CARD_ALT = "#0e3146"
BORDER = "#29546c"
TEXT = "#ecf7ff"
MUTED = "#7fa8bf"
ACCENT = "#7ad8ff"
ACCENT_SOFT = "#36b7e3"
WARM = "#ffc66d"
SUCCESS = "#98ffd4"
ERROR = "#ff9f9f"

WINDOW_SIZE = "1180x760"
STATUS_IDLE = "Passive Listening"
STATUS_ACTIVE = "Jarvis Activated"
STATUS_SPEAKING = "Speaking"
STATUS_THINKING = "Thinking"
STATUS_ERROR = "Attention Required"

DEFAULT_CONFIG = {
    "assistant_name": "Jarvis",
    "owner_name": "Omair",
    "workspace_dir": "",
    "voice": {
        "energy_threshold": 250,
        "pause_threshold": 0.8,
        "tts_rate": 175,
        "vosk_model_path": "vosk-model-en-us-0.22",
        "allow_google_fallback": False,
        "google_fallback_cooldown_seconds": 90,
    },
    "ai": {
        "enabled": True,
        "base_url": "http://localhost:11434/v1",
        "model": "llama3",
        "api_key": "ollama",
        "api_key_env": "",
        "timeout_seconds": 30,
        "max_tokens": 80,
        "max_history_messages": 8,
        "system_prompt": "You are Jarvis, Omair's personal local voice assistant. Be warm, direct, and useful. Use remembered facts and recent conversation when helpful. Answer naturally in one or two short sentences.",
    },
    "weather": {
        "openweather_api_key": "",
        "openweather_api_key_env": "OPENWEATHER_API_KEY",
        "weather_api_key": "",
        "weather_api_key_env": "WEATHER_API_KEY",
    },
}

root: tk.Tk | None = None
status_var: tk.StringVar | None = None
subtitle_var: tk.StringVar | None = None
heard_var: tk.StringVar | None = None
response_var: tk.StringVar | None = None
focus_var: tk.StringVar | None = None
memory_var: tk.StringVar | None = None
reminders_var: tk.StringVar | None = None
conversation_box: ScrolledText | None = None
command_entry: tk.Entry | None = None
orb_canvas: tk.Canvas | None = None
orb_outer: int | None = None
orb_inner: int | None = None
ui_queue: queue.Queue[tuple[str, str, str, str, str, bool]] = queue.Queue()
app_state_lock = threading.Lock()
config: dict = {}
voice_engine: VoiceEngine | None = None
command_handler: CommandHandler | None = None
music_file: Path | None = None
jarvis_active = False
selected_action_key = "chrome"
last_open_target = "chrome"
chat_history: list[dict] = []
shutdown_requested = False
reminder_thread: threading.Thread | None = None
reminder_stop_event = threading.Event()
_last_memory_panel_refresh = 0.0
request_state_lock = threading.Lock()
latest_ai_request_id = 0
completed_ai_request_id = 0
pending_ai_ack_request_id = 0
pending_ai_ack_phrase: str | None = None
THINKING_PHRASES = [
    "Checking...",
    "One moment...",
]
WAKE_GREETINGS = [
    "I'm online, Master {owner}.",
    "Always ready, Master {owner}.",
    "Jarvis is awake and ready, Master {owner}.",
    "At your service, Master {owner}.",
]
AI_ACK_DELAY_SECONDS = 8.0
last_thinking_phrase: str | None = None
last_wake_greeting: str | None = None


def log_event(message: str) -> None:
    try:
        with RUNTIME_LOG.open("a", encoding="utf-8") as handle:
            handle.write(message + "\n")
    except OSError:
        pass


def schedule_shutdown_after_speech() -> None:
    if root is None:
        on_close()
        return

    if voice_engine is None:
        on_close()
        return

    is_speaking = voice_engine.speaking_event.is_set()
    has_pending_speech = not voice_engine.tts_queue.empty()
    if is_speaking or has_pending_speech:
        root.after(120, schedule_shutdown_after_speech)
        return
    on_close()


def refresh_memory_panels(force: bool = False) -> None:
    global _last_memory_panel_refresh

    if memory_var is None or reminders_var is None:
        return
    if not force and time.time() - _last_memory_panel_refresh < 1.5:
        return

    memories = list_memories()
    reminders = list_reminders()

    name = str(memories.get("name") or "").strip()
    notes = memories.get("notes") if isinstance(memories.get("notes"), list) else []
    memory_lines: list[str] = []
    if name:
        memory_lines.append(f"Name: {name}")
    if notes:
        memory_lines.append(f"Note: {str(notes[0])}")
    if not memory_lines:
        memory_lines.append("No personal memory stored yet.")
    memory_var.set(" | ".join(memory_lines[:2]))

    if reminders:
        preview = []
        for reminder in reminders[:2]:
            try:
                when_text = datetime.fromisoformat(str(reminder.get("time"))).strftime("%I:%M %p").lstrip("0")
            except Exception:
                when_text = "soon"
            preview.append(f"{when_text} {reminder.get('text', 'Reminder')}")
        reminders_var.set(" | ".join(preview))
    else:
        reminders_var.set("No active reminders.")

    _last_memory_panel_refresh = time.time()


def reminder_loop() -> None:
    while not reminder_stop_event.is_set():
        triggered = check_reminders(logger=log_event)
        for reminder in triggered:
            reminder_text = str(reminder.get("text") or "It's time for your reminder.")
            spoken = f"Reminder: {reminder_text}."
            update_ui(
                STATUS_ACTIVE if jarvis_active else STATUS_IDLE,
                spoken,
                assistant_name(),
                subtitle="Reminder triggered.",
                focus_text="Reminder delivery in progress.",
                log_message=True,
            )
            if voice_engine is not None:
                voice_engine.speak_async(spoken)
        reminder_stop_event.wait(15)


def invalidate_pending_ai_requests() -> int:
    global latest_ai_request_id, completed_ai_request_id, pending_ai_ack_request_id, pending_ai_ack_phrase
    with request_state_lock:
        latest_ai_request_id += 1
        completed_ai_request_id = 0
        pending_ai_ack_request_id = 0
        pending_ai_ack_phrase = None
        return latest_ai_request_id


def is_current_ai_request(request_id: int) -> bool:
    with request_state_lock:
        return request_id == latest_ai_request_id


def try_register_ai_acknowledgement(request_id: int, phrase: str) -> bool:
    global pending_ai_ack_request_id, pending_ai_ack_phrase
    with request_state_lock:
        if request_id != latest_ai_request_id or completed_ai_request_id == request_id or pending_ai_ack_request_id:
            return False
        pending_ai_ack_request_id = request_id
        pending_ai_ack_phrase = phrase
        return True


def mark_ai_request_complete(request_id: int) -> None:
    global completed_ai_request_id
    with request_state_lock:
        if request_id == latest_ai_request_id:
            completed_ai_request_id = request_id


def clear_ai_acknowledgement(request_id: int) -> None:
    global pending_ai_ack_request_id, pending_ai_ack_phrase
    with request_state_lock:
        if pending_ai_ack_request_id == request_id:
            pending_ai_ack_request_id = 0
            pending_ai_ack_phrase = None


def should_interrupt_ai_acknowledgement(request_id: int) -> bool:
    with request_state_lock:
        return pending_ai_ack_request_id == request_id and bool(pending_ai_ack_phrase)


def choose_thinking_phrase() -> str:
    global last_thinking_phrase
    choices = [phrase for phrase in THINKING_PHRASES if phrase != last_thinking_phrase]
    phrase = random.choice(choices or THINKING_PHRASES)
    last_thinking_phrase = phrase
    return phrase


def choose_wake_greeting() -> str:
    global last_wake_greeting
    choices = [phrase for phrase in WAKE_GREETINGS if phrase != last_wake_greeting]
    phrase = random.choice(choices or WAKE_GREETINGS)
    last_wake_greeting = phrase
    return phrase.format(owner=owner_name())


def maybe_play_ai_acknowledgement(request_id: int, request_started_at: float) -> None:
    if time.perf_counter() - request_started_at < AI_ACK_DELAY_SECONDS:
        return
    phrase = choose_thinking_phrase()
    if not try_register_ai_acknowledgement(request_id, phrase):
        return
    respond(
        phrase,
        subtitle="Consulting the reasoning module.",
        focus_text="Preparing an AI response.",
        remember_history=False,
    )


def schedule_ai_acknowledgement(request_id: int, request_started_at: float) -> None:
    elapsed = time.perf_counter() - request_started_at
    remaining_delay = max(0.0, AI_ACK_DELAY_SECONDS - elapsed)
    timer = threading.Timer(
        remaining_delay,
        maybe_play_ai_acknowledgement,
        args=(request_id, request_started_at),
    )
    timer.daemon = True
    timer.start()


def store_ai_history(user_input: str, response_text: str) -> None:
    chat_history.append({"role": "user", "content": user_input})
    chat_history.append({"role": "assistant", "content": response_text})
    del chat_history[:-20]


def complete_ai_response_async(
    *,
    request_id: int,
    user_input: str,
    spoken_text: str,
    full_text: str | None,
    full_response_future,
) -> None:
    completed_text = full_text or spoken_text
    if full_response_future is not None:
        try:
            completed_text = full_response_future.result()
        except Exception as exc:
            log_event(f"ai_full_response_error:{exc}")
            completed_text = spoken_text

    if not is_current_ai_request(request_id):
        log_event(f"ai_full_response_discarded:{request_id}")
        return

    completed_text = str(completed_text or spoken_text).strip() or spoken_text
    if not completed_text:
        return

    store_ai_history(user_input, completed_text)

    if completed_text != spoken_text:
        log_event(f"full_text:{completed_text}")
        placeholder_responses = {"", "I'm working on that.", "Give me a moment... I'm processing that."}
        if str(spoken_text or "").strip() in placeholder_responses:
            respond(
                completed_text,
                subtitle="Full AI response completed.",
                focus_text="Final AI answer delivered.",
            )
            return
        update_ui(
            STATUS_ACTIVE,
            completed_text,
            assistant_name(),
            subtitle="Full AI response completed.",
            focus_text="Full response saved to the conversation.",
            log_message=True,
        )


def handle_ai_response_async(user_input: str, request_id: int) -> None:
    if command_handler is None:
        return

    try:
        ai_result = command_handler.handle(
            user_input,
            chat_history=chat_history,
            selected_action=selected_action_key,
            last_action=last_open_target,
        )
    except Exception as exc:
        log_event(f"ai_background_error:{exc}")
        if not is_current_ai_request(request_id):
            return
        mark_ai_request_complete(request_id)
        if voice_engine is not None and should_interrupt_ai_acknowledgement(request_id):
            voice_engine.interrupt_speech()
        clear_ai_acknowledgement(request_id)
        respond(
            "I ran into an internal AI error while processing that request.",
            subtitle="AI request failed.",
        )
        update_ui(STATUS_ACTIVE, "Ready for the next request.", "System", subtitle="Listening for your next instruction.", log_message=False)
        return

    if not is_current_ai_request(request_id):
        log_event(f"ai_response_discarded:{request_id}")
        return

    mark_ai_request_complete(request_id)
    if voice_engine is not None and should_interrupt_ai_acknowledgement(request_id):
        voice_engine.interrupt_speech()
    clear_ai_acknowledgement(request_id)
    if str(ai_result.response or "").strip():
        respond(
            ai_result.response,
            subtitle="AI response generated.",
            focus_text=ai_result.focus_text,
            remember_history=False,
            user_input=user_input,
        )
    threading.Thread(
        target=complete_ai_response_async,
        kwargs={
            "request_id": request_id,
            "user_input": user_input,
            "spoken_text": ai_result.response,
            "full_text": ai_result.full_response,
            "full_response_future": ai_result.full_response_future,
        },
        daemon=True,
    ).start()
    update_ui(STATUS_ACTIVE, "Ready for the next request.", "System", subtitle="Listening for your next instruction.", log_message=False)



def deep_merge(base: dict, incoming: dict) -> dict:
    merged = dict(base)
    for key, value in incoming.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged



def load_config() -> dict:
    if not CONFIG_PATH.exists():
        save_config(DEFAULT_CONFIG)
        return deep_merge({}, DEFAULT_CONFIG)

    try:
        raw = json.loads(CONFIG_PATH.read_text(encoding="utf-8-sig"))
        if not isinstance(raw, dict):
            raise ValueError("Config root must be a JSON object")
    except Exception as exc:
        log_event(f"config_load_error:{exc}")
        save_config(DEFAULT_CONFIG)
        return deep_merge({}, DEFAULT_CONFIG)

    merged = deep_merge(DEFAULT_CONFIG, raw)
    save_config(merged)
    return merged



def save_config(current: dict) -> None:
    CONFIG_PATH.write_text(json.dumps(current, indent=2), encoding="utf-8")



def assistant_name() -> str:
    return str(config.get("assistant_name") or "Jarvis")



def owner_name() -> str:
    return str(config.get("owner_name") or "Omair")



def find_music_file() -> Path | None:
    candidates = [MUSIC_BASE_NAME, "theme", "Theme"]
    for base_name in candidates:
        for extension in MUSIC_EXTENSIONS:
            candidate = APP_DIR / f"{base_name}{extension}"
            if candidate.exists():
                return candidate
    return None



def append_text(line: str, tone: str) -> None:
    if conversation_box is None:
        return
    conversation_box.configure(state="normal")
    conversation_box.insert(tk.END, line + "\n", tone)
    conversation_box.see(tk.END)
    conversation_box.configure(state="disabled")



def update_ui(
    status: str,
    message: str,
    speaker: str = "System",
    *,
    subtitle: str | None = None,
    focus_text: str | None = None,
    log_message: bool = True,
) -> None:
    ui_queue.put((status, message, speaker, subtitle or "", focus_text or "", log_message))



def set_orb_state(status: str) -> None:
    if orb_canvas is None or orb_outer is None or orb_inner is None:
        return
    if status == STATUS_ACTIVE:
        outer = "#1f4f66"
        inner = ACCENT
    elif status == STATUS_SPEAKING:
        outer = "#5c4f2a"
        inner = WARM
    elif status == STATUS_THINKING:
        outer = "#334b66"
        inner = "#8fc2ff"
    elif status == STATUS_ERROR:
        outer = "#5a2d35"
        inner = ERROR
    else:
        outer = "#183848"
        inner = ACCENT_SOFT
    orb_canvas.itemconfig(orb_outer, fill=outer)
    orb_canvas.itemconfig(orb_inner, fill=inner)



def process_ui_queue() -> None:
    while True:
        try:
            status, message, speaker, subtitle, focus_text, log_message = ui_queue.get_nowait()
        except queue.Empty:
            break

        if status_var is not None:
            status_var.set(status)
        if subtitle_var is not None and subtitle:
            subtitle_var.set(subtitle)
        if focus_var is not None and focus_text:
            focus_var.set(focus_text)
        if heard_var is not None and speaker == owner_name():
            heard_var.set(message)
        if response_var is not None and speaker == assistant_name():
            response_var.set(message)

        if log_message:
            tone = "system"
            if speaker == owner_name():
                tone = "user"
            elif speaker == assistant_name():
                tone = "assistant"
            append_text(f"{speaker}: {message}", tone)

        set_orb_state(status)

    refresh_memory_panels()

    if root is not None:
        root.after(100, process_ui_queue)



def ensure_audio_engine_ready() -> bool:
    if pygame.mixer.get_init():
        return True
    try:
        pygame.mixer.init()
        log_event("audio_engine_ready")
        return True
    except pygame.error as exc:
        log_event(f"audio_engine_error:{exc}")
        return False


def play_background_music(volume: float = AMBIENT_THEME_VOLUME, force_restart: bool = False) -> bool:
    if music_file is None:
        return False
    if not ensure_audio_engine_ready():
        return False
    try:
        busy = pygame.mixer.music.get_busy()
        if force_restart or not busy:
            pygame.mixer.music.load(str(music_file))
            pygame.mixer.music.play(-1)
        pygame.mixer.music.set_volume(volume)
        return True
    except pygame.error as exc:
        log_event(f"audio_engine_error:{exc}")
        return False


def stop_background_music() -> bool:
    if not pygame.mixer.get_init():
        return False
    was_busy = pygame.mixer.music.get_busy()
    pygame.mixer.music.stop()
    return bool(was_busy)


def execute_music_action(action: str) -> tuple[str, str]:
    if action == "stop":
        if stop_background_music():
            return "Stopping the theme music.", "Ambient theme paused."
        return "The theme music is already stopped.", "Ambient theme already paused."

    if music_file is None:
        log_event("theme_song_unavailable")
        return "I couldn't play the theme music because the audio file is unavailable.", "Theme audio file unavailable."
    if not ensure_audio_engine_ready():
        return "I couldn't start the audio engine for the theme music.", "Audio engine unavailable."

    if action == "restart":
        if play_background_music(force_restart=True):
            return "Restarting the theme music.", "Ambient theme restarting."
        return "I couldn't restart the theme music right now.", "Ambient theme restart failed."

    already_playing = pygame.mixer.music.get_busy()
    if play_background_music(force_restart=False):
        if already_playing:
            return "The theme music is already playing.", "Ambient theme already active."
        return "Playing the theme music.", "Ambient theme engaged."
    return "I couldn't play the theme music right now.", "Ambient theme start failed."



def duck_background_music() -> None:
    if pygame.mixer.get_init() and pygame.mixer.music.get_busy():
        try:
            pygame.mixer.music.set_volume(MUSIC_VOLUME)
        except pygame.error as exc:
            log_event(f"audio_engine_error:{exc}")



def restore_background_music() -> None:
    if pygame.mixer.get_init() and pygame.mixer.music.get_busy():
        try:
            pygame.mixer.music.set_volume(AMBIENT_THEME_VOLUME)
        except pygame.error as exc:
            log_event(f"audio_engine_error:{exc}")



def update_status_only(status: str, detail: str) -> None:
    update_ui(status, detail, "System", subtitle=detail, log_message=False)



def set_selected_action(action_key: str) -> None:
    global selected_action_key
    selected_action_key = action_key if action_key in APP_TARGETS else "chrome"
    label = APP_TARGETS[selected_action_key]["label"]
    update_ui(
        STATUS_ACTIVE if jarvis_active else STATUS_IDLE,
        f"{label} is selected.",
        "System",
        subtitle="Type a command or say open this to launch the selected option.",
        focus_text=f"Focused action: {label}",
        log_message=False,
    )


def respond(text: str, *, subtitle: str | None = None, focus_text: str | None = None, remember_history: bool = False, user_input: str | None = None) -> None:
    update_ui(
        STATUS_SPEAKING if jarvis_active else STATUS_IDLE,
        text,
        assistant_name(),
        subtitle=subtitle or "Voice response queued.",
        focus_text=focus_text or "Jarvis is replying.",
        log_message=True,
    )
    log_event(f"speak:{text}")
    if remember_history and user_input:
        chat_history.append({"role": "user", "content": user_input})
        chat_history.append({"role": "assistant", "content": text})
        del chat_history[:-20]
    if voice_engine is not None:
        voice_engine.speak_async(text)



def is_wake_phrase(text: str) -> bool:
    lowered = " ".join(text.lower().strip().split())
    return any(phrase in lowered for phrase in WAKE_PHRASES)



def handle_text_async(text: str) -> None:
    threading.Thread(target=handle_text, args=(text,), daemon=True).start()



def handle_text(text: str) -> None:
    global jarvis_active, last_open_target, shutdown_requested

    request_started_at = time.perf_counter()
    normalized = " ".join(text.lower().strip().split())
    if not normalized:
        return

    update_ui(STATUS_ACTIVE if jarvis_active else STATUS_IDLE, text, owner_name(), subtitle="Command received.")
    log_event(f"command:{normalized}")
    request_id = invalidate_pending_ai_requests()

    if is_wake_phrase(normalized):
        jarvis_active = True
        log_event("wake_phrase_detected")
        play_background_music(force_restart=False)
        respond(
            choose_wake_greeting(),
            subtitle="Wake phrase accepted.",
            focus_text="Voice and typed commands are both active.",
        )
        update_ui(
            STATUS_ACTIVE,
            "Jarvis systems are awake.",
            "System",
            subtitle="Offline-first recognition is active, with cloud fallback when available.",
            focus_text="Ask for apps, weather, files, time, location, or open-ended AI help.",
            log_message=False,
        )
        return

    if not jarvis_active:
        update_ui(
            STATUS_IDLE,
            "Waiting for the wake phrase.",
            "System",
            subtitle="Say 'Are you up Jarvis' or 'Jarvis are you up' to activate the assistant.",
            log_message=False,
        )
        return

    if EXIT_PHRASE in normalized:
        shutdown_requested = True
        respond(f"Alright Master {owner_name()}, call me if you need anything", subtitle="Shutting down Jarvis.")
        if root is not None:
            root.after(120, schedule_shutdown_after_speech)
        return

    update_ui(
        STATUS_THINKING,
        "Routing request.",
        "System",
        subtitle="Deciding whether to execute a system command or ask the AI model.",
        log_message=False,
    )

    try:
        result = command_handler.handle_local(
            text,
            selected_action=selected_action_key,
            last_action=last_open_target,
        )
    except Exception as exc:
        log_event(f"command_handler_error:{exc}")
        respond("I ran into an internal command error while processing that request.", subtitle="Command handling failed.")
        return

    if result is not None:
        if result.selected_action:
            set_selected_action(result.selected_action)
        if result.last_action:
            last_open_target = result.last_action
        if result.music_action:
            result.response, result.focus_text = execute_music_action(result.music_action)

        respond(
            result.response,
            subtitle="Command executed.",
            focus_text=result.focus_text,
            remember_history=False,
            user_input=text,
        )
        update_ui(STATUS_ACTIVE, "Ready for the next request.", "System", subtitle="Listening for your next instruction.", log_message=False)
        return

    schedule_ai_acknowledgement(request_id, request_started_at)
    threading.Thread(
        target=handle_ai_response_async,
        args=(text, request_id),
        daemon=True,
    ).start()



def on_voice_error(message: str) -> None:
    update_ui(STATUS_ERROR, message, "System", subtitle="Voice listener issue detected.")



def on_manual_command(_event=None) -> None:
    if command_entry is None:
        return
    text = command_entry.get().strip()
    if not text:
        return
    command_entry.delete(0, tk.END)
    handle_text_async(text)



def initialize_services() -> None:
    global voice_engine, command_handler, music_file, reminder_thread

    log_event("initialize_services")
    music_file = find_music_file()
    try:
        pygame.mixer.init()
        log_event("audio_engine_ready")
    except pygame.error as exc:
        log_event(f"audio_engine_error:{exc}")
        update_ui(STATUS_ERROR, "Background audio engine could not start.", subtitle="Jarvis can still listen and speak without the music layer.")

    if music_file is None:
        log_event("background_music_missing")
        update_ui(STATUS_IDLE, "Background music file not found.", subtitle="Add theme.mp3 beside the script if you want the cinematic background track.")
    else:
        try:
            pygame.mixer.music.load(str(music_file))
        except pygame.error as exc:
            log_event(f"background_music_corrupt:{exc}")
            music_file = None
        play_background_music(force_restart=True)

    initialize_ai_health(config, log_event)
    command_handler = CommandHandler(config=config, project_dir=APP_DIR, logger=log_event)
    voice_engine = VoiceEngine(
        config=config,
        logger=log_event,
        on_status=update_status_only,
        on_speech_start=duck_background_music,
        on_speech_end=restore_background_music,
    )
    voice_engine.listen_continuously(on_text=handle_text_async, on_error=on_voice_error)
    log_event("listener_thread_starting")
    reminder_stop_event.clear()
    reminder_thread = threading.Thread(target=reminder_loop, daemon=True)
    reminder_thread.start()
    refresh_memory_panels(force=True)



def create_info_card(parent: tk.Widget, title: str, value_var: tk.StringVar, width: int) -> tk.Frame:
    card = tk.Frame(parent, bg=CARD_ALT, highlightbackground=BORDER, highlightthickness=1, bd=0)
    card.configure(width=width, height=96)
    card.pack_propagate(False)
    tk.Label(card, text=title, bg=CARD_ALT, fg=MUTED, font=("Segoe UI", 9, "bold")).pack(anchor="w", padx=14, pady=(14, 6))
    tk.Label(card, textvariable=value_var, bg=CARD_ALT, fg=TEXT, justify="left", wraplength=width - 28, font=("Segoe UI Semibold", 11)).pack(anchor="w", padx=14)
    return card



def build_quick_actions(parent: tk.Widget) -> None:
    quick_actions = tk.Frame(parent, bg=PANEL)
    quick_actions.pack(fill="x", pady=(14, 0))
    labels = [
        ("chrome", "Chrome"),
        ("whatsapp", "WhatsApp"),
        ("notepad", "Notepad"),
        ("calculator", "Calculator"),
        ("explorer", "Explorer"),
        ("settings", "Settings"),
        ("youtube", "YouTube"),
    ]
    for index, (action_key, label) in enumerate(labels):
        button = tk.Button(
            quick_actions,
            text=label,
            command=lambda key=action_key: set_selected_action(key),
            bg=CARD,
            fg=TEXT,
            activebackground=ACCENT_SOFT,
            activeforeground=BG,
            relief="flat",
            padx=12,
            pady=8,
            font=("Segoe UI Semibold", 10),
            cursor="hand2",
        )
        button.grid(row=index // 4, column=index % 4, sticky="ew", padx=6, pady=6)
    for column in range(4):
        quick_actions.grid_columnconfigure(column, weight=1)



def build_gui() -> None:
    global root, status_var, subtitle_var, heard_var, response_var, focus_var, memory_var, reminders_var
    global conversation_box, command_entry, orb_canvas, orb_outer, orb_inner

    root = tk.Tk()
    root.title("Jarvis Command Center")
    root.geometry(WINDOW_SIZE)
    root.minsize(1100, 720)
    root.configure(bg=BG)
    root.protocol("WM_DELETE_WINDOW", on_close)

    status_var = tk.StringVar(value=STATUS_IDLE)
    subtitle_var = tk.StringVar(value="Initializing command center.")
    heard_var = tk.StringVar(value="No voice input detected yet.")
    response_var = tk.StringVar(value="No assistant response yet.")
    focus_var = tk.StringVar(value="Focused action: Google Chrome")
    memory_var = tk.StringVar(value="No personal memory stored yet.")
    reminders_var = tk.StringVar(value="No active reminders.")

    root.grid_columnconfigure(0, weight=3)
    root.grid_columnconfigure(1, weight=2)
    root.grid_rowconfigure(0, weight=1)

    left = tk.Frame(root, bg=BG)
    left.grid(row=0, column=0, sticky="nsew", padx=(22, 10), pady=22)
    left.grid_rowconfigure(2, weight=1)
    left.grid_columnconfigure(0, weight=1)

    hero = tk.Frame(left, bg=PANEL, highlightbackground=BORDER, highlightthickness=1, bd=0)
    hero.grid(row=0, column=0, sticky="ew")
    tk.Label(hero, text="STARK DESKTOP INTERFACE", bg=PANEL, fg=MUTED, font=("Segoe UI", 10, "bold")).pack(anchor="w", padx=20, pady=(18, 6))
    tk.Label(hero, text="JARVIS COMMAND CENTER", bg=PANEL, fg=TEXT, font=("Segoe UI Semibold", 24)).pack(anchor="w", padx=20)
    tk.Label(hero, textvariable=subtitle_var, bg=PANEL, fg=ACCENT, font=("Segoe UI", 11), wraplength=620, justify="left").pack(anchor="w", padx=20, pady=(8, 18))

    metrics = tk.Frame(left, bg=BG)
    metrics.grid(row=1, column=0, sticky="ew", pady=(14, 14))
    for column in range(4):
        metrics.grid_columnconfigure(column, weight=1)

    create_info_card(metrics, "System State", status_var, 190).grid(row=0, column=0, sticky="ew", padx=(0, 8))
    create_info_card(metrics, "Last Heard", heard_var, 190).grid(row=0, column=1, sticky="ew", padx=8)
    create_info_card(metrics, "Last Response", response_var, 190).grid(row=0, column=2, sticky="ew", padx=8)
    create_info_card(metrics, "Current Focus", focus_var, 190).grid(row=0, column=3, sticky="ew", padx=(8, 0))
    create_info_card(metrics, "Memory", memory_var, 396).grid(row=1, column=0, columnspan=2, sticky="ew", padx=(0, 8), pady=(10, 0))
    create_info_card(metrics, "Reminders", reminders_var, 396).grid(row=1, column=2, columnspan=2, sticky="ew", padx=(8, 0), pady=(10, 0))

    transcript_wrap = tk.Frame(left, bg=PANEL, highlightbackground=BORDER, highlightthickness=1, bd=0)
    transcript_wrap.grid(row=2, column=0, sticky="nsew")
    transcript_wrap.grid_rowconfigure(1, weight=1)
    transcript_wrap.grid_columnconfigure(0, weight=1)

    tk.Label(transcript_wrap, text="Conversation Feed", bg=PANEL, fg=TEXT, font=("Segoe UI Semibold", 14)).grid(row=0, column=0, sticky="w", padx=18, pady=(16, 6))

    conversation_box = ScrolledText(
        transcript_wrap,
        bg="#081722",
        fg=TEXT,
        insertbackground=ACCENT,
        relief="flat",
        borderwidth=0,
        wrap="word",
        font=("Segoe UI", 11),
        padx=10,
        pady=10,
    )
    conversation_box.grid(row=1, column=0, sticky="nsew", padx=18, pady=(0, 14))
    conversation_box.configure(state="disabled")
    conversation_box.tag_config("assistant", foreground=ACCENT)
    conversation_box.tag_config("user", foreground=SUCCESS)
    conversation_box.tag_config("system", foreground=MUTED)

    entry_row = tk.Frame(transcript_wrap, bg=PANEL)
    entry_row.grid(row=2, column=0, sticky="ew", padx=18, pady=(0, 18))
    entry_row.grid_columnconfigure(0, weight=1)

    command_entry = tk.Entry(entry_row, bg="#0b1d2b", fg=TEXT, insertbackground=ACCENT, relief="flat", font=("Segoe UI", 11))
    command_entry.grid(row=0, column=0, sticky="ew", ipady=10)
    command_entry.bind("<Return>", on_manual_command)

    send_button = tk.Button(
        entry_row,
        text="Send Command",
        command=on_manual_command,
        bg=ACCENT_SOFT,
        fg=BG,
        activebackground=ACCENT,
        activeforeground=BG,
        relief="flat",
        padx=16,
        pady=10,
        font=("Segoe UI Semibold", 10),
        cursor="hand2",
    )
    send_button.grid(row=0, column=1, padx=(10, 0))

    right = tk.Frame(root, bg=BG)
    right.grid(row=0, column=1, sticky="nsew", padx=(10, 22), pady=22)
    right.grid_columnconfigure(0, weight=1)

    status_panel = tk.Frame(right, bg=PANEL_ALT, highlightbackground=BORDER, highlightthickness=1, bd=0)
    status_panel.grid(row=0, column=0, sticky="ew")
    tk.Label(status_panel, text="Core Interface", bg=PANEL_ALT, fg=MUTED, font=("Segoe UI", 10, "bold")).pack(pady=(16, 4))

    orb_canvas = tk.Canvas(status_panel, width=210, height=210, bg=PANEL_ALT, highlightthickness=0, bd=0)
    orb_canvas.pack(pady=(4, 8))
    orb_outer = orb_canvas.create_oval(20, 20, 190, 190, fill="#183848", outline="")
    orb_inner = orb_canvas.create_oval(58, 58, 152, 152, fill=ACCENT_SOFT, outline="")

    tk.Label(status_panel, textvariable=status_var, bg=PANEL_ALT, fg=TEXT, font=("Segoe UI Semibold", 18)).pack()
    tk.Label(status_panel, text="Voice-first AI with offline speech fallback, command routing, and contextual replies.", bg=PANEL_ALT, fg=MUTED, wraplength=320, justify="center", font=("Segoe UI", 10)).pack(padx=18, pady=(8, 18))

    actions_panel = tk.Frame(right, bg=PANEL, highlightbackground=BORDER, highlightthickness=1, bd=0)
    actions_panel.grid(row=1, column=0, sticky="ew", pady=(14, 14))
    tk.Label(actions_panel, text="Quick Actions", bg=PANEL, fg=TEXT, font=("Segoe UI Semibold", 14)).pack(anchor="w", padx=18, pady=(16, 2))
    tk.Label(actions_panel, text="Select an option, then say 'open this' or type it directly.", bg=PANEL, fg=MUTED, font=("Segoe UI", 10)).pack(anchor="w", padx=18)
    build_quick_actions(actions_panel)

    bottom_panel = tk.Frame(right, bg=PANEL_ALT, highlightbackground=BORDER, highlightthickness=1, bd=0)
    bottom_panel.grid(row=2, column=0, sticky="ew")
    tk.Label(bottom_panel, text="Natural Prompts", bg=PANEL_ALT, fg=TEXT, font=("Segoe UI Semibold", 14)).pack(anchor="w", padx=18, pady=(16, 4))
    tk.Label(
        bottom_panel,
        text="Try: 'Jarvis are you up', 'what time is it', 'where am I', 'what\'s the weather', 'create folder notes', 'write hello into notes.txt', or ask a general AI question.",
        bg=PANEL_ALT,
        fg=MUTED,
        wraplength=320,
        justify="left",
        font=("Segoe UI", 10),
    ).pack(anchor="w", padx=18, pady=(0, 16))



def on_close() -> None:
    reminder_stop_event.set()
    if voice_engine is not None:
        voice_engine.stop()
    stop_background_music()
    if root is not None:
        root.destroy()



def main() -> None:
    global config
    log_event("main_start")
    config = load_config()
    build_gui()
    initialize_services()
    set_selected_action(selected_action_key)
    update_ui(
        STATUS_IDLE,
        "Jarvis command center is starting up.",
        "System",
        subtitle="Preparing offline-first voice recognition, command routing, and AI context.",
    )
    if root is not None:
        log_event("tk_mainloop_enter")
        root.after(100, process_ui_queue)
        root.mainloop()


if __name__ == "__main__":
    main()
