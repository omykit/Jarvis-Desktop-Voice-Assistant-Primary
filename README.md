# Jarvis — Intelligent Desktop Voice Automation & Response System

A privacy-first, fully local voice assistant for Windows. Jarvis listens for a wake phrase, executes desktop automation commands directly, and escalates open-ended queries to a locally hosted LLM — all without sending data to the cloud.

Inspired by the Marvel Cinematic Universe's J.A.R.V.I.S., built with Python, Tkinter, and Ollama.

![Status](https://img.shields.io/badge/status-active-brightgreen) ![Python](https://img.shields.io/badge/python-3.13-blue) ![Platform](https://img.shields.io/badge/platform-Windows-lightgrey)

---

## Why Jarvis

Commercial assistants (Siri, Alexa, Google Assistant) are convenient but cloud-dependent — every command leaves your machine. Jarvis was built to answer a simple question: **can a single developer build a genuinely useful voice assistant that never phones home?**

- **Fully offline core** — speech recognition (Vosk), command routing, memory, and reminders all run locally with zero network dependency.
- **Local LLM conversation** — Qwen3:4b / LLaMA via Ollama for open-ended queries, so nothing leaves the machine unless you explicitly configure a cloud endpoint.
- **Deep desktop integration** — opens apps, manages files, sets reminders, and more, directly on Windows.
- **Daily-driver stable** — 13 sessions across 12 days of real use with zero crashes during evaluation.

## Features

- Wake-phrase activation ("Are you up Jarvis", "Hey Jarvis") with confidence-scored offline transcription (Vosk)
- TTS echo guard to prevent the mic from picking up Jarvis's own voice
- Local LLM conversation via Ollama (Qwen3, LLaMA) with streaming responses and an "acknowledgement" filler phrase for long-running queries
- Desktop automation: launch Chrome, YouTube, WhatsApp, Notepad, Calculator, File Explorer, Settings
- File/folder creation and note-writing by voice
- Persistent JSON-backed memory and reminder scheduling
- Live graphical dashboard (Tkinter) — conversation feed, status orb, weather, quick actions
- Optional neural TTS via Piper for higher-quality voice output

## Architecture

Jarvis is organized into six layers: UI, speech recognition, command processing, AI processing, automation/execution, and configuration/data. A multi-threaded design (Tkinter main thread + daemon threads for listening, TTS, AI inference, and reminders) keeps the UI responsive under slow I/O.

| Module | Responsibility |
|---|---|
| `jarvis_desktop.py` | Entry point, GUI, thread coordination, config management |
| `voice_engine.py` | Vosk + pyttsx3/Piper, echo guard, continuous listener |
| `command_handler.py` | Intent routing, app launching, file operations |
| `memory_module.py` | JSON-backed memory, reminder scheduling & polling |
| `ai_module.py` | Ollama API client, health checks, streaming responses |
| `jarvis_config.json` | User-editable voice, AI, and weather settings |

## Tech Stack

Python 3.13 · Tkinter · Vosk (offline speech recognition) · pyttsx3 / Piper (TTS) · Ollama (Qwen3:4b, LLaMA) · SpeechRecognition · WeatherAPI

## Getting Started

**Requirements:** Windows 10/11 (64-bit), Python 3.11+ (3.13 recommended), 8GB+ RAM (16GB recommended), ~3GB storage for Vosk + Ollama models, a microphone. Internet is only required for weather lookups.

```bash
git clone https://github.com/omair-kittur/<repo-name>.git
cd <repo-name>
pip install pygame pyttsx3 SpeechRecognition vosk requests

# Pull the local LLM (requires Ollama installed: https://ollama.com)
ollama pull qwen3:4b

# Run
python jarvis_desktop.py
```

Configure your assistant name, AI model, voice settings, and WeatherAPI key in `jarvis_config.json` — see the full reference in the [project report]([https://docs.google.com/document/d/1HIkcicidkideLtPx7-u8gQ0YpSWQAg5D7N_y12jShbk/edit?usp=drivesdk]).

## Voice Commands (selected)

| Say | Action |
|---|---|
| "Are you up Jarvis" | Wake phrase |
| "Open Chrome / YouTube / Calculator / ..." | Launches the app |
| "Create folder [name]" | Creates a folder in the workspace |
| "Set a reminder for [time] to [task]" | Creates a timed reminder |
| "What's the weather" | Fetches current weather |
| *(anything else)* | Escalated to the local LLM |

Full command reference is in the project report (Appendix B).

## Results

- Reliable wake-phrase detection across supported phrase variants; high transcription accuracy on 5–15 word commands
- All 7 quick-action app targets launched successfully in every test, sub-1s latency
- 13 stable sessions over 12 days of daily use, zero crashes
- Known limitations: single-turn conversation context (configurable), requires Ollama running before launch, voice accuracy degrades in noisy environments

See the full [project report]([https://docs.google.com/document/d/1HIkcicidkideLtPx7-u8gQ0YpSWQAg5D7N_y12jShbk/edit?usp=drivesdk]) for methodology, architecture diagrams, and detailed results.

## Roadmap

- Multi-turn conversation history (config flag already exists, needs tuning)
- Bundle Piper neural TTS by default for a more natural voice
- GUI settings editor (no more manual JSON editing)
- Smart home / IoT command bridge (MQTT, Home Assistant)
- Wake-word-only passive mode for lower CPU usage

## Author

**Omair Ahmed Kittur** — [github.com/omair-kittur](https://github.com/omair-kittur) · [linkedin.com/in/omair-ahmed-kittur](https://linkedin.com/in/omair-ahmed-544ab62b1) · omairkittur2@gmail.com

Built as a final-year project, BCA (Artificial Intelligence), Jain University, 2026.
