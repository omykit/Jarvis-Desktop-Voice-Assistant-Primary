from __future__ import annotations

import json
import os
import queue
import random
import re
import subprocess
import sys
import tempfile
import threading
import time
from collections import deque
from difflib import SequenceMatcher
from pathlib import Path
from typing import Callable

import pygame
import pyttsx3
import speech_recognition as sr
try:
    import audioop
except ImportError:
    try:
        import audioop_lts as audioop
    except ImportError:
        audioop = None

try:
    from vosk import KaldiRecognizer, Model
except ImportError:
    KaldiRecognizer = None
    Model = None


Logger = Callable[[str], None]
TTS_ENGINE = "piper"  # options: "piper", "pyttsx3"
DEFAULT_PIPER_MODEL = Path("voices/en_US-lessac-medium.onnx")
DEFAULT_PIPER_CONFIG = Path("voices/en_US-lessac-medium.onnx.json")
_LISTENER_LOCK = threading.Lock()
_ACTIVE_LISTENER: "VoiceEngine | None" = None
DEFAULT_VOSK_MODELS = [
    Path("vosk-model-en-us-0.22"),
    Path("vosk-model-small-en-us-0.15"),
]
STT_ALLOWED_SHORT_COMMANDS = {
    "date",
    "help",
    "jarvis",
    "no",
    "stop",
    "thanks",
    "time",
    "yes",
}
JARVIS_PREFIXES = ["Certainly.", "Right away.", "Of course.", "Understood."]
INTERRUPT_KEYWORDS = ("stop", "wait", "jarvis")
CHUNK_PAUSE_SECONDS = 0.05


class VoiceEngine:
    """Thread-safe speech input and output with offline-first recognition."""

    def __init__(
        self,
        *,
        config: dict,
        logger: Logger | None = None,
        on_status: Callable[[str, str], None] | None = None,
        on_speech_start: Callable[[], None] | None = None,
        on_speech_end: Callable[[], None] | None = None,
    ) -> None:
        self.config = config
        self.logger = logger
        self.on_status = on_status
        self.on_speech_start = on_speech_start
        self.on_speech_end = on_speech_end
        self.recognizer = sr.Recognizer()
        self.recognizer.dynamic_energy_threshold = True
        self.recognizer.pause_threshold = float(config.get("voice", {}).get("pause_threshold", 0.8))
        self.recognizer.energy_threshold = int(config.get("voice", {}).get("energy_threshold", 250))
        self.recognizer.non_speaking_duration = float(
            config.get("voice", {}).get("non_speaking_duration", 0.5)
        )
        self.recognizer.operation_timeout = float(config.get("voice", {}).get("operation_timeout", 8))
        self.stop_event = threading.Event()
        self.speaking_event = threading.Event()
        self.stop_speaking_event = threading.Event()
        self.tts_queue: queue.Queue[str] = queue.Queue()
        self.listener_thread: threading.Thread | None = None
        self._engine: pyttsx3.Engine | None = None
        self._engine_lock = threading.Lock()
        self._playback_lock = threading.Lock()
        self._tts_engine_name = str(config.get("voice", {}).get("tts_engine", TTS_ENGINE)).lower()
        self._piper_model_path = self._resolve_path(
            config.get("voice", {}).get("piper_model_path") or DEFAULT_PIPER_MODEL
        )
        self._piper_config_path = self._resolve_path(
            config.get("voice", {}).get("piper_config_path") or DEFAULT_PIPER_CONFIG
        )
        self._vosk_confidence_floor = float(config.get("voice", {}).get("vosk_confidence_floor", 0.45))
        self._vosk_short_confidence_floor = float(
            config.get("voice", {}).get("vosk_short_confidence_floor", 0.65)
        )
        self._vosk_debug_raw = bool(config.get("voice", {}).get("vosk_debug_raw", False))
        self._min_transcript_chars = int(config.get("voice", {}).get("min_transcript_chars", 4))
        self._min_transcript_words = int(config.get("voice", {}).get("min_transcript_words", 1))
        self._vosk_phrase_bias = list(
            config.get("voice", {}).get(
                "vosk_phrase_bias",
                [
                    "jarvis",
                    "are you up jarvis",
                    "jarvis are you up",
                    "open",
                    "what time is it",
                    "time",
                    "date",
                    "what is the date",
                    "create file",
                    "create folder",
                    "[unk]",
                ],
            )
        )
        self._vosk_runtime_grammar_enabled = bool(
            config.get("voice", {}).get("vosk_runtime_grammar_enabled", False)
        )
        self._tts_prefix_probability = float(config.get("voice", {}).get("tts_prefix_probability", 0.0))
        self._echo_guard_seconds = float(config.get("voice", {}).get("echo_guard_seconds", 1.8))
        self._echo_similarity_floor = float(config.get("voice", {}).get("echo_similarity_floor", 0.72))
        self._allow_google_fallback = bool(config.get("voice", {}).get("allow_google_fallback", False))
        self._google_fallback_cooldown_seconds = float(
            config.get("voice", {}).get("google_fallback_cooldown_seconds", 90)
        )
        self._google_fallback_retry_after = 0.0
        self._recent_spoken_chunks: deque[tuple[float, str]] = deque(maxlen=18)
        self._last_speech_ended_at = 0.0
        self._tts_channel: pygame.mixer.Channel | None = None
        self._vosk_model = self._load_vosk_model()
        self._ensure_pygame_mixer()
        self._ensure_engine()
        self.tts_thread = threading.Thread(target=self._tts_worker, daemon=True)
        self.tts_thread.start()

    def _log(self, message: str) -> None:
        if self.logger is not None:
            self.logger(message)

    def _set_status(self, state: str, detail: str) -> None:
        if self.on_status is not None:
            self.on_status(state, detail)

    def _resolve_path(self, value: str | os.PathLike[str]) -> Path:
        path = Path(value)
        if not path.is_absolute():
            path = Path(__file__).resolve().parent / path
        return path

    def _load_vosk_model(self):
        voice_config = self.config.get("voice", {}) if isinstance(self.config, dict) else {}
        configured_model = voice_config.get("vosk_model_path")
        candidate_paths = []
        if configured_model:
            candidate_paths.append(Path(configured_model))
        candidate_paths.extend(DEFAULT_VOSK_MODELS)
        resolved_paths: list[Path] = []
        for candidate in candidate_paths:
            path = Path(candidate)
            if not path.is_absolute():
                path = Path(__file__).resolve().parent / path
            if path not in resolved_paths:
                resolved_paths.append(path)

        if Model is None:
            self._log("[STT ERROR] VOSK package unavailable")
            return None
        for path in resolved_paths:
            if not path.exists() or not path.is_dir():
                self._log(f"[STT ERROR] Model not found: {path}")
                continue
            try:
                model = Model(str(path))
                self._log(f"[STT] VOSK model loaded: {path}")
                return model
            except Exception as exc:
                self._log(f"[STT ERROR] VOSK model load failed: {path}: {exc}")
        self._log("[STT ERROR] No usable VOSK model found; Google speech fallback remains available")
        return None

    def _ensure_engine(self) -> pyttsx3.Engine | None:
        if self._engine is not None:
            return self._engine
        try:
            engine = pyttsx3.init()
            rate = int(self.config.get("voice", {}).get("tts_rate", 175))
            engine.setProperty("rate", rate)
            self._engine = engine
            self._log("speech_engine_ready")
        except Exception as exc:
            self._engine = None
            self._log(f"speech_engine_error:{exc}")
        return self._engine

    def _ensure_pygame_mixer(self) -> None:
        if not pygame.mixer.get_init():
            pygame.mixer.pre_init(22050, -16, 1, 512)
            pygame.mixer.init()
            self._log("pygame_mixer_ready")

    def speak_async(self, text: str) -> None:
        cleaned = text.strip()
        if cleaned:
            self.tts_queue.put(cleaned)

    def _clear_tts_queue(self) -> None:
        cleared_any = False
        while True:
            try:
                self.tts_queue.get_nowait()
                self.tts_queue.task_done()
                cleared_any = True
            except queue.Empty:
                break
        if cleared_any:
            self._log("[TTS] Queue Cleared")

    def _stop_tts_playback(self) -> None:
        with self._playback_lock:
            if self._tts_channel is None:
                return
            self._tts_channel.stop()

    def interrupt_speech(self) -> None:
        self._log("tts_interrupt_requested")
        self.stop_speaking_event.set()
        self._clear_tts_queue()
        try:
            self._stop_tts_playback()
            self._log("[TTS] Interrupted")
        except Exception as exc:
            self._log(f"tts_interrupt_music_error:{exc}")

        if self._engine is not None:
            try:
                self._engine.stop()
            except Exception as exc:
                self._log(f"tts_interrupt_engine_error:{exc}")

    def _prepare_tts_text(self, text: str) -> str:
        cleaned = re.sub(r"<think>.*?</think>", " ", str(text or ""), flags=re.DOTALL | re.IGNORECASE)
        cleaned = re.sub(r"</?think>", " ", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"```.*?```", " ", cleaned, flags=re.DOTALL)
        cleaned = re.sub(r"https?://\S+|www\.\S+", " link ", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"`([^`]*)`", r"", cleaned)
        cleaned = re.sub(r"^\s*(?:[-*+]|\d+[.)])\s+", "", cleaned, flags=re.MULTILINE)
        cleaned = cleaned.replace("&", " and ")
        cleaned = re.sub(r"(?<=\w)/(?=\w)", " or ", cleaned)
        cleaned = re.sub(r"[*_`~#<>[\]{}|^]", " ", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        cleaned = re.sub(r"(current|right now|currently|approximately|basically|actually|just|simply)", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"(in your local timezone)", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s+", " ", cleaned).strip(" ,.")
        cleaned = re.sub(r"(?i)^the current time(?: right now)? is\s+", "The time is ", cleaned)
        cleaned = re.sub(r"(?i)^the current date(?: right now)? is\s+", "The date is ", cleaned)
        cleaned = re.sub(r"(?i)^the weather (?:right now )?is\s+", "The weather is ", cleaned)
        cleaned = re.sub(r"(?i)^currently,?\s*", "", cleaned)
        cleaned = re.sub(r"(?i)please note that", "", cleaned)
        cleaned = re.sub(r"(?i)i can tell you that", "", cleaned)
        cleaned = re.sub(r"(?i)it looks like", "", cleaned)
        cleaned = re.sub(r"(?i)for your reference", "", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip(" ,.")
        sentences = [
            sentence.strip(" ,.")
            for sentence in re.split(r"(?<=[.!?])\s+", cleaned)
            if sentence.strip(" ,.")
        ]
        if len(sentences) > 3:
            sentences = sentences[:3]
        normalized_sentences: list[str] = []
        for sentence in sentences:
            sentence = sentence[0].upper() + sentence[1:] if sentence else sentence
            if sentence and sentence[-1] not in ".!?":
                sentence += "."
            normalized_sentences.append(sentence)
        cleaned = " ".join(normalized_sentences).strip()
        if len(cleaned) > 420:
            cleaned = cleaned[:420].rsplit(" ", 1)[0].strip() or cleaned[:420]
        if cleaned and random.random() < self._tts_prefix_probability:
            cleaned = f"{random.choice(JARVIS_PREFIXES[:2])} {cleaned}"
        return cleaned

    def _split_spoken_sentences(self, text: str) -> list[str]:
        parts = re.split(r"(?<=[.!?])\s+", text)
        return [part.strip() for part in parts if part.strip()]

    def _split_tts_chunks(self, text: str) -> list[str]:
        prepared = self._prepare_tts_text(text)
        if not prepared:
            return []

        if len(prepared) <= 260:
            return [prepared]

        raw_chunks = self._split_spoken_sentences(prepared)
        if not raw_chunks:
            raw_chunks = [prepared]

        chunks: list[str] = []
        buffer = ""
        for part in raw_chunks:
            if len(part) > 260:
                if buffer:
                    chunks.append(buffer)
                    buffer = ""
                chunks.extend(self._split_long_tts_text(part))
                continue
            if not buffer:
                buffer = part
                continue
            if len(buffer) + 1 + len(part) <= 260:
                buffer = f"{buffer} {part}"
                continue
            chunks.append(buffer)
            buffer = part
        if buffer:
            chunks.append(buffer)
        return chunks

    def _split_long_tts_text(self, text: str) -> list[str]:
        words = text.split()
        chunks: list[str] = []
        buffer = ""
        for word in words:
            candidate = f"{buffer} {word}".strip()
            if len(candidate) <= 220:
                buffer = candidate
                continue
            if buffer:
                chunks.append(buffer)
            buffer = word
        if buffer:
            chunks.append(buffer)
        return chunks

    def _normalize_match_text(self, text: str) -> str:
        return " ".join(re.sub(r"[^a-z0-9\s]", " ", str(text or "").casefold()).split())

    def _remember_spoken_chunk(self, text: str) -> None:
        normalized = self._normalize_match_text(text)
        if normalized:
            self._recent_spoken_chunks.append((time.monotonic(), normalized))

    def _looks_like_tts_echo(self, text: str, *, speaking_now: bool) -> bool:
        normalized = self._normalize_match_text(text)
        if not normalized:
            return False

        now = time.monotonic()
        while self._recent_spoken_chunks and now - self._recent_spoken_chunks[0][0] > self._echo_guard_seconds:
            self._recent_spoken_chunks.popleft()

        if not speaking_now and now - self._last_speech_ended_at > self._echo_guard_seconds:
            return False

        transcript_words = normalized.split()
        for _, spoken in self._recent_spoken_chunks:
            if normalized == spoken or normalized in spoken or spoken in normalized:
                return True
            if len(transcript_words) >= 2 and SequenceMatcher(None, normalized, spoken).ratio() >= self._echo_similarity_floor:
                return True
            spoken_words = spoken.split()
            overlap = len(set(transcript_words) & set(spoken_words))
            if len(transcript_words) >= 2 and overlap >= max(2, min(len(set(transcript_words)), len(set(spoken_words))) - 1):
                return True
        return False

    def speak_pyttsx3(self, text: str) -> None:
        engine = self._ensure_engine()
        if engine is None:
            raise RuntimeError("pyttsx3 engine is unavailable")
        with self._engine_lock:
            try:
                engine.stop()
            except Exception as exc:
                self._log(f"speech_engine_stop_warning:{exc}")
            engine.say(text)
            engine.runAndWait()

    def speak_piper(self, text: str) -> None:
        if not self._piper_model_path.exists():
            raise FileNotFoundError(f"Piper model not found: {self._piper_model_path}")
        if not self._piper_config_path.exists():
            raise FileNotFoundError(f"Piper config not found: {self._piper_config_path}")

        self._ensure_pygame_mixer()
        prepared_text = self._prepare_tts_text(text)
        if not prepared_text:
            return

        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
            temp_path = Path(temp_audio.name)

        try:
            command = [
                sys.executable,
                "-m",
                "piper",
                "-m",
                str(self._piper_model_path),
                "-c",
                str(self._piper_config_path),
                "-f",
                str(temp_path),
            ]
            subprocess.run(
                command,
                input=prepared_text + "\n",
                text=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=True,
            )

            sound = pygame.mixer.Sound(str(temp_path))
            with self._playback_lock:
                self._tts_channel = pygame.mixer.find_channel(True)
                if self._tts_channel is None:
                    raise RuntimeError("No playback channel available for Piper")
                self._tts_channel.play(sound)
            while self._tts_channel is not None and self._tts_channel.get_busy() and not self.stop_event.is_set():
                if self.stop_speaking_event.is_set():
                    self._stop_tts_playback()
                    break
                time.sleep(0.02)
            self._stop_tts_playback()
        finally:
            try:
                temp_path.unlink(missing_ok=True)
            except OSError:
                pass

    def _tts_worker(self) -> None:
        while not self.stop_event.is_set():
            try:
                text = self.tts_queue.get(timeout=0.2)
            except queue.Empty:
                continue

            self.stop_speaking_event.clear()
            self.speaking_event.set()
            self._set_status("Speaking", "Jarvis is speaking.")
            if self.on_speech_start is not None:
                self.on_speech_start()
            try:
                self._log("[TTS] Started")
                chunks = self._split_tts_chunks(text)
                total_chunks = len(chunks)
                for index, chunk in enumerate(chunks, start=1):
                    if self.stop_speaking_event.is_set():
                        self._log("[TTS] Interrupted between chunks")
                        break
                    self._log(f"[TTS] Chunk {index}/{total_chunks}")
                    self._remember_spoken_chunk(chunk)
                    if self._tts_engine_name == "piper":
                        self._log("[TTS] Using Piper")
                        try:
                            self.speak_piper(chunk)
                        except Exception as exc:
                            self._log(f"[PIPER ERROR] {exc}")
                            self._log("[TTS] Using pyttsx3")
                            self.speak_pyttsx3(chunk)
                    else:
                        self._log("[TTS] Using pyttsx3")
                        self.speak_pyttsx3(chunk)

                    if index < total_chunks and not self.stop_event.is_set() and not self.stop_speaking_event.is_set():
                        time.sleep(CHUNK_PAUSE_SECONDS)
                if self.stop_speaking_event.is_set():
                    self._clear_tts_queue()
                else:
                    self._log("[TTS] Completed")
            except Exception as exc:
                self._log(f"tts_runtime_error:{exc}")
                self._engine = None
                try:
                    self._log("[TTS] Using pyttsx3")
                    self.speak_pyttsx3(text)
                except Exception as retry_exc:
                    self._log(f"tts_retry_error:{retry_exc}")
                    self._set_status("Attention Required", "Speech output is unavailable.")
            finally:
                self._last_speech_ended_at = time.monotonic()
                if self.on_speech_end is not None:
                    self.on_speech_end()
                self.speaking_event.clear()
                self.stop_speaking_event.clear()
                self.tts_queue.task_done()

    def listen_continuously(
        self,
        *,
        on_text: Callable[[str], None],
        on_error: Callable[[str], None] | None = None,
    ) -> None:
        global _ACTIVE_LISTENER
        with _LISTENER_LOCK:
            if _ACTIVE_LISTENER is not None and _ACTIVE_LISTENER is not self:
                _ACTIVE_LISTENER.stop()
            if self.listener_thread and self.listener_thread.is_alive():
                return
            _ACTIVE_LISTENER = self
            self.listener_thread = threading.Thread(
                target=self._listen_loop,
                kwargs={"on_text": on_text, "on_error": on_error},
                daemon=True,
            )
            self.listener_thread.start()

    def _listen_loop(self, *, on_text: Callable[[str], None], on_error: Callable[[str], None] | None) -> None:
        while not self.stop_event.is_set():
            try:
                with sr.Microphone() as source:
                    self.recognizer.adjust_for_ambient_noise(source, duration=1)
                    self._log("microphone_ready")
                    self._set_status("Passive Listening", "Microphone ready. Offline-first recognition is active.")

                    while not self.stop_event.is_set():
                        if self.speaking_event.is_set():
                            try:
                                audio = self.recognizer.listen(source, timeout=0.15, phrase_time_limit=2.5)
                            except sr.WaitTimeoutError:
                                continue
                            except OSError as exc:
                                self._log(f"microphone_stream_error:{exc}")
                                break

                            text = self._transcribe(audio)
                            if not text:
                                continue
                            if self._looks_like_tts_echo(text, speaking_now=True):
                                self._log(f"tts_echo_ignored:{text}")
                                continue
                            lowered = text.casefold()
                            is_interrupt_keyword = any(keyword in lowered for keyword in INTERRUPT_KEYWORDS)
                            self._log(f"tts_interrupt_detected:{text}")
                            self._log(
                                f"tts_interrupt_reason:{'keyword' if is_interrupt_keyword else 'new_speech'}"
                            )
                            self.interrupt_speech()
                            self._set_status("Passive Listening", "Speech interrupted. Listening for the new command.")
                            on_text(text)
                            continue

                        try:
                            audio = self.recognizer.listen(source, timeout=1, phrase_time_limit=8)
                        except sr.WaitTimeoutError:
                            continue
                        except OSError as exc:
                            self._log(f"microphone_stream_error:{exc}")
                            break

                        text = self._transcribe(audio)
                        if not text:
                            continue
                        if self._looks_like_tts_echo(text, speaking_now=False):
                            self._log(f"post_tts_echo_ignored:{text}")
                            continue
                        on_text(text)
            except OSError as exc:
                self._log(f"microphone_not_ready:{exc}")
                if on_error is not None:
                    on_error(f"Microphone not ready yet: {exc}")
            except Exception as exc:
                self._log(f"listener_startup_error:{exc}")
                if on_error is not None:
                    on_error(f"Listener startup error: {exc}")

            if self.stop_event.is_set():
                break
            self._set_status("Attention Required", "Voice listener will retry in 5 seconds.")
            time.sleep(5)

    def _transcribe(self, audio: sr.AudioData) -> str | None:
        text = self._transcribe_vosk(audio)
        if text:
            if self._is_transcript_usable(text):
                self._log(f"speech_source:vosk:{text}")
                return text
            self._log(f"[STT] Ignored short VOSK transcript: {text}")

        should_try_google = self._vosk_model is None or self._allow_google_fallback
        if not should_try_google:
            return None

        text = self._transcribe_google(audio)
        if text:
            if self._is_transcript_usable(text):
                self._log(f"speech_source:google:{text}")
                return text
            self._log(f"[STT] Ignored short Google transcript: {text}")
        return None

    def _is_transcript_usable(self, text: str) -> bool:
        cleaned = " ".join(str(text or "").strip().split())
        if not cleaned:
            return False
        lowered = cleaned.casefold()
        if lowered in STT_ALLOWED_SHORT_COMMANDS:
            return True
        if len(cleaned) < self._min_transcript_chars:
            return False
        if len(cleaned.split()) < self._min_transcript_words:
            return False
        return True

    def _transcribe_vosk(self, audio: sr.AudioData) -> str | None:
        if self._vosk_model is None or KaldiRecognizer is None:
            return None
        try:
            raw = self._prepare_vosk_audio(audio)
            if not raw:
                self._log("vosk_audio_empty")
                return None

            if audioop is not None:
                rms = audioop.rms(raw, 2)
                silence_floor = int(self.config.get("voice", {}).get("vosk_silence_floor", 120))
                if rms < silence_floor:
                    self._log(f"vosk_silence_ignored:{rms}")
                    return None

            if self._vosk_runtime_grammar_enabled and self._vosk_phrase_bias:
                recognizer = KaldiRecognizer(
                    self._vosk_model,
                    16000,
                    json.dumps(self._vosk_phrase_bias),
                )
            else:
                recognizer = KaldiRecognizer(self._vosk_model, 16000)
            recognizer.SetWords(True)
            recognizer.AcceptWaveform(raw)
            result_raw = recognizer.Result()
            if self._vosk_debug_raw:
                self._log(f"vosk_raw:{result_raw}")
            result = json.loads(result_raw)
            text = str(result.get("text") or "").strip()
            confidence = self._extract_vosk_confidence(result)
            if text:
                self._log(f"recognized_text:{text}")
                self._log(f"confidence_score:{confidence:.2f}")
                if confidence < self._vosk_confidence_floor:
                    self._log(f"vosk_low_confidence:{confidence:.2f}:{text}")
                    return None
                if not self._is_transcript_usable(text):
                    self._log(f"vosk_short_transcript_ignored:{text}")
                    return None
                if len(text.split()) == 1 and text.casefold() not in STT_ALLOWED_SHORT_COMMANDS:
                    if confidence < self._vosk_short_confidence_floor:
                        self._log(f"vosk_short_low_confidence:{confidence:.2f}:{text}")
                        return None
                return text

            partial_raw = recognizer.PartialResult()
            if self._vosk_debug_raw:
                self._log(f"vosk_partial_raw:{partial_raw}")
            partial = str(json.loads(partial_raw).get("partial", "")).strip()
            if partial:
                self._log(f"vosk_partial_ignored:{partial}")
            return None
        except Exception as exc:
            self._log(f"vosk_transcription_error:{exc}")
            return None

    def _prepare_vosk_audio(self, audio: sr.AudioData) -> bytes:
        raw = audio.get_raw_data(convert_rate=16000, convert_width=2)
        if audioop is None:
            return raw
        try:
            rms = max(audioop.rms(raw, 2), 1)
            target_rms = int(self.config.get("voice", {}).get("vosk_target_rms", 1800))
            gain = min(max(target_rms / rms, 0.8), 3.0)
            normalized = audioop.mul(raw, 2, gain)
            return audioop.bias(normalized, 2, 0)
        except Exception as exc:
            self._log(f"vosk_audio_preprocess_error:{exc}")
            return raw

    def _extract_vosk_confidence(self, result: dict) -> float:
        words = result.get("result")
        if not isinstance(words, list) or not words:
            return 1.0
        confidences = [
            float(item.get("conf", 0.0))
            for item in words
            if isinstance(item, dict) and "conf" in item
        ]
        if not confidences:
            return 1.0
        return sum(confidences) / len(confidences)

    def _transcribe_google(self, audio: sr.AudioData) -> str | None:
        now = time.monotonic()
        if now < self._google_fallback_retry_after:
            return None
        try:
            text = self.recognizer.recognize_google(audio)
            self._google_fallback_retry_after = 0.0
            return text
        except sr.UnknownValueError:
            return None
        except sr.RequestError as exc:
            self._google_fallback_retry_after = now + self._google_fallback_cooldown_seconds
            self._log(f"speech_request_error:{exc}")
            self._log(f"speech_request_cooldown:{int(self._google_fallback_cooldown_seconds)}")
            return None
        except Exception as exc:
            self._log(f"speech_unknown_error:{exc}")
            return None

    def stop(self) -> None:
        self.stop_event.set()
        self.stop_speaking_event.set()
        self.speaking_event.clear()
        try:
            self.tts_queue.put_nowait("")
        except queue.Full:
            pass
        if self._tts_channel is not None:
            try:
                self._tts_channel.stop()
            except Exception:
                pass
        if self._engine is not None:
            try:
                self._engine.stop()
            except Exception:
                pass
