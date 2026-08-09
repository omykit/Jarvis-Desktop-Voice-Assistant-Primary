from __future__ import annotations

import json
import os
import random
import re
import threading
import time
import urllib.request
from concurrent.futures import Future, TimeoutError as FutureTimeoutError
from dataclasses import dataclass
from urllib.parse import urlparse, urlunparse
from typing import Callable

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

from memory_module import list_memories


DEFAULT_SYSTEM_PROMPT = (
    "You are Jarvis, a concise voice assistant. Answer naturally in one or two short sentences."
)
RESPONSE_RULE_SUFFIX = (
    "Reply with only the final spoken answer. "
    "Never include analysis or reasoning."
)


Logger = Callable[[str], None]
FALLBACK_MODELS = ["llama3", "phi3"]
MAX_TIMEOUT_SECONDS = 45
DEFAULT_TIMEOUT_SECONDS = 9
MAX_RETRIES = 1
RETRY_DELAY_SECONDS = 0.7
BACKGROUND_STREAM_TIMEOUT_SECONDS = 60
DEFAULT_MAX_TOKENS = 160
MIN_STREAM_RESPONSE_CHARS = 32
FIRST_RESPONSE_FALLBACK_SECONDS = 8.0
MIN_PARTIAL_RESPONSE_CHARS = 28
_AI_HEALTH: "OllamaHealth | None" = None
_AI_CLIENT_CACHE: dict[tuple[str, str, int], object] = {}
ollama_available = False
ollama_degraded = False
_FALLBACK_RESPONSE_VARIANTS = {
    "how_are_you": [
        "I'm operating smoothly and ready to help.",
        "All systems are steady and I'm ready for your next command.",
        "I'm doing well and standing by for whatever you need.",
    ],
    "interesting_fact": [
        "Here is one: Venus spins in the opposite direction from most planets.",
        "Here is one: octopuses have three hearts and blue blood.",
        "Here is one: honey can last for years without spoiling when sealed.",
        "Here is one: a day on Venus is longer than a year on Venus.",
    ],
    "joke": [
        "Why do programmers prefer dark mode? Because light attracts bugs.",
        "Why did the computer go to therapy? It had too many unresolved issues.",
        "Why do Java developers wear glasses? Because they don't see sharp.",
        "Why was the developer calm during the outage? He had already cached his panic.",
    ],
}
_LAST_FALLBACK_RESPONSE_BY_KEY: dict[str, str] = {}


@dataclass
class OllamaHealth:
    checked_at: float
    available: bool
    models: set[str]
    error: str = ""
    degraded: bool = False


@dataclass
class AIResponse:
    spoken_text: str
    full_text: str
    full_text_future: Future[str] | None = None


def _safe_log(logger: Logger | None, message: str) -> None:
    if logger is not None:
        logger(message)


def _resolve_api_key(ai_config: dict) -> str:
    api_key = str(ai_config.get("api_key") or "").strip()
    api_key_env = str(ai_config.get("api_key_env") or "").strip()
    if not api_key and api_key_env:
        api_key = os.environ.get(api_key_env, "").strip()
    return api_key or "ollama"


def _get_ai_client(base_url: str, api_key: str, timeout_seconds: int):
    cache_key = (base_url, api_key, timeout_seconds)
    client = _AI_CLIENT_CACHE.get(cache_key)
    if client is None:
        client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout_seconds)
        _AI_CLIENT_CACHE[cache_key] = client
    return client


def _trim_history(chat_history: list[dict], max_messages: int) -> list[dict]:
    if max_messages <= 0:
        return []
    return chat_history[-max_messages:]


def _choose_fallback_variant(key: str) -> str:
    options = _FALLBACK_RESPONSE_VARIANTS.get(key, [])
    if not options:
        return ""
    if len(options) == 1:
        choice = options[0]
    else:
        previous = _LAST_FALLBACK_RESPONSE_BY_KEY.get(key)
        available = [option for option in options if option != previous]
        choice = random.choice(available or options)
    _LAST_FALLBACK_RESPONSE_BY_KEY[key] = choice
    return choice


def _fallback_response(user_input: str) -> str:
    text = user_input.lower()

    if "how are you" in text:
        return _choose_fallback_variant("how_are_you")
    if "interesting" in text:
        return _choose_fallback_variant("interesting_fact")
    if "joke" in text:
        return _choose_fallback_variant("joke")
    if "weather" in text:
        return "I can answer weather questions once your weather API key is configured."
    if "time" in text or "date" in text:
        return "I can answer time and date locally through the command system."
    return "Give me a moment... I'm processing that."


def _fallback_ai_response(user_input: str) -> AIResponse:
    text = _fallback_response(user_input)
    return AIResponse(spoken_text=text, full_text=text)


def _ollama_root_url(base_url: str) -> str:
    parsed = urlparse(base_url)
    path = parsed.path.rstrip("/")
    if path.endswith("/v1"):
        path = path[:-3]
    return urlunparse((parsed.scheme, parsed.netloc, path.rstrip("/"), "", "", "")).rstrip("/")


def _model_names_from_tags(data: dict) -> set[str]:
    names: set[str] = set()
    models = data.get("models", [])
    if not isinstance(models, list):
        return names
    for model in models:
        if not isinstance(model, dict):
            continue
        name = str(model.get("name") or model.get("model") or "").strip()
        if name:
            names.add(name)
            names.add(name.split(":", 1)[0])
    return names


def _should_use_ollama_native_chat(base_url: str) -> bool:
    parsed = urlparse(base_url)
    host = (parsed.hostname or "").casefold()
    return host in {"localhost", "127.0.0.1"} and (parsed.port in {11434, None})


def _ollama_chat_url(base_url: str) -> str:
    return f"{_ollama_root_url(base_url)}/api/chat"


def _prepare_ollama_messages(messages: list[dict]) -> list[dict]:
    prepared: list[dict] = []
    for index, message in enumerate(messages):
        role = str(message.get("role") or "user")
        content = str(message.get("content") or "")
        if index == 0 and role == "system":
            content = f"{content} {RESPONSE_RULE_SUFFIX}".strip()
        prepared.append({"role": role, "content": content})
    return prepared


def _clean_streaming_candidate(text: str) -> str:
    raw = str(text or "")
    lowered = raw.casefold()
    if "<think>" in lowered and "</think>" not in lowered:
        return ""
    return _clean_ai_response(raw)


def initialize_ai_health(config: dict, logger: Logger | None = None) -> bool:
    """Run the Ollama health check once during startup and store the result."""
    global _AI_HEALTH, ollama_available, ollama_degraded
    ai_config = config.get("ai", {}) if isinstance(config, dict) else {}
    if not ai_config.get("enabled", True):
        _AI_HEALTH = OllamaHealth(
            checked_at=time.perf_counter(),
            available=False,
            models=set(),
            error="AI disabled",
            degraded=True,
        )
        ollama_available = False
        ollama_degraded = True
        _safe_log(logger, "[AI] Health check skipped: AI disabled")
        return False

    base_url = str(ai_config.get("base_url") or "http://localhost:11434/v1").rstrip("/")
    api_key = _resolve_api_key(ai_config)
    configured_timeout = int(ai_config.get("timeout_seconds", DEFAULT_TIMEOUT_SECONDS))
    timeout_seconds = max(1, min(configured_timeout, MAX_TIMEOUT_SECONDS))
    _AI_HEALTH = _check_ollama_health(base_url, timeout_seconds, logger)
    if _AI_HEALTH.available and OpenAI is not None:
        _get_ai_client(base_url, api_key, BACKGROUND_STREAM_TIMEOUT_SECONDS)
        _safe_log(logger, "[AI] Client warmed")
    ollama_available = _AI_HEALTH.available
    ollama_degraded = _AI_HEALTH.degraded
    return _AI_HEALTH.available


def _get_ai_health(base_url: str, timeout_seconds: float, logger: Logger | None = None) -> OllamaHealth:
    global _AI_HEALTH
    if _AI_HEALTH is None:
        _safe_log(logger, "[AI] Health state missing; running one-time initialization")
        _AI_HEALTH = _check_ollama_health(base_url, timeout_seconds, logger)
    return _AI_HEALTH


def _mark_ai_degraded(error: str, logger: Logger | None = None) -> None:
    global _AI_HEALTH, ollama_available, ollama_degraded
    previous_models = _AI_HEALTH.models if _AI_HEALTH is not None else set()
    _AI_HEALTH = OllamaHealth(
        checked_at=time.perf_counter(),
        available=False,
        models=previous_models,
        error=error,
        degraded=True,
    )
    ollama_available = False
    ollama_degraded = True
    _safe_log(logger, f"[AI ERROR] System marked degraded: {error}")


def _check_ollama_health(base_url: str, timeout_seconds: float, logger: Logger | None = None) -> OllamaHealth:
    root_url = _ollama_root_url(base_url)

    tags_url = f"{root_url}/api/tags"
    request = urllib.request.Request(tags_url, headers={"Accept": "application/json"})
    started_at = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=min(timeout_seconds, 3.0)) as response:
            import json

            data = json.loads(response.read().decode("utf-8"))
        models = _model_names_from_tags(data)
        health = OllamaHealth(
            checked_at=time.perf_counter(),
            available=bool(models),
            models=models,
            error="" if models else "No Ollama models installed",
            degraded=not bool(models),
        )
        elapsed = time.perf_counter() - started_at
        if models:
            _safe_log(logger, f"[AI] Health check OK in {elapsed:.2f}s models={','.join(sorted(health.models))}")
        else:
            _safe_log(logger, f"[AI ERROR] Ollama is running but no models are installed ({elapsed:.2f}s)")
    except Exception as exc:
        health = OllamaHealth(
            checked_at=time.perf_counter(),
            available=False,
            models=set(),
            error=str(exc),
            degraded=True,
        )
        _safe_log(logger, f"[AI ERROR] Ollama health check failed: {exc}")

    return health


def _candidate_models(primary_model: str, available_models: set[str]) -> list[str]:
    candidates: list[str] = []
    for model in [primary_model, *FALLBACK_MODELS]:
        model = str(model or "").strip()
        if model and model not in candidates:
            candidates.append(model)

    if not available_models:
        return candidates

    filtered = [
        model
        for model in candidates
        if model in available_models or model.split(":", 1)[0] in available_models
    ]
    return filtered or candidates


def _extract_quoted_candidate(text: str) -> str:
    matches = re.findall(r'"([^\"]{2,80})"', text)
    for candidate in reversed(matches):
        cleaned = " ".join(candidate.strip().split())
        if cleaned and re.search(r"[A-Za-z]", cleaned):
            return cleaned
    return ""


def _is_meta_reasoning_sentence(sentence: str) -> bool:
    lowered = sentence.casefold()
    meta_markers = (
        "the user",
        "user request",
        "user message",
        "they want",
        "i need",
        "i should",
        "i must",
        "i'm supposed",
        "i am supposed",
        "first thought",
        "second thought",
        "the key here",
        "the key is",
        "keep it",
        "tone",
        "instructions",
        "prompt",
        "roleplay",
        "voice assistant",
        "final answer",
        "one sentence",
        "two sentences",
        "concise",
        "overcomplicating",
        "friendly touch",
        "must be under",
        "but maybe",
        "let's unpack",
        "we are given",
        "since i'm",
        "since i am",
    )
    return any(marker in lowered for marker in meta_markers)


def _normalize_answer_sentence(sentence: str) -> str:
    cleaned = sentence.strip().strip('"').strip()
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" ,")
    if not cleaned:
        return ""
    if cleaned[-1] not in ".!?":
        cleaned += "."
    return cleaned[0].upper() + cleaned[1:]


def _salvage_meta_sentence(sentence: str) -> str:
    quoted = _extract_quoted_candidate(sentence)
    if quoted:
        return _normalize_answer_sentence(quoted)
    like_match = re.search(r"\blike\s+([A-Za-z][^.!?]{1,80})", sentence, flags=re.IGNORECASE)
    if like_match:
        candidate = like_match.group(1).split(" - ", 1)[0].strip(" ,")
        if candidate and not _is_meta_reasoning_sentence(candidate):
            return _normalize_answer_sentence(candidate)
    if ":" in sentence:
        candidate = sentence.rsplit(":", 1)[-1].strip(" ,")
        if candidate and not _is_meta_reasoning_sentence(candidate):
            return _normalize_answer_sentence(candidate)
    return ""


def _clean_ai_response(text: str) -> str:
    cleaned = str(text or "").strip().strip('"').strip()
    if "</think>" in cleaned.casefold():
        cleaned = cleaned.rsplit("</think>", 1)[-1].strip()
    cleaned = re.sub(r"<think>.*?</think>", "", cleaned, flags=re.DOTALL | re.IGNORECASE)
    cleaned = re.sub(r"</?think>", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"```.*?```", " ", cleaned, flags=re.DOTALL)
    cleaned = re.sub(r"`([^`]*)`", r"\1", cleaned)
    cleaned = re.sub(r"^\s*[-*+]\s+", "", cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    cleaned = re.sub(
        r"\b(currently|basically|actually|simply|right now|just)\b",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" ,")
    if not cleaned or not re.search(r"[A-Za-z0-9]", cleaned):
        return ""

    quoted_candidate = _extract_quoted_candidate(cleaned)
    sentences = [
        sentence.strip(" ,")
        for sentence in re.split(r"(?<=[.!?])\s+", cleaned)
        if sentence.strip(" ,")
    ]
    if not sentences:
        return _normalize_answer_sentence(quoted_candidate) if quoted_candidate else ""

    spoken_sentences: list[str] = []
    for sentence in sentences:
        if _is_meta_reasoning_sentence(sentence):
            salvaged = _salvage_meta_sentence(sentence)
            if salvaged:
                spoken_sentences.append(salvaged)
            continue
        if re.search(r"\b(i should|i need|i must|i'm supposed|prompt|instructions|roleplay|tone|sentence)\b", sentence, re.IGNORECASE):
            continue
        if sentence.count(",") >= 2 and len(sentence.split()) <= 8:
            continue
        normalized = _normalize_answer_sentence(sentence)
        if normalized:
            spoken_sentences.append(normalized)

    if not spoken_sentences and quoted_candidate:
        return _normalize_answer_sentence(quoted_candidate)
    if not spoken_sentences:
        return ""

    deduped: list[str] = []
    seen: set[str] = set()
    for sentence in spoken_sentences:
        key = sentence.casefold()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(sentence)

    return " ".join(deduped[:2]).strip()


def _is_detail_request(user_input: str) -> bool:
    text = user_input.casefold()
    detail_markers = [
        "explain",
        "detail",
        "deep",
        "step by step",
        "full",
        "long",
        "elaborate",
        "why",
        "how does",
        "how do",
    ]
    return any(marker in text for marker in detail_markers)


def _requested_max_tokens(ai_config: dict, user_input: str) -> int:
    configured = int(ai_config.get("max_tokens", DEFAULT_MAX_TOKENS))
    cap = 180 if _is_detail_request(user_input) else DEFAULT_MAX_TOKENS
    return max(40, min(configured, cap))


def _stream_has_usable_response(text: str) -> bool:
    raw = str(text or "").strip()
    cleaned = _clean_ai_response(text)
    if not cleaned:
        return False
    if len(cleaned) >= 32 and re.search(r"[.!?]\s*$", raw):
        return True
    return len(cleaned) >= MIN_STREAM_RESPONSE_CHARS


def _first_sentence_from_stream(text: str) -> str:
    cleaned = str(text or "").strip().strip('"').strip()
    match = re.search(r"(.+?[.!?])(?:\s|$)", cleaned, flags=re.DOTALL)
    if not match:
        return ""
    first_sentence = _clean_ai_response(match.group(1))
    if len(first_sentence) < 24:
        return ""
    if len(first_sentence.split()) < 6:
        return ""
    return first_sentence


def _partial_response_from_stream(text: str) -> str:
    cleaned = _clean_streaming_candidate(text)
    if cleaned and _is_safe_partial_response(cleaned):
        return cleaned
    return ""


def _is_safe_partial_response(text: str) -> bool:
    cleaned = _clean_ai_response(text)
    if len(cleaned) < MIN_PARTIAL_RESPONSE_CHARS:
        return False
    if re.search(r"\b(and|or|but|with|for|to|of|in|on|at|any|the|a|an)$", cleaned, re.IGNORECASE):
        return False
    return True


def _start_full_stream_future(stream, chunks: list[str], logger: Logger | None) -> Future[str]:
    full_text_future: Future[str] = Future()
    thread = threading.Thread(
        target=_finish_stream_in_background,
        args=(stream, chunks, full_text_future, logger),
        daemon=True,
    )
    thread.start()
    return full_text_future


def _extract_stream_delta(chunk) -> str:
    try:
        return str(chunk.choices[0].delta.content or "")
    except Exception:
        return ""


def _finish_stream_in_background(stream, chunks: list[str], future: Future[str], logger: Logger | None = None) -> None:
    try:
        for chunk in stream:
            delta = _extract_stream_delta(chunk)
            if delta:
                chunks.append(delta)
        full_text = _clean_ai_response("".join(chunks))
        if full_text:
            future.set_result(full_text)
        else:
            future.set_exception(ValueError("AI stream completed without usable text"))
    except Exception as exc:
        _safe_log(logger, f"[AI ERROR] stream_background_error:{exc}")
        if not future.done():
            partial = _clean_ai_response("".join(chunks))
            if partial:
                future.set_result(partial)
            else:
                future.set_exception(exc)


def _stream_completion_worker(
    client,
    *,
    model: str,
    messages: list[dict],
    max_tokens: int,
    first_sentence_future: Future[str],
    full_text_future: Future[str],
    logger: Logger | None = None,
) -> None:
    chunks: list[str] = []
    try:
        stream = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.7,
            top_p=0.9,
            max_tokens=max_tokens,
            stream=True,
        )
        for chunk in stream:
            delta = _extract_stream_delta(chunk)
            if delta:
                chunks.append(delta)
        full_text = _clean_ai_response("".join(chunks))
        if full_text:
            if not first_sentence_future.done():
                first_sentence_future.set_result(full_text)
            full_text_future.set_result(full_text)
            _safe_log(logger, f"[AI] Full streamed response completed chars={len(full_text)}")
        else:
            raise ValueError("AI stream completed without usable text")
    except Exception as exc:
        _safe_log(logger, f"[AI ERROR] stream_worker_error:{exc}")
        if not first_sentence_future.done():
            first_sentence_future.set_exception(exc)
        if not full_text_future.done():
            partial = _clean_ai_response("".join(chunks))
            if partial:
                full_text_future.set_result(partial)
            else:
                full_text_future.set_exception(exc)


def _ollama_stream_completion_worker(
    base_url: str,
    *,
    model: str,
    messages: list[dict],
    max_tokens: int,
    first_sentence_future: Future[str],
    full_text_future: Future[str],
    logger: Logger | None = None,
) -> None:
    payload = {
        "model": model,
        "messages": _prepare_ollama_messages(messages),
        "stream": False,
        "think": False,
        "keep_alive": "10m",
        "options": {
            "num_predict": max_tokens,
            "temperature": 0.5,
            "top_p": 0.85,
        },
    }
    request = urllib.request.Request(
        _ollama_chat_url(base_url),
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Accept": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=BACKGROUND_STREAM_TIMEOUT_SECONDS) as response:
            data = json.loads(response.read().decode("utf-8"))
        content = str(data.get("message", {}).get("content") or "")
        full_text = _clean_ai_response(content)
        if full_text:
            if not first_sentence_future.done():
                first_sentence_future.set_result(full_text)
            full_text_future.set_result(full_text)
            _safe_log(logger, f"[AI] Full Ollama response completed chars={len(full_text)}")
        else:
            raise ValueError("AI response completed without usable text")
    except Exception as exc:
        _safe_log(logger, f"[AI ERROR] ollama_stream_worker_error:{exc}")
        if not first_sentence_future.done():
            first_sentence_future.set_exception(exc)
        if not full_text_future.done():
            full_text_future.set_exception(exc)


def _request_ollama_completion(
    base_url: str,
    *,
    model: str,
    messages: list[dict],
    max_tokens: int,
    timeout_seconds: int,
    logger: Logger | None = None,
) -> AIResponse:
    payload = {
        "model": model,
        "messages": _prepare_ollama_messages(messages),
        "stream": False,
        "think": False,
    }
    request = urllib.request.Request(
        _ollama_chat_url(base_url),
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Accept": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        data = json.loads(response.read().decode("utf-8"))
    content = str(data.get("message", {}).get("content") or "")
    full_text = _clean_ai_response(content)
    if not full_text:
        raise ValueError("AI response completed without usable text")
    _safe_log(logger, f"[AI] Full Ollama response completed chars={len(full_text)}")
    return AIResponse(spoken_text=full_text, full_text=full_text, full_text_future=None)


def _request_ai_completion(
    client,
    *,
    model: str,
    messages: list[dict],
    max_tokens: int,
    timeout_seconds: int,
    logger: Logger | None = None,
) -> AIResponse:
    first_sentence_future: Future[str] = Future()
    full_text_future: Future[str] = Future()
    thread = threading.Thread(
        target=_stream_completion_worker,
        kwargs={
            "client": client,
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "first_sentence_future": first_sentence_future,
            "full_text_future": full_text_future,
            "logger": logger,
        },
        daemon=True,
    )
    thread.start()
    try:
        spoken_text = first_sentence_future.result(timeout=FIRST_RESPONSE_FALLBACK_SECONDS)
    except FutureTimeoutError:
        spoken_text = "I'm working on that."
    return AIResponse(
        spoken_text=spoken_text,
        full_text=spoken_text,
        full_text_future=full_text_future,
    )


def _build_memory_context(config: dict) -> str:
    memories = list_memories()
    parts: list[str] = []
    owner_name = str(config.get("owner_name") or "").strip() if isinstance(config, dict) else ""
    if owner_name:
        parts.append(f"The user's preferred name is {owner_name}.")
    name = memories.get("name")
    if name:
        parts.append(f"The user's name is {name}.")
    notes = memories.get("notes")
    if isinstance(notes, list) and notes:
        parts.append("Remembered notes: " + "; ".join(str(item) for item in notes[:4]) + ".")
    return " ".join(parts)


def get_ai_response_streaming(user_input: str, chat_history: list[dict], config: dict, logger: Logger | None = None) -> AIResponse:
    """Return a conversational response using Ollama's OpenAI-compatible chat API."""
    ai_config = config.get("ai", {}) if isinstance(config, dict) else {}
    if not ai_config.get("enabled", True):
        return _fallback_ai_response(user_input)

    base_url = str(ai_config.get("base_url") or "http://localhost:11434/v1").rstrip("/")
    api_key = _resolve_api_key(ai_config)
    use_ollama_native = _should_use_ollama_native_chat(base_url)

    if OpenAI is None and not use_ollama_native:
        _safe_log(logger, "[AI ERROR] openai package is not installed")
        return _fallback_ai_response(user_input)

    model = str(ai_config.get("model") or "qwen3:4b")
    configured_timeout = int(ai_config.get("timeout_seconds", DEFAULT_TIMEOUT_SECONDS))
    timeout_seconds = max(1, min(configured_timeout, MAX_TIMEOUT_SECONDS))
    max_tokens = _requested_max_tokens(ai_config, user_input)
    max_history_messages = int(ai_config.get("max_history_messages", 12))
    system_prompt = str(ai_config.get("system_prompt") or DEFAULT_SYSTEM_PROMPT)
    if RESPONSE_RULE_SUFFIX not in system_prompt:
        system_prompt = f"{system_prompt} {RESPONSE_RULE_SUFFIX}".strip()
    memory_context = _build_memory_context(config)
    if memory_context:
        system_prompt = f"{system_prompt} {memory_context}".strip()

    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(_trim_history(chat_history, max_history_messages))
    messages.append({"role": "user", "content": user_input})

    health = _get_ai_health(base_url, timeout_seconds, logger)
    if not health.available:
        if health.degraded:
            _safe_log(logger, f"[AI ERROR] Request skipped: degraded={health.error}")
        return _fallback_ai_response(user_input)

    client = None if use_ollama_native else _get_ai_client(base_url, api_key, BACKGROUND_STREAM_TIMEOUT_SECONDS)
    request_started_at = time.perf_counter()
    _safe_log(
        logger,
        f"[AI] Request start model={model} first_response_timeout={FIRST_RESPONSE_FALLBACK_SECONDS}s "
        f"stream_timeout={BACKGROUND_STREAM_TIMEOUT_SECONDS}s max_tokens={max_tokens}",
    )

    ai_response: AIResponse | None = None
    last_error: Exception | None = None
    retry_count = 0
    candidate_models = _candidate_models(model, health.models)
    if len(candidate_models) == 1 and MAX_RETRIES:
        candidate_models.append(candidate_models[0])
    for attempt, candidate_model in enumerate(candidate_models, start=1):
        if attempt > MAX_RETRIES + 1:
            break
        try:
            if attempt > 1:
                retry_count += 1
                _safe_log(logger, f"[AI] Retry {retry_count} model={candidate_model}")
                time.sleep(RETRY_DELAY_SECONDS)
            _safe_log(logger, f"[AI] Request sent model={candidate_model} attempt={attempt}")
            if use_ollama_native:
                ai_response = _request_ollama_completion(
                    base_url,
                    model=candidate_model,
                    messages=messages,
                    max_tokens=max_tokens,
                    timeout_seconds=timeout_seconds,
                    logger=logger,
                )
            else:
                ai_response = _request_ai_completion(
                    client,
                    model=candidate_model,
                    messages=messages,
                    max_tokens=max_tokens,
                    timeout_seconds=timeout_seconds,
                    logger=logger,
                )
            elapsed = time.perf_counter() - request_started_at
            _safe_log(logger, f"[AI] Response received model={candidate_model} response_time={elapsed:.2f}s retries={retry_count}")
            break
        except Exception as exc:
            last_error = exc
            elapsed = time.perf_counter() - request_started_at
            _safe_log(logger, f"[AI ERROR] model={candidate_model} attempt={attempt} elapsed={elapsed:.2f}s error={exc}")

    if ai_response is None:
        total_elapsed = time.perf_counter() - request_started_at
        _safe_log(logger, f"[AI ERROR] all_models_failed response_time={total_elapsed:.2f}s retries={retry_count} last_error={last_error}")
        _mark_ai_degraded(str(last_error or "AI request failed"), logger)
        return _fallback_ai_response(user_input)

    try:
        spoken_text = _clean_ai_response(ai_response.spoken_text)
        if spoken_text:
            ai_response.spoken_text = spoken_text
            ai_response.full_text = _clean_ai_response(ai_response.full_text) or spoken_text
            return ai_response
        _mark_ai_degraded("AI returned an unusable response", logger)
        return _fallback_ai_response(user_input)
    except Exception as exc:
        _safe_log(logger, f"[AI ERROR] parse_failure:{exc}")
        _mark_ai_degraded(f"parse_failure:{exc}", logger)
        return AIResponse(
            spoken_text="Give me a moment... I'm processing that.",
            full_text="Give me a moment... I'm processing that.",
        )


def get_ai_response(user_input: str, chat_history: list[dict], config: dict, logger: Logger | None = None) -> str:
    """Compatibility wrapper for callers that still expect a plain string."""
    response = get_ai_response_streaming(user_input, chat_history, config, logger)
    return response.spoken_text
