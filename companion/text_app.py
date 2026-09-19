from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from companion.config import ROOT, load_config
from companion.speak import SpeakError, Speaker
from companion.speech_tags import build_speech_tag_instructions


class TextGenerationError(RuntimeError):
    pass


def _steering_instructions(draft: str, interruption: str) -> str:
    return (
        "You were already in the middle of speaking this reply:\n"
        f"{draft.strip()}\n\n"
        "The user talked over you with:\n"
        f"{interruption.strip()}\n\n"
        "Continue the same thought. Do not restart, recap from the beginning, "
        "or apologize for the interruption. Do not repeat what you already said "
        "unless a short glue phrase is needed. Fold the new remark into the rest "
        "of the answer, the way a person would if someone spoke while they were "
        "still talking."
    )


def _memory_terms(text: str) -> set[str]:
    ignored = {
        "about",
        "again",
        "first",
        "from",
        "have",
        "said",
        "say",
        "that",
        "this",
        "what",
        "when",
        "where",
        "which",
        "with",
        "would",
        "your",
    }
    return {
        term
        for term in re.findall(r"[a-z0-9']{3,}", text.lower())
        if term not in ignored
    }


def _fit_memory_context(
    rows: list[tuple[int, str, str]],
    pinned_ids: set[int],
    priority_ids: set[int],
    max_chars: int,
) -> list[tuple[int, str, str]]:
    def cost(row: tuple[int, str, str]) -> int:
        return len(str(row[1])) + len(str(row[2])) + 16

    if sum(cost(row) for row in rows) <= max_chars:
        return rows

    chosen: dict[int, tuple[int, str, str]] = {}
    used = 0
    for group in (
        [row for row in rows if int(row[0]) in pinned_ids],
        [row for row in rows if int(row[0]) in priority_ids],
        list(reversed(rows)),
    ):
        for row in group:
            row_id = int(row[0])
            row_cost = cost(row)
            if row_id in chosen or used + row_cost > max_chars:
                continue
            chosen[row_id] = row
            used += row_cost
    return [chosen[key] for key in sorted(chosen)]


class ConversationMemory:
    def __init__(self, settings: dict[str, Any]) -> None:
        self.enabled = bool(settings.get("enabled", True))
        raw_path = Path(str(settings.get("path") or "data/text_memory.sqlite3"))
        self.path = raw_path if raw_path.is_absolute() else ROOT / raw_path
        self.max_messages = max(0, int(settings.get("max_messages", 12)))
        self.first_messages = max(0, int(settings.get("first_messages", 4)))
        self.relevant_messages = max(0, int(settings.get("relevant_messages", 8)))
        self.max_context_chars = max(
            1000, int(settings.get("max_context_chars", 12000))
        )
        if self.enabled:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self._connect() as connection:
                connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS messages (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
                        content TEXT NOT NULL,
                        created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                    )
                    """
                )

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path)

    def load(self, query: str = "") -> list[dict[str, str]]:
        if not self.enabled or self.max_messages == 0:
            return []
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT id, role, content
                FROM messages
                ORDER BY id
                """
            ).fetchall()
        if not rows:
            return []

        first = rows[: self.first_messages]
        recent = rows[-self.max_messages :]
        query_terms = _memory_terms(query)
        relevant: list[tuple[int, str, str]] = []
        if query_terms and self.relevant_messages:
            scored = [
                (len(query_terms & _memory_terms(str(content))), row)
                for row in rows
                for _id, _role, content in (row,)
            ]
            relevant = [
                row
                for score, row in sorted(
                    scored,
                    key=lambda item: (item[0], item[1][0]),
                    reverse=True,
                )
                if score > 0
            ][: self.relevant_messages]

        selected = {int(row[0]): row for row in (*first, *relevant, *recent)}
        ordered = [selected[key] for key in sorted(selected)]
        fitted = _fit_memory_context(
            ordered,
            pinned_ids={int(row[0]) for row in first},
            priority_ids={int(row[0]) for row in relevant},
            max_chars=self.max_context_chars,
        )
        return [
            {"role": str(role), "content": str(content)}
            for _id, role, content in fitted
        ]

    def add_exchange(self, user_text: str, assistant_text: str) -> None:
        if not self.enabled:
            return
        with self._connect() as connection:
            connection.executemany(
                "INSERT INTO messages (role, content) VALUES (?, ?)",
                (("user", user_text), ("assistant", assistant_text)),
            )

    def clear(self) -> None:
        if not self.enabled:
            return
        with self._connect() as connection:
            connection.execute("DELETE FROM messages")


class TextGenerator:
    def __init__(
        self,
        settings: dict[str, Any],
        provider_override: str | None = None,
        disable_memory: bool = False,
        tts_settings: dict[str, Any] | None = None,
    ) -> None:
        self.settings = settings
        self.tts_settings = tts_settings or {}
        self.provider = (
            provider_override or str(settings.get("provider") or "ollama")
        ).strip().lower()
        if self.provider not in {"ollama", "openai"}:
            raise TextGenerationError(
                f"Unknown text_generation.provider {self.provider!r}; "
                "use 'ollama' or 'openai'."
            )
        memory_settings = dict(settings.get("memory") or {})
        if disable_memory:
            memory_settings["enabled"] = False
        self.memory = ConversationMemory(memory_settings)

    @property
    def model(self) -> str:
        provider_settings = self.settings.get(self.provider) or {}
        return str(provider_settings.get("model") or "")

    def _system_prompt(self) -> str:
        parts = [str(self.settings.get("system_prompt") or "").strip()]
        speech_settings = self.settings.get("speech_tags")
        tag_prompt = build_speech_tag_instructions(
            speech_settings if isinstance(speech_settings, dict) else None,
            self.tts_settings if isinstance(self.tts_settings, dict) else None,
        )
        if tag_prompt:
            parts.append(tag_prompt)
        return "\n\n".join(part for part in parts if part)

    def generate(self, prompt: str, continue_from: str | None = None) -> str:
        messages: list[dict[str, str]] = []
        system_prompt = self._system_prompt()
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        memory = self.memory.load(prompt)
        if memory:
            transcript = "\n".join(
                f"{index}. {item['role'].title()}: {item['content']}"
                for index, item in enumerate(memory, start=1)
            )
            messages.append(
                {
                    "role": "system",
                    "content": (
                        "Saved conversation memory follows in chronological order. "
                        "Use it directly when answering questions about prior messages, "
                        "including what was said first. Do not claim that you lack access "
                        f"to it.\n\n{transcript}"
                    ),
                }
            )
        if continue_from and continue_from.strip():
            messages.append(
                {
                    "role": "system",
                    "content": _steering_instructions(continue_from, prompt),
                }
            )
        messages.append({"role": "user", "content": prompt})

        if self.provider == "ollama":
            text = self._generate_ollama(messages)
        else:
            text = self._generate_openai(messages)
        if not text:
            raise TextGenerationError("The provider returned an empty response.")
        self.memory.add_exchange(prompt, text)
        return text

    def _generate_ollama(self, messages: list[dict[str, str]]) -> str:
        settings = self.settings.get("ollama") or {}
        model = str(settings.get("model") or "").strip()
        if not model:
            raise TextGenerationError("text_generation.ollama.model is required.")
        base_url = str(settings.get("base_url") or "http://localhost:11434").rstrip("/")
        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": float(self.settings.get("temperature", 0.7)),
                "num_predict": int(self.settings.get("max_tokens", 512)),
            },
        }
        result = _post_json(
            f"{base_url}/api/chat",
            payload,
            timeout=int(settings.get("timeout_sec", 120)),
        )
        message = result.get("message") or {}
        return str(message.get("content") or "").strip()

    def _generate_openai(self, messages: list[dict[str, str]]) -> str:
        settings = self.settings.get("openai") or {}
        api_key = (
            os.environ.get("OPENAI_API_KEY", "").strip()
            or str(settings.get("api_key") or "").strip()
        )
        if not api_key:
            raise TextGenerationError(
                "Set OPENAI_API_KEY or text_generation.openai.api_key."
            )
        model = str(settings.get("model") or "").strip()
        if not model:
            raise TextGenerationError("text_generation.openai.model is required.")
        base_url = str(
            settings.get("base_url") or "https://api.openai.com/v1"
        ).rstrip("/")
        payload = {
            "model": model,
            "messages": messages,
            "temperature": float(self.settings.get("temperature", 0.7)),
            "max_completion_tokens": int(self.settings.get("max_tokens", 512)),
        }
        result = _post_json(
            f"{base_url}/chat/completions",
            payload,
            timeout=int(settings.get("timeout_sec", 120)),
            headers={"Authorization": f"Bearer {api_key}"},
        )
        choices = result.get("choices") or []
        if not choices:
            raise TextGenerationError("OpenAI returned no choices.")
        message = choices[0].get("message") or {}
        return str(message.get("content") or "").strip()


def _post_json(
    url: str,
    payload: dict[str, Any],
    timeout: int,
    headers: dict[str, str] | None = None,
) -> dict[str, Any]:
    request_headers = {"Content-Type": "application/json", **(headers or {})}
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=request_headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace").strip()
        raise TextGenerationError(
            f"Provider request failed with HTTP {exc.code}: {detail}"
        ) from exc
    except urllib.error.URLError as exc:
        raise TextGenerationError(f"Could not reach provider at {url}: {exc.reason}") from exc
    try:
        result = json.loads(body)
    except json.JSONDecodeError as exc:
        raise TextGenerationError("Provider returned invalid JSON.") from exc
    if not isinstance(result, dict):
        raise TextGenerationError("Provider returned an unexpected response.")
    return result


def run(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="OpenCompanion text generator")
    parser.add_argument("--provider", choices=("ollama", "openai"))
    parser.add_argument("--no-memory", action="store_true")
    parser.add_argument("--clear-memory", action="store_true")
    args = parser.parse_args(argv)

    full_settings = load_config()
    settings = full_settings.get("text_generation") or {}
    try:
        generator = TextGenerator(
            settings,
            provider_override=args.provider,
            disable_memory=args.no_memory,
            tts_settings=full_settings.get("tts") or {},
        )
        speaker = Speaker(full_settings) if full_settings.get("speak_aloud") else None
    except (OSError, sqlite3.Error, SpeakError, TextGenerationError, ValueError) as exc:
        print(f"Could not start text generation: {exc}", file=sys.stderr)
        return 1

    if args.clear_memory:
        generator.memory.clear()
        print("Conversation memory cleared.")
        return 0

    memory_status = "on" if generator.memory.enabled else "off"
    print(
        f"OpenCompanion text generation - {generator.provider}:{generator.model} "
        f"(memory {memory_status})"
    )
    if speaker is not None:
        print(f"TTS: {speaker.label}")
        print("You can type while he is talking; that steers the same reply.")
    print("Commands: /clear, /help, /exit")
    while True:
        tts_error = speaker.take_error() if speaker is not None else ""
        if tts_error:
            print(f"TTS error: {tts_error}", file=sys.stderr)
        try:
            prompt = input("\nYou> ").strip()
        except (EOFError, KeyboardInterrupt):
            if speaker is not None:
                speaker.interrupt()
            print("\nStopped.")
            return 0
        if not prompt:
            continue
        command = prompt.lower()
        if command in {"/exit", "/quit"}:
            if speaker is not None:
                speaker.interrupt()
            return 0
        if command == "/clear":
            if speaker is not None:
                speaker.interrupt()
            generator.memory.clear()
            print("Memory cleared.")
            continue
        if command == "/help":
            print(
                "/clear forgets saved conversation; /exit stops the program. "
                "Type while he is talking to steer the same thought."
            )
            continue
        continue_from = ""
        if speaker is not None and speaker.is_speaking():
            continue_from = speaker.current_text()
            speaker.interrupt()
            print("(steering the reply he was already saying)")
        try:
            response = generator.generate(
                prompt,
                continue_from=continue_from or None,
            )
        except (OSError, sqlite3.Error, TextGenerationError, ValueError) as exc:
            print(f"Error: {exc}", file=sys.stderr)
            continue
        print(f"\nAssistant> {response}")
        if speaker is not None:
            speaker.say_async(response)


if __name__ == "__main__":
    raise SystemExit(run())
