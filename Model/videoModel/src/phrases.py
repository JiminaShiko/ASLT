"""Stage 5 - word buffer and phrase assembly; deterministic rewrite driven by phrases.json.

Signing order is preserved, and REST/OTHER never reach the buffer.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from . import config


@dataclass(frozen=True)
class Rule:
    pattern: tuple[str, ...]
    text: str
    needs: str = ""

    @property
    def size(self) -> int:
        return len(self.pattern)


@dataclass
class PhraseBook:
    rules: list[Rule] = field(default_factory=list)
    word_forms: dict[str, str] = field(default_factory=dict)
    max_buffer_words: int = 8
    notes: list[str] = field(default_factory=list)
    how_it_works: list[str] = field(default_factory=list)

    def longest_first(self) -> list[Rule]:
        return sorted(self.rules, key=lambda r: (-r.size, r.pattern))

    def inactive(self, labels: list[str]) -> list[Rule]:
        """Rules that can never fire because a word is not in the vocabulary."""
        known = set(labels)
        return [r for r in self.rules
                if not all(w in known for w in r.pattern)]


def load_book(path: Path | None = None) -> PhraseBook:
    path = path or config.PHRASES_JSON
    raw = json.loads(path.read_text(encoding="utf-8"))
    rules = [Rule(pattern=tuple(str(w).lower() for w in r["pattern"]),
                  text=str(r["text"]), needs=str(r.get("needs", "")))
             for r in raw.get("rules", [])]
    seen: set[tuple[str, ...]] = set()
    for r in rules:
        if r.pattern in seen:
            raise ValueError(f"phrases.json lists {list(r.pattern)} twice")
        seen.add(r.pattern)
        if not r.pattern:
            raise ValueError("phrases.json has a rule with an empty pattern")
    return PhraseBook(
        rules=rules,
        word_forms={str(k).lower(): str(v)
                    for k, v in raw.get("word_forms", {}).items()},
        max_buffer_words=int(raw.get("max_buffer_words", 8)),
        notes=list(raw.get("grammar_notes", [])),
        how_it_works=list(raw.get("how_it_works", [])),
    )


class PhraseBuilder:
    """Holds committed words and re-renders text greedily, longest rule first."""

    def __init__(self, book: PhraseBook | None = None,
                 path: Path | None = None) -> None:
        self.book = book or load_book(path)
        self.rules = self.book.longest_first()
        self.words: list[str] = []
        self.last_rule: Rule | None = None

    def reset(self) -> None:
        self.words.clear()
        self.last_rule = None

    def render(self, words: list[str]) -> tuple[list[str], Rule | None]:
        out: list[str] = []
        last: Rule | None = None
        i = 0
        while i < len(words):
            for r in self.rules:
                if tuple(words[i:i + r.size]) == r.pattern:
                    out.append(r.text)
                    last = r if i + r.size == len(words) else None
                    i += r.size
                    break
            else:
                w = words[i]
                out.append(self.book.word_forms.get(w, w))
                last = None
                i += 1
        return out, last

    def commit(self, word: str) -> str:
        w = str(word).strip().lower()
        if not w or w in (config.REST_LABEL.lower(), config.OTHER_LABEL.lower()):
            return self.text()
        self.words.append(w)
        while len(self.words) > self.book.max_buffer_words:
            self.words.pop(0)
        _, self.last_rule = self.render(self.words)
        return self.text()

    def text(self) -> str:
        return " ".join(self.render(self.words)[0])

    def as_dict(self) -> dict:
        items, rule = self.render(self.words)
        return {"text": " ".join(items), "chunks": items,
                "words": self.words.copy(),
                "rule": list(rule.pattern) + [rule.text] if rule else None}


def sentence(words: list[str], book: PhraseBook | None = None) -> str:
    """One-shot helper for tests and the report: feed words, get the text."""
    b = PhraseBuilder(book)
    for w in words:
        b.commit(w)
    return b.text()


if __name__ == "__main__":
    from . import vocab as _v
    book = load_book()
    labels = _v.load_vocab().labels
    demos = [["hello"], ["good", "friend"], ["me", "name"],
             ["my", "name"], ["school", "friend"], ["home", "school"],
             ["good", "good", "friend"], ["please", "help", "me"],
             ["me", "sad"], ["no", "good"]]
    known = set(labels)
    out = {}
    for d in demos:
        s = sentence(d, book)
        # demos with untrainable words are tagged so they are not read as working
        out[", ".join(d)] = s if all(w in known for w in d) else s + "  [batch 2]"
    print(json.dumps({
        "max_buffer_words": book.max_buffer_words,
        "rules": [list(r.pattern) + ["->", r.text] for r in book.rules],
        "word_forms": book.word_forms,
        "sentences": out,
        "rules_that_cannot_fire_with_the_current_vocab": [
            " ".join(r.pattern) + f" (needs {r.needs})"
            for r in book.inactive(labels)],
    }, indent=1))
