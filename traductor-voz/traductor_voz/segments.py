"""Frases: agrupa los segmentos del reconocedor y elige el audio de referencia de cada frase."""

from __future__ import annotations

import re
from dataclasses import dataclass

SENTENCE_END = (".", "!", "?", "…", "。", "！", "？")


@dataclass
class Phrase:
    start: float
    end: float
    text: str

    @property
    def duration(self) -> float:
        return self.end - self.start


def merge_segments(
    segments: list[tuple[float, float, str]], max_gap: float = 0.35, max_len: float = 12.0
) -> list[Phrase]:
    """Une segmentos cortados a mitad de oración (sin pausa real entre ellos) en una sola frase."""
    phrases: list[Phrase] = []
    for start, end, text in segments:
        text = text.strip()
        if not text:
            continue
        if phrases:
            last = phrases[-1]
            if (
                not last.text.endswith(SENTENCE_END)
                and start - last.end <= max_gap
                and end - last.start <= max_len
            ):
                last.end = end
                last.text = f"{last.text} {text}"
                continue
        phrases.append(Phrase(start, end, text))
    return phrases


def reference_window(phrases: list[Phrase], i: int, min_sec: float, max_sec: float = 15.0) -> tuple[int, int]:
    """Índices [lo, hi] de las frases que forman el audio de referencia para la frase i.

    Lo ideal es usar la misma frase como referencia: así el motor copia la emoción de ese momento.
    Si es muy corta, se le suman las frases vecinas (la más cercana en el tiempo primero)
    hasta llegar a min_sec, sin pasar de max_sec.
    """
    lo = hi = i
    while phrases[hi].end - phrases[lo].start < min_sec:
        options = []
        if lo > 0:
            options.append((phrases[lo].start - phrases[lo - 1].end, "prev"))
        if hi < len(phrases) - 1:
            options.append((phrases[hi + 1].start - phrases[hi].end, "next"))
        grown = False
        for _, side in sorted(options):
            new_lo, new_hi = (lo - 1, hi) if side == "prev" else (lo, hi + 1)
            if phrases[new_hi].end - phrases[new_lo].start <= max_sec:
                lo, hi = new_lo, new_hi
                grown = True
                break
        if not grown:
            break
    return lo, hi


def split_text(text: str, max_chars: int = 250) -> list[str]:
    """Parte un texto largo en trozos por oraciones (los motores de voz se pierden con textos largos)."""
    text = " ".join(text.split())
    if len(text) <= max_chars:
        return [text] if text else []
    sentences = re.split(r"(?<=[.!?…;:])\s+", text)
    chunks: list[str] = []
    for sentence in sentences:
        while len(sentence) > max_chars:
            cut = sentence.rfind(",", 0, max_chars)
            if cut <= 0:
                cut = sentence.rfind(" ", 0, max_chars)
            if cut <= 0:
                cut = max_chars
            chunks.append(sentence[: cut + 1].strip())
            sentence = sentence[cut + 1 :].strip()
        if chunks and len(chunks[-1]) + 1 + len(sentence) <= max_chars:
            chunks[-1] = f"{chunks[-1]} {sentence}"
        elif sentence:
            chunks.append(sentence)
    return chunks
