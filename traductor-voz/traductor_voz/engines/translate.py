"""Traducción de texto local (sin internet una vez descargados los modelos).

- MarianMT (Helsinki-NLP/opus-mt): modelos chicos (~300 MB) y buenos para cada par de idiomas.
- NLLB-200 (Meta): un solo modelo para cualquier par, se usa cuando no hay MarianMT para el par.
"""

from __future__ import annotations

MARIAN_LANGS = ("es", "en", "fr", "de")
MARIAN_PAIRS = frozenset((a, b) for a in MARIAN_LANGS for b in MARIAN_LANGS if a != b)

NLLB_MODEL = "facebook/nllb-200-distilled-600M"
NLLB_CODES = {
    "es": "spa_Latn",
    "en": "eng_Latn",
    "fr": "fra_Latn",
    "de": "deu_Latn",
    "it": "ita_Latn",
    "pt": "por_Latn",
    "nl": "nld_Latn",
    "ca": "cat_Latn",
    "ru": "rus_Cyrl",
    "zh": "zho_Hans",
    "ja": "jpn_Jpan",
}
BATCH = 16


def choose_model(src: str, tgt: str, engine: str = "auto") -> tuple[str, str]:
    """Devuelve (tipo, id del modelo) para traducir de src a tgt."""
    if engine in ("auto", "marian") and (src, tgt) in MARIAN_PAIRS:
        return "marian", f"Helsinki-NLP/opus-mt-{src}-{tgt}"
    if engine == "marian":
        raise ValueError(f"No hay modelo MarianMT para {src}->{tgt}; usa el motor 'nllb'.")
    if src not in NLLB_CODES or tgt not in NLLB_CODES:
        raise ValueError(f"No sé traducir {src}->{tgt}.")
    return "nllb", NLLB_MODEL


class Translator:
    def __init__(self, engine: str = "auto", device: str | None = None):
        self.engine = engine
        self.device = device
        self._models: dict[str, tuple[object, object]] = {}

    def _load(self, model_id: str):
        if model_id not in self._models:
            import torch
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

            device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            model = AutoModelForSeq2SeqLM.from_pretrained(model_id).to(device).eval()
            self._models[model_id] = (tokenizer, model)
        return self._models[model_id]

    def translate(self, texts: list[str], src: str, tgt: str) -> list[str]:
        if src == tgt or not texts:
            return list(texts)
        import torch

        kind, model_id = choose_model(src, tgt, self.engine)
        tokenizer, model = self._load(model_id)
        gen_kwargs: dict = {"num_beams": 4, "max_new_tokens": 512}
        if kind == "nllb":
            tokenizer.src_lang = NLLB_CODES[src]
            gen_kwargs["forced_bos_token_id"] = tokenizer.convert_tokens_to_ids(NLLB_CODES[tgt])
        out: list[str] = []
        for i in range(0, len(texts), BATCH):
            batch = texts[i : i + BATCH]
            inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to(model.device)
            with torch.inference_mode():
                tokens = model.generate(**inputs, **gen_kwargs)
            out.extend(t.strip() for t in tokenizer.batch_decode(tokens, skip_special_tokens=True))
        return out
