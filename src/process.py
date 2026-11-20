import json
import time
from pathlib import Path
from typing import Protocol

import ollama
import pandas as pd
from faster_whisper import WhisperModel

from log_runs import (
    DictionaryConfig,
    Replacement,
    SlmConfig,
    TranscribeArtifact,
    WhisperConfig,
)
from utils import get_sha256

SLM_SYSTEM_PROMPT = """\
Poprawiasz pojedyncze terminy IT w transkrypcji rozmowy rekrutacyjnej (mowa po polsku, nagrana i spisana automatycznie).

Dostajesz:
- Zdanie: kontekst,
- Fragment: dokładny kawałek tekstu do oceny (może mieć kilka słów),
- Prawdopodobny termin: podstawowa forma terminu IT.

Zwracasz JSON z polami "r_to" i "status".
"r_to" ZASTĘPUJE CAŁY Fragment, więc nie powtarzaj w nim słów, które są błędnym zapisem.

Zasady:
1. Jeśli Fragment jest już poprawnie zapisany, zwróć go DOKŁADNIE bez zmian (status "podmiana").
2. Zachowaj przypadek i formę słowa z Fragmentu: jeśli w tekście jest rzeczownik odczasownikowy, przymiotnik albo inny przypadek, zostaw tę samą formę, popraw tylko pisownię.
3. Skrótowce i nazwy kończące się nietypowo odmieniaj z łącznikiem: "PR-ów", "AWS-ie", "GraphQL-u". Zwykłe nazwy bez łącznika: "Jirze", "Kubernetesie".
4. Nie dodawaj słów, nie zmieniaj terminu na inny (np. "deploy" to nie "deployment").
5. Jeśli nie znasz poprawnego zapisu: status "nie_wiem".
6. Jeśli Fragment w tym zdaniu nie oznacza terminu IT (np. wąż w zoo): status "nie_termin".

Przykłady:
Zdanie: Wrzuciłem poprawkę przez pul rekłest.
Fragment: pul rekłest
Prawdopodobny termin: pull request
{"r_to": "pull request", "status": "podmiana"}

Zdanie: Logi zbieramy w Kibanie.
Fragment: Kibanie
Prawdopodobny termin: Kibana
{"r_to": "Kibanie", "status": "podmiana"}

Zdanie: Mamy problem z refaktorowaniem starego kodu.
Fragment: refaktorowaniem
Prawdopodobny termin: refactoring
{"r_to": "refaktorowaniem", "status": "podmiana"}

Zdanie: Wszystko stoi na kubernetesie.
Fragment: kubernetesie
Prawdopodobny termin: Kubernetes
{"r_to": "Kubernetesie", "status": "podmiana"}

Zdanie: Kolejka wiadomości szła przez kafce.
Fragment: kafce
Prawdopodobny termin: Kafka
{"r_to": "Kafce", "status": "podmiana"}

Zdanie: W terrarium mieszka pyton.
Fragment: pyton
Prawdopodobny termin: Python
{"r_to": "pyton", "status": "nie_termin"}
"""


SLM_SCHEMA = {
    "type": "object",
    "properties": {
        "status": {"enum": ["podmiana", "nie_wiem", "nie_termin"]},
        "r_to": {"type": "string"},
    },
    "required": ["status", "r_to"],
}


def transcribe(
    whisper_model: WhisperModel, whisper_conf: WhisperConfig, audio_path: Path | str
) -> TranscribeArtifact:
    start = time.perf_counter()
    segments, info = whisper_model.transcribe(
        audio=str(audio_path), **whisper_conf.get_transcribe_args()
    )

    rec_id = Path(audio_path).stem

    words = []
    text_parts = []
    for seg in segments:
        text_parts.append(seg.text.strip())
        if seg.words:
            words.extend(seg.words)

    whisper_text = " ".join(text_parts)
    final_text = whisper_text
    duration_s = time.perf_counter() - start

    return TranscribeArtifact(
        rec_id=rec_id,
        final_text=final_text,
        words=words,
        transcribe_s=duration_s,
    )


# Every correction step only PROPOSES replacements (list[Replacement]).
# apply_replacements is the only function that changes the text.
# None of these functions read files or know about run_id / RunRepository.


def find_dictionary_replacements(
    text: str,
    dictionary: pd.DataFrame,
    config: DictionaryConfig,
) -> list[Replacement]:
    """Variant 3 Find misrecognised dictionary terms in `text`.

    `dictionary`: ProjectRepository.get_dictionary(), hidden terms already removed.
    Returns replacements with source="dictionary"; `position` refers to `text`.
    """
    raise NotImplementedError


class SlmClient(Protocol):
    """Minimal interface to the local SLM, so tests can pass a fake client.

    A real implementation wraps ollama.chat(model=..., format=schema, options=...)
    and returns the raw response text (expected to be JSON matching `schema`).
    """

    def chat(self, system: str, user: str, schema: dict) -> str: ...


class OllamaSlmClient:
    def __init__(self, config):
        self.config = config

    def chat(self, system: str, user: str, schema: dict) -> str:
        response = ollama.chat(
            model=self.config.model,
            messages=[
                {"role": "system", "content": system},
                {
                    "role": "user",
                    "content": user,
                },
            ],
            format=schema,
            options={"temperature": self.config.temperature, "num_predict": 1000},
        )
        return response.message.content


def make_slm_config(model: str, temperature: float) -> SlmConfig:
    config_json_string = json.dumps(SLM_SCHEMA, sort_keys=True)

    prompt_sha = get_sha256(config_json_string + SLM_SYSTEM_PROMPT, length=8)

    digest = None
    for m in ollama.list().models:
        if m.model == model:
            digest = m.digest
    if digest is None:
        raise ValueError("Model is not downloaded in Ollama")

    return SlmConfig(
        model=model, digest=digest, prompt_sha=prompt_sha, temperature=temperature
    )


def find_slm_replacements(
    text: str,
    candidates: list[Replacement] | None,
    config: SlmConfig,
    client: SlmClient,
) -> list[Replacement]:
    """Variant 4 Ask the SLM for replacements, with Polish inflection.

    - candidates=None: the SLM sees the whole `text` and returns a JSON list of
      replacements (approach 1): Not implemented
    - candidates given: the SLM gets each marked fragment (r_from at position) with
      its sentence and returns only the correct spelling for it (approach 2). Today
      the candidates come from the dictionary, in variant 5 from HerBERT.
    Returns RAW proposals with source="slm"; validation is done by
    validate_slm_replacements.
    """
    if candidates is None:
        raise NotImplementedError(
            "approach 1 is not implemented; pass a list of candidates (approach 2)"
        )

    proposals = []
    for candidate in candidates:
        start = text.rfind(".", 0, candidate.position)
        end = text.find(".", candidate.position)
        stop = end + 1 if end != -1 else len(text)
        sentence = text[start + 1 : stop].strip()

        user = (
            f"Zdanie: {sentence}\n"
            f"Fragment: {candidate.r_from}\n"
            f"Prawdopodobny termin: {candidate.base_term}"
        )
        raw = client.chat(SLM_SYSTEM_PROMPT, user, SLM_SCHEMA)
        try:
            answer = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if (
            answer.get("status") != "podmiana"
            or not answer.get("r_to")
            or candidate.r_from == answer.get("r_to")
        ):
            continue
        else:
            proposal = Replacement(
                r_from=candidate.r_from,
                r_to=answer.get("r_to"),
                base_term=candidate.base_term,
                position=candidate.position,
                source="slm",
            )
            proposals.append(proposal)
    return proposals


def validate_slm_replacements(
    text: str,
    proposals: list[Replacement],
) -> tuple[list[Replacement], list[Replacement]]:
    """Split SLM proposals into (accepted, rejected).

    Rules (poc-etap1-ustalenia.md, "Słownik i podmiana"):
    - text[position:position + len(r_from)] == r_from
    - r_to starts with base_term without its last 2 characters
      (Jira -> Jirze, Kafka -> Kafce); terms of up to 3 characters must appear
      in full (AWS -> AWS-ie)
    """
    accepted = []
    rejected = []

    for prop in proposals:
        position_ok = (
            text[prop.position : prop.position + len(prop.r_from)] == prop.r_from
        )
        if len(prop.base_term) <= 3:
            prefix = prop.base_term
        else:
            prefix = prop.base_term[:-2]
        prefix_ok = prop.r_to.startswith(prefix)

        if position_ok and prefix_ok:
            accepted.append(prop)
        else:
            rejected.append(prop)

    return accepted, rejected


def apply_replacements(text: str, replacements: list[Replacement]) -> str:
    """Return `text` with every replacement applied.

    Everything outside the replaced fragments must stay byte-for-byte unchanged.
    Apply from the highest position down, so earlier positions stay valid.
    Overlapping replacements are a conflict: decide and document the rule.
    """
    raise NotImplementedError
