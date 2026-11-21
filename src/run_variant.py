import time

from faster_whisper import WhisperModel
from faster_whisper.tokenizer import Tokenizer

from constants import KEY_PROJECT_PACKAGES
from log_runs import AudioSplit, ReplaceArtifact, RunConfig, RunRepository
from process import (
    apply_replacements,
    build_hotwords,
    ensure_hotwords_fit,
    find_dictionary_replacements,
    transcribe,
)
from project_data import ProjectRepository
from validate_data import find_missing_hidden_terms

HOTWORDS_SOURCE = "terms_it.csv term column minus hidden_terms.txt"


def run_variant_first(
    config: RunConfig,
    split: AudioSplit,
    run_repo: RunRepository,
    project_repo: ProjectRepository,
    key_packages: list[str] = KEY_PROJECT_PACKAGES,
) -> str:
    config_id = run_repo.save_config(config)
    recordings = project_repo.get_recordings(split)

    whisper_model = WhisperModel(
        config.whisper.model,
        device=config.whisper.device,
        compute_type=config.whisper.compute_type,
    )

    tokenizer = Tokenizer(
        whisper_model.hf_tokenizer,
        whisper_model.model.is_multilingual,
        task="transcribe",
        language=config.whisper.language,
    )
    ensure_hotwords_fit(config.whisper.hotwords, tokenizer, whisper_model.max_length)

    with run_repo.start_run(
        config_id=config_id, split=split, key_packages=key_packages
    ) as run:
        for rec in recordings:
            artifact = transcribe(whisper_model, config.whisper, rec)
            run.save_artifact(artifact)

        return run.run_id

    raise RuntimeError(
        "Returned run_id is empty. Something went wrong during run execution"
    )


def run_variant_second(
    config: RunConfig,
    split: AudioSplit,
    run_repo: RunRepository,
    project_repo: ProjectRepository,
    key_packages: list[str] = KEY_PROJECT_PACKAGES,
) -> str:
    """Run variant 2: Whisper with the dictionary terms as hotwords, hidden terms left out.

    Keeps every Whisper setting from `config` and only fills in the hotwords,
    so the hotwords are the one difference from variant 1. Raises ValueError
    before anything is saved if hidden_terms.txt names a term that is not in
    the dictionary, since such a term would silently stay in the hotwords.
    """
    all_terms = project_repo.get_dictionary(exclude_hidden=False)["term"].tolist()
    missing = find_missing_hidden_terms(all_terms, project_repo.get_hidden_terms())
    if missing:
        raise ValueError(f"Hidden terms not in dictionary: {missing}")

    hotwords = build_hotwords(project_repo.get_dictionary())
    whisper = config.whisper.model_copy(
        update={"hotwords": hotwords, "hotwords_source": HOTWORDS_SOURCE}
    )
    config = config.model_copy(update={"variant": 2, "whisper": whisper})

    return run_variant_first(
        config,
        split,
        run_repo,
        project_repo,
        key_packages=key_packages
    )


def run_variant_third(
    config: RunConfig,
    split: AudioSplit,
    run_repo: RunRepository,
    project_repo: ProjectRepository,
    key_packages: list[str] = KEY_PROJECT_PACKAGES,
) -> str:
    """Run variant 3: dictionary correction of the latest variant 2 run, no Whisper.

    `config` needs `parent_config_id` (the variant 2 config) and
    `dictionary.rapidfuzz_threshold` on the 0-100 scale. `variant`, `whisper`
    (copied from the parent config) and the dictionary hashes are filled in here.
    Corrects every recording of the latest "ok" run of the parent config on
    `split` and saves one ReplaceArtifact per recording. `rejected` stays empty,
    because find_dictionary_replacements never proposes overlapping replacements.

    Raises ValueError before anything is saved if the parent config or its run
    is missing, the threshold is not on the 0-100 scale, or hidden_terms.txt
    names a term that is not in the dictionary.
    """
    if config.parent_config_id is None or config.dictionary is None:
        raise ValueError("Variant 3 needs parent_config_id and dictionary in config")

    threshold = config.dictionary.rapidfuzz_threshold
    if not 1 < threshold <= 100:
        raise ValueError(
            f"rapidfuzz_threshold must be on the 0-100 scale, got {threshold}"
        )

    parent = run_repo.get_config(config.parent_config_id)
    if parent is None:
        raise ValueError(f"Parent config not found: {config.parent_config_id}")

    source_run = run_repo.get_latest_run(config.parent_config_id, split)
    if source_run is None:
        raise ValueError(
            f"No 'ok' run of config {config.parent_config_id} on {split}, "
            "run variant 2 first"
        )

    all_terms = project_repo.get_dictionary(exclude_hidden=False)["term"].tolist()
    missing = find_missing_hidden_terms(all_terms, project_repo.get_hidden_terms())
    if missing:
        raise ValueError(f"Hidden terms not in dictionary: {missing}")

    dictionary_config = config.dictionary.model_copy(
        update={
            "terms_sha": project_repo.get_dictionary_sha(),
            "hidden_terms_sha": project_repo.get_hidden_terms_sha(),
        }
    )
    config = config.model_copy(
        update={
            "variant": 3,
            "whisper": parent.whisper,
            "dictionary": dictionary_config,
        }
    )
    config_id = run_repo.save_config(config)
    dictionary = project_repo.get_dictionary()

    with run_repo.start_run(
        config_id=config_id,
        split=split,
        key_packages=key_packages,
        source_run_id=source_run.run_id,
    ) as run:
        for rec_id in source_run.rec_ids:
            text = run_repo.get_artifact(source_run.run_id, rec_id).final_text

            start = time.perf_counter()
            replacements = find_dictionary_replacements(
                text, dictionary, dictionary_config
            )
            final_text = apply_replacements(text, replacements)
            replace_s = time.perf_counter() - start

            run.save_artifact(
                ReplaceArtifact(
                    rec_id=rec_id,
                    final_text=final_text,
                    replacements=replacements,
                    rejected=[],
                    replace_s=replace_s,
                )
            )

        return run.run_id

    raise RuntimeError(
        "Returned run_id is empty. Something went wrong during run execution"
    )
