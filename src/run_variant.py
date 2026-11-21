from faster_whisper import WhisperModel
from faster_whisper.tokenizer import Tokenizer

from constants import KEY_PROJECT_PACKAGES
from log_runs import AudioSplit, RunConfig, RunRepository
from process import build_hotwords, ensure_hotwords_fit, transcribe
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
