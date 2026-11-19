from faster_whisper import WhisperModel
from faster_whisper.tokenizer import Tokenizer

from constants import KEY_PROJECT_PACKAGES, RECORDINGS_PATH
from log_runs import AudioSplit, RunConfig, RunRepository
from process import ensure_hotwords_fit, transcribe
from project_data import ProjectRepository

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
