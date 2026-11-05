import pytest
from log_runs import WhisperConfig


@pytest.fixture(scope="function")
def whisper_model_and_config():
    config: WhisperConfig
