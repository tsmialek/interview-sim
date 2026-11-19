import pytest
import subprocess
from pathlib import Path


@pytest.fixture
def git_repo(tmp_path) -> Path:
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()

    subprocess.run(["git", "init"], cwd=repo_dir, check=True)
    subprocess.run(
        ["git", "config", "user.email", "user@example.com"], cwd=repo_dir, check=True
    )
    subprocess.run(
        ["git", "config", "user.name", "Test User"], cwd=repo_dir, check=True
    )

    dummy_file_0 = repo_dir / "test_file_0.txt"
    dummy_file_1 = repo_dir / "test_file_1.txt"
    dummy_file_0.write_text("Hello world!\n", encoding="utf-8")
    dummy_file_1.write_text("Hello world from second file\n", encoding="utf-8")

    subprocess.run(["git", "add", "."], cwd=repo_dir, check=True)
    subprocess.run(["git", "commit", "-m", "Iinitial commit"], cwd=repo_dir, check=True)

    return repo_dir


@pytest.fixture
def make_data_dir(tmp_path):
    def _factory(rec_ids: list[str] | None = None) -> Path:
        if not rec_ids:
            rec_ids = []

        recordings = (
            "file_id,speaker,sample_type,question_id,style,device,environment,split\n"
            "rec_203,KJ,positive,9,casual,laptop,quiet,dev\n"
            "rec_202,KJ,positive,6,casual,laptop,quiet,test\n"
            "rec_201,KJ,positive,3,casual,laptop,quiet,dev\n"
            "rec_204,KJ,positive,12,casual,laptop,quiet,test\n"
        )

        terms = (
            "term,aliases,category,description\n"
            "Jira,dżira|dżiry|dżirze|dżirą|dżirę|jiry|jirze,tool,system do zarządzania zadaniami\n"
            "Confluence,konfluens|konfluensa|konfluensie|konfluencji|confluensie,tool,firmowa baza wiedzy i dokumentacji\n"
            "GitHub,githab|gitab|githuba|githubie|githubem|gitchab,tool,serwis do hostowania repozytoriów Git\n"
            "GitLab,git lab|gitlaba|gitlabie|gitlabem|gitlabu,tool,platforma do repozytoriów i CI/CD\n"
            "Jenkins,dżenkins|dżenkinsa|dżenkinsie|dżenkinsem|jenkinsa,tool,serwer automatyzacji buildów\n"
            "Slack,slak|slaka|slaku|slacku|slackiem|slakiem,tool,komunikator zespołowy\n"
        )

        data_dir = tmp_path / "data"
        data_dir.mkdir()

        audio_path = data_dir / "audio"
        audio_path.mkdir()

        transcript_path = data_dir / "transcripts"
        transcript_path.mkdir()

        (data_dir / "terms_it.csv").write_text(terms, encoding="utf-8")

        (transcript_path / "rec_1.txt").write_text(
            "Korzystałem z {AWS|AWS} z systemów {EC2|EC2}", encoding="utf-8"
        )
        (transcript_path / "rec_101.txt").write_text(
            "Korzystałem z {dotnetu|.NET} oraz {Javy|Java}", encoding="utf-8"
        )
        (data_dir / "hidden_terms.txt").write_text(
            "Jira \n\nGitHub\n", encoding="utf-8"
        )

        recordings_path = data_dir / "recordings.csv"
        recordings_path.write_text(recordings, encoding="utf-8")
        for id in rec_ids:
            (audio_path / f"{id}.wav").touch()

        return data_dir

    return _factory
