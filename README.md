# Vocare Interview Simulator

Speech analysis and candidate evaluation module for a job interview simulator: recording, transcription and assessment of the candidate's answers (speech fluency, professionalism, style and semantics).

**Authors: Tomasz Śmiałek, Bartłomiej Kłuś, Kacper Jóźwik**

_Project description: [docs/NLP - Opis projektu.pdf](docs/NLP%20-%20Opis%20projektu.pdf)_

## Setup

To start working with the project you need conda installed. Create the environment and register the Jupyter kernel:

```bash
# create the environment
conda env create -f environment.yml

# activate it
conda activate interview-sim

# register the kernel (for Jupyter Notebook)
python -m ipykernel install --user --name interview-sim

# enable the project git hooks (data validation before commit)
git config core.hooksPath .githooks
```

The Polish `spaCy` model (`pl_core_news_md`) is installed together with the environment.

After `environment.yml` changes, update an existing environment with:

```bash
conda env update -f environment.yml --prune
```

## Data

Recordings and annotations are versioned with [DVC](https://dvc.org/) and stored in a private Google Drive folder. Git keeps only the small `.dvc` pointer files. Recordings are personal data, so never commit them to Git.

### DVC setup

The remote is already defined in `.dvc/config`. You only need access to the Drive folder and the OAuth client secret, which you get from a project member outside the repository. Your Google account must also be added as a test user of the OAuth app.

Save the secret in `.dvc/config.local`, which is ignored by Git:

```bash
dvc remote modify --local gdrive gdrive_client_secret <CLIENT_SECRET>
```

Never put the secret in `.dvc/config`.

Download the data (on first use a browser window opens for Google sign-in):

```bash
dvc pull
```

### Adding or changing data

```bash
# start tracking a folder, or record changes in it
dvc add data/audio

# commit the pointer file
git add data/audio.dvc data/.gitignore
git commit -m "Update audio data"

# upload the data, then share the pointer
dvc push
git push
```

After someone else changes the data, run `git pull` and then `dvc pull`.

### Validating data

Run the validator after adding or editing recordings, transcripts, `recordings.csv` or `terms_it.csv`:

```bash
python src/validate_data.py               # full check
python src/validate_data.py --skip-audio  # without data/audio, e.g. before dvc pull
```

It checks that every recording has a transcript and a manifest row, that terms in `{form|term}` exist in the dictionary, that fillers are only `yyy` / `eee` / `mmm`, and that audio is 16 kHz mono 16-bit WAV. Errors make it exit with code 1, warnings (duration outside 45–90 s, digits, a dictionary term without braces) are for review.

With the hooks enabled (see [Setup](#setup)), it also runs automatically before every commit that touches `data/`. The hook blocks the commit when there are errors and prints the report (in VS Code: the error dialog, or Output → Git). Audio is checked only when `data/audio.dvc` is staged. To commit anyway, use `git commit --no-verify`.

### Handling DVC conflicts

```bash
git config merge.dvc.name 'DVC merge driver'
git config merge.dvc.driver 'dvc git-hook merge-driver --ancestor %O --our %A --their %B'
```

This handles adding new files to the same folder on different branches. It does not resolve simultaneous changes to the same file.
