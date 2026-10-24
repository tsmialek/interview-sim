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
```

The Polish `spaCy` model (`pl_core_news_md`) is installed together with the environment.

After `environment.yml` changes, update an existing environment with:

```bash
conda env update -f environment.yml --prune
```
