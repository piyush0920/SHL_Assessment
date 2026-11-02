# Spoken-English Grammar Scoring Engine

A multimodal regression project that predicts a spoken-English grammar score from short audio recordings. The system combines linguistic features extracted from speech transcripts with representations learned directly from audio.

**Task:** Predict a continuous score from 0 to 5 using information about both what the speaker says and how the speaker speaks.

## Problem and data

The project README records the following dataset split:

| Partition | Samples |
|---|---:|
| Training | 769 |
| Test | 216 |
| Audio duration | Approximately 45–60 seconds per clip |
| Target | Grammar score on a 0–5 scale |

The audio dataset and pretrained model files are not included in this repository. The notebook discovers source files from a Kaggle input environment, so reproducing the complete pipeline requires access to the original assessment data and a compatible runtime.

## Modeling pipeline

The project approaches grammar scoring as a multimodal regression problem.

### 1. Speech transcription

- Transcribe audio using **Whisper-small**.
- Retain useful speech disfluencies such as fillers, repetitions, and false starts.
- Cache transcription outputs to avoid repeating expensive inference.

### 2. Linguistic and grammar features

Engineer interpretable features from transcripts, including:

- Grammar-error signals using **LanguageTool**.
- Syntactic structure and dependency features using **spaCy**.
- Sentence length and complexity.
- Vocabulary richness and lexical diversity.
- Repetition, filler words, and disfluency patterns.
- GPT-2 language-model perplexity.

### 3. Text representations

- Generate sentence embeddings with **MPNet**.
- Fine-tune **DeBERTa-v3-base** for task-specific regression on transcript text.

### 4. Acoustic representations

Use pretrained speech encoders to capture information that may be lost during transcription:

- **WavLM** representations.
- **Whisper encoder** hidden states.
- Statistical pooling over temporal representations to create fixed-length feature vectors.

### 5. Base regressors and stacking

The notebook compares complementary model families, including Ridge Regression, Support Vector Regression, Gradient Boosting, multimodal Ridge, and fine-tuned DeBERTa.

Out-of-fold predictions are generated with five-fold cross-validation and combined using a **non-negative linear stacking model**. The transformer branch also includes multi-seed experiments to examine prediction stability.

## Architecture at a glance

```text
Audio recording
      |
      +----------------------------+
      |                            |
Whisper transcription        Audio encoders
      |                       WavLM / Whisper
      |                            |
Grammar + syntax features          |
      |                            |
MPNet / DeBERTa                    |
      |                            |
      +-------------+--------------+
                    |
             Base regressors
                    |
          Five-fold OOF predictions
                    |
        Non-negative linear stacker
                    |
          Predicted grammar score
                  (0–5)
```

## Reported result

The existing project README records a best observed score of **0.3668**. The repository does not currently identify the leaderboard metric, scoring direction, or a baseline alongside that number. Treat it as a recorded project result rather than a directly interpretable accuracy or RMSE until the evaluation metric and competition context are added.

For a meaningful model comparison, report the validation/leaderboard metric name, whether lower or higher is better, and the baseline score next to the ensemble result.

## Technology stack

| Area | Tools |
|---|---|
| Audio transcription | Whisper |
| Acoustic representations | WavLM, Whisper encoder |
| NLP features | LanguageTool, spaCy, GPT-2 |
| Text representations | MPNet, DeBERTa-v3-base |
| Regression | Ridge, SVR, Gradient Boosting |
| Ensemble | Out-of-fold predictions, non-negative linear stacking |
| Core libraries | Python, PyTorch, Transformers, scikit-learn, pandas, NumPy |

## Repository contents

```text
.
├── shl-assessment-grammer-scoring-final.ipynb
├── piyush_sharma.csv
└── README.md
```

The notebook contains the modeling workflow. Audio clips, cached embeddings, checkpoints, and other intermediate artifacts are not committed here.

## Execution notes

The notebook is designed around a Kaggle-style input environment and expects the original audio and label files to be available under the input directory.

The workflow uses large pretrained models and audio feature extraction, so a GPU-backed runtime is recommended for practical execution. Install compatible versions of the libraries used in the notebook, including:

```bash
pip install numpy pandas scikit-learn scipy matplotlib seaborn   librosa soundfile transformers sentence-transformers   torch spacy language-tool-python jupyter
```

Additional setup may be needed for the LanguageTool Java runtime and model downloads. This is a dependency starting point, not a pinned, fully reproducible environment; the repository does not currently contain a `requirements.txt`.

## Limitations and responsible use

- The audio dataset is small, so generalisation to different accents, recording conditions, ages, or speech contexts is not established.
- Grammar scores can be influenced by transcription quality and speech-recognition errors.
- A predicted score should not be treated as a definitive measure of a person's communication ability.
- The current repository does not include the original audio inputs or a complete environment lockfile.
- The reported 0.3668 score needs its metric name and evaluation context documented.

## Author

**Piyush Sharma**  
[GitHub](https://github.com/piyush0920) · [Portfolio](https://piyush0920.github.io/) · [LinkedIn](https://linkedin.com/in/piyush-sharma-6591171ba)
