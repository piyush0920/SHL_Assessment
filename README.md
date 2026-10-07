# SHL Hiring Assessment 2026 — Grammar Scoring Engine

A multimodal machine learning solution for predicting **continuous spoken grammar scores (0–5)** from 45–60 second `.wav` speech recordings.

## Approach

The pipeline combines **audio, linguistic, and semantic representations**:

- **Whisper-small** for speech-to-text transcription
- **Handcrafted linguistic and audio features** such as sentence length, lexical diversity, repetition, fillers, speaking rate, and pauses
- **GPT-2 perplexity** as a language-model-based signal
- **LanguageTool + spaCy** for grammar and syntactic features
- **MPNet embeddings** for text representation
- **DeBERTa-v3-base** fine-tuned for text regression
- **WavLM-base-plus, Whisper encoder, and WavLM-large** embeddings using mean + standard-deviation pooling
- **Ridge, SVR, and Gradient Boosting** regression models
- **OOF-based non-negative stacking** to combine complementary model predictions

## Validation

Given the small training set of **769 labelled samples**, the solution uses **5-fold cross-validation and out-of-fold predictions** for model evaluation and stacking.

### V4 Performance

- **CV RMSE:** 0.4779
- **Pearson Correlation:** 0.9228
- **Kaggle Leaderboard Score:** 0.3694

Ablation experiments showed that **pretrained audio representations provided the strongest predictive signal**, while text-based models contributed complementary information.

## Files

- `shl_grammar_scoring_v4.ipynb` — complete end-to-end solution
- `submission.csv` — competition submission

## Key Takeaway

The solution combines **speech representations, linguistic analysis, and ensemble learning** to estimate human-rated grammar proficiency from spoken language.
