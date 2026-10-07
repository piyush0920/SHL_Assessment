# 🎙️ Spoken English Grammar Scoring Engine

An end-to-end **multimodal machine learning system** for predicting spoken English grammar scores from 45–60 second audio clips.

The objective is to predict a continuous grammar score on a **0–5 scale**, combining information from both **what the speaker says** and **how the speaker speaks**.

---

## 🎯 Problem Statement

Given a short `.wav` speech recording, predict the corresponding **grammar proficiency score (0–5)**.

The dataset contains:

- **769 training samples**
- **216 test samples**
- Audio duration: approximately **45–60 seconds**
- Evaluation based on **RMSE and Pearson correlation**

---

## 🧠 Methodology

I approached the problem as a **multimodal regression task**, since spoken grammar quality is influenced by both linguistic content and acoustic characteristics.

### 1. Speech-to-Text with Whisper

The audio recordings are first transcribed using **Whisper-small**.

The transcription provides the textual representation required for linguistic and grammar analysis, while retaining useful disfluencies such as fillers, repetitions, and false starts.

### 2. Linguistic & Grammar Features

From the transcripts, I engineered interpretable features covering:

- Grammar error density using **LanguageTool**
- Syntactic structure using **spaCy**
- Sentence length and complexity
- Vocabulary richness and lexical diversity
- Repetition and disfluency patterns
- Filler-word usage
- Clause and dependency characteristics
- Language-model perplexity using **GPT-2**

These features provide explicit signals related to grammatical quality and fluency.

### 3. Text Embeddings

I used **MPNet sentence embeddings** to capture semantic and contextual information that handcrafted features cannot fully represent.

A fine-tuned **DeBERTa-v3-base regression model** was also trained directly on the transcripts to learn task-specific linguistic representations.

### 4. Acoustic Representation Learning

Because transcription can lose information about pronunciation, pauses, fluency and delivery, I also modeled the original audio directly.

Pretrained representations were extracted from:

- **WavLM**
- **Whisper encoder**

Temporal hidden representations were aggregated using statistical pooling to create fixed-length acoustic feature vectors.

### 5. Multiple Regression Models

Different model families were used to capture complementary patterns:

- Ridge Regression
- Support Vector Regression (SVR)
- Gradient Boosting
- Fine-tuned DeBERTa

A multimodal Ridge model was also trained using combinations of handcrafted, text and audio representations.

### 6. Out-of-Fold Stacking

All base models were evaluated using **5-fold cross-validation** and generated **out-of-fold (OOF) predictions**.

These predictions were then combined using a **non-negative linear stacking model**.

This allows the ensemble to learn how much each model contributes while reducing the risk of overfitting on the relatively small training set.

### 7. Model Stability

The fine-tuned DeBERTa branch was trained using **multiple random seeds**, and the predictions were averaged to improve stability and reduce sensitivity to initialization.

---

## 🔬 Key Design Decisions

### Why multimodal?

Grammar is not purely a text problem. Acoustic information can capture aspects of fluency and speech delivery that may not survive transcription.

### Why pretrained speech representations?

Models such as WavLM provide rich representations learned from large-scale speech data, allowing the system to capture acoustic information without training an audio encoder from scratch.

### Why handcrafted features?

They provide interpretable signals such as grammar errors, sentence structure, repetition and disfluency, complementing neural embeddings.

### Why stacking?

With only 769 labelled samples, combining several complementary models through OOF stacking was more robust than relying on a single model.

### Why non-negative stacking?

Constraining ensemble weights to be non-negative reduces unstable cancellations between correlated models and provides a more controlled ensemble.

---

## 📊 Final Performance

**Best observed leaderboard score: 0.3668**

| Component | Role |
|---|---|
| Whisper-small | Speech transcription |
| LanguageTool | Grammar error analysis |
| spaCy | Syntactic analysis |
| GPT-2 | Linguistic fluency / perplexity |
| MPNet | Semantic text representation |
| WavLM | Acoustic representation |
| Whisper Encoder | Speech representation |
| Ridge / SVR / Gradient Boosting | Base regression models |
| DeBERTa-v3-base | Task-specific text regression |
| Non-negative Linear Stack | Final ensemble |

---

## 🏗️ Overall Pipeline

```text
                Audio (.wav)
                      │
             ┌────────┴────────┐
             │                 │
        Whisper-small      Audio Encoders
             │                 │
        Transcript        WavLM / Whisper
             │                 │
     ┌───────┼────────┐        │
     │       │        │        │
LanguageTool spaCy   GPT-2     │
     │       │        │        │
     └───────┴────────┘        │
             │                 │
     Handcrafted Features      │
             │                 │
         MPNet Embedding       │
             │                 │
       Fine-tuned DeBERTa      │
             │                 │
             └────────┬────────┘
                      │
             Multiple Regressors
                      │
             5-Fold OOF Predictions
                      │
          Non-Negative Linear Stacker
                      │
                 Final Score
                   (0–5)
