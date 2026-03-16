# LLM Science Exam — Kaggle Competition

## Overview

The [Kaggle LLM Science Exam](https://www.kaggle.com/competitions/kaggle-llm-science-exam) competition (2023) required answering science multiple-choice questions (5 options, MAP@3 metric). The key constraint was a 9-hour GPU inference budget with no internet at submission time, making retrieval-augmented inference the dominant paradigm.

Kaggle profile: [illidan7](https://www.kaggle.com/illidan7)

## Approach

### 1. Wikipedia Corpus Construction

Filtered 5.7M Wikipedia articles to ~270k STEM-relevant articles using Cohere embeddings and hierarchical KMeans clustering. This curated corpus became the knowledge base for all retrieval-augmented inference.

### 2. Training Data Engineering

Augmented the limited competition data (200 samples) with MMLU (100k+), ScienceQA, and ARC datasets. Used GPT-3.5 to generate plausible 5th options for 4-option datasets. Applied BERTopic distribution matching to ensure training data reflected test-set topic distribution.

### 3. Retrieval Pipeline

Built a two-stage retrieval system: (1) FAISS similarity search with SentenceTransformers over the 270k article corpus, then (2) TF-IDF re-ranking to select the most relevant sentences. Experimented with dual-context retrieval combining filtered and original Wikipedia.

### 4. Multi-Model Inference

Ran multiple fine-tuned models in parallel — DeBERTa-v3-large (shorter context, higher accuracy), LongFormer-large (4096-token context window, broader coverage), and an AWP-trained DeBERTa variant. An "openbook" fallback model handled low-confidence predictions.

### 5. Ensemble Weight Optimization

Combined model predictions via softmax probability fusion with learned weights, optimized through scipy.minimize (Nelder-Mead) and custom hill-climbing search against MAP@3 on validation sets.

### 6. 70B LLM Experiment

Tested direct inference with Xwin-LM-70B using RAG (layer-by-layer model loading), demonstrating the retriever+reader paradigm as an alternative, though classifier ensembles proved more practical within compute constraints.

## Repository Structure

```
├── baseline/
│   └── deberta-v3-baseline-training.ipynb        # Initial DeBERTa fine-tune
├── data-preparation/
│   ├── wikipedia-270k-corpus-filtering.ipynb     # ★ 5.7M → 270k STEM articles (Cohere + KMeans)
│   ├── mmlu-option-e-generation.ipynb            # GPT-3.5 generates 5th options
│   ├── science-question-curation-faiss.ipynb     # FAISS-based training data curation
│   ├── topic-distribution-filtering.ipynb        # BERTopic distribution matching
│   └── validation-set-construction.ipynb         # Multi-dataset validation set
├── retrieval-pipeline/
│   ├── tfidf-longformer-openbook.ipynb           # TF-IDF retrieval + LongFormer + fallback
│   ├── faiss-retrieval-pipeline.ipynb            # FAISS retrieval variant
│   └── dual-context-ensemble.ipynb               # Combined filtered + original Wikipedia
├── ensemble/
│   ├── multi-model-ensemble.ipynb                # ★ Peak ensemble: 3+ specialized models
│   ├── scipy-weight-optimization.ipynb           # Weight optimization via scipy
│   └── hillclimb-weight-search.ipynb             # Hill-climbing weight search
└── experiments/
    └── xwin-70b-rag-inference.ipynb              # 70B LLM with RAG experiment
```



## Kaggle Notebooks

Published notebooks on Kaggle ([illidan7](https://www.kaggle.com/illidan7)):

- [LLMSE - Study for the right exam!](https://www.kaggle.com/code/illidan7/llmse-study-for-the-right-exam) (21 upvotes)
- [LLMSE - ContextEnsemble](https://www.kaggle.com/code/illidan7/llmse-contextensemble) (3 upvotes)

## Tech Stack

- **Models**: DeBERTa-v3-large, LongFormer-large, AWP-trained DeBERTa, Xwin-LM-70B
- **Retrieval**: FAISS, SentenceTransformers, TF-IDF, Cohere embeddings
- **Data Engineering**: GPT-3.5 (option generation), BERTopic, HDBSCAN, UMAP
- **Optimization**: scipy.minimize, hill climbing, MAP@3
- **Infrastructure**: Kaggle Notebooks (GPU, 9-hour budget)

## Competition

- **Name**: [Kaggle LLM Science Exam](https://www.kaggle.com/competitions/kaggle-llm-science-exam)
- **Type**: Multiple-choice QA (5 options)
- **Metric**: MAP@3
- **Timeline**: July — October 2023
