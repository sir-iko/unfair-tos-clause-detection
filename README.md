# Unfair Clause Detection in Terms of Service

A comparative study of four machine learning approaches for identifying unfair clauses in consumer Terms of Service agreements, evaluated on the CLAUDETTE ToS-100 corpus. Fine-tuned BERT achieved a macro F1 of 0.8270, with a Linear SVM baseline close behind.

## Problem

Consumer Terms of Service documents are long, dense, and rarely read in full. Buried inside them are clauses that can be unfair to consumers: limitation of liability, unilateral changes, mandatory arbitration, and so on. Automatically flagging such clauses helps consumers and regulators identify problematic agreements without manual review of every document.

## Dataset

The project uses the CLAUDETTE ToS-100 corpus, a publicly available dataset of 100 online Terms of Service documents annotated at the sentence level for unfair clauses. The dataset is available from [the CLAUDETTE project](http://claudette.eui.eu/).

## Approach

Four models on the binary classification task (fair vs unfair sentence):

1. **Multinomial Naive Bayes** as a probabilistic baseline using TF-IDF features.
2. **Linear SVM** (LinearSVC) using the same TF-IDF representation, chosen for its strong baseline performance on text classification.
3. **Bi-LSTM** with pre-trained word embeddings to capture sequential context within clauses.
4. **Fine-tuned BERT** (`bert-base-uncased`) implemented in PyTorch via the HuggingFace Trainer API, providing contextual embeddings.

Preprocessing for the classical models used gensim for tokenisation and spaCy for POS filtering. The pipeline retains auxiliary verbs (`AUX`) because modal verbs ("shall", "may", "must") carry meaningful signal in legal language.

## Results

| Model | Macro F1 |
|-------|----------|
| Multinomial Naive Bayes | 0.6894 |
| Linear SVM | 0.8173 |
| Bi-LSTM | 0.7654 |
| **Fine-tuned BERT** | **0.8270** |

Fine-tuned BERT achieved the highest macro F1 at 0.8270, narrowly outperforming Linear SVM at 0.8173. The Bi-LSTM underperformed both, likely reflecting the small corpus size where contextual transformer features and well-engineered TF-IDF representations both outperform a sequence model trained from limited examples. Naive Bayes provided a useful lower-bound baseline.

## Repository Structure

```
unfair-tos-clause-detection/
├── notebooks/
│   └── tos-unfairness-detection.ipynb    # full pipeline: preprocessing, training, evaluation
├── reports/
│   └── final-report.pdf                  # full written report
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

## How to Run

1. Clone the repository:
```bash
   git clone https://github.com/[your-username]/unfair-tos-clause-detection.git
   cd unfair-tos-clause-detection
```
2. Install dependencies:
```bash
   pip install -r requirements.txt
```
3. Download the CLAUDETTE ToS-100 dataset and place `ToS-100.csv` in the project root.
4. Open the notebook:
```bash
   jupyter notebook notebooks/tos-unfairness-detection.ipynb
```

## Report

The full written report with methodology, related work, and analysis is available in [`reports/final-report.pdf`](reports/final-report.pdf).

## Tech Stack

Python, scikit-learn, PyTorch, HuggingFace Transformers, gensim, spaCy, Jupyter
