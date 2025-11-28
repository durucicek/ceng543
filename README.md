# CENG 543 – Take Home Assignment

This repository contains implementations of advanced NLP models including RNN-based classifiers, attention-equipped seq2seq models, Transformers, and a Retrieval-Augmented Generation (RAG) pipeline.

## 📦 Project Structure

* `Q1.ipynb` – Recurrent architectures & embedding paradigms (IMDb sentiment)
* `Q2-Q3-Q5.ipynb` – Attention mechanisms, Transformer vs RNN, and interpretability (Multi30k EN→DE translation)
* `Q4.ipynb` – Retrieval-Augmented Generation (RAG) on HotpotQA

---

## 🔧 Environment Setup

### Prerequisites

* Python 3.10
* Conda
* CUDA-capable GPU 

### Installation

```bash
conda create -n 543 python=3.10.19
conda activate 543

cd 543          # or the root of this repo
pip install -r requirements.txt
```

## 📓 Notebook Overview

### Q1.ipynb – Recurrent Architectures & Embeddings (IMDb)

**Task:** Binary sentiment analysis on IMDb.
**Data:** ~10,000 training and ~1,000 test samples (subset of IMDb).

**Models:**

* Architectures:

  * BiLSTM
  * BiGRU
* Embeddings:

  * Word2Vec (static, Google News, 300-dim)
  * DistilBERT (contextual, frozen encoder)

This yields 4 variants:

* `BiLSTM_Word2Vec`
* `BiGRU_Word2Vec`
* `BiLSTM_BERT`
* `BiGRU_BERT`

**Metrics & Outputs:**

* Accuracy, Macro-F1, training time per model
* Convergence curves (loss vs epoch)
* Simple embedding visualizations (e.g., t-SNE)

**Key takeaway (high-level):**

* Contextual embeddings (DistilBERT) consistently outperform static Word2Vec.
* Among RNNs, GRU variants generally achieve slightly better performance and optimization than LSTM under the same embedding.

---

### Q2-Q3-Q5.ipynb – Attention, Transformers & Interpretability (Multi30k EN→DE)

#### Q2 – Attention Mechanisms

**Task:** English→German machine translation (Multi30k).
**Backbone:** BiGRU encoder–decoder with attention.

**Attention variants:**

* Additive (Bahdanau)
* Multiplicative (Luong)
* Scaled Dot-Product

**Training:**

* Default `EPOCHS = 5` (can be increased to 10–15 if time allows).

**Metrics & Outputs:**

* BLEU, ROUGE-L, perplexity
* Attention heatmaps for qualitative inspection

**High-level findings:**

* Additive and scaled dot-product attention produce sharper, more selective alignments and better BLEU/ROUGE than the multiplicative variant.

---

#### Q3 – From Seq2Seq to Transformer

**Task:** Same Multi30k EN→DE setup.

**Comparison:**

* Best-attention RNN model from Q2 vs. Transformer encoder–decoder.
* Ablations on:

  * Number of layers (e.g., 1 / 3 / 6)
  * Number of heads (e.g., 2 / 8 / 16)

**Metrics:**

* Perplexity (PPL)
* BLEU, ROUGE-L
* Training time & (approximate) GPU memory

**High-level findings:**

* Transformer provides better or comparable BLEU at similar or lower perplexity than the RNN baseline.
* A 3-layer, 8-head “Base” configuration is a good sweet spot: deeper or very wide-head configurations only mildly improve PPL and can hurt BLEU or training stability on this small dataset.

---

#### Q5 – Interpretability & Trust

**Focus:**

* Diagnostic analysis of the trained translation models using:

  * Token-level uncertainty (entropy over output distribution)
  * Attention maps
  * Integrated Gradients (IG) for input importance

**What you get:**

* Examples of confident vs uncertain errors
* Cases where the model attends to wrong source tokens vs. cases where vocabulary coverage is the real issue
* Discussion of how entropy, attention, and IG together improve transparency and help identify when to involve human oversight

---

### Q4.ipynb – Retrieval-Augmented Generation (HotpotQA)

**Task:** Multi-hop question answering with RAG on a subset of HotpotQA.

**Components:**

* Retrievers:

  * BM25 (sparse)
  * Sentence-BERT `all-MiniLM-L6-v2` (dense)
* Generator:

  * FLAN-T5 (small)
* Data:

  * `SUBSET_SIZE = 5000` questions from HotpotQA validation split

**Metrics:**

* Retrieval: Precision@K, Recall@K (relaxed: at least one relevant doc),
* Generation: BLEU, ROUGE-L, BERTScore

**Outputs:**

* Comparison table for BM25 vs SBERT (retrieval & generation metrics)
* Example questions with:

  * Retrieved documents
  * Model answers
  * Ground-truth answers
  * Labels for faithful vs hallucinated responses

---

## 🚀 How to Run

1. Activate environment:

   ```bash
   conda activate 543
   ```

2. Launch Jupyter:

   ```bash
   cd 543   # repo root
   jupyter notebook
   ```

3. Open and run notebooks:

   Recommended order:

   1. `Q1.ipynb` (independent)
   2. `Q2-Q3-Q5.ipynb` (run Q2 → Q3 → Q5 sections in sequence)
   3. `Q4.ipynb` (independent RAG pipeline)

Run cell-by-cell if you want to monitor training progress and metrics more closely.

---

## 👤 Author

* Duru Çiçek – 290201043