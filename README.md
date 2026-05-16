# Multi-Label Romanian Emotion Classification using Statistical and Neural Approaches with Model Interpretability

## 1. Project Objective & Scope

The core goal of this project is to build and evaluate a system capable of detecting **multiple overlapping emotions** in Romanian text simultaneously. Instead of treating text as having only one emotional tone, the project maps Romanian tweets to an emotion inventory derived from Plutchik’s framework, while respecting the **actual label set provided by REDv2**.

**In scope**

- Multi-label classification on REDv2 (Romanian Emotion Dataset v2)
- TF-IDF text vectorization (shared feature space for all models)
- **Method 1:** Linear SVM with **Binary Relevance** on TF-IDF
- **Method 2:** Multi-layer perceptron (MLP) on TF-IDF (with optional dimensionality reduction)
- LIME-based interpretability and side-by-side comparison
- Standard multi-label metrics and a documented train/validation/test protocol

**Out of scope (unless added later as stretch goals)**

- Transformer / BERT-style contextual models (fair comparison would use different features; not part of the core “same features, different learner” design)
- Real-time deployment or API serving

---

## 2. The Data & Feature Engineering Phase

### 2.1 The Corpus (REDv2)

We use the **Romanian Emotion Dataset (REDv2)**: human-annotated tweets that capture nuanced language, slang, and cultural context. Because it is **multi-label**, each tweet can activate several emotion labels at once (e.g., Sadness and Anger).

**Official REDv2 emotion labels (7 classes)**

| REDv2 label | Present in Plutchik’s 8 basic emotions? |
|-------------|----------------------------------------|
| Anger       | Yes                                    |
| Fear        | Yes                                    |
| Joy         | Yes                                    |
| Sadness     | Yes                                    |
| Trust       | Yes                                    |
| Surprise    | Yes                                    |
| Neutral     | **No** — not one of Plutchik’s eight basic emotions |

**Plutchik’s eight basic emotions (for reference)**

Joy, Trust, Fear, Surprise, Sadness, Disgust, Anger, Anticipation.

### 2.2 REDv2 vs Plutchik: Differences and How We Tackle Them

| Issue | Description | How we handle it in this project |
|-------|-------------|----------------------------------|
| **Neutral** | REDv2 includes Neutral; Plutchik’s wheel has no direct “neutral” basic emotion. | Treat **Neutral as a first-class REDv2 label** in all training and evaluation. In the report, state explicitly that the task is **REDv2 multi-label emotion detection**, informed by Plutchik, not a strict 1:1 Plutchik replication. |
| **Missing Plutchik labels** | REDv2 has no **Disgust** or **Anticipation** annotations. | Do **not** invent these labels. Report results only on the seven REDv2 labels. Discuss in limitations that the corpus cannot evaluate disgust/anticipation. |
| **Naming** | Joy, Trust, Fear, Surprise, Sadness, Anger align with Plutchik. | Use REDv2 names in code and tables; add a short mapping table in the document/thesis linking to Plutchik where applicable. |
| **Framing** | Project title mentions Plutchik’s wheel. | Frame Plutchik as **conceptual motivation** (multi-dimensional, co-occurring emotions); **operational labels** are REDv2’s seven. |

**Recommended wording for the thesis/report**

> “We adopt Plutchik’s idea of multiple coexisting emotions as motivation. Annotations follow REDv2’s seven labels (Anger, Fear, Joy, Sadness, Neutral, Trust, Surprise). Neutral is retained as an emotion-related stance label. Disgust and Anticipation are out of scope because they are not annotated in REDv2.”

### 2.3 Data Protocol (Before Modeling)

- Use the **official REDv2 train/validation/test split** if provided; otherwise define a **fixed, stratified split** (document seed and proportions).
- Check for **duplicate tweets** or near-duplicates across splits; remove or assign duplicates to one split only.
- Run **exploratory data analysis (EDA)**:
  - Per-label frequency (class imbalance)
  - **Label co-occurrence matrix** (which emotions appear together)
  - Tweet length distribution (informs `max_features` and preprocessing)
- Save preprocessing choices (lowercasing, diacritics, tokenization) in the report.

### 2.4 Text Numerical Vectorization (TF-IDF)

Computers cannot read words directly; text is transformed into a numerical matrix:

- Each **row** = one tweet  
- Each **column** = a weighted importance of a token (word or n-gram) in that tweet relative to the corpus  

**Pipeline decisions to document**

| Parameter | Purpose |
|-----------|---------|
| `max_features` | Cap vocabulary size (e.g., 10k–50k) to control sparsity and memory |
| `ngram_range` | Often `(1, 2)` for unigrams + bigrams; justify in report |
| `min_df` / `max_df` | Remove very rare or overly common tokens |
| Romanian-specific tokenization | Consistent handling of diacritics (ă, â, î, ș, ț); align with REDv2 authors’ recommendations if available |

The **same fitted TF-IDF vectorizer** (trained on training text only) must be used for SVM, MLP, and LIME so comparisons are fair.

---

## 3. The Two-Pronged Modeling Strategy

Both methods consume the **same TF-IDF features** (or the same TF-IDF → SVD pipeline for the MLP if dimensionality reduction is applied). Hyperparameter tuning uses the **validation set** only; final numbers come from the **held-out test set** once.

### 3.1 Method 1: Geometrical Baseline — Linear SVM with Binary Relevance

**Concept:** TF-IDF defines a high-dimensional space. A **linear** classifier learns a hyperplane per emotion: which side of the plane indicates presence vs absence of that label.

**Multi-label formulation: Binary Relevance (BR)**

- Train **one binary classifier per label** (7 classifiers for REDv2).
- Each classifier learns independently: “Is Anger present?” “Is Joy present?” etc.
- This is **not** a single multi-class SVM (which assumes exactly one label per instance).

**Implementation sketch**

- Base estimator: **LinearSVC** or **SGDClassifier** with hinge/log loss on sparse TF-IDF.
- Wrap with `sklearn.multiclass.OneVsRestClassifier` (equivalent to Binary Relevance for multi-label).
- Tune **`C`** (regularization strength) on validation data.
- Use **`class_weight='balanced'`** or per-label weights if imbalance is severe.

**Optional sanity check:** Logistic Regression with Binary Relevance often performs similarly; can be mentioned if results are close.

### 3.2 Method 2: Pattern-Learning Network — MLP on TF-IDF

**Concept:** Hidden layers learn **non-linear** combinations of input features instead of a single linear boundary per label.

**Challenge:** Raw TF-IDF is **high-dimensional and sparse**. A naive MLP on the full matrix often trains poorly or overfits. We address this with **one or more** of the following (document which options we use):

#### A. Cap vocabulary size (`max_features`)

- Limit the TF-IDF matrix to the top *N* terms by corpus frequency (e.g., 10,000–30,000).
- Reduces width of the input layer and training cost.
- Apply the **same** `max_features` when comparing SVM and MLP unless the report justifies different caps.

#### B. Truncated SVD (LSA) on TF-IDF before the MLP

- After TF-IDF, apply **`TruncatedSVD`** (latent semantic analysis) to compress to *k* dimensions (e.g., 200–500).
- Fit SVD on **training data only**, then transform validation/test.
- **SVM path:** can use raw TF-IDF (linear models handle sparsity well) **or** the same SVD features for a stricter “same representation” comparison—state the choice clearly.
- **MLP path:** strongly recommended to use TF-IDF → SVD → MLP.

#### C. Strong regularization (MLP architecture and training)

| Technique | Role |
|-----------|------|
| **Small hidden layers** | e.g., one or two layers with 128–256 units |
| **Dropout** | e.g., 0.3–0.5 between layers |
| **Early stopping** | Monitor validation loss; restore best weights |
| **L2 weight decay** | Penalize large weights |
| **Batch normalization** | Optional; can stabilize training |
| **Lower learning rate + Adam** | Standard stable optimizer |

**Multi-label output:** Sigmoid activation on 7 outputs + **binary cross-entropy** loss (independent labels, consistent with Binary Relevance philosophy).

### 3.3 Simple Baseline (Required for Credibility)

Before claiming SVM or MLP “works,” include at least one trivial baseline:

- **Most frequent label(s)** per training distribution, or  
- **Binary Relevance with constant / majority prediction** per label  

This shows improvement is not only due to metric choice.

### 3.4 Threshold Tuning (Multi-Label)

Default probability threshold **0.5** is often suboptimal, especially for rare labels.

- Tune **per-label thresholds** on the **validation set** (e.g., grid search to maximize F1 per label or macro-F1).
- Apply tuned thresholds for **test-set** evaluation and for LIME case studies.

---

## 4. Model Limitations (TF-IDF, SVM, MLP)

These must appear in the report so results are interpreted correctly.

### 4.1 Limitations of TF-IDF (Affects Both Models)

| Limitation | Impact on REDv2 |
|------------|-----------------|
| **Bag-of-words** | Word order is ignored (“not happy” vs “happy” can be confused depending on tokenization). |
| **No deep context** | Sarcasm, irony, and culture-specific slang may be missed. |
| **Negation** | Romanian negation (“nu”, “n-am”) may not flip emotion scores reliably. |
| **OOV / typos** | Rare or misspelled words may be dropped or poorly represented. |
| **Neutral vs emotional** | Neutral may correlate with generic words; models may learn lexical shortcuts. |

Both SVM and MLP share these limits because they share (or nearly share) the same features.

### 4.2 Limitations of Linear SVM + Binary Relevance

| Limitation | Notes |
|------------|--------|
| **Linear boundaries only** | Cannot model XOR-like interactions between features without explicit feature engineering. |
| **Label independence (BR)** | Each emotion is trained separately; co-occurrence structure (e.g., Sadness + Anger) is not modeled jointly. |
| **Sparse high-dimensional inputs** | Works well in practice for text, but explanations may emphasize frequent emotion words. |

### 4.3 Limitations of MLP on TF-IDF (+ SVD)

| Limitation | Notes |
|------------|--------|
| **Many hyperparameters** | Layers, dropout, *k* for SVD, learning rate—unfair comparison if MLP is undertuned while SVM is well tuned. |
| **SVD information loss** | Compression helps training but may discard rare discriminative tokens. |
| **Still no sequential understanding** | MLP on TF-IDF/SVD is not a substitute for contextual embeddings. |
| **Overfitting risk** | Mitigated by SVD, dropout, early stopping, and small networks. |
| **Interpretability** | Harder to read than linear weights; LIME becomes more important. |

### 4.4 Limitations of LIME

| Limitation | Notes |
|------------|--------|
| **Local approximations** | Explains one prediction, not global model behavior. |
| **Instability** | Slight changes in sampling can change highlighted words; consider multiple runs or fixed seeds. |
| **Correlated words** | LIME may attribute effect to one of several co-occurring emotion cues. |
| **Disruption artifacts** | Masking words can produce unnatural strings; less realistic for slang-heavy tweets. |

**Complement for linear SVM:** For each label, report **top positive/negative TF-IDF coefficients** (when using linear models) as a stable global check alongside LIME.

---

## 5. Model Interpretability & Evaluation (LIME Integration)

High accuracy alone is insufficient; we verify that models decide for **plausible linguistic reasons**.

### 5.1 LIME Protocol

1. Select **5–10 test tweets** with clear criteria (document in appendix):
   - Multi-label (2+ emotions)
   - High model confidence vs uncertain cases
   - Cases where **SVM and MLP disagree**
   - At least one **Neutral** + emotion example
2. Run LIME with the **same TF-IDF vectorizer** (and SVD transformer for MLP if used).
3. For each predicted label, show **top contributing tokens** (positive/negative).
4. Compare SVM vs MLP: same trigger words vs different contextual tokens.

### 5.2 Qualitative Success Criteria (Define Up Front)

- **Good:** Highlighted tokens are interpretable emotion cues (e.g., “furios”, “trist”, “minunat”) or clear stance markers for Neutral.
- **Weak:** Explanations dominated by stopwords, usernames, URLs, or unrelated high-frequency terms.
- **Comparative question:** Does the MLP attribute weight to **different** or **additional** tokens than the linear model on the same tweet?

---

## 6. Evaluation Metrics & Comparative Analysis

### 6.1 Quantitative Metrics (Test Set)

| Metric | Purpose |
|--------|---------|
| **Micro-averaged F1** | Overall performance with label frequency influence |
| **Macro-averaged F1** | Equal weight per label; sensitive to rare classes |
| **Per-label F1, precision, recall** | Table for all 7 REDv2 labels |
| **Hamming loss** | Fraction of wrong label decisions (lower is better) |
| **Subset accuracy (exact match)** | Strict: entire label set must match exactly |
| **Label-wise PR-AUC** (optional) | Useful under imbalance |

Report **validation** results for tuning; report **test** results once for final comparison.

### 6.2 Comparative Analysis & Expected Outcomes

**Quantitative:** Compare SVM (BR) vs MLP on the same test split with the same metrics and threshold policy. Discuss whether gains on macro-F1 (rare labels) differ from micro-F1.

**Qualitative (LIME):** Assess whether the MLP captures subtle context better than the linear model, or whether both models rely on the same high-frequency emotion triggers.

**Fairness checklist**

- Same train/val/test splits
- Same text preprocessing
- Same TF-IDF fit on train only
- Comparable hyperparameter search effort (document grids)
- Threshold tuning applied consistently

---

## 7. Reproducibility & Implementation Checklist

Before coding, confirm:

- [ ] REDv2 downloaded; label columns verified: Anger, Fear, Joy, Sadness, Neutral, Trust, Surprise  
- [ ] Plutchik vs REDv2 mapping section written (Section 2.2)  
- [ ] EDA notebook/script: label counts + co-occurrence heatmap  
- [ ] TF-IDF pipeline with documented `max_features`, ngrams, `min_df`/`max_df`  
- [ ] Binary Relevance + **Linear SVM** (or linear SGD) with `C` tuning  
- [ ] MLP: TF-IDF → (recommended) **TruncatedSVD** → MLP with dropout + early stopping  
- [ ] Per-label threshold tuning on validation set  
- [ ] Baseline model implemented  
- [ ] Test evaluation: micro/macro F1, Hamming, per-label F1, subset accuracy  
- [ ] LIME on fixed test examples; linear coefficients for SVM where applicable  
- [ ] Limitations section (Section 4) included in final report  
- [ ] Fixed random seeds; `requirements.txt` with library versions  

**Suggested stack:** Python 3, `scikit-learn`, `numpy`, `scipy`, `lime`, `matplotlib` / `seaborn`, `pandas`; MLP via `sklearn.neural_network.MLPClassifier` or PyTorch (choose one and document).

---

## 8. Project Structure (Suggested)

```
project/
├── data/                 # REDv2 (not committed if large)
├── notebooks/            # EDA, LIME visualizations
├── src/
│   ├── preprocess.py
│   ├── features.py       # TF-IDF, optional SVD
│   ├── train_svm.py      # Binary Relevance + Linear SVM
│   ├── train_mlp.py
│   ├── evaluate.py       # metrics, thresholds
│   └── lime_explain.py
├── models/               # saved vectorizer, SVD, classifiers
├── results/              # metrics tables, figures
└── requirements.txt
```

---

## 9. Summary

This project detects **multiple overlapping emotions** in Romanian tweets using **REDv2’s seven labels**, motivated by Plutchik but **not claiming a full Plutchik label set** (Neutral is included; Disgust and Anticipation are absent). **TF-IDF** feeds a **Binary Relevance linear SVM** baseline and an **MLP** with explicit strategies for sparse high-dimensional inputs (`max_features`, **Truncated SVD**, regularization). **LIME** and linear coefficients audit whether predictions align with meaningful words. Evaluation combines **multi-label metrics**, **threshold tuning**, **baselines**, and a **fair comparison protocol**, with limitations stated clearly before implementation begins.
