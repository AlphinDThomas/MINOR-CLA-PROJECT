# Forensic Stylometry: Recognizing Human and LLM-Generated Texts in Short Samples

> **Primary Theoretical & Empirical Foundation:**  
> Przystalski, K., Argasinski, J. K., Grabska-Gradzinska, I., & Ochab, J. K. (2026).  
> **"Stylometry recognizes human and LLM-generated texts in short samples"**  
> *Expert Systems with Applications*, 296, Article 129001.  
> DOI: [10.1016/j.eswa.2025.129001](https://doi.org/10.1016/j.eswa.2025.129001)

---

## 1. Project Overview & Architectural Pivot

This project focuses on **Computational Forensic Stylometry** to reliably distinguish between human-authored writing and machine-generated texts (MGT) synthesized by modern Large Language Models (including **GPT-3.5**, **GPT-4**, **LLaMa 2**, **LLaMa 3**, **Orca**, and **Falcon**).

### Why the Shift Away from Recommender Systems (SVD / ALS)?
Earlier work attempted to solve fraudulent review injection via Collaborative Filtering matrix factorizations (**FunkSVD** and **ALS**). However, rating matrices ($R \approx P Q^\top$) only observe scalar ratings and fail completely when adversaries generate syntactically flawless, high-volume synthetic text reviews with LLMs. 

By pivoting to **computational stylometry**, the system analyzes the **subconscious authorial and syntactic DNA** of the text itself. As proven by Przystalski et al. (2026), stylometric features achieve **98% to 100% accuracy** even on short text samples (10 sentences / ~150–250 tokens) and remain robust against adversarial paraphrasers like DIPPER (11B) and Parrot (T5).

---

## 2. Core Stylometric Dimensions

Our pipeline implements both **StyloMetrix** (human-designed linguistic rules) and **CLARIN-PL** (statistical n-gram frequencies):

1. **Fact-Packing Density (`PROPN` & `NUM`):** Human encyclopedic and experiential writing exhibits heavy concentrations of proper names, dates, and numbers. LLMs substitute concrete facts with abstract generalizations.
2. **Grammatical Standardization:** LLM decoders exhibit tightly constrained POS n-gram frequencies with minimal authorial variance, whereas human texts show rich stylistic dispersion and long-tail outliers.
3. **Overused LLM Lexical Markers:** Quantitative tracking of chronic LLM tropes: `significant`, `notable`, `despite`, `furthermore`, `testament`, `crucial`, `legacy`, and `various`.
4. **Syntactic Fronting (`FOS_FRONTING`):** Frequency of placing adverbial/prepositional clauses prior to the sentence subject.
5. **Lemma Type-Token Ratio (`L_TYPE_TOKEN_RATIO_LEMMAS`):** Measures vocabulary diversity and lexical richness over lemmatized root forms.
6. **Generation & Formatting Artifacts (`SPACE` Token):** Detection of double spaces and paragraph-initial whitespace tokens characteristic of specific LLMs (e.g., LLaMa 2).

---

## 3. Repository Structure

- [`app.py`](file:///c:/Users/hp/OneDrive/Desktop/College/7th%20Sem/Minor%20Project/MINOR-CLA-PROJECT-main/app.py) — Interactive Streamlit Forensic Stylometry & LLM Detection Suite.
- [`docs/stylometry_ai_vs_human_detection.md`](file:///c:/Users/hp/OneDrive/Desktop/College/7th%20Sem/Minor%20Project/MINOR-CLA-PROJECT-main/docs/stylometry_ai_vs_human_detection.md) — Comprehensive technical report, mathematical formulations, and complete empirical benchmark comparisons.
- [`Fake_Review_Detector_Pro_Colab.ipynb`](file:///c:/Users/hp/OneDrive/Desktop/College/7th%20Sem/Minor%20Project/MINOR-CLA-PROJECT-main/Fake_Review_Detector_Pro_Colab.ipynb) — DistilBERT transformer training notebook for Google Colab (free T4 GPU).
- `distilbert_model/` — Local weights for offline deep transformer sequence classification.
- `svm_classifier.joblib`, `dt_classifier.joblib`, `nb_classifier.joblib` — Pre-trained classical and tree-based baseline classifiers.

---

## 4. Running the Streamlit Application

```bash
# Activate the virtual environment
.\venv\Scripts\activate

# Run the Streamlit web app
streamlit run app.py --server.port 8501
```

Access the live interface at `http://localhost:8501`.

---

## 5. Summary Benchmark Performance (Przystalski et al., 2026)

| Generator Pair / Benchmark | StyloMetrix (196 features) | Frequency N-Grams (3000 features) |
| :--- | :---: | :---: |
| **Wiki (Human) vs. GPT-4** | 94.0% | **98.0%** |
| **Wiki (Human) vs. GPT-3.5** | 97.0% | **99.0%** |
| **Wiki (Human) vs. LLaMa 2** | 99.0% | **100.0%** |
| **Wiki (Human) vs. LLaMa 3** | 95.0% | **99.0%** |
| **Wiki (Human) vs. Orca** | 99.0% | **100.0%** |
| **Wiki (Human) vs. Falcon** | 98.0% | **100.0%** |
| **Multiclass Attribution (MCC)** | 0.72 | **0.87** |
| **Paraphrased with DIPPER (11B)** | — | **>99.8% Recall** |
