
# CrisisLens
### Multilingual and Crosslingual Fact-Checked Claim Retrieval

CrisisLens is a multilingual fact-checked claim retrieval system designed to identify relevant fact-checks for social media posts across multiple languages. It leverages fine-tuned multilingual transformer models along with FAISS vector search to provide fast and accurate semantic retrieval for both monolingual and crosslingual scenarios.

---

##  Features

-  Multilingual & Crosslingual Retrieval
-  Fine-tuned MPNet-based Bi-Encoder Models
-  FAISS Vector Similarity Search
-  Evaluated on SemEval 2025 Task 7
-  Interactive Streamlit Web Interface

---

##  Results (SemEval 2025 Task 7)

| Model | Mono S@10 | Cross S@10 |
|------|-----------:|-----------:|
| BM25 Baseline | 0.65 | 0.45 |
| GTR-T5-Large | 0.76 | 0.58 |
| **CrisisLens** | **0.8382** | **0.6867** |

---

##  Hugging Face Models

- **Crosslingual Model:** https://huggingface.co/Sayyam-1/crislens-cross-mpnet
- **Monolingual Model:** https://huggingface.co/Sayyam-1/crislens-mono-mpnet

---

##  Dataset

**SemEval 2025 Task 7 � MultiClaim**

- Dataset was restricted and by the approval of the author i was given access at ZENODO
- 281,367 Fact-Checked Claims
- 39 Languages
- Multilingual & Crosslingual Retrieval Benchmark

---

##  Installation

Clone the repository:

```bash
git clone https://github.com/askSayyam/CrisisLens.git
```

Navigate to the project:

```bash
cd CrisisLens
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

---

##  Demo

A short demonstration of the application is available below.


📹 **Demo Video:** [Demo/CrisisLens-Demo.mp4](Demo/CrisisLens-Demo.mp4)
```


---

##  Tech Stack

- Python
- Streamlit
- Hugging Face Transformers
- Sentence Transformers
- FAISS
- PyTorch
- Pandas
- NumPy

---

##  Author

**Sayyam Khalid Satti**

AI & Machine Learning Developer

---

##  License

This project is intended for research and educational purposes.
