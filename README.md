# ResumeIQ — Resume Category Predictor

An ML-powered resume classification system that predicts job categories from any resume using TF-IDF vectorization and Scikit-learn — with a clean **ResumeIQ** dashboard built in Streamlit.

Paste resume text or upload a PDF and instantly get the top matching job category predictions with confidence scores.

---

## Features

- ML-based resume job category prediction
- 3 models trained and compared (LR · SVM · Naive Bayes)
- Top 3 predictions with confidence scores and progress bars
- PDF upload support with automatic text extraction
- Professional light-themed UI (White + Blue + Cyan)
- Works completely offline — no API or internet required
- Uses Kaggle Resume Dataset (2,484 resumes · 24 categories)
- Real-time prediction engine with saved model artifacts

---

## How It Works

### 1️⃣ Dataset

**Resume.csv** — 2,484 resumes across 24 job categories

| Column | Description |
|---|---|
| ID | Unique resume ID |
| Resume_str | Raw resume text content |
| Resume_html | HTML version of resume |
| Category | **Target** — Job category label |

**24 Job Categories:**
`HR` · `Designer` · `Information-Technology` · `Teacher` · `Advocate` · `Business-Development` · `Healthcare` · `Fitness` · `Agriculture` · `BPO` · `Sales` · `Consultant` · `Digital-Media` · `Automobile` · `Chef` · `Finance` · `Apparel` · `Engineering` · `Accountant` · `Construction` · `Public-Relations` · `Banking` · `Arts` · `Aviation`

---

### 2️⃣ Data Processing (Notebook)

- Loaded and inspected Resume.csv
- Checked for missing values and class distribution
- Text cleaning → removed URLs, emails, special characters
- Stopword removal + Lemmatization (NLTK)
- TF-IDF vectorization → `max_features=15000`, bigrams, sublinear_tf
- Train-test split → 80/20, stratified by category
- Label encoding for target categories

---

### 3️⃣ EDA Performed

- Resume count per category (bar chart)
- Resume word count distribution (histogram)
- Word clouds for top 6 categories
- Class imbalance ratio analysis
- Token count before vs after cleaning

---

### 4️⃣ ML Models

- **3 Models Trained** → Logistic Regression, Linear SVM, Naive Bayes
- **Evaluation** → Accuracy, F1-Score (Weighted), 5-Fold Cross Validation
- **Best Model** → Auto-selected via Cross Validation (typically LR or SVM)
- **Saved as** → `model/resume_classifier.pkl` + `model/label_encoder.pkl`

---

## Model Results

| Model | CV F1 Score | Stability |
|---|---|---|
| **Logistic Regression** | **~99%** | **High** |
| Linear SVM | ~99% | High |
| Naive Bayes | ~95% | Medium |

> Best model auto-selected via 5-Fold Stratified Cross Validation — highest weighted F1 score wins.

---

## Key Findings

- **Linear models (LR + SVM)** outperform Naive Bayes significantly on resume text
- **TF-IDF with bigrams** captures skill phrases better than unigrams alone
- **Sublinear TF** normalization improves performance on long resume documents
- `class_weight='balanced'` helps handle minor class imbalance
- **CalibratedClassifierCV** on SVM gives proper probability estimates

---

## Tech Stack

| Tool | Purpose |
|---|---|
| Python | Core language |
| Pandas & NumPy | Data manipulation |
| Matplotlib & Seaborn | EDA visualizations |
| NLTK | Text preprocessing |
| Scikit-learn | ML models, TF-IDF, evaluation |
| WordCloud | Category word clouds |
| pdfplumber | PDF text extraction |
| Joblib | Model serialization |
| Streamlit | Interactive web dashboard |

---

## Project Structure
```
resumeiq/
│
├── app.py                              ← Streamlit dashboard (ResumeIQ UI)
├── Resume_Classifier_EndToEnd.ipynb    ← Full ML pipeline notebook
├── Resume.csv                          ← Raw dataset
│
├── model/
│   ├── resume_classifier.pkl           ← Trained best model pipeline
│   └── label_encoder.pkl               ← Category LabelEncoder
│
├── data/                               ← (Optional) Category-wise resume folders
│   ├── HR/
│   ├── Data Science/
│   └── ...
│
├── requirements.txt
└── README.md
```

---

## Installation & Setup

### 1️⃣ Clone the repo
```bash
git clone https://github.com/your-username/resumeiq.git
cd resumeiq
```

### 2️⃣ Create virtual environment
```bash
python -m venv venv
```

### 3️⃣ Activate environment

**Windows**
```bash
venv\Scripts\activate
```

**Mac/Linux**
```bash
source venv/bin/activate
```

### 4️⃣ Install requirements
```bash
pip install -r requirements.txt
```

### 5️⃣ Run the notebook first (to generate model files)
```bash
jupyter notebook Resume_Classifier_EndToEnd.ipynb
```

### 6️⃣ Run the Streamlit app
```bash
streamlit run app.py
```

---

## requirements.txt
```
streamlit
pandas
numpy
scikit-learn
nltk
wordcloud
matplotlib
seaborn
pdfplumber
joblib
```

---

## Dataset

Available on Kaggle: https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset
---

## Screenshots

<!-- Add your screenshots here -->
![img alt](https://github.com/nikhil-kumarrr/images/blob/main/Screenshot%202026-03-16%20003707.png?raw=true)
![img alt](https://github.com/nikhil-kumarrr/images/blob/main/Screenshot%202026-03-16%20003757.png?raw=true)
