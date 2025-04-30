# Intrusion Detection System using Machine Learning

This project applies a variety of machine learning algorithms—including Random Forest, XGBoost, a Multi-Layer Perceptron (MLP), and a combined ensemble model—to detect malicious network traffic using the NSL-KDD dataset. The objective is to effectively distinguish between normal and attack traffic, forming the foundation for a real-world intrusion detection system (IDS).

---


## Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/cybersecurity_anomaly_detection.git
cd cybersecurity_anomaly_detection
```

### 2. Create and Activate Virtual Environment
#### On Windows
```bash
python -m venv .venv
.venv\Scripts\activate
```
#### On Mac/Linux
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Requirements
```bash 
pip install -r requirements.txt
```

---

## Project Structure

```
cybersecurity_anomaly_detection/
│
├── data/                # Place datasets here (must be downloaded manually)
│   ├── KDDTrain+.txt
│   └── KDDTest+.txt
│
├── notebooks/           # Jupyter notebooks for EDA, training, evaluation
├── results/             # Outputs from training and evaluation
├── src/                 # Python scripts for CLI tools or modular code
├── requirements.txt     # Python dependencies
├── .gitignore
└── README.md
```

---

## Model Performance Summary

The following table summarizes performance across all trained models, evaluated on the NSL-KDD test set.

| Model                      | Accuracy | Precision (Macro) | Recall (Macro) |
|---------------------------|----------|--------------------|----------------|
| Logistic Regression       | 54.4%    | 28.0%              | 48.0%          |
| Random Forest (Tuned)     | 78.3%    | 82.0%              | 81.0%          |
| XGBoost (Tuned)           | 78.8%    | 82.0%              | 81.0%          |
| MLP (Tuned)               | 80.0%    | 83.0%              | 82.0%          |
| Ensemble (RF + XGB + MLP) | 80.0%    | 83.0%              | 82.0%          |

> Detailed logs, loss curves, and F1-Score comparisons are available in the project notebooks.

---

## Contributions

Feel free to fork this repository or open an issue if you have feedback or ideas for improvement.

---

## Author

**Ayotunde Ogunnaiya**