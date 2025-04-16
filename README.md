# 🔐 Intrusion Detection System using Machine Learning

This project applies machine learning algorithms to detect malicious network traffic using the NSL-KDD dataset. The goal is to distinguish between normal and attack traffic effectively, providing the foundation for a real-world intrusion detection system (IDS).

---

## 📦 Setup Instructions

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

## Project Structure
cybersecurity_anomaly_detection/
│
├── data/                # Place datasets here (must be downloaded manually)
│   ├── KDDTrain+.txt
│   └── KDDTest+.txt
│
├── notebooks/           # Jupyter notebooks for EDA, training, and evaluation
├── results/             # Outputs from model training and evaluation (currently), expandable to store runtime detection results later
├── src/                 # Python scripts (if any used for modular code)
├── requirements.txt     # Python dependencies
├── .gitignore
└── README.md

## Contributions
**Feel free to fork this repo or open an issue if you have suggestions for improvement.**

## Author
**Ayotunde Ogunnaiya**
