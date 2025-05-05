from flask import Flask, render_template, request, send_file
import pandas as pd
import torch
import joblib
import os
from io import BytesIO
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.models import MLP

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

app = Flask(__name__)

# Load model and preprocessing artifacts
scaler = joblib.load("results/mlp_scaler.pkl")
feature_columns = joblib.load("data/mlp_feature_columns.pkl")
model = MLP(len(feature_columns))
model.load_state_dict(torch.load("results/mlp_model.pth", map_location=device))
model.eval()

# Preprocessing function
def preprocess(df):
    categorical_cols = ["protocol_type", "service", "flag"]
    df = pd.get_dummies(df, columns=categorical_cols)
    df = df.reindex(columns=feature_columns, fill_value=0)
    return scaler.transform(df)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files["file"]
        if not file:
            return "No file uploaded", 400
    
        df = pd.read_csv(file, header=None)
        df.columns = [
            "duration","protocol_type","service","flag","src_bytes","dst_bytes","land","wrong_fragment","urgent",
            "hot","num_failed_logins","logged_in","num_compromised","root_shell","su_attempted","num_root",
            "num_file_creations","num_shells","num_access_files","num_outbound_cmds","is_host_login",
            "is_guest_login","count","srv_count","serror_rate","srv_serror_rate","rerror_rate","srv_rerror_rate",
            "same_srv_rate","diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count",
            "dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_src_port_rate",
            "dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate",
            "dst_host_rerror_rate","dst_host_srv_rerror_rate", "label", "difficulty"
        ]
        
        df_original = df.copy()
        df = df.drop(columns=["label", "difficulty"])
        x_scaled = preprocess(df)
        
        x_tensor = torch.tensor(x_scaled, dtype=torch.float32).to(device)
        
        with torch.no_grad():
            outputs = model(x_tensor).squeeze()
            probs = torch.sigmoid(outputs).cpu().numpy()
            preds = (probs > 0.01).astype(int)
        
        df_original["prediction"] = preds
        df_original["confidence"] = probs
        df_original["prediction_label"] = df_original["prediction"].map({0: "normal", 1: "attack"})
    
        # Save to a BytesIO object
        output = BytesIO()
        df_original.to_csv(output, index=False)
        output.seek(0)
        
        return send_file(output, as_attachment=True, download_name="predictions.csv", mimetype="text/csv")
    
    return render_template("index.html")
