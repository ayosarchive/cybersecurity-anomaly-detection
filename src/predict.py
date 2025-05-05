import argparse
import pandas as pd
import joblib
import torch
import xgboost as xgb
from models import MLP

# Note: Ensemble mode included for experimentation only. Not used in final deployment due to calibration mismatch.

# Preprocessing 
def preprocess_input(df, feature_columns):
    categorical_cols = ["protocol_type", "service", "flag"]
    df_encoded = pd.get_dummies(df, columns=categorical_cols)
    df_encoded = df_encoded.reindex(columns=feature_columns, fill_value=0)
    return df_encoded

def predict_mlp(x_scaled, device, input_dim):
    model = MLP(input_dim)
    model.load_state_dict(torch.load("results/mlp_model.pth"))
    model.to(device)
    model.eval()

    x_tensor = torch.tensor(x_scaled, dtype=torch.float32).to(device)
    with torch.no_grad():
        outputs = model(x_tensor).squeeze()
        probs = torch.sigmoid(outputs).cpu().numpy()
    return probs

# def predict_ensemble(x_unscaled, x_scaled, device, feature_columns):
#     # Load models
#     rf_model = joblib.load("results/rf_model.pkl")
#     xgb_model = xgb.Booster()
#     xgb_model.load_model("results/xgb_model.json")

#     mlp_model = MLP(x_scaled.shape[1])
#     mlp_model.load_state_dict(torch.load("results/mlp_model.pth"))
#     mlp_model.to(device)
#     mlp_model.eval()

#     x_tensor = torch.tensor(x_scaled, dtype=torch.float32).to(device)

#     # Predict
#     rf_probs = rf_model.predict_proba(x_unscaled)[:, 1]
#     xgb_dmatrix = xgb.DMatrix(x_unscaled, feature_names=feature_columns)
#     xgb_probs = xgb_model.predict(xgb_dmatrix)

#     with torch.no_grad():
#         mlp_outputs = mlp_model(x_tensor).squeeze()
#         mlp_probs = torch.sigmoid(mlp_outputs).cpu().numpy()

#     # Average
#     ensemble_probs = (0.05 * rf_probs + 0.05 * xgb_probs + 0.9 * mlp_probs) 
#     return ensemble_probs

def main():
    parser = argparse.ArgumentParser(description="Predict using MLP or Ensemble model")
    parser.add_argument('--input', type=str, required=True, help="Input CSV path")
    parser.add_argument('--output', type=str, required=True, help="Output CSV path")
    # parser.add_argument('--model', type=str, choices=['mlp', 'ensemble'], required=True, help="Model to use")
    parser.add_argument('--device', type=str, default='cpu', help="Device to use (cpu or cuda)")
    args = parser.parse_args()

    # Fallback to CPU if CUDA not available
    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    THRESHOLD = 0.01

    # Load input CSV
    df = pd.read_csv(args.input, header=None)
    df.columns = [
        "duration","protocol_type","service","flag","src_bytes","dst_bytes","land","wrong_fragment","urgent",
        "hot","num_failed_logins","logged_in","num_compromised","root_shell","su_attempted","num_root",
        "num_file_creations","num_shells","num_access_files","num_outbound_cmds","is_host_login",
        "is_guest_login","count","srv_count","serror_rate","srv_serror_rate","rerror_rate","srv_rerror_rate",
        "same_srv_rate","diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count",
        "dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_src_port_rate",
        "dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate",
        "dst_host_rerror_rate","dst_host_srv_rerror_rate","label", "difficulty"
    ]
    df = df.drop(columns=["label", "difficulty"])

    # Load preprocessing artifacts
    feature_columns = joblib.load("data/mlp_feature_columns.pkl")
    scaler = joblib.load("results/mlp_scaler.pkl")

    # Preprocess
    x_encoded = preprocess_input(df, feature_columns)
    x_scaled = scaler.transform(x_encoded)
    x_unscaled = x_encoded

    # Predict
    # if args.model == "mlp":
    #     probs = predict_mlp(x_scaled, device, input_dim=len(feature_columns))
    # elif args.model == "ensemble":
    #     probs = predict_ensemble(x_unscaled, x_scaled, device, feature_columns)

    probs = predict_mlp(x_scaled, device, input_dim=len(feature_columns))
    
    # Final prediction
    predictions = (probs >= THRESHOLD).astype(int)

    # Save output
    df["prediction"] = predictions
    df["confidence"] = probs
    df["prediction_label"] = df["prediction"].map({0: "normal", 1: "attack"})
    df.to_csv(args.output, index=False)
    print(f"Predictions saved to {args.output}")

if __name__ == "__main__":
    main()
