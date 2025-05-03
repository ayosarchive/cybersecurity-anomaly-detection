import argparse
import pandas as pd
import joblib
import torch
from models import MLP  # assuming models.py contains the MLP class

# Preprocess input
def preprocess_input(df, feature_columns):
    categorical_cols = ["protocol_type", "service", "flag"]
    df_encoded = pd.get_dummies(df, columns=categorical_cols)
    df_encoded = df_encoded.reindex(columns=feature_columns, fill_value=0)
    return df_encoded

def main():
    parser = argparse.ArgumentParser(description="Predict using the trained MLP model")
    parser.add_argument('--input', type=str, required=True, help="Input CSV path")
    parser.add_argument('--output', type=str, required=True, help="Output CSV path")
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'], help="Device to run the model on (cpu or cuda)")
    args = parser.parse_args()

    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # Load feature columns
    feature_columns = joblib.load("data/mlp_feature_columns.pkl")
    input_dim = len(feature_columns)

    # Load model
    model = MLP(input_dim)
    model.load_state_dict(torch.load("results/mlp_model.pth", map_location=device))
    model = model.to(device)
    model.eval()

    # Load scaler
    scaler = joblib.load("results/mlp_scaler.pkl")

    # Load and prepare data
    df = pd.read_csv(args.input, header=None)

    # Assign correct column names
    column_names = [
        "duration","protocol_type","service","flag","src_bytes","dst_bytes","land","wrong_fragment","urgent",
        "hot","num_failed_logins","logged_in","num_compromised","root_shell","su_attempted","num_root",
        "num_file_creations","num_shells","num_access_files","num_outbound_cmds","is_host_login",
        "is_guest_login","count","srv_count","serror_rate","srv_serror_rate","rerror_rate","srv_rerror_rate",
        "same_srv_rate","diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count",
        "dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_src_port_rate",
        "dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate",
        "dst_host_rerror_rate","dst_host_srv_rerror_rate","label", "difficulty"
    ]
    df.columns = column_names
    df = df.drop(columns=["label", "difficulty"])
    
    # Preprocess input data
    x_input = preprocess_input(df, feature_columns)

    # Ensure the input data has the same columns as the model was trained on
    if x_input.shape[1] != len(feature_columns):
        raise ValueError(f"Input data must have {len(feature_columns)} features after encoding. Found {x_input.shape[1]} features.")
    
    x_scaled = scaler.transform(x_input)
    x_tensor = torch.tensor(x_scaled, dtype=torch.float32).to(device)

    # Predict
    with torch.no_grad():
        outputs = model(x_tensor).squeeze()
        probs = torch.sigmoid(outputs).cpu().numpy()
        predictions = (probs >= 0.1).astype(int)

    # Save predictions
    df["prediction"] = predictions
    df["confidence"] = probs
    df["prediction_label"] = df["prediction"].map({0: "normal", 1: "attack"})
    df.to_csv(args.output, index=False)
    print(f"Predictions saved to {args.output}")

if __name__ == "__main__":
    main()
