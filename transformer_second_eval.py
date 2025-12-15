# evaluate_regressor.py
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer
from peft import PeftModel, LoraConfig, get_peft_model, TaskType
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from tqdm import tqdm
import argparse
import os


# --------------------
# Data Processing
# --------------------
class EssayDataset(Dataset):
    def __init__(self, essays, tokenizer, max_length=256):
        self.essays = essays.tolist() if hasattr(essays, "tolist") else essays
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.essays)

    def __getitem__(self, idx):
        text = self.essays[idx]
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "index": idx,
        }


# --------------------
# Model Architecture (must match training)
# --------------------
class LLMRegressor(nn.Module):
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        super().__init__()
        # Load the base transformer directly
        self.encoder = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Freeze encoder
        for param in self.encoder.parameters():
            param.requires_grad = False

        # Regression head
        self.regressor = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(384, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def mean_pooling(self, token_embeddings, attention_mask):
        """Mean pooling - take attention mask into account for correct averaging"""
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    def forward(self, input_ids, attention_mask):
        # Get embeddings from the encoder
        with torch.no_grad():
            encoder_output = self.encoder(
                input_ids=input_ids, attention_mask=attention_mask
            )
            token_embeddings = encoder_output.last_hidden_state

        # Apply mean pooling
        embeddings = self.mean_pooling(token_embeddings, attention_mask)

        # Regression head
        output = self.regressor(embeddings).squeeze(-1)
        return output


def load_model(checkpoint_path, device):
    """Load model from checkpoint, handling LoRA if present"""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Get model config
    config = checkpoint.get("config", {})
    model_name = config.get("model_name", "sentence-transformers/all-MiniLM-L6-v2")
    use_lora = config.get("use_lora", False)

    print(f"Model config: {model_name}")
    print(f"LoRA enabled during training: {use_lora}")

    # Initialize model
    model = LLMRegressor(model_name)

    # Check if checkpoint has LoRA weights
    state_dict = checkpoint["model_state_dict"]
    has_lora = any("lora" in key for key in state_dict.keys())

    if has_lora:
        print("Detected LoRA weights in checkpoint, applying LoRA to model...")
        # Apply LoRA to match the saved state
        lora_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=16,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["dense"],
        )
        model.encoder = get_peft_model(model.encoder, lora_config)

    # Load weights
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    print(f"Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
    rmse = checkpoint.get("rmse", "N/A")
    mae = checkpoint.get("mae", "N/A")
    r2 = checkpoint.get("r2", "N/A")

    if rmse != "N/A":
        print(f"Model RMSE: {rmse:.4f}")
    if mae != "N/A":
        print(f"Model MAE: {mae:.4f}")
    if r2 != "N/A":
        print(f"Model R²: {r2:.4f}")

    return model, config


def predict_all(model, dataloader, device, max_samples=None):
    """Generate predictions for all samples"""
    model.eval()
    all_preds = []
    all_indices = []

    with torch.no_grad():
        sample_count = 0
        for batch in tqdm(dataloader, desc="Generating predictions"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            indices = batch["index"]

            predictions = model(input_ids, attention_mask)

            # Clamp predictions to [0, 2] range
            preds_clamped = torch.clamp(predictions, 0, 2).cpu()

            all_preds.extend(preds_clamped.numpy())
            all_indices.extend(indices.numpy())

            sample_count += len(preds_clamped)

            # Check AFTER adding samples
            if max_samples is not None and sample_count >= max_samples:
                print(f"Reached max_samples limit: {sample_count}")
                break

    return np.array(all_preds), np.array(all_indices)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help='Path to CSV with "user_prompt" column (and optionally "error" for MSE)',
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to saved model checkpoint (.pt file)",
    )
    # add a limit on the number of samples to process
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--output_dir", type=str, default="./evaluation_output")
    parser.add_argument(
        "--essay_column",
        type=str,
        default="user_prompt",
        help="Name of essay/text column in CSV",
    )
    parser.add_argument(
        "--label_column",
        type=str,
        default="error",
        help="Name of label column in CSV (optional)",
    )
    args = parser.parse_args()

    # --------------------
    # Setup
    # --------------------
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using device: {device}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Set plotting parameters for poster quality
    plt.rcParams.update(
        {
            "font.size": 24,
            "axes.labelsize": 28,
            "axes.titlesize": 32,
            "xtick.labelsize": 22,
            "ytick.labelsize": 22,
            "legend.fontsize": 22,
            "figure.figsize": (12, 8),
            "lines.linewidth": 3,
            "axes.linewidth": 2,
            "xtick.major.width": 2,
            "ytick.major.width": 2,
            "xtick.major.size": 8,
            "ytick.major.size": 8,
        }
    )
    sns.set_style("dark")
    sns.set_context("poster")

    # --------------------
    # Load Model
    # --------------------
    model, config = load_model(args.model_path, device)
    max_length = config.get("max_length", 256)

    # --------------------
    # Load Data
    # --------------------
    df = pd.read_csv(args.data_path)
    print(f"\nLoaded {len(df)} samples")

    # Check if label column exists
    has_labels = args.label_column in df.columns

    if has_labels:
        print(f"Found label column '{args.label_column}'")

    # --------------------
    # Create DataLoader
    # --------------------
    dataset = EssayDataset(df[args.essay_column], model.tokenizer, max_length)

    if device.type == "mps":
        args.batch_size = min(args.batch_size, 8)
        print(f"Adjusted batch size to {args.batch_size} for MPS")

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    # --------------------
    # Generate Predictions
    # --------------------
    predictions, indices = predict_all(
        model, dataloader, device, max_samples=args.max_samples
    )

    # IMPORTANT: Trim dataframe to match actual predictions
    if args.max_samples is not None and len(predictions) < len(df):
        df = df.iloc[: len(predictions)].copy()
        print(f"Trimmed dataframe to {len(df)} samples to match predictions")

    # Add predictions to dataframe
    df_results = df.copy()
    df_results["prediction"] = predictions

    # Calculate MSE if labels available
    if has_labels:
        labels = df[args.label_column].values[: len(predictions)]  # Also trim labels
        mse_model = np.mean((labels - predictions) ** 2)
        rmse_model = np.sqrt(mse_model)
        mae_model = np.mean(np.abs(labels - predictions))

        # Baseline: predict mean
        mean_label = np.mean(labels)
        mse_baseline = np.mean((labels - mean_label) ** 2)
        rmse_baseline = np.sqrt(mse_baseline)

        print(f"\nModel Performance (on {len(predictions)} samples):")
        print(f"MSE: {mse_model:.4f}")
        print(f"RMSE: {rmse_model:.4f}")
        print(f"MAE: {mae_model:.4f}")
        print(f"\nBaseline (predict mean={mean_label:.4f}):")
        print(f"MSE: {mse_baseline:.4f}")
        print(f"RMSE: {rmse_baseline:.4f}")

        # Add diagnostic info
        print(f"\nPrediction Statistics:")
        print(f"Mean: {np.mean(predictions):.4f}")
        print(f"Std: {np.std(predictions):.4f}")
        print(f"Min: {np.min(predictions):.4f}")
        print(f"Max: {np.max(predictions):.4f}")

        print(f"\nLabel Statistics:")
        print(f"Mean: {np.mean(labels):.4f}")
        print(f"Std: {np.std(labels):.4f}")
        print(f"Min: {np.min(labels):.4f}")
        print(f"Max: {np.max(labels):.4f}")

    # --------------------
    # Print Top 10 Highest and Lowest Predictions
    # --------------------
    df_sorted = df_results.sort_values("prediction", ascending=False)

    print("\n" + "=" * 80)
    print("TOP 10 HIGHEST PREDICTIONS:")
    print("=" * 80)
    for i, (idx, row) in enumerate(df_sorted.head(10).iterrows(), 1):
        print(f"\n[{i}] Prediction: {row['prediction']:.4f}")
        if has_labels:
            print(f"    Actual: {row[args.label_column]:.4f}")
        print(f"    Essay preview: {str(row[args.essay_column])[:200]}...")
        print("-" * 80)

    print("\n" + "=" * 80)
    print("TOP 10 LOWEST PREDICTIONS:")
    print("=" * 80)
    for i, (idx, row) in enumerate(df_sorted.tail(10).iterrows(), 1):
        print(f"\n[{i}] Prediction: {row['prediction']:.4f}")
        if has_labels:
            print(f"    Actual: {row[args.label_column]:.4f}")
        print(f"    Essay preview: {str(row[args.essay_column])[:200]}...")
        print("-" * 80)

    # --------------------
    # Plot Histogram
    # --------------------
    fig, ax = plt.subplots(figsize=(14, 9))
    ax.hist(
        predictions,
        bins=50,
        color="#E63946",
        edgecolor="black",
        linewidth=1.2,
        alpha=0.85,
    )
    ax.set_xlabel("Predicted Error Score", fontweight="bold")
    ax.set_ylabel("Frequency", fontweight="bold")
    ax.set_title("Distribution of Model Predictions", fontweight="bold", pad=20)
    ax.grid(True, alpha=0.3, axis="y", linewidth=1.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Add statistics text box
    if has_labels:
        textstr = (
            f"Model MSE: {mse_model:.4f}\n"
            f"Model RMSE: {rmse_model:.4f}\n"
            f"Model MAE: {mae_model:.4f}\n"
            f"Baseline MSE: {mse_baseline:.4f}"
        )
    else:
        textstr = (
            f"Mean: {np.mean(predictions):.4f}\n"
            f"Std: {np.std(predictions):.4f}\n"
            f"Min: {np.min(predictions):.4f}\n"
            f"Max: {np.max(predictions):.4f}"
        )

    props = dict(boxstyle="round", facecolor="wheat", alpha=0.8)
    ax.text(
        0.65,
        0.95,
        textstr,
        transform=ax.transAxes,
        fontsize=20,
        verticalalignment="top",
        bbox=props,
    )

    plt.tight_layout()
    output_filename = os.path.join(
        args.output_dir,
        f"prediction_histogram_{os.path.basename(args.data_path).split('.')[0]}.png",
    )
    plt.savefig(output_filename, dpi=300, bbox_inches="tight")
    print(f"\nHistogram saved to: {output_filename}")
    plt.close()

    # --------------------
    # Save Results
    # --------------------
    results_csv = os.path.join(
        args.output_dir, f"predictions_{os.path.basename(args.data_path)}"
    )
    df_results.to_csv(results_csv, index=False)
    print(f"Predictions saved to: {results_csv}")


if __name__ == "__main__":
    main()
