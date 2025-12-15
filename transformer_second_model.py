# train_regressor.py
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sentence_transformers import SentenceTransformer
from peft import LoraConfig, get_peft_model, TaskType
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tqdm import tqdm
import argparse
import json
import os
import numpy as np


# --------------------
# Data Processing
# --------------------
class EssayDataset(Dataset):
    def __init__(self, essays, labels, tokenizer, max_length=256):
        self.essays = essays.tolist() if hasattr(essays, "tolist") else essays
        self.labels = labels.tolist() if hasattr(labels, "tolist") else labels
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
            "error": torch.tensor(self.labels[idx], dtype=torch.float),
        }


class LLMRegressor(nn.Module):
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        super().__init__()
        from transformers import AutoModel, AutoTokenizer

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


# --------------------
# Training Utilities
# --------------------
def train_epoch(model, dataloader, optimizer, criterion, device, scaler):
    model.train()
    total_loss = 0

    for batch in tqdm(dataloader, desc="Training"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["error"].to(device)

        optimizer.zero_grad()

        # Mixed precision training
        with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
            predictions = model(input_ids, attention_mask)
            loss = criterion(predictions, labels)

        if scaler:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["error"].to(device)

            predictions = model(input_ids, attention_mask)
            loss = criterion(predictions, labels)

            total_loss += loss.item()

            # Clamp predictions to [0, 2] range for evaluation
            preds_clamped = torch.clamp(predictions, 0, 2).cpu()

            all_preds.extend(preds_clamped.numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calculate regression metrics
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    mse = mean_squared_error(all_labels, all_preds)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(all_labels, all_preds)
    r2 = r2_score(all_labels, all_preds)

    return total_loss / len(dataloader), mse, rmse, mae, r2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help='Path to CSV with "user_prompt" and "error" columns (error in range 0-2)',
    )
    parser.add_argument(
        "--model_name", type=str, default="sentence-transformers/all-MiniLM-L12-v2"
    )
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--use_lora", action="store_true", default=False)
    parser.add_argument(
        "--use_hack",
        action="store_true",
        default=False,
        help="Remove stopwords to preserve info at shorter lengths",
    )
    parser.add_argument("--output_dir", type=str, default="./model_output")
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

    # --------------------
    # Load Data
    # --------------------
    df = pd.read_csv(args.data_path)
    print(f"Loaded {len(df)} samples")

    # Check for required columns
    if "user_prompt" not in df.columns or "error" not in df.columns:
        raise ValueError(
            "CSV must contain 'user_prompt' and 'error' columns. "
            f"Found columns: {df.columns.tolist()}"
        )

    print(f"Label range: [{df['error'].min():.2f}, {df['error'].max():.2f}]")
    print(f"Label mean: {df['error'].mean():.2f}, std: {df['error'].std():.2f}")

    # Validate label range
    if df["error"].min() < 0 or df["error"].max() > 2:
        print("WARNING: Some labels are outside the expected [0, 2] range!")

    # Simple hack: Remove stopwords to fit more content in 256 tokens
    if args.use_hack:
        import nltk

        try:
            nltk.data.find("corpora/stopwords")
        except LookupError:
            nltk.download("stopwords")
        from nltk.corpus import stopwords

        stop_words = set(stopwords.words("english"))

        def clean_text(text):
            return " ".join(
                [word for word in str(text).split() if word.lower() not in stop_words]
            )

        df["user_prompt"] = df["user_prompt"].apply(clean_text)
        print("Applied stopword removal hack")

    # Split data
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

    # --------------------
    # Create Model
    # --------------------
    model = LLMRegressor(args.model_name)

    # Apply LoRA for parameter-efficient fine-tuning
    if args.use_lora:
        lora_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=16,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["dense", "dense_h_to_4h", "dense_4h_to_h"],
        )
        model.encoder = get_peft_model(model.encoder, lora_config)
        print("Applied LoRA")
        model.encoder.print_trainable_parameters()

    model.to(device)

    # --------------------
    # Create DataLoaders
    # --------------------
    tokenizer = model.tokenizer

    train_dataset = EssayDataset(
        train_df["user_prompt"], train_df["error"], tokenizer, args.max_length
    )
    val_dataset = EssayDataset(
        val_df["user_prompt"], val_df["error"], tokenizer, args.max_length
    )

    # Adjust batch size for MacBook Air M3
    if device.type == "mps":
        args.batch_size = min(args.batch_size, 8)
        print(f"Adjusted batch size to {args.batch_size} for MPS device")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

    # --------------------
    # Training Setup
    # --------------------
    # Use MSE loss for regression
    criterion = nn.MSELoss()

    # Only optimize the regressor head (and LoRA if enabled)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=0.01,
    )

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )

    # Mixed precision scaler
    scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None

    # Early stopping
    best_rmse = float("inf")
    patience = 3
    patience_counter = 0

    # --------------------
    # Training Loop
    # --------------------
    print("Starting training...")
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")

        train_loss = train_epoch(
            model, train_loader, optimizer, criterion, device, scaler
        )
        val_loss, val_mse, val_rmse, val_mae, val_r2 = validate(
            model, val_loader, criterion, device
        )

        print(f"Train Loss: {train_loss:.4f}")
        print(
            f"Val Loss: {val_loss:.4f} | MSE: {val_mse:.4f} | "
            f"RMSE: {val_rmse:.4f} | MAE: {val_mae:.4f} | R²: {val_r2:.4f}"
        )

        # Update learning rate
        scheduler.step(val_rmse)

        # Save best model (based on RMSE)
        if val_rmse < best_rmse:
            best_rmse = val_rmse
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "rmse": val_rmse,
                    "mae": val_mae,
                    "r2": val_r2,
                    "config": args.__dict__,
                },
                f"{args.output_dir}/best_model.pt",
            )
            print(f"✓ Saved new best model (RMSE: {val_rmse:.4f})")
            patience_counter = 0
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= patience:
            print(f"Early stopping triggered after {epoch + 1} epochs")
            break

    print(f"\nTraining complete! Best RMSE: {best_rmse:.4f}")
    print(f"Model saved to {args.output_dir}/best_model.pt")


if __name__ == "__main__":
    main()
