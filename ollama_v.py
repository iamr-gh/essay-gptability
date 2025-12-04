# sonnet 4.5
import pandas as pd
import requests
import json
from typing import List
import argparse
from pathlib import Path
from tqdm import tqdm
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading


def load_prompts(input_csv: str) -> pd.DataFrame:
    """Load prompts from CSV file."""
    df = pd.read_csv(input_csv)
    required_cols = ["system_prompt", "user_prompt"]

    if not all(col in df.columns for col in required_cols):
        raise ValueError(
            f"CSV must contain columns: {required_cols}. Found: {df.columns.tolist()}"
        )

    return df


def generate_response(
    system_prompt: str,
    user_prompt: str,
    model: str,
    temperature: float,
    max_tokens: int,
    top_p: float,
    base_url: str = "http://localhost:11434",
    index: int = None,
) -> tuple:
    """Generate a single response using Ollama API."""

    url = f"{base_url}/api/chat"

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
            "top_p": top_p,
        },
    }

    try:
        response = requests.post(url, json=payload, timeout=600)
        response.raise_for_status()
        result = response.json()
        return (index, result["message"]["content"])
    except requests.exceptions.RequestException as e:
        return (index, f"[ERROR: {str(e)}]")


def batch_generate_parallel(
    df: pd.DataFrame,
    model: str,
    num_workers: int,
    temperature: float,
    max_tokens: int,
    top_p: float,
    checkpoint_file: str = None,
    checkpoint_interval: int = 50,
) -> List[str]:
    """Generate responses in parallel with checkpointing."""

    responses = [None] * len(df)
    start_idx = 0

    # Load checkpoint if exists
    if checkpoint_file and Path(checkpoint_file).exists():
        checkpoint_df = pd.read_csv(checkpoint_file)
        if "response" in checkpoint_df.columns:
            for i, resp in enumerate(checkpoint_df["response"].tolist()):
                if pd.notna(resp) and resp != "":
                    responses[i] = resp
                    start_idx = i + 1
            print(f"Resuming from checkpoint at row {start_idx}")

    # Get remaining rows to process
    remaining_indices = [i for i in range(start_idx, len(df))]

    if not remaining_indices:
        print("All prompts already processed!")
        return responses

    print(
        f"\nProcessing {len(remaining_indices)} prompts with {num_workers} parallel workers..."
    )
    print(f"Model: {model}")
    print(f"Temperature: {temperature}, Max tokens: {max_tokens}, Top-p: {top_p}\n")

    start_time = time.time()
    completed_count = start_idx
    lock = threading.Lock()

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Submit all tasks
        futures = {}
        for idx in remaining_indices:
            row = df.iloc[idx]
            future = executor.submit(
                generate_response,
                system_prompt="",
                user_prompt=row["user_prompt"],
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                index=idx,
            )
            futures[future] = idx

        # Process completed tasks with progress bar
        with tqdm(total=len(remaining_indices), initial=0) as pbar:
            for future in as_completed(futures):
                idx, response = future.result()

                with lock:
                    responses[idx] = response
                    completed_count += 1

                    # Save checkpoint periodically
                    if checkpoint_file and completed_count % checkpoint_interval == 0:
                        temp_df = df.copy()
                        temp_df["response"] = responses
                        temp_df.to_csv(checkpoint_file, index=False)

                    # Update progress bar with stats
                    elapsed = time.time() - start_time
                    rate = completed_count / elapsed if elapsed > 0 else 0
                    remaining = len(df) - completed_count
                    eta_minutes = (remaining / rate / 60) if rate > 0 else 0

                    pbar.set_postfix(
                        {"rate": f"{rate:.2f} req/s", "ETA": f"{eta_minutes:.1f}m"}
                    )
                    pbar.update(1)

    elapsed_total = time.time() - start_time
    print(f"\nCompleted in {elapsed_total / 60:.1f} minutes")
    print(f"Average rate: {len(remaining_indices) / elapsed_total:.2f} requests/second")

    return responses


def save_results(df: pd.DataFrame, responses: List[str], output_csv: str) -> None:
    """Save results to CSV with the same structure as input."""
    df["response"] = responses
    df.to_csv(output_csv, index=False)
    print(f"Results saved to: {output_csv}")


def test_ollama_connection(base_url: str, model: str) -> bool:
    """Test connection to Ollama and check if model is available."""
    try:
        # Check if Ollama is running
        response = requests.get(f"{base_url}/api/tags", timeout=5)
        response.raise_for_status()

        # Check if model is available
        models = response.json().get("models", [])
        model_names = [m["name"] for m in models]

        if not any(model in name for name in model_names):
            print(f"⚠ Model '{model}' not found. Available models:")
            for name in model_names:
                print(f"  - {name}")
            print(f"\nPull the model with: ollama pull {model}")
            return False

        print("✓ Connected to Ollama")
        print(f"✓ Model '{model}' is available")
        return True

    except requests.exceptions.RequestException as e:
        print(f"✗ Cannot connect to Ollama at {base_url}")
        print(f"  Make sure Ollama is running: ollama serve")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Batch process prompts using Ollama with parallel workers"
    )
    parser.add_argument(
        "input_csv",
        type=str,
        help="Path to input CSV with system_prompt and user_prompt columns",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Path to output CSV (default: input_csv with _output suffix)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="llama3.2",
        help="Ollama model name (default: llama3.2)",
    )
    parser.add_argument(
        "--workers",
        "-w",
        type=int,
        default=4,
        help="Number of parallel workers (default: 4)",
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=50,
        help="Save checkpoint every N completions (default: 50)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (default: 0.7)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Maximum tokens to generate (default: 512)",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Top-p sampling parameter (default: 0.9)",
    )
    parser.add_argument(
        "--checkpoint",
        action="store_true",
        help="Enable checkpointing to resume interrupted runs",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default="http://localhost:11434",
        help="Ollama API base URL (default: http://localhost:11434)",
    )

    args = parser.parse_args()

    # Set output path
    if args.output is None:
        input_path = Path(args.input_csv)
        args.output = str(
            input_path.parent / f"{input_path.stem}_output{input_path.suffix}"
        )

    # Set checkpoint file
    checkpoint_file = None
    if args.checkpoint:
        input_path = Path(args.input_csv)
        checkpoint_file = str(
            input_path.parent / f"{input_path.stem}_checkpoint{input_path.suffix}"
        )

    # Test Ollama connection
    if not test_ollama_connection(args.base_url, args.model):
        return

    # Warn about worker count
    if args.workers > 8:
        print(f"\n⚠ Warning: {args.workers} workers may be too many for M3 MacBook Air")
        print("  Recommended: 2-6 workers depending on model size")
        response = input("Continue? (y/n): ")
        if response.lower() != "y":
            return

    # Load prompts
    print("\nLoading prompts...")
    df = load_prompts(args.input_csv)
    print(f"Loaded {len(df)} prompts")

    # Generate responses
    responses = batch_generate_parallel(
        df,
        model=args.model,
        num_workers=args.workers,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        top_p=args.top_p,
        checkpoint_file=checkpoint_file,
        checkpoint_interval=args.checkpoint_interval,
    )

    # Save results
    save_results(df, responses, args.output)

    # Clean up checkpoint file
    if checkpoint_file and Path(checkpoint_file).exists():
        Path(checkpoint_file).unlink()
        print("Checkpoint file removed")


if __name__ == "__main__":
    main()
