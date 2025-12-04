# sonnet 4.5
import pandas as pd
from vllm import LLM, SamplingParams
from typing import List
import argparse
from pathlib import Path
from tqdm import tqdm
import os

# Enable memory optimization
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


def load_prompts(input_csv: str) -> pd.DataFrame:
    """Load prompts from CSV file."""
    df = pd.read_csv(input_csv)
    required_cols = ["system_prompt", "user_prompt"]

    if not all(col in df.columns for col in required_cols):
        raise ValueError(
            f"CSV must contain columns: {required_cols}. Found: {df.columns.tolist()}"
        )

    return df


def load_checkpoint(checkpoint_path: str) -> pd.DataFrame:
    """Load existing checkpoint if available."""
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from: {checkpoint_path}")
        return pd.read_csv(checkpoint_path)
    return None


def format_prompts(df: pd.DataFrame) -> List[str]:
    """Format system and user prompts into single strings."""
    prompts = []
    for _, row in df.iterrows():
        formatted = (
            f"{row['system_prompt']}\n\nUser: {row['user_prompt']}\n\nAssistant:"
        )
        prompts.append(formatted)

    return prompts


def batch_generate(
    prompts: List[str],
    model_name: str = "Qwen/Qwen2.5-7B-Instruct",
    batch_size: int = 16,
    temperature: float = 0.7,
    max_tokens: int = 512,
    top_p: float = 0.9,
    start_idx: int = 0,
    checkpoint_path: str = None,
    df: pd.DataFrame = None,
) -> List[str]:
    """Generate responses using vLLM in batches with checkpointing."""

    # Initialize vLLM with minimal memory settings for 7B model
    print(f"Loading model: {model_name}")
    llm = LLM(
        model=model_name,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.88,  # Aggressive but safe
        max_model_len=1024,  # Reduced context window
        trust_remote_code=True,
        dtype="float16",
        max_num_batched_tokens=512,  # Very conservative
        block_size=8,  # Smaller block size for more flexibility
        swap_space=2,  # Enable small CPU swap
    )

    # Set sampling parameters
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )

    # Process in batches with progress bar and checkpointing
    print(
        f"Generating responses for {len(prompts) - start_idx} prompts (starting from {start_idx})..."
    )
    print(f"Batch size: {batch_size}")
    all_responses = [""] * len(prompts)  # Initialize with empty strings

    # Track progress
    checkpoint_frequency = max(1, batch_size * 10)  # Checkpoint every 10 batches

    for i in tqdm(range(start_idx, len(prompts), batch_size)):
        batch = prompts[i : i + batch_size]
        batch_indices = list(range(i, min(i + batch_size, len(prompts))))

        try:
            outputs = llm.generate(batch, sampling_params)
            responses = [output.outputs[0].text for output in outputs]

            # Store responses
            for idx, response in zip(batch_indices, responses):
                all_responses[idx] = response

            # Checkpoint periodically
            if checkpoint_path and (i - start_idx) % checkpoint_frequency == 0:
                save_checkpoint(df, all_responses, checkpoint_path, i + batch_size)

        except Exception as e:
            print(f"\nError processing batch starting at index {i}: {e}")
            # Save checkpoint on error
            if checkpoint_path:
                save_checkpoint(df, all_responses, checkpoint_path, i)
            raise

    # Final checkpoint
    if checkpoint_path:
        save_checkpoint(df, all_responses, checkpoint_path, len(prompts))

    return all_responses


def save_checkpoint(
    df: pd.DataFrame, responses: List[str], checkpoint_path: str, current_idx: int
):
    """Save checkpoint with current progress."""
    df_checkpoint = df.copy()
    df_checkpoint["response"] = responses
    df_checkpoint["processed"] = [bool(r) for r in responses]
    df_checkpoint.to_csv(checkpoint_path, index=False)
    print(f"\nCheckpoint saved: {current_idx} prompts processed")


def save_results(df: pd.DataFrame, responses: List[str], output_csv: str) -> None:
    """Save results to CSV with the same structure as input."""
    df["response"] = responses
    # Remove checkpoint column if it exists
    if "processed" in df.columns:
        df = df.drop(columns=["processed"])
    df.to_csv(output_csv, index=False)
    print(f"Results saved to: {output_csv}")


def main():
    parser = argparse.ArgumentParser(
        description="Batch process prompts using vLLM with checkpointing"
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
        "--checkpoint",
        "-c",
        type=str,
        default=None,
        help="Path to checkpoint file (default: output path with _checkpoint suffix)",
    )
    parser.add_argument(
        "--resume", action="store_true", help="Resume from checkpoint if available"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-7B-Instruct",
        help="Model name (default: Qwen/Qwen2.5-7B-Instruct)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for processing prompts (default: 16 for 7B model)",
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

    args = parser.parse_args()

    # Set output path
    if args.output is None:
        input_path = Path(args.input_csv)
        args.output = str(
            input_path.parent / f"{input_path.stem}_output{input_path.suffix}"
        )

    # Set checkpoint path
    if args.checkpoint is None:
        output_path = Path(args.output)
        args.checkpoint = str(
            output_path.parent / f"{output_path.stem}_checkpoint{output_path.suffix}"
        )

    # Load or resume from checkpoint
    start_idx = 0
    existing_responses = None

    if args.resume and os.path.exists(args.checkpoint):
        checkpoint_df = load_checkpoint(args.checkpoint)
        if checkpoint_df is not None and "response" in checkpoint_df.columns:
            existing_responses = checkpoint_df["response"].tolist()
            # Find first unprocessed index
            if "processed" in checkpoint_df.columns:
                processed = checkpoint_df["processed"].tolist()
                try:
                    start_idx = processed.index(False)
                    print(f"Resuming from index {start_idx}")
                except ValueError:
                    print("All prompts already processed!")
                    return

    # Load prompts
    print("Loading prompts...")
    df = load_prompts(args.input_csv)
    print(f"Loaded {len(df)} prompts")

    # If resuming, use existing responses as starting point
    if existing_responses and len(existing_responses) == len(df):
        df["response"] = existing_responses

    # Format prompts
    prompts = format_prompts(df)

    # Generate responses
    responses = batch_generate(
        prompts,
        model_name=args.model,
        batch_size=args.batch_size,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        top_p=args.top_p,
        start_idx=start_idx,
        checkpoint_path=args.checkpoint,
        df=df,
    )

    # Save final results
    save_results(df, responses, args.output)

    # Clean up checkpoint
    if os.path.exists(args.checkpoint):
        os.remove(args.checkpoint)
        print(f"Checkpoint file removed: {args.checkpoint}")


if __name__ == "__main__":
    main()
