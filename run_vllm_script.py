# sonnet 4.5
import pandas as pd
from vllm import LLM, SamplingParams
from typing import List
import argparse
from pathlib import Path
from tqdm import tqdm


def load_prompts(input_csv: str) -> pd.DataFrame:
    """Load prompts from CSV file."""
    df = pd.read_csv(input_csv)
    required_cols = ["system_prompt", "user_prompt"]

    if not all(col in df.columns for col in required_cols):
        raise ValueError(
            f"CSV must contain columns: {required_cols}. Found: {df.columns.tolist()}"
        )

    return df


def format_prompts(df: pd.DataFrame) -> List[str]:
    """Format system and user prompts into single strings."""
    prompts = []
    for _, row in df.iterrows():
        # Adjust this format based on your model's chat template
        formatted = (
            f"{row['system_prompt']}\n\nUser: {row['user_prompt']}\n\nAssistant:"
        )
        prompts.append(formatted)

    return prompts


def batch_generate(
    prompts: List[str],
    model_name: str = "gpt-oss/gpt-oss-20b",
    batch_size: int = 256,
    temperature: float = 0.7,
    max_tokens: int = 512,
    top_p: float = 0.9,
) -> List[str]:
    """Generate responses using vLLM in batches."""

    # Initialize vLLM
    print(f"Loading model: {model_name}")
    llm = LLM(
        model=model_name,
        tensor_parallel_size=1,  # Adjust based on your GPU setup
        gpu_memory_utilization=0.9,
        max_model_len=4096,  # Adjust based on your needs
    )

    # Set sampling parameters
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )

    # Process in batches with progress bar
    print(
        f"Generating responses for {len(prompts)} prompts in batches of {batch_size}..."
    )
    all_responses = []

    for i in tqdm(range(0, len(prompts), batch_size)):
        batch = prompts[i : i + batch_size]
        outputs = llm.generate(batch, sampling_params)
        responses = [output.outputs[0].text for output in outputs]
        all_responses.extend(responses)

    return all_responses


def save_results(df: pd.DataFrame, responses: List[str], output_csv: str) -> None:
    """Save results to CSV with the same structure as input."""
    df["response"] = responses
    df.to_csv(output_csv, index=False)
    print(f"Results saved to: {output_csv}")


def main():
    parser = argparse.ArgumentParser(description="Batch process prompts using vLLM")
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
        default="gpt-oss/gpt-oss-20b",
        help="Model name (default: gpt-oss/gpt-oss-20b)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Batch size for processing prompts (default: 256)",
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

    # Load prompts
    print("Loading prompts...")
    df = load_prompts(args.input_csv)
    print(f"Loaded {len(df)} prompts")

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
    )

    # Save results
    save_results(df, responses, args.output)


if __name__ == "__main__":
    main()
