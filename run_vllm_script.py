#!/usr/bin/env python3
"""
Batch inference script for gpt-oss-20b with vLLM.
Processes a CSV of prompts and saves responses with the same keys.

Usage:
    python batch_inference.py --input prompts.csv --output responses.csv \
        --model openai/gpt-oss-20b --batch-size 64
"""

import argparse
import csv
import gc
import json
import logging
import os
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd
from tqdm import tqdm

# vLLM imports
from vllm import LLM, SamplingParams
from vllm.utils import FlexibleArgumentParser

# For gpt-oss harmony format
try:
    from openai_harmony import (
        HarmonyEncodingName,
        load_harmony_encoding,
        Conversation,
        Message,
        Role,
        SystemContent,
        DeveloperContent,
    )

    HARMONY_AVAILABLE = True
except ImportError:
    print("Warning: openai-harmony not found. Using standard chat template.")
    HARMONY_AVAILABLE = False
    from transformers import AutoTokenizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("batch_inference.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def load_prompts(csv_path: str) -> List[Dict[str, Any]]:
    """Load prompts from CSV file."""
    logger.info(f"Loading prompts from {csv_path}")
    df = pd.read_csv(csv_path)

    # Validate required columns
    required_cols = {"id", "system_prompt", "user_prompt"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"CSV must contain columns: {required_cols}")

    # Convert to list of dicts for easier processing
    prompts = df.to_dict("records")
    logger.info(f"Loaded {len(prompts)} prompts")
    return prompts


def prepare_gpt_oss_prompt(system_prompt: str, user_prompt: str) -> List[int]:
    """Prepare prompt using Harmony encoding for gpt-oss."""
    if HARMONY_AVAILABLE:
        encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)

        convo = Conversation.from_messages(
            [
                Message.from_role_and_content(Role.SYSTEM, SystemContent.new()),
                Message.from_role_and_content(
                    Role.DEVELOPER,
                    DeveloperContent.new().with_instructions(system_prompt),
                ),
                Message.from_role_and_content(Role.USER, user_prompt),
            ]
        )

        prefill_ids = encoding.render_conversation_for_completion(convo, Role.ASSISTANT)
        return prefill_ids
    else:
        # Fallback to standard chat template
        tokenizer = AutoTokenizer.from_pretrained("openai/gpt-oss-20b")
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        return tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True
        )


def process_batch(
    llm: LLM,
    batch: List[Dict[str, Any]],
    sampling_params: SamplingParams,
    use_harmony: bool = True,
) -> List[Dict[str, Any]]:
    """Process a batch of prompts and return results."""
    results = []

    if use_harmony and HARMONY_AVAILABLE:
        # Prepare tokenized prompts
        prompt_token_ids = [
            prepare_gpt_oss_prompt(item["system_prompt"], item["user_prompt"])
            for item in batch
        ]

        # Generate with pre-filled token IDs
        outputs = llm.generate(
            prompt_token_ids=prompt_token_ids,
            sampling_params=sampling_params,
            use_tqdm=False,
        )
    else:
        # Use text prompts with chat template
        prompts = [
            f"<|im_start|>system\n{item['system_prompt']}\n<|im_start|>user\n{item['user_prompt']}\n<|im_start|>assistant\n"
            for item in batch
        ]
        outputs = llm.generate(prompts, sampling_params, use_tqdm=False)

    # Process outputs
    for i, output in enumerate(outputs):
        result = {
            "id": batch[i]["id"],
            "system_prompt": batch[i]["system_prompt"],
            "user_prompt": batch[i]["user_prompt"],
            "generated_text": output.outputs[0].text,
            "finish_reason": output.outputs[0].finish_reason,
            "prompt_tokens": len(output.prompt_token_ids),
            "completion_tokens": len(output.outputs[0].token_ids),
        }
        results.append(result)

    return results


def main():
    parser = argparse.ArgumentParser(description="Batch inference with gpt-oss-20b")
    parser.add_argument("--input", required=True, help="Input CSV file path")
    parser.add_argument("--output", required=True, help="Output CSV file path")
    parser.add_argument("--model", default="openai/gpt-oss-20b", help="Model ID")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument(
        "--max-tokens", type=int, default=1024, help="Max tokens to generate"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.7, help="Sampling temperature"
    )
    parser.add_argument("--top-p", type=float, default=0.95, help="Top-p sampling")
    parser.add_argument(
        "--tensor-parallel-size", type=int, default=1, help="Tensor parallelism"
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.85,
        help="GPU memory utilization",
    )
    parser.add_argument(
        "--resume", action="store_true", help="Resume from existing output"
    )
    parser.add_argument(
        "--max-model-len", type=int, default=32768, help="Max model length"
    )

    args = parser.parse_args()

    # Load prompts
    prompts = load_prompts(args.input)
    total_prompts = len(prompts)

    # Setup resume logic
    processed_ids = set()
    if args.resume and Path(args.output).exists():
        logger.info(f"Resuming from existing output: {args.output}")
        existing_df = pd.read_csv(args.output)
        processed_ids = set(existing_df["id"].astype(str).tolist())
        prompts = [p for p in prompts if str(p["id"]) not in processed_ids]
        logger.info(
            f"Found {len(processed_ids)} already processed, {len(prompts)} remaining"
        )

    if not prompts:
        logger.info("All prompts already processed!")
        return

    # Initialize vLLM
    logger.info(f"Initializing vLLM with model: {args.model}")
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        trust_remote_code=True,
        dtype="auto",
        quantization="mxfp4" if "gpt-oss" in args.model else None,
    )

    # Setup sampling params
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        stop_token_ids=[128009] if HARMONY_AVAILABLE else None,  # Harmony stop token
    )

    # Process in batches
    logger.info(f"Processing {len(prompts)} prompts in batches of {args.batch_size}")
    all_results = []

    # Create output directory if needed
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    # Open output file for incremental writing
    output_file = open(args.output, "a", newline="", encoding="utf-8")
    fieldnames = [
        "id",
        "system_prompt",
        "user_prompt",
        "generated_text",
        "finish_reason",
        "prompt_tokens",
        "completion_tokens",
    ]
    writer = csv.DictWriter(output_file, fieldnames=fieldnames)

    # Write header only if new file
    if not args.resume or not Path(args.output).exists():
        writer.writeheader()

    # Process batches with progress bar
    with tqdm(total=len(prompts), desc="Processing prompts") as pbar:
        for i in range(0, len(prompts), args.batch_size):
            batch = prompts[i : i + args.batch_size]

            try:
                # Process batch
                batch_results = process_batch(
                    llm, batch, sampling_params, use_harmony=HARMONY_AVAILABLE
                )

                # Write results immediately
                for result in batch_results:
                    writer.writerow(result)
                output_file.flush()

                # Update progress
                pbar.update(len(batch))

                # Optional: Clear cache periodically
                if i > 0 and i % (args.batch_size * 10) == 0:
                    gc.collect()

            except Exception as e:
                logger.error(f"Error processing batch {i // args.batch_size}: {e}")
                # Save failed batch info for retry
                failed_ids = [item["id"] for item in batch]
                logger.error(f"Failed batch IDs: {failed_ids}")
                continue

    output_file.close()
    logger.info(f"Completed! Results saved to {args.output}")

    # Print summary statistics
    if Path(args.output).exists():
        final_df = pd.read_csv(args.output)
        total_tokens = (
            final_df["prompt_tokens"].sum() + final_df["completion_tokens"].sum()
        )
        logger.info(f"Total tokens processed: {total_tokens:,}")
        logger.info(
            f"Average completion tokens: {final_df['completion_tokens'].mean():.1f}"
        )


if __name__ == "__main__":
    main()
