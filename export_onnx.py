"""
Export the regression head from the trained model to ONNX format.

This script extracts only the regression head (not the encoder) and exports it
to ONNX format with quantization for efficient browser-based inference.

The full inference pipeline will use:
1. Transformers.js with Xenova/all-MiniLM-L12-v2 for embeddings (from CDN)
2. This exported ONNX regression head for the final prediction
"""

import torch
import torch.nn as nn
import os
import argparse


class RegressionHead(nn.Module):
    """
    Standalone regression head that matches the architecture from transformer_second_model.py.
    
    Architecture: 384 -> 128 -> 64 -> 1
    With Dropout(0.2) and ReLU activations.
    """
    
    def __init__(self):
        super().__init__()
        self.regressor = nn.Sequential(
            nn.Dropout(0.2),      # index 0 - no weights
            nn.Linear(384, 128),  # index 1
            nn.ReLU(),            # index 2 - no weights
            nn.Dropout(0.2),      # index 3 - no weights
            nn.Linear(128, 64),   # index 4
            nn.ReLU(),            # index 5 - no weights
            nn.Linear(64, 1),     # index 6
        )
    
    def forward(self, x):
        """
        Forward pass through the regression head.
        
        Args:
            x: Tensor of shape (batch_size, 384) - sentence embeddings
            
        Returns:
            Tensor of shape (batch_size, 1) - predicted error scores
        """
        return self.regressor(x)


def load_regressor_weights(checkpoint_path: str) -> RegressionHead:
    """
    Load the regression head weights from a full model checkpoint.
    
    Args:
        checkpoint_path: Path to the model checkpoint (.pt file)
        
    Returns:
        RegressionHead model with loaded weights
    """
    print(f"Loading checkpoint from: {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    full_state_dict = checkpoint['model_state_dict']
    
    # Extract only regressor weights
    regressor_state_dict = {}
    for key, value in full_state_dict.items():
        if key.startswith('regressor.'):
            regressor_state_dict[key] = value
            print(f"  Extracted: {key} -> {value.shape}")
    
    # Create model and load weights
    model = RegressionHead()
    model.load_state_dict(regressor_state_dict)
    model.eval()  # Set to evaluation mode (disables dropout)
    
    print(f"\nLoaded {len(regressor_state_dict)} weight tensors")
    print(f"Model config from checkpoint: {checkpoint.get('config', 'N/A')}")
    print(f"Model metrics - RMSE: {checkpoint.get('rmse', 'N/A'):.4f}, "
          f"MAE: {checkpoint.get('mae', 'N/A'):.4f}, "
          f"R2: {checkpoint.get('r2', 'N/A'):.4f}")
    
    return model


def export_to_onnx(model: nn.Module, output_path: str):
    """
    Export the model to ONNX format using legacy exporter for better compatibility.
    
    Args:
        model: PyTorch model to export
        output_path: Path for the output ONNX file
    """
    print(f"\nExporting to ONNX: {output_path}")
    
    # Create dummy input with dynamic batch size
    # Shape: (batch_size, embedding_dim) = (1, 384)
    dummy_input = torch.randn(1, 384)
    
    # Export to ONNX using legacy exporter (dynamo=False) for better compatibility
    # with quantization and ONNX Runtime Web
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['embedding'],
        output_names=['score'],
        dynamic_axes={
            'embedding': {0: 'batch_size'},
            'score': {0: 'batch_size'}
        },
        dynamo=False  # Use legacy exporter for compatibility
    )
    
    print(f"ONNX model saved to: {output_path}")
    
    # Verify the model
    import onnx
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print("ONNX model validation passed!")
    
    return output_path


def quantize_onnx(input_path: str, output_path: str):
    """
    Quantize the ONNX model to int8 for smaller size and faster inference.
    
    Args:
        input_path: Path to the input ONNX model
        output_path: Path for the quantized output model
    """
    print(f"\nQuantizing model: {input_path} -> {output_path}")
    
    from onnxruntime.quantization import quantize_dynamic, QuantType
    
    quantize_dynamic(
        input_path,
        output_path,
        weight_type=QuantType.QUInt8
    )
    
    # Report file sizes
    original_size = os.path.getsize(input_path)
    quantized_size = os.path.getsize(output_path)
    reduction = (1 - quantized_size / original_size) * 100
    
    print(f"Original size: {original_size / 1024:.2f} KB")
    print(f"Quantized size: {quantized_size / 1024:.2f} KB")
    print(f"Size reduction: {reduction:.1f}%")
    
    return output_path


def verify_onnx_inference(onnx_path: str, pytorch_model: nn.Module):
    """
    Verify that ONNX model produces same outputs as PyTorch model.
    
    Args:
        onnx_path: Path to the ONNX model
        pytorch_model: Original PyTorch model for comparison
    """
    print(f"\nVerifying ONNX inference accuracy...")
    
    import onnxruntime as ort
    import numpy as np
    
    # Create test inputs
    test_inputs = [
        torch.randn(1, 384),
        torch.randn(5, 384),
        torch.randn(10, 384),
    ]
    
    # Create ONNX session
    session = ort.InferenceSession(onnx_path)
    
    pytorch_model.eval()
    max_diff = 0.0
    
    for i, test_input in enumerate(test_inputs):
        # PyTorch inference
        with torch.no_grad():
            pytorch_output = pytorch_model(test_input).numpy()
        
        # ONNX inference
        onnx_output = session.run(
            ['score'],
            {'embedding': test_input.numpy()}
        )[0]
        
        # Compare
        diff = np.abs(pytorch_output - onnx_output).max()
        max_diff = max(max_diff, diff)
        print(f"  Test {i+1} (batch={test_input.shape[0]}): max diff = {diff:.6f}")
    
    print(f"\nMax difference across all tests: {max_diff:.6f}")
    if max_diff < 1e-4:
        print("Verification PASSED - outputs match within tolerance")
    else:
        print("WARNING: Outputs differ more than expected (may be due to quantization)")


def main():
    parser = argparse.ArgumentParser(
        description="Export regression head to ONNX format for web deployment"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="model_output/best_model.pt",
        help="Path to the model checkpoint"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="web_demo/models",
        help="Output directory for ONNX models"
    )
    parser.add_argument(
        "--no-quantize",
        action="store_true",
        help="Skip quantization (keep full precision)"
    )
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model
    model = load_regressor_weights(args.checkpoint)
    
    # Export to ONNX
    onnx_path = os.path.join(args.output_dir, "regressor.onnx")
    export_to_onnx(model, onnx_path)
    
    # Verify full-precision model
    verify_onnx_inference(onnx_path, model)
    
    # Quantize
    if not args.no_quantize:
        quantized_path = os.path.join(args.output_dir, "regressor_quantized.onnx")
        quantize_onnx(onnx_path, quantized_path)
        
        # Verify quantized model
        verify_onnx_inference(quantized_path, model)
        
        print(f"\n{'='*50}")
        print("Export complete!")
        print(f"Full precision model: {onnx_path}")
        print(f"Quantized model: {quantized_path}")
        print(f"{'='*50}")
    else:
        print(f"\n{'='*50}")
        print("Export complete!")
        print(f"Model saved to: {onnx_path}")
        print(f"{'='*50}")


if __name__ == "__main__":
    main()
