"""
Benchmarking script for Transformer models.

This script implements end-to-end benchmarking of forward and backward passes
for Transformer language models, as described in task 1.1.3.
"""

import argparse
import timeit
from typing import Optional, Dict
import statistics
import torch
import torch.nn as nn
import torch.cuda.nvtx as nvtx

from cs336_basics.model import BasicsTransformerLM


# Model configurations from Table 1
MODEL_CONFIGS = {
    'small': {
        'd_model': 768,
        'd_ff': 3072,
        'num_layers': 12,
        'num_heads': 12,
    },
    'medium': {
        'd_model': 1024,
        'd_ff': 4096,
        'num_layers': 24,
        'num_heads': 16,
    },
    'large': {
        'd_model': 1280,
        'd_ff': 5120,
        'num_layers': 36,
        'num_heads': 20,
    },
    'xl': {
        'd_model': 1600,
        'd_ff': 6400,
        'num_layers': 48,
        'num_heads': 25,
    },
    '2.7B': {
        'd_model': 2560,
        'd_ff': 10240,
        'num_layers': 32,
        'num_heads': 32,
    },
}


def initialize_model(
    model_size: str,
    vocab_size: int = 10000,
    context_length: int = 128,
    rope_theta: float = 10000.0,
    device: str = 'cuda',
    dtype: Optional[torch.dtype] = None,
) -> BasicsTransformerLM:
    """
    Initialize a Transformer model with the given configuration.

    Args:
        model_size: Name of the model size from MODEL_CONFIGS
        vocab_size: Size of the vocabulary
        context_length: Maximum sequence length
        rope_theta: Theta value for RoPE positional encoding
        device: Device to place the model on
        dtype: Data type for model parameters (None for default float32)

    Returns:
        Initialized BasicsTransformerLM model
    """
    if model_size not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model size: {model_size}. Choose from {list(MODEL_CONFIGS.keys())}")

    config = MODEL_CONFIGS[model_size]

    model = BasicsTransformerLM(
        vocab_size=vocab_size,
        context_length=context_length,
        d_model=config['d_model'],
        num_layers=config['num_layers'],
        num_heads=config['num_heads'],
        d_ff=config['d_ff'],
        rope_theta=rope_theta,
    )

    model = model.to(device)
    if dtype is not None:
        model = model.to(dtype)

    return model


def generate_random_batch(
    batch_size: int,
    sequence_length: int,
    vocab_size: int,
    device: str = 'cuda',
) -> torch.Tensor:
    """
    Generate a random batch of token IDs.

    Args:
        batch_size: Number of sequences in the batch
        sequence_length: Length of each sequence
        vocab_size: Size of the vocabulary
        device: Device to place the batch on

    Returns:
        Random token IDs tensor of shape (batch_size, sequence_length)
    """
    return torch.randint(0, vocab_size, (batch_size, sequence_length), device=device)


def benchmark_forward(
    model: nn.Module,
    input_ids: torch.Tensor,
    warmup_steps: int,
    num_steps: int,
    use_mixed_precision: bool = False,
    dtype: Optional[torch.dtype] = None,
    device: str = 'cuda',
) -> Dict[str, float]:
    """
    Benchmark the forward pass of the model.

    Args:
        model: The model to benchmark
        input_ids: Input token IDs
        warmup_steps: Number of warmup iterations before timing
        num_steps: Number of iterations to time
        use_mixed_precision: Whether to use mixed precision (autocast)
        dtype: Data type for mixed precision (e.g., torch.bfloat16)
        device: Device being used (cuda or cpu)

    Returns:
        Dictionary with 'mean' and 'std' of timings in seconds
    """
    model.eval()
    use_cuda = device == 'cuda' and torch.cuda.is_available()
    print(f'using cuda: {use_cuda}')

    # Warmup
    with nvtx.range("warmup"):
        with torch.no_grad():
            for _ in range(warmup_steps):
                if use_mixed_precision and dtype is not None:
                    with torch.autocast(device_type=device, dtype=dtype):
                        _ = model(input_ids)
                else:
                    _ = model(input_ids)
                if use_cuda:
                    torch.cuda.synchronize()

    # Actual timing
    timings = []
    with nvtx.range("benchmark_forward"):
        with torch.no_grad():
            for _ in range(num_steps):
                start = timeit.default_timer()

                if use_mixed_precision and dtype is not None:
                    with torch.autocast(device_type=device, dtype=dtype):
                        _ = model(input_ids)
                else:
                    _ = model(input_ids)

                if use_cuda:
                    torch.cuda.synchronize()
                end = timeit.default_timer()
                timings.append(end - start)

    return {
        'mean': statistics.mean(timings),
        'std': statistics.stdev(timings) if len(timings) > 1 else 0.0,
        'timings': timings,
    }


def benchmark_forward_backward(
    model: nn.Module,
    input_ids: torch.Tensor,
    warmup_steps: int,
    num_steps: int,
    use_mixed_precision: bool = False,
    dtype: Optional[torch.dtype] = None,
    device: str = 'cuda',
) -> Dict[str, Dict[str, float]]:
    """
    Benchmark the forward and backward pass of the model separately.

    Args:
        model: The model to benchmark
        input_ids: Input token IDs
        warmup_steps: Number of warmup iterations before timing
        num_steps: Number of iterations to time
        use_mixed_precision: Whether to use mixed precision (autocast)
        dtype: Data type for mixed precision (e.g., torch.bfloat16)
        device: Device being used (cuda or cpu)

    Returns:
        Dictionary with 'forward' and 'backward' keys, each containing
        'mean', 'std', and 'timings' of the respective pass in seconds
    """
    model.train()
    use_cuda = device == 'cuda' and torch.cuda.is_available()

    # Warmup
    with nvtx.range("warmup"):
        for _ in range(warmup_steps):
            model.zero_grad()

            if use_mixed_precision and dtype is not None:
                with torch.autocast(device_type=device, dtype=dtype):
                    output = model(input_ids)
                    loss = output.sum()
            else:
                output = model(input_ids)
                loss = output.sum()

            loss.backward()
            if use_cuda:
                torch.cuda.synchronize()

    # Actual timing - separate forward and backward
    forward_timings = []
    backward_timings = []

    with nvtx.range("benchmark_forward_backward"):
        for _ in range(num_steps):
            model.zero_grad()

            # Time forward pass
            with nvtx.range("forward"):
                start_forward = timeit.default_timer()

                if use_mixed_precision and dtype is not None:
                    with torch.autocast(device_type=device, dtype=dtype):
                        output = model(input_ids)
                        loss = output.sum()
                else:
                    output = model(input_ids)
                    loss = output.sum()

                if use_cuda:
                    torch.cuda.synchronize()
                end_forward = timeit.default_timer()
                forward_timings.append(end_forward - start_forward)

            # Time backward pass
            with nvtx.range("backward"):
                start_backward = timeit.default_timer()
                loss.backward()
                if use_cuda:
                    torch.cuda.synchronize()
                end_backward = timeit.default_timer()
                backward_timings.append(end_backward - start_backward)

    return {
        'forward': {
            'mean': statistics.mean(forward_timings),
            'std': statistics.stdev(forward_timings) if len(forward_timings) > 1 else 0.0,
            'timings': forward_timings,
        },
        'backward': {
            'mean': statistics.mean(backward_timings),
            'std': statistics.stdev(backward_timings) if len(backward_timings) > 1 else 0.0,
            'timings': backward_timings,
        },
    }


def main():
    parser = argparse.ArgumentParser(description='Benchmark Transformer models')

    # Model configuration
    parser.add_argument('--model-size', type=str, default='small',
                        choices=list(MODEL_CONFIGS.keys()),
                        help='Model size to benchmark')
    parser.add_argument('--vocab-size', type=int, default=10000,
                        help='Vocabulary size')
    parser.add_argument('--context-length', type=int, default=512,
                        help='Maximum sequence length')
    parser.add_argument('--batch-size', type=int, default=4,
                        help='Batch size')

    # Benchmarking parameters
    parser.add_argument('--warmup-steps', type=int, default=3,
                        help='Number of warmup steps')
    parser.add_argument('--num-steps', type=int, default=10,
                        help='Number of steps to measure')
    parser.add_argument('--forward-only', action='store_true',
                        help='Only benchmark forward pass (no backward)')

    # Mixed precision
    parser.add_argument('--mixed-precision', action='store_true',
                        help='Use mixed precision (bfloat16)')

    # Device
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda or cpu)')

    args = parser.parse_args()
    # Check CUDA availability
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = 'cpu'


    print(f'device: {args.device}')
    # Initialize model
    print(f"Initializing {args.model_size} model...")
    dtype = torch.bfloat16 if args.mixed_precision else None
    with nvtx.range("initialize_model"):
        model = initialize_model(
            model_size=args.model_size,
            vocab_size=args.vocab_size,
            context_length=args.context_length,
            device=args.device,
            dtype=dtype,
        )

    with nvtx.range("generate_random_batch"):
        # Generate random batch
        input_ids = generate_random_batch(
            batch_size=args.batch_size,
            sequence_length=args.context_length,
            vocab_size=args.vocab_size,
            device=args.device,
        )

    print(f"\nBenchmarking configuration:")
    print(f"  Model size: {args.model_size}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Sequence length: {args.context_length}")
    print(f"  Warmup steps: {args.warmup_steps}")
    print(f"  Measurement steps: {args.num_steps}")
    print(f"  Mixed precision: {args.mixed_precision}")
    print(f"  Forward only: {args.forward_only}")

    # Run benchmark
    if args.forward_only:
        print("\nBenchmarking forward pass...")
        results = benchmark_forward(
            model=model,
            input_ids=input_ids,
            warmup_steps=args.warmup_steps,
            num_steps=args.num_steps,
            use_mixed_precision=args.mixed_precision,
            dtype=torch.bfloat16 if args.mixed_precision else None,
            device=args.device,
        )
        print(f"Forward pass: {results['mean']:.4f} ± {results['std']:.4f} seconds")

        # Print individual timings for debugging
        if args.num_steps <= 20:
            print(f"\nIndividual timings: {[f'{t:.4f}' for t in results['timings']]}")
    else:
        print("\nBenchmarking forward and backward passes separately...")
        results = benchmark_forward_backward(
            model=model,
            input_ids=input_ids,
            warmup_steps=args.warmup_steps,
            num_steps=args.num_steps,
            use_mixed_precision=args.mixed_precision,
            dtype=torch.bfloat16 if args.mixed_precision else None,
            device=args.device,
        )
        print(f"Forward pass:  {results['forward']['mean']:.4f} ± {results['forward']['std']:.4f} seconds")
        print(f"Backward pass: {results['backward']['mean']:.4f} ± {results['backward']['std']:.4f} seconds")
        print(f"Total (F+B):   {results['forward']['mean'] + results['backward']['mean']:.4f} seconds")

        # Print individual timings for debugging
        if args.num_steps <= 20:
            print(f"\nForward timings:  {[f'{t:.4f}' for t in results['forward']['timings']]}")
            print(f"Backward timings: {[f'{t:.4f}' for t in results['backward']['timings']]}")


if __name__ == '__main__':
    # cs336_basics.model.scaled_dot_product_attention = annotated_scaled_dot_product_attention
    main()

