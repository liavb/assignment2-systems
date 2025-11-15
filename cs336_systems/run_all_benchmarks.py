"""
Helper script to run benchmarks across all model sizes.

This script automates running benchmarks for all model configurations
from Table 1, making it easier to collect data for the assignment writeup.
"""

import subprocess
import sys
from typing import Dict
import pandas as pd

# Model configurations from Table 1
MODEL_SIZES = ['small', 'medium', 'large', 'xl', '2.7B']

def run_benchmark(
    model_size: str,
    context_length: int = 512,
    batch_size: int = 4,
    warmup_steps: int = 5,
    num_steps: int = 10,
    forward_only: bool = False,
    mixed_precision: bool = False,
    device: str = 'cuda',
) -> Dict[str, float]:
    """
    Run a single benchmark configuration.

    Args:
        model_size: Model size to benchmark
        context_length: Sequence length
        batch_size: Batch size
        warmup_steps: Number of warmup steps
        num_steps: Number of measurement steps
        forward_only: Whether to benchmark only forward pass
        mixed_precision: Whether to use mixed precision
        device: Device to use

    Returns:
        Dictionary with benchmark results
    """
    cmd = [
        sys.executable,
        '-m', 'cs336_systems.benchmark',
        '--model-size', model_size,
        '--context-length', str(context_length),
        '--batch-size', str(batch_size),
        '--warmup-steps', str(warmup_steps),
        '--num-steps', str(num_steps),
        '--device', device,
    ]

    if forward_only:
        cmd.append('--forward-only')

    if mixed_precision:
        cmd.append('--mixed-precision')

    print(f"Running: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        output = result.stdout

        # Parse the output to extract timing
        for line in output.split('\n'):
            if 'Forward pass:' in line or 'Forward + backward pass:' in line:
                parts = line.split(':')[1].strip().split('±')
                mean = float(parts[0].strip().split()[0])
                std = float(parts[1].strip().split()[0])
                return {'mean': mean, 'std': std, 'success': True}

        return {'mean': None, 'std': None, 'success': False, 'error': 'Could not parse output'}

    except subprocess.CalledProcessError as e:
        error_msg = e.stderr if e.stderr else str(e)
        if 'out of memory' in error_msg.lower() or 'oom' in error_msg.lower():
            return {'mean': None, 'std': None, 'success': False, 'error': 'OOM'}
        return {'mean': None, 'std': None, 'success': False, 'error': error_msg}
    except Exception as e:
        return {'mean': None, 'std': None, 'success': False, 'error': str(e)}


def run_all_forward_backward_benchmarks(
    context_length: int = 512,
    device: str = 'cuda',
    warmup_steps: int = 5,
    num_steps: int = 10,
) -> pd.DataFrame:
    """
    Run forward and backward benchmarks for all model sizes.

    Args:
        context_length: Sequence length to use
        device: Device to use
        warmup_steps: Number of warmup steps
        num_steps: Number of measurement steps

    Returns:
        DataFrame with benchmark results
    """
    results = []

    for model_size in MODEL_SIZES:
        print(f"\n{'='*60}")
        print(f"Benchmarking {model_size} model")
        print(f"{'='*60}")

        # Forward only
        print(f"\nForward pass only...")
        fwd_result = run_benchmark(
            model_size=model_size,
            context_length=context_length,
            forward_only=True,
            warmup_steps=warmup_steps,
            num_steps=num_steps,
            device=device,
        )

        # Forward + backward
        print(f"\nForward + backward pass...")
        fwd_bwd_result = run_benchmark(
            model_size=model_size,
            context_length=context_length,
            forward_only=False,
            warmup_steps=warmup_steps,
            num_steps=num_steps,
            device=device,
        )

        results.append({
            'model_size': model_size,
            'context_length': context_length,
            'forward_mean': fwd_result.get('mean'),
            'forward_std': fwd_result.get('std'),
            'forward_status': 'OOM' if fwd_result.get('error') == 'OOM' else ('success' if fwd_result.get('success') else 'failed'),
            'fwd_bwd_mean': fwd_bwd_result.get('mean'),
            'fwd_bwd_std': fwd_bwd_result.get('std'),
            'fwd_bwd_status': 'OOM' if fwd_bwd_result.get('error') == 'OOM' else ('success' if fwd_bwd_result.get('success') else 'failed'),
        })

    return pd.DataFrame(results)


def run_mixed_precision_comparison(
    context_length: int = 512,
    device: str = 'cuda',
    warmup_steps: int = 5,
    num_steps: int = 10,
) -> pd.DataFrame:
    """
    Compare FP32 vs mixed precision (BF16) for all model sizes.

    Args:
        context_length: Sequence length to use
        device: Device to use
        warmup_steps: Number of warmup steps
        num_steps: Number of measurement steps

    Returns:
        DataFrame with comparison results
    """
    results = []

    for model_size in MODEL_SIZES:
        print(f"\n{'='*60}")
        print(f"Mixed precision comparison for {model_size} model")
        print(f"{'='*60}")

        # FP32
        print(f"\nFP32...")
        fp32_result = run_benchmark(
            model_size=model_size,
            context_length=context_length,
            forward_only=False,
            mixed_precision=False,
            warmup_steps=warmup_steps,
            num_steps=num_steps,
            device=device,
        )

        # BF16
        print(f"\nBF16...")
        bf16_result = run_benchmark(
            model_size=model_size,
            context_length=context_length,
            forward_only=False,
            mixed_precision=True,
            warmup_steps=warmup_steps,
            num_steps=num_steps,
            device=device,
        )

        results.append({
            'model_size': model_size,
            'context_length': context_length,
            'fp32_mean': fp32_result.get('mean'),
            'fp32_std': fp32_result.get('std'),
            'bf16_mean': bf16_result.get('mean'),
            'bf16_std': bf16_result.get('std'),
            'speedup': fp32_result.get('mean') / bf16_result.get('mean') if (fp32_result.get('mean') and bf16_result.get('mean')) else None,
        })

    return pd.DataFrame(results)


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Run benchmarks across all model sizes')
    parser.add_argument('--mode', type=str, default='forward_backward',
                        choices=['forward_backward', 'mixed_precision', 'both'],
                        help='Which benchmarks to run')
    parser.add_argument('--context-length', type=int, default=512,
                        help='Sequence length')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')
    parser.add_argument('--warmup-steps', type=int, default=5,
                        help='Number of warmup steps')
    parser.add_argument('--num-steps', type=int, default=10,
                        help='Number of measurement steps')
    parser.add_argument('--output', type=str, default=None,
                        help='Output CSV file path')

    args = parser.parse_args()

    if args.mode in ['forward_backward', 'both']:
        print("\n" + "="*80)
        print("RUNNING FORWARD/BACKWARD BENCHMARKS")
        print("="*80)
        df = run_all_forward_backward_benchmarks(
            context_length=args.context_length,
            device=args.device,
            warmup_steps=args.warmup_steps,
            num_steps=args.num_steps,
        )
        print("\n" + "="*80)
        print("RESULTS")
        print("="*80)
        print(df.to_string(index=False))

        if args.output:
            output_file = args.output
        else:
            output_file = f'benchmark_results_ctx{args.context_length}.csv'
        df.to_csv(output_file, index=False)
        print(f"\nResults saved to {output_file}")

    if args.mode in ['mixed_precision', 'both']:
        print("\n" + "="*80)
        print("RUNNING MIXED PRECISION COMPARISON")
        print("="*80)
        df = run_mixed_precision_comparison(
            context_length=args.context_length,
            device=args.device,
            warmup_steps=args.warmup_steps,
            num_steps=args.num_steps,
        )
        print("\n" + "="*80)
        print("RESULTS")
        print("="*80)
        print(df.to_string(index=False))

        if args.output:
            output_file = args.output.replace('.csv', '_mixed_precision.csv')
        else:
            output_file = f'benchmark_mixed_precision_ctx{args.context_length}.csv'
        df.to_csv(output_file, index=False)
        print(f"\nResults saved to {output_file}")


if __name__ == '__main__':
    main()

