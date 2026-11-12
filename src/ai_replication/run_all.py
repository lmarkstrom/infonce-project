"""
run_all.py — Master runner for all paper experiments.

Usage:
  python run_all.py --quick     # Fast smoke-test (small epochs)
  python run_all.py             # Full reproduction (matches paper)

Runs:
  1. Synthetic experiments (Laplace, GMM, Binary) → Table 1 + Figs 2, 3
  2. CIFAR-10 MLP dynamics → Figure 4
  3. CIFAR-10 ResNet-18 Contrastive vs. Supervised → Table 1
  4. Final report (evaluate.py)
"""

import sys
import os
import argparse
import subprocess


def run(cmd: list[str]):
    """Run a command and exit on failure."""
    print('\n' + '='*70)
    print('RUNNING:', ' '.join(cmd))
    print('='*70)
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        print(f'\n[WARNING] Command exited with code {result.returncode}')


def main():
    parser = argparse.ArgumentParser(description='Run all paper experiments')
    parser.add_argument(
        '--quick', action='store_true',
        help='Run quick smoke-test with small epoch counts (not full reproduction)'
    )
    parser.add_argument(
        '--skip-cifar', action='store_true',
        help='Skip CIFAR-10 experiments (useful if torchvision/data not available)'
    )
    args = parser.parse_args()

    python = sys.executable
    here = os.path.dirname(os.path.abspath(__file__))
    os.chdir(here)

    quick_flag = ['--quick'] if args.quick else []

    # 1. Synthetic experiments
    run([python, 'train_synthetic.py'] + quick_flag)

    # 2. CIFAR-10 MLP dynamics
    if not args.skip_cifar:
        run([python, 'train_cifar_mlp.py'] + quick_flag)

    # 3. CIFAR-10 ResNet-18 contrastive vs supervised
    if not args.skip_cifar:
        run([python, 'train_cifar_resnet.py'] + quick_flag)

    # 4. Final consolidated report
    run([python, 'evaluate.py'])

    print('\n' + '='*70)
    print('ALL EXPERIMENTS COMPLETED')
    print('Results are in:', os.path.join(here, 'results'))
    print('='*70)


if __name__ == '__main__':
    main()
