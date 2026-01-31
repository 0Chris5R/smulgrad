#!/usr/bin/env python3
"""
MNIST Demo - Train your autodiff engine on real handwritten digits!

Run with: uv run python tests/cases/demo_mnist.py

This demo validates that your SmulGrad implementation can train on a real
machine learning benchmark - the MNIST handwritten digit dataset.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import plotext as pltx
import sys
import time
import io
from pathlib import Path

from rich.console import Console
from rich.table import Table
from rich_pixels import Pixels
from PIL import Image

from tests.mnist_utils import load_mnist, compute_accuracy


def plot_digit_grid(images, predictions, actuals, confidences, title, console):
    """Plot a grid of digits with predictions using matplotlib and rich-pixels."""
    n_images = len(images)
    cols = min(4, n_images)
    rows = (n_images + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.5, rows * 3), facecolor='#1e1e1e')
    if rows == 1 and cols == 1:
        axes = np.array([axes])
    axes = np.atleast_2d(axes).flatten()

    # Custom colormap: dark to bright blue/white
    cmap = LinearSegmentedColormap.from_list('mnist', ['#1e1e1e', '#3d6a99', '#7eb3e0', '#ffffff'])

    for i, ax in enumerate(axes):
        ax.set_facecolor('#1e1e1e')
        if i < n_images:
            img = images[i].reshape(28, 28)
            pred = predictions[i]
            actual = actuals[i]
            conf = confidences[i]

            ax.imshow(img, cmap=cmap, vmin=0, vmax=1, interpolation='lanczos')
            ax.set_xticks([])
            ax.set_yticks([])

            # Color based on correct/incorrect
            if pred == actual:
                border_color = '#4ade80'  # Green
                text_color = '#4ade80'
                status = f"{pred}"
            else:
                border_color = '#f87171'  # Red
                text_color = '#f87171'
                status = f"{pred} (actual: {actual})"

            ax.set_title(f"{status}\n{conf*100:.0f}%", fontsize=10, color=text_color, pad=5)

            for spine in ax.spines.values():
                spine.set_color(border_color)
                spine.set_linewidth(3)
        else:
            ax.set_visible(False)

    fig.suptitle(title, fontsize=12, color='white', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # Save to buffer at high DPI and display
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=600, bbox_inches='tight',
                facecolor='#1e1e1e', edgecolor='none')
    buf.seek(0)
    plt.close(fig)

    # Display in terminal using rich-pixels
    img = Image.open(buf)
    img = img.resize((80, 45), Image.Resampling.LANCZOS)
    pixels = Pixels.from_image(img)
    console.print(pixels)


def plot_comparison_grid(images, predictions, actuals, title, console, show_correct=True):
    """Plot correct or incorrect predictions in a row."""
    mask = np.array(predictions) == np.array(actuals)
    if not show_correct:
        mask = ~mask

    indices = np.where(mask)[0]
    if len(indices) == 0:
        console.print(f"  [dim]No {'correct' if show_correct else 'incorrect'} predictions to show[/dim]")
        return

    # Take first 5
    indices = indices[:5]
    n_images = len(indices)

    fig, axes = plt.subplots(1, n_images, figsize=(n_images * 2, 2.5), facecolor='#1e1e1e')
    if n_images == 1:
        axes = [axes]

    cmap = LinearSegmentedColormap.from_list('mnist', ['#1e1e1e', '#3d6a99', '#7eb3e0', '#ffffff'])
    border_color = '#4ade80' if show_correct else '#f87171'
    text_color = border_color

    for i, (ax, idx) in enumerate(zip(axes, indices)):
        ax.set_facecolor('#1e1e1e')
        img = images[idx].reshape(28, 28)
        pred = predictions[idx]
        actual = actuals[idx]

        ax.imshow(img, cmap=cmap, vmin=0, vmax=1, interpolation='lanczos')
        ax.set_xticks([])
        ax.set_yticks([])

        if show_correct:
            ax.set_title(f"{pred}", fontsize=11, color=text_color, pad=5)
        else:
            ax.set_title(f"{pred}\n(was {actual})", fontsize=10, color=text_color, pad=5)

        for spine in ax.spines.values():
            spine.set_color(border_color)
            spine.set_linewidth(3)

    fig.suptitle(title, fontsize=11, color='white', y=1.0)
    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=600, bbox_inches='tight',
                facecolor='#1e1e1e', edgecolor='none')
    buf.seek(0)
    plt.close(fig)

    img = Image.open(buf)
    img = img.resize((70, 20), Image.Resampling.LANCZOS)
    pixels = Pixels.from_image(img)
    console.print(pixels)


def main():
    console = Console()

    # Import adapter
    try:
        from tests.adapters import run_mnist_training
        from smulgrad.nn import softmax
        from smulgrad.engine import Tensor
    except ImportError as e:
        console.print(f"[red]Error importing: {e}[/red]")
        console.print("Make sure you have completed Parts 1-6 and the MNIST bonus.")
        sys.exit(1)

    # Header
    console.print()
    console.print("=" * 60, style="bold")
    console.print("     SMULGRAD MNIST DEMO - HANDWRITTEN DIGITS", style="bold cyan")
    console.print("=" * 60, style="bold")
    console.print()
    console.print("  [dim]Using hyperparameters from run_mnist_training in adapters.py[/dim]")

    # Load data for visualization
    console.print()
    console.print("[bold]Loading MNIST dataset...[/bold]")

    try:
        train_X, train_y, test_X, test_y = load_mnist()
    except Exception as e:
        console.print(f"[red]Error loading MNIST: {e}[/red]")
        console.print("Please check your internet connection.")
        sys.exit(1)

    # Convert one-hot to labels for display
    train_labels = np.argmax(train_y, axis=1)
    test_labels = np.argmax(test_y, axis=1)

    console.print(f"  Train: {train_X.shape[0]:,} samples")
    console.print(f"  Test:  {test_X.shape[0]:,} samples")

    # Show sample digits
    console.print()
    console.print("-" * 60)
    console.print("  SAMPLE TRAINING DIGITS", style="bold yellow")
    console.print("-" * 60)
    console.print()
    console.print("  [dim]These are real handwritten digits your network will learn:[/dim]")
    console.print()

    sample_idx = np.random.choice(len(train_X), 8, replace=False)
    sample_images = train_X[sample_idx]
    sample_labels_display = train_labels[sample_idx]

    plot_digit_grid(
        sample_images,
        sample_labels_display,  # predictions = actuals for display
        sample_labels_display,
        [1.0] * 8,  # 100% confidence
        "Training Data Samples",
        console
    )

    input("\n  Press Enter to start training...")

    # Training via adapter
    console.print()
    console.print("-" * 60)
    console.print("  TRAINING", style="bold green")
    console.print("-" * 60)
    console.print()
    console.print("  [dim]Training model via your train_mnist implementation...[/dim]")
    console.print()

    start_time = time.time()
    mlp, accuracy = run_mnist_training()
    total_time = time.time() - start_time

    console.print(f"  Training completed in {total_time:.1f}s")
    console.print(f"  Final test accuracy: [bold green]{accuracy*100:.1f}%[/bold green]")

    # Compute predictions for visualization
    console.print()
    console.print("[bold]Computing predictions for visualization...[/bold]")

    logits = mlp(Tensor(test_X))
    probs = softmax(logits, axis=-1)
    predictions = np.argmax(probs.data, axis=1).tolist()
    confidences = np.max(probs.data, axis=1).tolist()

    input("\n  Press Enter to see predictions...")

    # Sample predictions
    console.print()
    console.print("-" * 60)
    console.print("  SAMPLE PREDICTIONS", style="bold yellow")
    console.print("-" * 60)
    console.print()
    console.print("  [dim]Random test digits with model predictions:[/dim]")
    console.print()

    # Show random test samples
    sample_idx = np.random.choice(len(test_X), 8, replace=False)
    plot_digit_grid(
        test_X[sample_idx],
        [predictions[i] for i in sample_idx],
        test_labels[sample_idx],
        [confidences[i] for i in sample_idx],
        "Model Predictions on Test Set",
        console
    )

    input("\n  Press Enter to see correct vs incorrect...")

    # Correct vs Incorrect
    console.print()
    console.print("-" * 60)
    console.print("  CORRECT PREDICTIONS", style="bold green")
    console.print("-" * 60)
    console.print()

    plot_comparison_grid(test_X, predictions, test_labels.tolist(),
                         "Correctly Classified Digits", console, show_correct=True)

    console.print()
    console.print("-" * 60)
    console.print("  INCORRECT PREDICTIONS", style="bold red")
    console.print("-" * 60)
    console.print()

    plot_comparison_grid(test_X, predictions, test_labels.tolist(),
                         "Misclassified Digits", console, show_correct=False)

    # Final summary
    console.print()
    console.print("=" * 60, style="bold")
    console.print("  SUMMARY", style="bold cyan")
    console.print("=" * 60, style="bold")

    results_table = Table(show_header=False, box=None, padding=(0, 2))
    results_table.add_row("Final accuracy:", f"[bold green]{accuracy*100:.1f}%[/bold green]")
    results_table.add_row("", "")
    results_table.add_row("Training samples:", "60,000")
    results_table.add_row("Test samples:", "10,000")
    results_table.add_row("Total time:", f"{total_time:.1f}s")
    console.print(results_table)

    console.print()
    console.print("=" * 60, style="bold")

    if accuracy >= 0.97:
        console.print("  [bold green]OUTSTANDING! >97% accuracy on MNIST![/bold green]")
    elif accuracy >= 0.95:
        console.print("  [bold green]EXCELLENT! >95% accuracy - production quality![/bold green]")
    elif accuracy >= 0.90:
        console.print("  [bold green]GREAT! >90% accuracy on real handwritten digits![/bold green]")
    elif accuracy >= 0.80:
        console.print("  [bold yellow]GOOD! Your model is learning well.[/bold yellow]")
    else:
        console.print("  [bold red]Keep training or adjust hyperparameters.[/bold red]")

    console.print("=" * 60, style="bold")

    console.print()
    console.print("  [dim]Your from-scratch autodiff engine just trained on 60,000[/dim]")
    console.print("  [dim]handwritten digits and can now recognize them![/dim]")
    console.print()


if __name__ == "__main__":
    main()
