#!/usr/bin/env python3
"""
MLP Demo - Watch your neural network learn!

Run with: uv run python tests/cases/demo_mlp.py
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from rich.console import Console
from rich_pixels import Pixels
from PIL import Image
import io
import sys
import time


def generate_circles_data(n_points: int, noise: float = 0.1):
    """Generate concentric circles dataset."""
    np.random.seed(42)
    n_inner = n_points // 2
    n_outer = n_points - n_inner

    theta_inner = np.random.uniform(0, 2 * np.pi, n_inner)
    r_inner = 0.4 + np.random.randn(n_inner) * noise
    X_inner = np.c_[r_inner * np.cos(theta_inner), r_inner * np.sin(theta_inner)]

    theta_outer = np.random.uniform(0, 2 * np.pi, n_outer)
    r_outer = 1.0 + np.random.randn(n_outer) * noise
    X_outer = np.c_[r_outer * np.cos(theta_outer), r_outer * np.sin(theta_outer)]

    X = np.vstack([X_inner, X_outer])
    y = np.zeros((n_points, 2))
    y[:n_inner, 0] = 1.0
    y[n_inner:, 1] = 1.0

    idx = np.random.permutation(n_points)
    return X[idx], y[idx]


def plot_decision_boundary(mlp, X_data, y_data, title, softmax_fn, Tensor, console):
    """Plot decision boundary with matplotlib and display in terminal."""
    bounds = [X_data[:, 0].min() - 0.3, X_data[:, 0].max() + 0.3,
              X_data[:, 1].min() - 0.3, X_data[:, 1].max() + 0.3]

    # Create high-res grid for decision boundary
    resolution = 200
    xx, yy = np.meshgrid(
        np.linspace(bounds[0], bounds[1], resolution),
        np.linspace(bounds[2], bounds[3], resolution)
    )
    grid = np.c_[xx.ravel(), yy.ravel()]

    logits = mlp(Tensor(grid))
    probs = softmax_fn(logits, axis=-1)
    preds = np.argmax(probs.data, axis=1).reshape(xx.shape)

    # Create matplotlib figure
    fig, ax = plt.subplots(figsize=(10, 10), facecolor='#1e1e1e')
    ax.set_facecolor('#1e1e1e')

    # Custom colormap for decision regions
    cmap_bg = ListedColormap(['#2d4a6f', '#6f3a3a'])  # Dark blue, dark red
    ax.contourf(xx, yy, preds, alpha=0.8, cmap=cmap_bg)

    # Separate data by class
    class_0 = y_data[:, 0] == 1.0
    X0, X1 = X_data[class_0], X_data[~class_0]

    # Plot data points
    ax.scatter(X0[:, 0], X0[:, 1], c='#4a90d9', edgecolors='white',
               s=80, linewidths=1.5, label='Inner circle', zorder=5)
    ax.scatter(X1[:, 0], X1[:, 1], c='#d94a4a', edgecolors='white',
               s=80, linewidths=1.5, label='Outer circle', zorder=5)

    ax.set_xlim(bounds[0], bounds[1])
    ax.set_ylim(bounds[2], bounds[3])
    ax.set_title(title, fontsize=14, color='white', pad=10)
    ax.legend(loc='upper right', facecolor='#2e2e2e', edgecolor='white',
              labelcolor='white')
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_color('white')

    # Save to buffer and display
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=600, bbox_inches='tight',
                facecolor='#1e1e1e', edgecolor='none')
    buf.seek(0)
    plt.close(fig)

    # Display in terminal using rich-pixels
    img = Image.open(buf)
    img = img.resize((80, 40), Image.Resampling.LANCZOS)
    pixels = Pixels.from_image(img)
    console.print(pixels)


def plot_loss_curve(losses):
    """Plot loss curve with plotext."""
    import plotext as pltx
    pltx.clf()
    pltx.cld()
    pltx.plot(losses, marker="braille", color="red")
    pltx.title("Training Loss")
    pltx.xlabel("Epoch")
    pltx.ylabel("Loss")
    pltx.plotsize(60, 15)
    pltx.show()


def main():
    console = Console()

    # Import user's implementation
    try:
        from smulgrad.engine import Tensor
        from smulgrad.nn import MLP, softmax, cross_entropy, SGD
    except ImportError as e:
        console.print(f"[red]Error importing smulgrad: {e}[/red]")
        console.print("Make sure you have implemented MLP, softmax, cross_entropy, and SGD")
        sys.exit(1)

    def compute_accuracy(mlp, X_data, y_data):
        X = Tensor(X_data)
        logits = mlp(X)
        probs = softmax(logits, axis=-1)
        preds = np.argmax(probs.data, axis=1)
        targets = np.argmax(y_data, axis=1)
        return np.mean(preds == targets)

    # Header
    console.print("=" * 60, style="bold")
    console.print("     SMULGRAD MLP DEMO - CONCENTRIC CIRCLES", style="bold cyan")
    console.print("=" * 60, style="bold")

    # Generate data
    n_points = 200
    X_data, y_data = generate_circles_data(n_points, noise=0.08)

    console.print(f"\n  Dataset: {n_points} points")
    console.print("  Task: Separate inner circle from outer circle")
    console.print("  (This requires learning a non-linear boundary!)")

    # Create model
    mlp = MLP(2, 32, 2)
    params = mlp.parameters()
    optimizer = SGD(params, lr=0.5)

    console.print(f"\n  Model: MLP(2 -> 32 -> 2)")
    console.print(f"  Parameters: {sum(p.data.size for p in params)}")

    initial_acc = compute_accuracy(mlp, X_data, y_data)
    console.print(f"  Initial accuracy: {initial_acc*100:.1f}% (random = 50%)")

    input("\n  Press Enter to see initial prediction...")

    # Show initial prediction
    console.print("\n" + "-" * 60)
    console.print("  INITIAL PREDICTION (before training)", style="bold yellow")
    console.print("-" * 60)
    console.print("\n  [dim]Blue region = predicts inner, Red region = predicts outer[/dim]")
    console.print("  [dim]Blue dots = inner data, Red dots = outer data[/dim]\n")
    plot_decision_boundary(mlp, X_data, y_data,
                          f"Before Training | Accuracy: {initial_acc*100:.1f}%",
                          softmax, Tensor, console)

    input("\n  Press Enter to start training...")

    # Training
    console.print("\n" + "-" * 60)
    console.print("  TRAINING", style="bold green")
    console.print("-" * 60)

    epochs = 200
    losses = []

    console.print(f"\n  {'Epoch':<8} {'Loss':<12} {'Accuracy':<12} {'Progress'}")
    console.print("  " + "-" * 50)

    for epoch in range(epochs):
        optimizer.zero_grad()
        logits = mlp(Tensor(X_data))
        loss = cross_entropy(logits, Tensor(y_data))
        loss.backward()
        optimizer.step()

        loss_val = float(loss.data)
        losses.append(loss_val)

        if epoch % 10 == 0 or epoch == epochs - 1:
            acc = compute_accuracy(mlp, X_data, y_data)
            bar_len = int(acc * 20)
            bar = "[green]" + "#" * bar_len + "[/green]" + "[dim]" + "-" * (20 - bar_len) + "[/dim]"
            console.print(f"  {epoch:<8} {loss_val:<12.4f} {acc*100:<11.1f}% [{bar}]")
            time.sleep(0.03)

    final_acc = compute_accuracy(mlp, X_data, y_data)

    input("\n  Training complete! Press Enter to see final prediction...")

    console.print("\n" + "-" * 60)
    console.print("  FINAL PREDICTION (after training)", style="bold yellow")
    console.print("-" * 60)
    console.print("\n  [dim]Blue region = predicts inner, Red region = predicts outer[/dim]")
    console.print("  [dim]Blue dots = inner data, Red dots = outer data[/dim]\n")
    plot_decision_boundary(mlp, X_data, y_data,
                          f"After Training | Accuracy: {final_acc*100:.1f}%",
                          softmax, Tensor, console)

    input("\n  Press Enter to see loss curve...")

    console.print("\n" + "-" * 60)
    console.print("  LOSS CURVE", style="bold magenta")
    console.print("-" * 60 + "\n")
    plot_loss_curve(losses)

    input("\n  Press Enter to see summary...")

    # Summary
    console.print("\n" + "=" * 60, style="bold")
    console.print("  SUMMARY", style="bold cyan")
    console.print("=" * 60, style="bold")
    console.print(f"\n  Initial accuracy: {initial_acc*100:.1f}%")
    console.print(f"  Final accuracy:   [green]{final_acc*100:.1f}%[/green]")
    console.print(f"  Improvement:      [green]+{(final_acc - initial_acc)*100:.1f}%[/green]")
    console.print(f"\n  Initial loss:     {losses[0]:.4f}")
    console.print(f"  Final loss:       [green]{losses[-1]:.4f}[/green]")
    console.print(f"  Reduction:        [green]{(1 - losses[-1]/losses[0])*100:.1f}%[/green]")

    console.print("\n" + "=" * 60, style="bold")
    if final_acc >= 0.95:
        console.print("  [bold green]EXCELLENT! Your MLP perfectly separated the circles![/bold green]")
    elif final_acc >= 0.85:
        console.print("  [bold green]GREAT! Your MLP learned the circular boundary![/bold green]")
    elif final_acc >= 0.70:
        console.print("  [bold yellow]GOOD! Your MLP is learning.[/bold yellow]")
    else:
        console.print("  [bold red]Keep working - try more epochs or hidden units.[/bold red]")
    console.print("=" * 60 + "\n", style="bold")


if __name__ == "__main__":
    main()
