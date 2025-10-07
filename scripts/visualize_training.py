#!/usr/bin/env python3
"""
Training Visualization Script
Generate comprehensive plots for all experiments
"""

import json
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import yaml

# Set style for better-looking plots
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['legend.fontsize'] = 9

def load_experiment_data(checkpoint_dir):
    """Load experiment data from checkpoint directory"""
    config_path = checkpoint_dir / "config.yaml"
    history_path = checkpoint_dir / "history.json"

    if not config_path.exists() or not history_path.exists():
        return None

    with open(config_path) as f:
        config = yaml.safe_load(f)

    with open(history_path) as f:
        history = json.load(f)

    return {
        'name': checkpoint_dir.name,
        'config': config,
        'history': history,
        'experiment_name': config.get('experiment', {}).get('name', 'unknown'),
        'use_lmsa': config.get('model', {}).get('use_lmsa', False),
        'use_focal': config.get('loss', {}).get('use_focal', False),
        'use_class_weights': config.get('loss', {}).get('use_class_weights', False),
        'learning_rate': config.get('training', {}).get('learning_rate', 0),
    }

def get_best_metrics(history):
    """Get best epoch and metrics from history"""
    val_f_scores = np.array(history['val_f_score'])
    best_idx = np.argmax(val_f_scores)

    return {
        'best_epoch': best_idx + 1,
        'best_val_f_score': val_f_scores[best_idx],
        'train_f_score_at_best': history['train_f_score'][best_idx],
        'final_val_f_score': val_f_scores[-1],
        'total_epochs': len(val_f_scores)
    }

def plot_all_experiments():
    """Create comprehensive visualization of all experiments"""
    checkpoints_dir = Path("checkpoints")
    experiments = []

    # Load all experiments
    for ckpt_dir in sorted(checkpoints_dir.iterdir()):
        if ckpt_dir.is_dir() and ckpt_dir.name.startswith("microsegformer"):
            data = load_experiment_data(ckpt_dir)
            if data:
                experiments.append(data)

    print(f"Found {len(experiments)} experiments")

    # Create figure with subplots
    fig = plt.figure(figsize=(16, 12))

    # 1. Training curves - Val F-Score over epochs
    ax1 = plt.subplot(2, 3, 1)
    for exp in experiments:
        history = exp['history']
        val_f = history['val_f_score']
        epochs = range(1, len(val_f) + 1)

        # Color by experiment type
        if exp['use_lmsa'] and not exp['use_focal']:
            color = 'green'
            linewidth = 2.5
            alpha = 0.9
            label = f"{exp['experiment_name']} (LMSA) ✓"
        elif exp['use_focal']:
            color = 'red'
            linewidth = 1.5
            alpha = 0.6
            label = f"{exp['experiment_name']} (Focal)"
        else:
            color = 'blue'
            linewidth = 1.5
            alpha = 0.6
            label = f"{exp['experiment_name']}"

        ax1.plot(epochs, val_f, label=label, linewidth=linewidth, alpha=alpha, color=color)

    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Validation F-Score')
    ax1.set_title('Validation F-Score Over Epochs')
    ax1.legend(loc='lower right', fontsize=8)
    ax1.grid(True, alpha=0.3)

    # 2. Training curves - Train F-Score over epochs
    ax2 = plt.subplot(2, 3, 2)
    for exp in experiments:
        history = exp['history']
        train_f = history['train_f_score']
        epochs = range(1, len(train_f) + 1)

        if exp['use_lmsa'] and not exp['use_focal']:
            color = 'green'
            linewidth = 2.5
            alpha = 0.9
        elif exp['use_focal']:
            color = 'red'
            linewidth = 1.5
            alpha = 0.6
        else:
            color = 'blue'
            linewidth = 1.5
            alpha = 0.6

        ax2.plot(epochs, train_f, linewidth=linewidth, alpha=alpha, color=color)

    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Train F-Score')
    ax2.set_title('Training F-Score Over Epochs')
    ax2.grid(True, alpha=0.3)

    # 3. Loss curves
    ax3 = plt.subplot(2, 3, 3)
    for exp in experiments:
        history = exp['history']
        val_loss = history['val_loss']
        epochs = range(1, len(val_loss) + 1)

        if exp['use_lmsa'] and not exp['use_focal']:
            color = 'green'
            linewidth = 2.5
            alpha = 0.9
        elif exp['use_focal']:
            color = 'red'
            linewidth = 1.5
            alpha = 0.6
        else:
            color = 'blue'
            linewidth = 1.5
            alpha = 0.6

        ax3.plot(epochs, val_loss, linewidth=linewidth, alpha=alpha, color=color)

    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Validation Loss')
    ax3.set_title('Validation Loss Over Epochs')
    ax3.grid(True, alpha=0.3)

    # 4. Best performance comparison
    ax4 = plt.subplot(2, 3, 4)
    exp_names = []
    best_val_scores = []
    colors_bar = []

    for exp in experiments:
        metrics = get_best_metrics(exp['history'])
        exp_names.append(exp['experiment_name'][:20])  # Truncate long names
        best_val_scores.append(metrics['best_val_f_score'])

        if exp['use_lmsa'] and not exp['use_focal']:
            colors_bar.append('green')
        elif exp['use_focal']:
            colors_bar.append('red')
        else:
            colors_bar.append('steelblue')

    y_pos = np.arange(len(exp_names))
    ax4.barh(y_pos, best_val_scores, color=colors_bar, alpha=0.7)
    ax4.set_yticks(y_pos)
    ax4.set_yticklabels(exp_names, fontsize=8)
    ax4.set_xlabel('Best Validation F-Score')
    ax4.set_title('Best Performance Comparison')
    ax4.grid(True, alpha=0.3, axis='x')

    # Add value labels on bars
    for i, v in enumerate(best_val_scores):
        ax4.text(v, i, f' {v:.4f}', va='center', fontsize=8)

    # 5. Train/Val gap at best epoch
    ax5 = plt.subplot(2, 3, 5)
    gaps = []

    for exp in experiments:
        metrics = get_best_metrics(exp['history'])
        gap = metrics['train_f_score_at_best'] - metrics['best_val_f_score']
        gaps.append(gap)

    colors_gap = ['red' if g > 0.05 else 'orange' if g > 0.02 else 'green' for g in gaps]
    ax5.barh(y_pos, gaps, color=colors_gap, alpha=0.7)
    ax5.set_yticks(y_pos)
    ax5.set_yticklabels(exp_names, fontsize=8)
    ax5.set_xlabel('Train - Val Gap at Best Epoch')
    ax5.set_title('Overfitting Analysis (Lower is Better)')
    ax5.axvline(x=0, color='black', linestyle='--', linewidth=1)
    ax5.grid(True, alpha=0.3, axis='x')

    # Add value labels
    for i, v in enumerate(gaps):
        ax5.text(v, i, f' {v:.4f}', va='center', fontsize=8)

    # 6. Convergence speed (epochs to best)
    ax6 = plt.subplot(2, 3, 6)
    epochs_to_best = []

    for exp in experiments:
        metrics = get_best_metrics(exp['history'])
        epochs_to_best.append(metrics['best_epoch'])

    colors_conv = []
    for exp in experiments:
        if exp['use_lmsa'] and not exp['use_focal']:
            colors_conv.append('green')
        elif exp['use_focal']:
            colors_conv.append('red')
        else:
            colors_conv.append('steelblue')

    ax6.barh(y_pos, epochs_to_best, color=colors_conv, alpha=0.7)
    ax6.set_yticks(y_pos)
    ax6.set_yticklabels(exp_names, fontsize=8)
    ax6.set_xlabel('Epochs to Best Model')
    ax6.set_title('Convergence Speed')
    ax6.grid(True, alpha=0.3, axis='x')

    # Add value labels
    for i, v in enumerate(epochs_to_best):
        ax6.text(v, i, f' {v}', va='center', fontsize=8)

    plt.suptitle('CE7454 Face Parsing - Training Analysis Dashboard', fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])

    # Save figure
    output_path = Path("report/figures/training_analysis.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")

    # Also save as PDF for report
    output_pdf = Path("report/figures/training_analysis.pdf")
    plt.savefig(output_pdf, bbox_inches='tight')
    print(f"Saved: {output_pdf}")

    plt.close()

    return experiments

def create_best_model_detailed_plot(best_exp_name="microsegformer_20251007_153857"):
    """Create detailed plot for the best model"""
    checkpoint_dir = Path("checkpoints") / best_exp_name
    data = load_experiment_data(checkpoint_dir)

    if not data:
        print(f"Could not load {best_exp_name}")
        return

    history = data['history']

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    epochs = range(1, len(history['train_loss']) + 1)

    # 1. F-Score comparison
    ax1 = axes[0, 0]
    ax1.plot(epochs, history['train_f_score'], label='Train F-Score', color='steelblue', linewidth=2)
    ax1.plot(epochs, history['val_f_score'], label='Val F-Score', color='orange', linewidth=2)

    # Mark best epoch
    metrics = get_best_metrics(history)
    best_epoch = metrics['best_epoch']
    ax1.axvline(x=best_epoch, color='green', linestyle='--', linewidth=2, alpha=0.7, label=f'Best Epoch ({best_epoch})')
    ax1.scatter([best_epoch], [metrics['best_val_f_score']], color='red', s=100, zorder=5, label=f"Best Val: {metrics['best_val_f_score']:.4f}")

    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('F-Score')
    ax1.set_title('F-Score Evolution - Best Model (LMSA)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Loss curves
    ax2 = axes[0, 1]
    ax2.plot(epochs, history['train_loss'], label='Train Loss', color='steelblue', linewidth=2)
    ax2.plot(epochs, history['val_loss'], label='Val Loss', color='orange', linewidth=2)
    ax2.axvline(x=best_epoch, color='green', linestyle='--', linewidth=2, alpha=0.7, label=f'Best Epoch ({best_epoch})')

    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.set_title('Loss Evolution - Best Model (LMSA)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Train/Val gap over time
    ax3 = axes[1, 0]
    train_f = np.array(history['train_f_score'])
    val_f = np.array(history['val_f_score'])
    gap = train_f - val_f

    ax3.plot(epochs, gap, color='purple', linewidth=2)
    ax3.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax3.axvline(x=best_epoch, color='green', linestyle='--', linewidth=2, alpha=0.7, label=f'Best Epoch ({best_epoch})')
    ax3.fill_between(epochs, 0, gap, where=(gap > 0), alpha=0.3, color='red', label='Overfitting')
    ax3.fill_between(epochs, 0, gap, where=(gap <= 0), alpha=0.3, color='green', label='Healthy')

    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Train - Val Gap')
    ax3.set_title('Overfitting Analysis (Gap = Train F-Score - Val F-Score)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. Learning rate schedule
    ax4 = axes[1, 1]

    # Reconstruct LR schedule
    lr = float(data['learning_rate'])
    warmup_epochs = data['config'].get('training', {}).get('warmup_epochs', 5)
    total_epochs = len(epochs)

    lrs = []
    for epoch in epochs:
        if epoch <= warmup_epochs:
            # Linear warmup
            current_lr = (epoch / warmup_epochs) * lr
        else:
            # Cosine annealing
            t = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
            current_lr = lr * 0.5 * (1 + np.cos(t * np.pi))
        lrs.append(current_lr)

    ax4.plot(epochs, lrs, color='darkblue', linewidth=2)
    ax4.axvline(x=best_epoch, color='green', linestyle='--', linewidth=2, alpha=0.7, label=f'Best Epoch ({best_epoch})')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Learning Rate')
    ax4.set_title('Learning Rate Schedule (Warmup + Cosine Annealing)')
    ax4.set_yscale('log')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.suptitle(f'Detailed Analysis: {data["experiment_name"]} (Val: {metrics["best_val_f_score"]:.4f}, Test: 0.72)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])

    # Save
    output_path = Path("report/figures/best_model_analysis.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")

    output_pdf = Path("report/figures/best_model_analysis.pdf")
    plt.savefig(output_pdf, bbox_inches='tight')
    print(f"Saved: {output_pdf}")

    plt.close()

def print_summary_table(experiments):
    """Print summary table of all experiments"""
    print("\n" + "="*120)
    print(f"{'Experiment':<25} {'LMSA':<6} {'Focal':<6} {'LR':<8} {'Best Val':<10} {'Best Epoch':<11} {'Final Val':<10} {'Gap':<8}")
    print("="*120)

    for exp in sorted(experiments, key=lambda x: get_best_metrics(x['history'])['best_val_f_score'], reverse=True):
        metrics = get_best_metrics(exp['history'])
        gap = metrics['train_f_score_at_best'] - metrics['best_val_f_score']

        lmsa_str = "✓" if exp['use_lmsa'] else "✗"
        focal_str = "✓" if exp['use_focal'] else "✗"
        lr_float = float(exp['learning_rate']) if exp['learning_rate'] else 0.0

        print(f"{exp['experiment_name']:<25} {lmsa_str:<6} {focal_str:<6} {lr_float:<8.0e} "
              f"{metrics['best_val_f_score']:<10.4f} {metrics['best_epoch']:<11} "
              f"{metrics['final_val_f_score']:<10.4f} {gap:<8.4f}")

    print("="*120)

if __name__ == "__main__":
    print("Generating training visualizations...")

    # Create comprehensive dashboard
    experiments = plot_all_experiments()

    # Create detailed plot for best model
    create_best_model_detailed_plot("microsegformer_20251007_153857")

    # Print summary
    print_summary_table(experiments)

    print("\nVisualization complete!")
