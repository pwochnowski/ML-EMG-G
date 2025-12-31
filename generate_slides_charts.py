#!/usr/bin/env python3
"""
Generate charts for EMG classification presentation slides.

Charts:
1. Model Comparison (LOSO) - Grouped bar chart
2. Preprocessing Configurations - Horizontal bar chart
3. Within-Subject CV Results - Heatmap
4. LOSO vs Within-Subject Comparison - Side-by-side bar chart
5. Cross-Dataset Comparison - Compare results across datasets
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# Set style for presentation-quality figures
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Output directory for charts
OUTPUT_DIR = Path("datasets/analysis/slides")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# All datasets
DATASETS = ['rami', 'db1', 'myo']
BASE_DIR = Path("datasets")


def load_data():
    """Load all relevant CSV files for all datasets."""
    all_data = {}

    for dataset in DATASETS:
        data = {}
        training_dir = BASE_DIR / dataset / "training"
        
        # LOSO model comparison - use real data if available, else synthetic
        loso_path = training_dir / "loso_model_comparison.csv"
        if loso_path.exists():
            data['loso_models'] = pd.read_csv(loso_path)
            print(f"  [{dataset}] Loaded real LOSO data")
        
        # Preprocessing evaluation - use real data if available, else synthetic
        preproc_path = training_dir / "preprocessing_eval.csv"
        if preproc_path.exists():
            data['preprocessing'] = pd.read_csv(preproc_path)
            print(f"  [{dataset}] Loaded real preprocessing data")
        
        # Within-subject CV results - use real data if available, else synthetic
        within_subject_path = training_dir / "loso_summary_within_subject.csv"
        if within_subject_path.exists():
            data['within_subject'] = pd.read_csv(within_subject_path)
            print(f"  [{dataset}] Loaded real within-subject data")
        
        if data:
            all_data[dataset] = data
    
    return all_data


def chart_loso_model_comparison(data, dataset_name, save=True):
    """
    Chart 1: Grouped bar chart comparing model performance on LOSO.
    """
    if 'loso_models' not in data:
        print(f"No LOSO model comparison data found for {dataset_name}")
        return
    
    df = data['loso_models'].copy()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create bar chart
    x = np.arange(len(df))
    bars = ax.bar(x, df['mean_acc'] * 100, yerr=df['std_acc'] * 100, 
                  capsize=5, color=sns.color_palette("husl", len(df)),
                  edgecolor='black', linewidth=1.2)
    
    # Customize
    ax.set_xlabel('Model', fontsize=14, fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontsize=14, fontweight='bold')
    ax.set_title(f'LOSO Cross-Validation: Model Comparison\n({dataset_name.upper()} Dataset)', 
                 fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(df['model'].str.upper(), fontsize=12)
    ax.set_ylim(0, 100)
    
    # Add value labels on bars
    for bar, acc, std in zip(bars, df['mean_acc'], df['std_acc']):
        height = bar.get_height()
        ax.annotate(f'{acc*100:.1f}%\n±{std*100:.1f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Add horizontal line for reference
    ax.axhline(y=50, color='red', linestyle='--', alpha=0.5, label='Random guess')
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    
    if save:
        fig.savefig(OUTPUT_DIR / f"{dataset_name}_1_loso_model_comparison.png", dpi=300, bbox_inches='tight')
        fig.savefig(OUTPUT_DIR / f"{dataset_name}_1_loso_model_comparison.pdf", bbox_inches='tight')
        print(f"Saved: {dataset_name}_1_loso_model_comparison.png/pdf")
    
    return fig


def chart_preprocessing_comparison(data, dataset_name, save=True):
    """
    Chart 2: Horizontal bar chart comparing preprocessing configurations.
    """
    if 'preprocessing' not in data:
        print(f"No preprocessing data found for {dataset_name}")
        return
    
    df = data['preprocessing'].copy()
    df = df.sort_values('accuracy', ascending=True)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create horizontal bar chart
    y = np.arange(len(df))
    colors = sns.color_palette("viridis", len(df))
    bars = ax.barh(y, df['accuracy'] * 100, xerr=df['std'] * 100,
                   capsize=4, color=colors, edgecolor='black', linewidth=1.2)
    
    # Customize
    ax.set_ylabel('Preprocessing Configuration', fontsize=14, fontweight='bold')
    ax.set_xlabel('Accuracy (%)', fontsize=14, fontweight='bold')
    ax.set_title(f'Impact of Signal Preprocessing on Classification\n({dataset_name.upper()} Dataset)', 
                 fontsize=16, fontweight='bold')
    ax.set_yticks(y)
    
    # Clean up config names for display
    labels = df['config_name'].str.replace('_', ' ').str.title()
    ax.set_yticklabels(labels, fontsize=11)
    ax.set_xlim(0, 100)
    
    # Add value labels
    for bar, acc in zip(bars, df['accuracy']):
        width = bar.get_width()
        ax.annotate(f'{acc*100:.2f}%',
                    xy=(width, bar.get_y() + bar.get_height() / 2),
                    xytext=(5, 0), textcoords="offset points",
                    ha='left', va='center', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    
    if save:
        fig.savefig(OUTPUT_DIR / f"{dataset_name}_2_preprocessing_comparison.png", dpi=300, bbox_inches='tight')
        fig.savefig(OUTPUT_DIR / f"{dataset_name}_2_preprocessing_comparison.pdf", bbox_inches='tight')
        print(f"Saved: {dataset_name}_2_preprocessing_comparison.png/pdf")
    
    return fig


def chart_within_subject_heatmap(data, dataset_name, save=True):
    """
    Chart 3: Heatmap showing within-subject CV accuracy per subject and model.
    """
    if 'within_subject' not in data:
        print(f"No within-subject data found for {dataset_name}")
        return
    
    df = data['within_subject'].copy()
    
    # Pivot to create matrix
    pivot_df = df.pivot(index='subject', columns='model', values='mean_acc') * 100
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create heatmap
    sns.heatmap(pivot_df, annot=True, fmt='.1f', cmap='RdYlGn',
                vmin=90, vmax=100, ax=ax,
                cbar_kws={'label': 'Accuracy (%)', 'shrink': 0.8},
                linewidths=0.5, linecolor='white')
    
    ax.set_title(f'Within-Subject CV Accuracy (%) by Subject and Model\n({dataset_name.upper()} Dataset)', 
                 fontsize=16, fontweight='bold')
    ax.set_xlabel('Model', fontsize=14, fontweight='bold')
    ax.set_ylabel('Subject', fontsize=14, fontweight='bold')
    
    # Rotate x labels
    plt.xticks(rotation=0, fontsize=12)
    plt.yticks(rotation=0, fontsize=10)
    
    plt.tight_layout()
    
    if save:
        fig.savefig(OUTPUT_DIR / f"{dataset_name}_3_within_subject_heatmap.png", dpi=300, bbox_inches='tight')
        fig.savefig(OUTPUT_DIR / f"{dataset_name}_3_within_subject_heatmap.pdf", bbox_inches='tight')
        print(f"Saved: {dataset_name}_3_within_subject_heatmap.png/pdf")
    
    return fig


def chart_loso_vs_within_subject(data, dataset_name, save=True):
    """
    Chart 4: Side-by-side comparison of LOSO vs Within-Subject accuracy.
    Shows the generalization gap.
    """
    if 'loso_models' not in data or 'within_subject' not in data:
        print(f"Missing data for comparison chart for {dataset_name}")
        return
    
    # Calculate mean within-subject accuracy per model
    within_subject_mean = data['within_subject'].groupby('model')['mean_acc'].mean() * 100
    
    # Get LOSO accuracy
    loso_df = data['loso_models'].set_index('model')['mean_acc'] * 100
    
    # Find common models
    common_models = list(set(within_subject_mean.index) & set(loso_df.index))
    
    if not common_models:
        print(f"No common models found between LOSO and within-subject results for {dataset_name}")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(common_models))
    width = 0.35
    
    within_vals = [within_subject_mean[m] for m in common_models]
    loso_vals = [loso_df[m] for m in common_models]
    
    bars1 = ax.bar(x - width/2, within_vals, width, label='Within-Subject CV',
                   color='#2ecc71', edgecolor='black', linewidth=1.2)
    bars2 = ax.bar(x + width/2, loso_vals, width, label='LOSO (Cross-Subject)',
                   color='#e74c3c', edgecolor='black', linewidth=1.2)
    
    # Customize
    ax.set_xlabel('Model', fontsize=14, fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontsize=14, fontweight='bold')
    ax.set_title(f'Generalization Gap: Within-Subject vs LOSO\n({dataset_name.upper()} Dataset)', 
                 fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([m.upper() for m in common_models], fontsize=12)
    ax.set_ylim(0, 105)
    ax.legend(loc='upper right', fontsize=12)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}%',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Add gap annotations
    for i, model in enumerate(common_models):
        gap = within_vals[i] - loso_vals[i]
        mid_y = (within_vals[i] + loso_vals[i]) / 2
        ax.annotate(f'Δ{gap:.1f}%', xy=(i, mid_y), ha='center', va='center',
                    fontsize=9, color='gray', style='italic')
    
    plt.tight_layout()
    
    if save:
        fig.savefig(OUTPUT_DIR / f"{dataset_name}_4_loso_vs_within_subject.png", dpi=300, bbox_inches='tight')
        fig.savefig(OUTPUT_DIR / f"{dataset_name}_4_loso_vs_within_subject.pdf", bbox_inches='tight')
        print(f"Saved: {dataset_name}_4_loso_vs_within_subject.png/pdf")
    
    return fig


def chart_within_subject_boxplot(data, dataset_name, save=True):
    """
    Chart 5: Box plot showing distribution of within-subject accuracy across subjects.
    """
    if 'within_subject' not in data:
        print(f"No within-subject data found for {dataset_name}")
        return
    
    df = data['within_subject'].copy()
    df['mean_acc'] = df['mean_acc'] * 100
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create box plot
    sns.boxplot(data=df, x='model', y='mean_acc', ax=ax, palette="husl")
    sns.stripplot(data=df, x='model', y='mean_acc', ax=ax, 
                  color='black', alpha=0.5, size=6)
    
    ax.set_xlabel('Model', fontsize=14, fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontsize=14, fontweight='bold')
    ax.set_title(f'Within-Subject CV: Accuracy Distribution Across Subjects\n({dataset_name.upper()} Dataset)', 
                 fontsize=16, fontweight='bold')
    ax.set_ylim(85, 100)
    
    # Update x-tick labels
    ax.set_xticklabels([t.get_text().upper() for t in ax.get_xticklabels()], fontsize=12)
    
    plt.tight_layout()
    
    if save:
        fig.savefig(OUTPUT_DIR / f"{dataset_name}_5_within_subject_boxplot.png", dpi=300, bbox_inches='tight')
        fig.savefig(OUTPUT_DIR / f"{dataset_name}_5_within_subject_boxplot.pdf", bbox_inches='tight')
        print(f"Saved: {dataset_name}_5_within_subject_boxplot.png/pdf")
    
    return fig


def chart_summary_table(data, dataset_name, save=True):
    """
    Chart 6: Summary table as a figure for slides.
    """
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axis('off')
    
    # Build summary data
    summary_data = []
    
    if 'loso_models' in data:
        best_loso = data['loso_models'].loc[data['loso_models']['mean_acc'].idxmax()]
        summary_data.append(['Best LOSO Model', best_loso['model'].upper(), 
                            f"{best_loso['mean_acc']*100:.1f}% ± {best_loso['std_acc']*100:.1f}%"])
    
    if 'within_subject' in data:
        ws_means = data['within_subject'].groupby('model')['mean_acc'].mean()
        best_ws_model = ws_means.idxmax()
        best_ws_acc = ws_means.max()
        summary_data.append(['Best Within-Subject Model', best_ws_model.upper(), 
                            f"{best_ws_acc*100:.1f}%"])
    
    if 'preprocessing' in data:
        best_preproc = data['preprocessing'].loc[data['preprocessing']['accuracy'].idxmax()]
        summary_data.append(['Best Preprocessing', best_preproc['config_name'].replace('_', ' ').title(),
                            f"{best_preproc['accuracy']*100:.2f}%"])
    
    if summary_data:
        table = ax.table(cellText=summary_data,
                        colLabels=['Metric', 'Configuration', 'Accuracy'],
                        loc='center',
                        cellLoc='center',
                        colColours=['#3498db'] * 3)
        
        table.auto_set_font_size(False)
        table.set_fontsize(14)
        table.scale(1.2, 2)
        
        # Style header
        for i in range(3):
            table[(0, i)].set_text_props(color='white', fontweight='bold')
        
        ax.set_title(f'Summary of Best Results ({dataset_name.upper()} Dataset)', 
                     fontsize=18, fontweight='bold', pad=20)
    
    plt.tight_layout()
    
    if save:
        fig.savefig(OUTPUT_DIR / f"{dataset_name}_6_summary_table.png", dpi=300, bbox_inches='tight')
        fig.savefig(OUTPUT_DIR / f"{dataset_name}_6_summary_table.pdf", bbox_inches='tight')
        print(f"Saved: {dataset_name}_6_summary_table.png/pdf")
    
    return fig


def chart_cross_dataset_comparison(all_data, save=True):
    """
    Chart 7: Cross-dataset comparison showing LOSO accuracy across all datasets.
    """
    fig, axes = plt.subplots(2, 1, figsize=(10, 12))
    
    # --- Top plot: Best model LOSO accuracy per dataset ---
    ax1 = axes[0]
    datasets_with_loso = []
    best_accs = []
    best_stds = []
    best_models = []
    
    for dataset, data in all_data.items():
        if 'loso_models' in data:
            best = data['loso_models'].loc[data['loso_models']['mean_acc'].idxmax()]
            datasets_with_loso.append(dataset.upper())
            best_accs.append(best['mean_acc'] * 100)
            best_stds.append(best['std_acc'] * 100)
            best_models.append(best['model'].upper())
    
    if datasets_with_loso:
        x = np.arange(len(datasets_with_loso))
        colors = sns.color_palette("Set2", len(datasets_with_loso))
        bars = ax1.bar(x, best_accs, yerr=best_stds, capsize=5, 
                       color=colors, edgecolor='black', linewidth=1.2)
        
        ax1.set_xlabel('Dataset', fontsize=14, fontweight='bold')
        ax1.set_ylabel('LOSO Accuracy (%)', fontsize=14, fontweight='bold')
        ax1.set_title('Best LOSO Accuracy by Dataset', fontsize=16, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(datasets_with_loso, fontsize=12)
        ax1.set_ylim(0, 100)
        ax1.axhline(y=50, color='red', linestyle='--', alpha=0.5, label='Random guess')
        ax1.legend()
        
        for bar, acc, model in zip(bars, best_accs, best_models):
            height = bar.get_height()
            ax1.annotate(f'{acc:.1f}%\n({model})',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # --- Bottom plot: Generalization gap per dataset ---
    ax2 = axes[1]
    gap_data = []
    
    for dataset, data in all_data.items():
        if 'loso_models' in data and 'within_subject' in data:
            ws_mean = data['within_subject']['mean_acc'].mean() * 100
            loso_best = data['loso_models']['mean_acc'].max() * 100
            gap = ws_mean - loso_best
            gap_data.append({
                'dataset': dataset.upper(),
                'within_subject': ws_mean,
                'loso': loso_best,
                'gap': gap
            })
    
    if gap_data:
        gap_df = pd.DataFrame(gap_data)
        x = np.arange(len(gap_df))
        width = 0.35
        
        bars1 = ax2.bar(x - width/2, gap_df['within_subject'], width, 
                       label='Within-Subject CV', color='#2ecc71', edgecolor='black')
        bars2 = ax2.bar(x + width/2, gap_df['loso'], width, 
                       label='LOSO', color='#e74c3c', edgecolor='black')
        
        ax2.set_xlabel('Dataset', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Accuracy (%)', fontsize=14, fontweight='bold')
        ax2.set_title('Generalization Gap Across Datasets', fontsize=16, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(gap_df['dataset'], fontsize=12)
        ax2.set_ylim(0, 105)
        ax2.legend(loc='upper right', fontsize=11)
        
        # Add gap annotations
        for i, row in gap_df.iterrows():
            mid_y = (row['within_subject'] + row['loso']) / 2
            ax2.annotate(f'Δ{row["gap"]:.1f}%', xy=(i, mid_y), ha='center', 
                        fontsize=10, color='gray', style='italic', fontweight='bold')
    
    plt.tight_layout()
    
    if save:
        fig.savefig(OUTPUT_DIR / "cross_dataset_comparison.png", dpi=300, bbox_inches='tight')
        fig.savefig(OUTPUT_DIR / "cross_dataset_comparison.pdf", bbox_inches='tight')
        print(f"Saved: cross_dataset_comparison.png/pdf")
    
    return fig


def chart_all_models_all_datasets(all_data, save=True):
    """
    Chart 8: Grouped bar chart comparing all models across all datasets.
    """
    # Collect data
    plot_data = []
    for dataset, data in all_data.items():
        if 'loso_models' in data:
            for _, row in data['loso_models'].iterrows():
                plot_data.append({
                    'Dataset': dataset.upper(),
                    'Model': row['model'].upper(),
                    'Accuracy': row['mean_acc'] * 100,
                    'Std': row['std_acc'] * 100
                })
    
    if not plot_data:
        print("No data for all models comparison")
        return
    
    df = pd.DataFrame(plot_data)
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Create grouped bar chart
    datasets = df['Dataset'].unique()
    models = df['Model'].unique()
    x = np.arange(len(models))
    width = 0.25
    offsets = np.linspace(-width, width, len(datasets))
    
    colors = sns.color_palette("Set2", len(datasets))
    
    for i, dataset in enumerate(datasets):
        subset = df[df['Dataset'] == dataset].set_index('Model')
        accs = [subset.loc[m, 'Accuracy'] if m in subset.index else 0 for m in models]
        stds = [subset.loc[m, 'Std'] if m in subset.index else 0 for m in models]
        
        bars = ax.bar(x + offsets[i], accs, width, yerr=stds, label=dataset,
                     color=colors[i], edgecolor='black', linewidth=1, capsize=3)
    
    ax.set_xlabel('Model', fontsize=14, fontweight='bold')
    ax.set_ylabel('LOSO Accuracy (%)', fontsize=14, fontweight='bold')
    ax.set_title('LOSO Performance: All Models Across All Datasets', fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=12)
    ax.set_ylim(0, 100)
    ax.axhline(y=50, color='red', linestyle='--', alpha=0.5, label='Random guess')
    ax.legend(loc='upper right', fontsize=11)
    
    plt.tight_layout()
    
    if save:
        fig.savefig(OUTPUT_DIR / "all_models_all_datasets.png", dpi=300, bbox_inches='tight')
        fig.savefig(OUTPUT_DIR / "all_models_all_datasets.pdf", bbox_inches='tight')
        print(f"Saved: all_models_all_datasets.png/pdf")
    
    return fig


def chart_cross_sensor_transfer(save=True):
    """
    Chart 9: Cross-sensor transfer learning results (baseline deep learning).
    Shows dramatic accuracy drop when transferring models between different hardware.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Data from the baseline deep learning study
    conditions = ['Same-Sensor\n(Intra-Domain)', 'Cross-Sensor\n(Inter-Domain)']
    accuracies = [90, 20]
    colors = ['#2ecc71', '#e74c3c']
    
    bars = ax.bar(conditions, accuracies, color=colors, edgecolor='black', linewidth=2, width=0.6)
    
    # Customize
    ax.set_ylabel('Accuracy (%)', fontsize=14, fontweight='bold')
    ax.set_title('Deep Learning Transfer: Same vs Cross-Sensor Performance\n(1D-CNN trained on individual datasets)', 
                 fontsize=16, fontweight='bold')
    ax.set_ylim(0, 105)
    
    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax.annotate(f'{acc}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 5), textcoords="offset points",
                    ha='center', va='bottom', fontsize=20, fontweight='bold')
    
    # Add reference lines
    ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5, linewidth=1.5)
    ax.text(1.4, 52, 'Random Guess', fontsize=10, color='gray', style='italic')
    
    # Add arrow showing the drop
    ax.annotate('', xy=(1, 25), xytext=(0, 85),
                arrowprops=dict(arrowstyle='->', color='black', lw=2))
    ax.text(0.5, 55, '70% drop!', fontsize=14, fontweight='bold', color='black',
            ha='center', rotation=-50)
    
    # Add explanatory text box
    textstr = 'Models overfit to hardware artifacts\n(sampling noise) rather than\nlearning anatomical muscle patterns'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.98, 0.02, textstr, transform=ax.transAxes, fontsize=11,
            verticalalignment='bottom', horizontalalignment='right', bbox=props)
    
    plt.tight_layout()
    
    if save:
        fig.savefig(OUTPUT_DIR / "cross_sensor_transfer.png", dpi=300, bbox_inches='tight')
        fig.savefig(OUTPUT_DIR / "cross_sensor_transfer.pdf", bbox_inches='tight')
        print(f"Saved: cross_sensor_transfer.png/pdf")
    
    return fig


def main():
    """Generate all charts."""
    print("=" * 60)
    print("Generating Presentation Charts for EMG Classification")
    print("=" * 60)
    
    # Load data
    print("\nLoading data...")
    all_data = load_data()
    print(f"Loaded datasets: {list(all_data.keys())}")
    
    # Generate per-dataset charts
    for dataset_name, data in all_data.items():
        print(f"\n--- Generating charts for {dataset_name.upper()} ---")
        chart_loso_model_comparison(data, dataset_name)
        chart_preprocessing_comparison(data, dataset_name)
        chart_within_subject_heatmap(data, dataset_name)
        chart_loso_vs_within_subject(data, dataset_name)
        chart_within_subject_boxplot(data, dataset_name)
        chart_summary_table(data, dataset_name)
    
    # Generate cross-dataset comparison charts
    print(f"\n--- Generating cross-dataset comparison charts ---")
    chart_cross_dataset_comparison(all_data)
    chart_all_models_all_datasets(all_data)
    chart_cross_sensor_transfer()
    
    print("-" * 40)
    print(f"\n✓ All charts saved to: {OUTPUT_DIR.absolute()}")
    print("\nChart files:")
    for f in sorted(OUTPUT_DIR.glob("*.png")):
        print(f"  - {f.name}")


if __name__ == "__main__":
    main()
