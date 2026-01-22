"""
Visualization Functions for Causal Uplift Analysis

Industry-standard plots for uplift modeling:
1. Qini Curve - Cumulative incremental gains
2. Decile Comparison - Treatment vs Control by segment
3. Net Value Curve - ROI optimization
4. Causal Quadrants - Ground truth validation
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional
from pathlib import Path

# Set global style
sns.set_style('whitegrid')


def plot_qini_curve(y_true, uplift_score, treatment, plot=True):
    """
    Plots the Qini curve (Cumulative Incremental Gains).
    Standard industry metric for Uplift Modeling.
    
    Args:
        y_true: Binary outcome (1 = converted, 0 = not converted)
        uplift_score: Predicted CATE from T-Learner
        treatment: Binary treatment indicator (1 = treated, 0 = control)
        plot: If True, returns matplotlib figure. If False, returns data.
    
    Returns:
        fig or DataFrame with Qini curve data
    """
    # 1. Create a DataFrame and sort by uplift score (High to Low)
    data = pd.DataFrame({
        'y': y_true,
        'cate': uplift_score,
        'w': treatment
    }).sort_values('cate', ascending=False).reset_index(drop=True)
    
    # 2. Calculate cumulative sums
    data['nt'] = data['w'].cumsum()
    data['nc'] = (1 - data['w']).cumsum()
    data['rt'] = (data['y'] * data['w']).cumsum()
    data['rc'] = (data['y'] * (1 - data['w'])).cumsum()
    
    # 3. Calculate Qini Curve
    total_nt = data['w'].sum()
    total_nc = (1 - data['w']).sum()
    
    scaling_factor = total_nt / total_nc if total_nc > 0 else 1
    data['qini'] = data['rt'] - (data['rc'] * scaling_factor)
    
    # 4. Calculate Random Curve (Baseline)
    total_uplift = data['qini'].iloc[-1]
    data['random'] = np.linspace(0, total_uplift, len(data))
    
    # 5. Plot
    if plot:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(data.index, data['qini'], label='T-Learner Model', color='#1f77b4', linewidth=2)
        ax.plot(data.index, data['random'], label='Random Targeting', color='gray', linestyle='--')
        
        ax.set_title('Qini Curve (Cumulative Incremental Transactions)', fontsize=14)
        ax.set_xlabel('Number of Users Targeted (Sorted by Uplift)', fontsize=12)
        ax.set_ylabel('Cumulative Incremental Gains', fontsize=12)
        ax.legend()
        
        # Add "Area Under Qini" (AUUC) text
        auuc = np.trapz(data['qini'], dx=1)
        ax.text(0.05, 0.95, f'AUUC: {auuc:,.0f}', transform=ax.transAxes, 
                fontsize=12, verticalalignment='top', 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        return fig
    
    return data


def plot_uplift_by_decile(y_true, uplift_score, treatment, strategy='decile'):
    """
    Plots the Conversion Rate for Treatment vs. Control across deciles.
    Visual Validation: Top deciles should show Treatment >> Control.
    
    Args:
        y_true: Binary outcome
        uplift_score: Predicted CATE
        treatment: Binary treatment indicator
        strategy: 'decile' for 10 bins
    
    Returns:
        matplotlib figure
    """
    df = pd.DataFrame({
        'y': y_true,
        'treatment': treatment,
        'uplift_score': uplift_score
    })
    
    # Binning
    if strategy == 'decile':
        df['bin'] = pd.qcut(df['uplift_score'], 10, labels=False, duplicates='drop')
        df['bin'] = df['bin'].max() - df['bin']  # Reverse so 0 is highest uplift
    
    # Aggregation
    agg = df.groupby('bin').apply(lambda x: pd.Series({
        'Treatment': x[x['treatment'] == 1]['y'].mean(),
        'Control': x[x['treatment'] == 0]['y'].mean(),
        'Count': len(x)
    })).reset_index()
    
    # Plotting
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Melting for Seaborn side-by-side bars
    plot_data = agg.melt(id_vars='bin', value_vars=['Treatment', 'Control'], 
                         var_name='Group', value_name='Conversion Rate')
    
    sns.barplot(x='bin', y='Conversion Rate', hue='Group', data=plot_data, 
                palette={'Treatment': '#2ecc71', 'Control': '#e74c3c'}, ax=ax)
    
    ax.set_title("Actual Conversion Rate by Uplift Decile", fontsize=14)
    ax.set_xlabel("Uplift Decile (0 = Top 10% Persuadables)", fontsize=12)
    ax.set_ylabel("Conversion Rate", fontsize=12)
    ax.legend(title="Group")
    
    plt.tight_layout()
    return fig


def plot_net_value_curve(y_true, uplift_score, treatment, 
                        benefit_per_conversion=50, 
                        cost_per_treatment=2):
    """
    Plots the Cumulative Financial Impact (ROI).
    Business Value: Shows exactly where to stop spending money.
    
    Args:
        y_true: Binary outcome
        uplift_score: Predicted CATE
        treatment: Binary treatment indicator
        benefit_per_conversion: Revenue gained per incremental conversion ($)
        cost_per_treatment: Cost to treat one customer ($)
    
    Returns:
        matplotlib figure
    """
    df = pd.DataFrame({
        'y': y_true,
        'uplift': uplift_score,
        'w': treatment
    }).sort_values('uplift', ascending=False).reset_index(drop=True)
    
    # Cumulative stats
    df['treated_cumsum'] = np.arange(len(df)) + 1
    df['cumulative_uplift'] = df['uplift'].cumsum()
    df['cumulative_revenue'] = df['cumulative_uplift'] * benefit_per_conversion
    df['cumulative_cost'] = df['treated_cumsum'] * cost_per_treatment
    df['net_value'] = df['cumulative_revenue'] - df['cumulative_cost']
    
    # Find Optimal Threshold
    max_idx = df['net_value'].argmax()
    max_val = df['net_value'].max()
    opt_users = df['treated_cumsum'].iloc[max_idx]
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df['treated_cumsum'], df['net_value'], label='Policy Net Value', 
            color='#1f77b4', linewidth=2)
    
    # Highlight Peak
    ax.scatter([opt_users], [max_val], color='red', s=100, zorder=5, 
               label=f'Max Profit (${max_val:,.0f})')
    ax.axvline(opt_users, color='red', linestyle='--', alpha=0.5)
    
    ax.set_title(f"Financial Impact Analysis (ROI)\n"
                 f"Benefit: ${benefit_per_conversion} | Cost: ${cost_per_treatment}", 
                 fontsize=14)
    ax.set_xlabel("Number of Customers Targeted", fontsize=12)
    ax.set_ylabel("Net Incremental Value ($)", fontsize=12)
    ax.legend()
    
    # Add annotation for optimal point
    ax.annotate(f'Optimal: {opt_users:,} customers', 
                xy=(opt_users, max_val), xytext=(opt_users + len(df)*0.1, max_val*0.9),
                arrowprops=dict(arrowstyle='->', color='red'),
                fontsize=11, color='red')
    
    plt.tight_layout()
    return fig


def plot_causal_quadrants(uplift_score, true_uplift=None):
    """
    Plots the 2x2 Causal Matrix (Persuadables vs Sleeping Dogs).
    Only works if Ground Truth (true_uplift) is available.
    
    Args:
        uplift_score: Predicted CATE from model
        true_uplift: Ground truth CATE (only available with synthetic data)
    
    Returns:
        matplotlib figure
    """
    if true_uplift is not None:
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Scatter plot of Predicted vs True
        sns.scatterplot(x=true_uplift, y=uplift_score, alpha=0.1, color='purple', ax=ax)
        
        # Draw Quadrant Lines
        ax.axhline(0, color='black', linestyle='--', linewidth=1)
        ax.axvline(0, color='black', linestyle='--', linewidth=1)
        
        # Get axis limits for text placement
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        
        # Annotate Quadrants
        ax.text(xlim[1]*0.6, ylim[1]*0.8, "TRUE PERSUADABLES\n(Model Correct)", 
                fontsize=11, color='green', fontweight='bold', ha='center')
        ax.text(xlim[0]*0.6, ylim[0]*0.8, "TRUE SLEEPING DOGS\n(Model Correct)", 
                fontsize=11, color='green', fontweight='bold', ha='center')
        ax.text(xlim[0]*0.6, ylim[1]*0.8, "Safe Error\n(Missed Do-Not-Disturb)", 
                fontsize=10, color='orange', ha='center')
        ax.text(xlim[1]*0.6, ylim[0]*0.8, "Costly Error\n(Missed Opportunity)", 
                fontsize=10, color='red', ha='center')
        
        # Calculate and display correlation
        corr = np.corrcoef(true_uplift, uplift_score)[0, 1]
        ax.text(0.05, 0.95, f'Correlation: {corr:.3f}', transform=ax.transAxes,
                fontsize=12, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax.set_title("Model Validation: Predicted vs. True Uplift", fontsize=14)
        ax.set_xlabel("True Causal Effect (Ground Truth)", fontsize=12)
        ax.set_ylabel("Predicted Uplift Score (Model)", fontsize=12)
        
        plt.tight_layout()
        return fig
    else:
        # Fallback: Just show distribution of predicted scores
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(uplift_score, bins=50, kde=True, color='purple', ax=ax)
        ax.axvline(0, color='red', linestyle='--', label='Zero Uplift')
        ax.set_title("Distribution of Predicted Uplift Scores", fontsize=14)
        ax.set_xlabel("Predicted Uplift (CATE)", fontsize=12)
        ax.legend()
        plt.tight_layout()
        return fig


def save_all_plots(
    y_test, 
    cate_predictions, 
    treatment_test,
    true_uplift=None,
    output_dir: str = "outputs/plots"
):
    """
    Generate and save all visualization plots.
    
    Args:
        y_test: Test set outcomes
        cate_predictions: Model CATE predictions
        treatment_test: Test set treatment indicators
        true_uplift: Ground truth (optional, for synthetic data)
        output_dir: Directory to save plots
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # 1. Qini Curve
    fig1 = plot_qini_curve(y_test, cate_predictions, treatment_test)
    fig1.savefig(f'{output_dir}/qini_curve.png', dpi=150, bbox_inches='tight')
    print(f"Saved: {output_dir}/qini_curve.png")
    plt.close(fig1)
    
    # 2. Decile Comparison
    fig2 = plot_uplift_by_decile(y_test, cate_predictions, treatment_test)
    fig2.savefig(f'{output_dir}/decile_comparison.png', dpi=150, bbox_inches='tight')
    print(f"Saved: {output_dir}/decile_comparison.png")
    plt.close(fig2)
    
    # 3. ROI Analysis
    fig3 = plot_net_value_curve(y_test, cate_predictions, treatment_test)
    fig3.savefig(f'{output_dir}/roi_analysis.png', dpi=150, bbox_inches='tight')
    print(f"Saved: {output_dir}/roi_analysis.png")
    plt.close(fig3)
    
    # 4. Causal Quadrants
    fig4 = plot_causal_quadrants(cate_predictions, true_uplift)
    fig4.savefig(f'{output_dir}/causal_quadrants.png', dpi=150, bbox_inches='tight')
    print(f"Saved: {output_dir}/causal_quadrants.png")
    plt.close(fig4)
    
    print(f"\nAll plots saved to {output_dir}/")
