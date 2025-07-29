import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set style for better plots
plt.style.use('default')
sns.set_palette("husl")

def create_summary_visualizations():
    """Create summary visualizations for the analysis"""
    
    # Model comparison data
    model_data = {
        'FROG': {
            'buy': {'linear_r2': 0.988, 'power_r2': 0.840, 'beta': 0.000047},
            'sell': {'linear_r2': 0.968, 'power_r2': 0.923, 'beta': 0.000079}
        },
        'SOUN': {
            'buy': {'linear_r2': 0.844, 'power_r2': 0.601, 'beta': 0.000001},
            'sell': {'linear_r2': 0.969, 'power_r2': 0.655, 'beta': 0.000002}
        },
        'CRWV': {
            'buy': {'linear_r2': 0.995, 'power_r2': 0.851, 'beta': 0.000145},
            'sell': {'linear_r2': 0.973, 'power_r2': 0.746, 'beta': 0.000142}
        }
    }
    
    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Model R² Comparison
    tickers = list(model_data.keys())
    linear_r2_buy = [model_data[t]['buy']['linear_r2'] for t in tickers]
    linear_r2_sell = [model_data[t]['sell']['linear_r2'] for t in tickers]
    power_r2_buy = [model_data[t]['buy']['power_r2'] for t in tickers]
    power_r2_sell = [model_data[t]['sell']['power_r2'] for t in tickers]
    
    x = np.arange(len(tickers))
    width = 0.2
    
    ax1.bar(x - 1.5*width, linear_r2_buy, width, label='Linear (Buy)', alpha=0.8)
    ax1.bar(x - 0.5*width, linear_r2_sell, width, label='Linear (Sell)', alpha=0.8)
    ax1.bar(x + 0.5*width, power_r2_buy, width, label='Power (Buy)', alpha=0.8)
    ax1.bar(x + 1.5*width, power_r2_sell, width, label='Power (Sell)', alpha=0.8)
    
    ax1.set_xlabel('Ticker')
    ax1.set_ylabel('R² Value')
    ax1.set_title('Model Performance Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels(tickers)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Impact Sensitivity (Beta) Comparison
    beta_buy = [model_data[t]['buy']['beta'] for t in tickers]
    beta_sell = [model_data[t]['sell']['beta'] for t in tickers]
    
    x = np.arange(len(tickers))
    width = 0.35
    
    ax2.bar(x - width/2, beta_buy, width, label='Buy Side', alpha=0.8)
    ax2.bar(x + width/2, beta_sell, width, label='Sell Side', alpha=0.8)
    
    ax2.set_xlabel('Ticker')
    ax2.set_ylabel('Beta (Impact Sensitivity)')
    ax2.set_title('Impact Sensitivity by Ticker')
    ax2.set_xticks(x)
    ax2.set_xticklabels(tickers)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')  # Log scale due to large differences
    
    # 3. Theoretical Impact Function Shapes
    order_sizes = np.linspace(1, 200, 100)
    
    for i, ticker in enumerate(tickers):
        beta = model_data[ticker]['buy']['beta']
        alpha = 0.05  # Approximate baseline
        impact = beta * order_sizes + alpha
        ax3.plot(order_sizes, impact, label=f'{ticker} (β={beta:.6f})', linewidth=2)
    
    ax3.set_xlabel('Order Size (Shares)')
    ax3.set_ylabel('Expected Impact')
    ax3.set_title('Theoretical Impact Functions (Buy Side)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Strategy Performance Summary
    strategies = ['Uniform', 'Front-loaded', 'Back-loaded', 'VWAP-style']
    # For linear models, all strategies perform identically
    performance = [1.0, 1.0, 1.0, 1.0]  # Normalized performance
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']
    ax4.bar(strategies, performance, color=colors, alpha=0.8)
    ax4.set_ylabel('Relative Performance')
    ax4.set_title('Execution Strategy Performance\\n(Linear Impact Models)')
    ax4.set_ylim(0.98, 1.02)
    ax4.grid(True, alpha=0.3)
    
    # Add text annotation
    ax4.text(0.5, 0.5, 'All strategies yield\\nidentical results\\nfor linear impact models', 
             transform=ax4.transAxes, ha='center', va='center',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('analysis_summary.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Summary analysis plot saved: analysis_summary.png")

def create_impact_comparison():
    """Create detailed impact function comparison"""
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Parameters from analysis
    params = {
        'FROG': {'beta': 0.000047, 'alpha': 0.051516, 'color': '#FF6B6B'},
        'SOUN': {'beta': 0.000001, 'alpha': 0.005637, 'color': '#4ECDC4'},
        'CRWV': {'beta': 0.000145, 'alpha': 0.056103, 'color': '#45B7D1'}
    }
    
    order_sizes = np.linspace(1, 200, 200)
    
    for i, (ticker, param) in enumerate(params.items()):
        ax = axes[i]
        
        # Linear impact
        impact_linear = param['beta'] * order_sizes + param['alpha']
        
        # Square root impact (for comparison)
        impact_sqrt = param['beta'] * 50 * np.sqrt(order_sizes)  # Scaled for visibility
        
        # Power law impact (approximate)
        impact_power = param['beta'] * 20 * (order_sizes ** 0.8)  # Scaled for visibility
        
        ax.plot(order_sizes, impact_linear, label='Linear Model', 
                color=param['color'], linewidth=3)
        ax.plot(order_sizes, impact_sqrt, label='Square Root Model', 
                linestyle='--', alpha=0.7, linewidth=2)
        ax.plot(order_sizes, impact_power, label='Power Law Model', 
                linestyle=':', alpha=0.7, linewidth=2)
        
        ax.set_xlabel('Order Size (Shares)')
        ax.set_ylabel('Expected Slippage')
        ax.set_title(f'{ticker}: Impact Function Comparison\\n'
                    f'β = {param["beta"]:.6f}, α = {param["alpha"]:.6f}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add parameter text
        ax.text(0.05, 0.95, f'Linear: g(x) = {param["beta"]:.6f}x + {param["alpha"]:.3f}', 
                transform=ax.transAxes, va='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('impact_function_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Impact function comparison saved: impact_function_comparison.png")

def create_practical_insights():
    """Create practical insights visualization"""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Cost comparison for different order sizes
    order_sizes = [100, 500, 1000, 2000, 5000]
    
    frog_costs = [0.000047 * s + 0.051516 for s in order_sizes]
    soun_costs = [0.000001 * s + 0.005637 for s in order_sizes]
    crwv_costs = [0.000145 * s + 0.056103 for s in order_sizes]
    
    x = np.arange(len(order_sizes))
    width = 0.25
    
    ax1.bar(x - width, frog_costs, width, label='FROG', alpha=0.8)
    ax1.bar(x, soun_costs, width, label='SOUN', alpha=0.8)
    ax1.bar(x + width, crwv_costs, width, label='CRWV', alpha=0.8)
    
    ax1.set_xlabel('Order Size (Shares)')
    ax1.set_ylabel('Expected Impact')
    ax1.set_title('Total Impact by Order Size')
    ax1.set_xticks(x)
    ax1.set_xticklabels(order_sizes)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Marginal cost (per share impact)
    marginal_costs_frog = [frog_costs[i]/order_sizes[i] for i in range(len(order_sizes))]
    marginal_costs_soun = [soun_costs[i]/order_sizes[i] for i in range(len(order_sizes))]
    marginal_costs_crwv = [crwv_costs[i]/order_sizes[i] for i in range(len(order_sizes))]
    
    ax2.plot(order_sizes, marginal_costs_frog, 'o-', label='FROG', linewidth=2, markersize=8)
    ax2.plot(order_sizes, marginal_costs_soun, 's-', label='SOUN', linewidth=2, markersize=8)
    ax2.plot(order_sizes, marginal_costs_crwv, '^-', label='CRWV', linewidth=2, markersize=8)
    
    ax2.set_xlabel('Order Size (Shares)')
    ax2.set_ylabel('Impact per Share')
    ax2.set_title('Average Impact per Share')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')
    
    # 3. Time allocation patterns
    time_periods = np.arange(1, 391)  # 390 minutes
    uniform_allocation = np.full(390, 10000/390)
    
    # VWAP-style (U-shaped)
    x_norm = np.linspace(0, 1, 390)
    vwap_weights = 2 * (x_norm - 0.5)**2 + 0.5
    vwap_weights = vwap_weights / np.sum(vwap_weights)
    vwap_allocation = 10000 * vwap_weights
    
    # Front-loaded
    front_weights = np.exp(-0.01 * np.arange(390))
    front_weights = front_weights / np.sum(front_weights)
    front_allocation = 10000 * front_weights
    
    ax3.plot(time_periods[:100], uniform_allocation[:100], label='Uniform', linewidth=2)
    ax3.plot(time_periods[:100], vwap_allocation[:100], label='VWAP-style', linewidth=2)
    ax3.plot(time_periods[:100], front_allocation[:100], label='Front-loaded', linewidth=2)
    
    ax3.set_xlabel('Time Period (First 100 minutes)')
    ax3.set_ylabel('Shares per Period')
    ax3.set_title('Allocation Strategy Patterns')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Risk-return trade-off illustration
    strategies = ['Uniform', 'Front-loaded', 'Back-loaded', 'VWAP-style']
    
    # For linear models, cost is identical but risk differs
    costs = [1.0, 1.0, 1.0, 1.0]  # Normalized costs
    risks = [0.5, 0.8, 0.9, 0.4]  # Hypothetical risk levels
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']
    scatter = ax4.scatter(risks, costs, s=[300, 250, 200, 350], 
                         c=colors, alpha=0.7, edgecolors='black', linewidth=2)
    
    for i, strategy in enumerate(strategies):
        ax4.annotate(strategy, (risks[i], costs[i]), 
                    xytext=(10, 10), textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor=colors[i], alpha=0.3))
    
    ax4.set_xlabel('Implementation Risk (Relative)')
    ax4.set_ylabel('Total Cost (Relative)')
    ax4.set_title('Strategy Risk-Return Profile\\n(Hypothetical for Linear Models)')
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim(0, 1.2)
    ax4.set_ylim(0.95, 1.05)
    
    plt.tight_layout()
    plt.savefig('practical_insights.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Practical insights plot saved: practical_insights.png")

def main():
    print("Creating comprehensive visualization suite...")
    
    create_summary_visualizations()
    create_impact_comparison()
    create_practical_insights()
    
    print("\\nAll visualizations completed!")
    print("Generated files:")
    print("- analysis_summary.png")
    print("- impact_function_comparison.png") 
    print("- practical_insights.png")
    print("- execution_strategy_comparison.png (from previous analysis)")

if __name__ == "__main__":
    main()
