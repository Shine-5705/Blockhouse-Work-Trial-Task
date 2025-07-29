import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

# Set matplotlib backend
plt.switch_backend('Agg')

class OptimalExecutionStrategy:
    """
    Implements optimal execution strategies to minimize total temporary impact
    """
    
    def __init__(self, ticker_params):
        """
        Initialize with ticker-specific impact parameters
        ticker_params: dict with structure {ticker: {side: {'beta': value, 'alpha': value}}}
        """
        self.ticker_params = ticker_params
    
    def linear_impact_function(self, x, beta, alpha=0):
        """Linear impact function: g(x) = beta * x + alpha"""
        return beta * x + alpha
    
    def sqrt_impact_function(self, x, beta):
        """Square root impact function: g(x) = beta * sqrt(x)"""
        return beta * np.sqrt(x)
    
    def power_impact_function(self, x, beta, alpha):
        """Power law impact function: g(x) = beta * x^alpha"""
        return beta * (x ** alpha)
    
    def total_impact_linear(self, allocation_vector, beta, alpha=0):
        """Calculate total impact for linear model"""
        return np.sum([self.linear_impact_function(xi, beta, alpha) for xi in allocation_vector])
    
    def uniform_allocation(self, S, N):
        """Uniform allocation strategy: equal amounts at each time step"""
        return np.full(N, S / N)
    
    def front_loaded_allocation(self, S, N, decay_rate=0.1):
        """Front-loaded strategy: more trading early in the day"""
        weights = np.exp(-decay_rate * np.arange(N))
        weights = weights / np.sum(weights)
        return S * weights
    
    def back_loaded_allocation(self, S, N, growth_rate=0.1):
        """Back-loaded strategy: more trading late in the day"""
        weights = np.exp(growth_rate * np.arange(N))
        weights = weights / np.sum(weights)
        return S * weights
    
    def vwap_style_allocation(self, S, N):
        """VWAP-style allocation: U-shaped pattern (higher at open/close)"""
        # Create U-shaped volume pattern
        x = np.linspace(0, 1, N)
        weights = 2 * (x - 0.5)**2 + 0.5  # U-shaped curve
        weights = weights / np.sum(weights)
        return S * weights
    
    def optimal_linear_allocation(self, S, N, beta, alpha=0, time_penalty=0):
        """
        Find optimal allocation for linear impact model using calculus
        For linear model g(x) = beta*x, the optimal solution is uniform allocation
        """
        # For pure linear model without time penalty, uniform is optimal
        if time_penalty == 0:
            return self.uniform_allocation(S, N)
        
        # With time penalty, we can use analytical solution
        # This assumes simple quadratic time penalty
        weights = np.ones(N)
        for i in range(N):
            # Adjust weights based on time penalty
            time_factor = 1 + time_penalty * (N - i - 1) / N
            weights[i] = 1 / time_factor
        
        weights = weights / np.sum(weights)
        return S * weights
    
    def optimize_allocation_numerical(self, S, N, beta, alpha=0, model_type='linear'):
        """
        Numerical optimization for complex impact models
        """
        def objective(x):
            if model_type == 'linear':
                return self.total_impact_linear(x, beta, alpha)
            elif model_type == 'sqrt':
                return np.sum([self.sqrt_impact_function(xi, beta) for xi in x])
            elif model_type == 'power':
                return np.sum([self.power_impact_function(xi, beta, alpha) for xi in x])
        
        # Constraints: sum of allocations must equal S, all allocations >= 0
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - S},  # Sum constraint
        ]
        
        bounds = [(0, S) for _ in range(N)]  # Non-negativity and upper bound
        
        # Initial guess: uniform allocation
        x0 = np.full(N, S / N)
        
        # Optimize
        result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)
        
        if result.success:
            return result.x
        else:
            print(f"Optimization failed: {result.message}")
            return self.uniform_allocation(S, N)

def simulate_execution_strategies():
    """
    Simulate different execution strategies and compare their performance
    """
    
    # Parameters from our analysis
    ticker_params = {
        'FROG': {
            'buy': {'beta': 0.000047, 'alpha': 0.051516},
            'sell': {'beta': 0.000079, 'alpha': 0.057907}
        },
        'SOUN': {
            'buy': {'beta': 0.000001, 'alpha': 0.005637},
            'sell': {'beta': 0.000002, 'alpha': 0.005535}
        },
        'CRWV': {
            'buy': {'beta': 0.000145, 'alpha': 0.056103},
            'sell': {'beta': 0.000142, 'alpha': 0.046059}
        }
    }
    
    # Simulation parameters
    S = 10000  # Total shares to execute
    N = 390    # Number of time periods (minutes in trading day)
    
    optimizer = OptimalExecutionStrategy(ticker_params)
    
    results = {}
    
    for ticker in ticker_params:
        print(f"\\nAnalyzing {ticker}...")
        results[ticker] = {}
        
        for side in ['buy', 'sell']:
            beta = ticker_params[ticker][side]['beta']
            alpha = ticker_params[ticker][side]['alpha']
            
            print(f"  {side.capitalize()} side (β={beta:.6f}, α={alpha:.6f}):")
            
            # Test different strategies
            strategies = {
                'uniform': optimizer.uniform_allocation(S, N),
                'front_loaded': optimizer.front_loaded_allocation(S, N, 0.01),
                'back_loaded': optimizer.back_loaded_allocation(S, N, 0.01),
                'vwap_style': optimizer.vwap_style_allocation(S, N),
                'optimal_linear': optimizer.optimal_linear_allocation(S, N, beta, alpha)
            }
            
            strategy_results = {}
            
            for strategy_name, allocation in strategies.items():
                total_impact = optimizer.total_impact_linear(allocation, beta, alpha)
                avg_impact_per_share = total_impact / S
                
                strategy_results[strategy_name] = {
                    'total_impact': total_impact,
                    'avg_impact_per_share': avg_impact_per_share,
                    'max_order_size': np.max(allocation),
                    'min_order_size': np.min(allocation),
                    'allocation_std': np.std(allocation)
                }
                
                print(f"    {strategy_name:12}: Total Impact = {total_impact:.4f}, "
                      f"Avg/Share = {avg_impact_per_share:.6f}")
            
            results[ticker][side] = strategy_results
    
    return results, optimizer

def create_strategy_visualization(results, optimizer, S=10000, N=390):
    """Create visualizations comparing different strategies"""
    
    fig, axes = plt.subplots(3, 2, figsize=(15, 18))
    axes = axes.flatten()
    
    ticker_params = {
        'FROG': {'buy': {'beta': 0.000047, 'alpha': 0.051516}},
        'SOUN': {'buy': {'beta': 0.000001, 'alpha': 0.005637}},
        'CRWV': {'buy': {'beta': 0.000145, 'alpha': 0.056103}}
    }
    
    for i, ticker in enumerate(['FROG', 'SOUN', 'CRWV']):
        ax1 = axes[i*2]
        ax2 = axes[i*2 + 1]
        
        beta = ticker_params[ticker]['buy']['beta']
        alpha = ticker_params[ticker]['buy']['alpha']
        
        # Generate allocations
        time_steps = np.arange(1, N+1)
        strategies = {
            'Uniform': optimizer.uniform_allocation(S, N),
            'Front-loaded': optimizer.front_loaded_allocation(S, N, 0.01),
            'VWAP-style': optimizer.vwap_style_allocation(S, N),
            'Back-loaded': optimizer.back_loaded_allocation(S, N, 0.01)
        }
        
        # Plot allocation patterns
        for strategy_name, allocation in strategies.items():
            ax1.plot(time_steps[:50], allocation[:50], label=strategy_name, linewidth=2)
        
        ax1.set_xlabel('Time Step (First 50 minutes)')
        ax1.set_ylabel('Shares per Time Step')
        ax1.set_title(f'{ticker}: Allocation Patterns')
        ax1.legend()
        ax1.grid(True)
        
        # Plot cumulative impact
        cumulative_impacts = {}
        for strategy_name, allocation in strategies.items():
            cumulative_impact = np.cumsum([optimizer.linear_impact_function(x, beta, alpha) for x in allocation])
            cumulative_impacts[strategy_name] = cumulative_impact
            ax2.plot(time_steps, cumulative_impact, label=strategy_name, linewidth=2)
        
        ax2.set_xlabel('Time Step')
        ax2.set_ylabel('Cumulative Impact')
        ax2.set_title(f'{ticker}: Cumulative Impact Over Time')
        ax2.legend()
        ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('execution_strategy_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Strategy comparison plot saved: execution_strategy_comparison.png")

def main():
    print("="*60)
    print("OPTIMAL EXECUTION STRATEGY ANALYSIS")
    print("="*60)
    
    # Run strategy simulation
    results, optimizer = simulate_execution_strategies()
    
    # Create visualizations
    create_strategy_visualization(results, optimizer)
    
    # Summary analysis
    print("\\n" + "="*60)
    print("STRATEGY PERFORMANCE SUMMARY")
    print("="*60)
    
    for ticker in results:
        print(f"\\n{ticker}:")
        for side in results[ticker]:
            print(f"  {side.capitalize()} Side:")
            
            strategies = results[ticker][side]
            
            # Find best strategy
            best_strategy = min(strategies.keys(), 
                              key=lambda k: strategies[k]['total_impact'])
            worst_strategy = max(strategies.keys(), 
                               key=lambda k: strategies[k]['total_impact'])
            
            best_impact = strategies[best_strategy]['total_impact']
            worst_impact = strategies[worst_strategy]['total_impact']
            improvement = (worst_impact - best_impact) / worst_impact * 100
            
            print(f"    Best Strategy: {best_strategy} (Impact: {best_impact:.4f})")
            print(f"    Worst Strategy: {worst_strategy} (Impact: {worst_impact:.4f})")
            print(f"    Improvement: {improvement:.2f}%")
    
    return results

if __name__ == "__main__":
    results = main()
