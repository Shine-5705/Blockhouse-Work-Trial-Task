import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.optimize import curve_fit
import warnings
warnings.filterwarnings('ignore')

DATASETS = ['FROG', 'SOUN', 'CRWV']
BASE_DIR = './'
LEVELS = 10

def load_all_csvs(ticker):
    """Load and concatenate all CSV files for a given ticker"""
    folder_path = os.path.join(BASE_DIR, ticker)
    dfs = []
    file_count = 0
    
    for filename in sorted(os.listdir(folder_path)):
        if filename.endswith('.csv'):
            try:
                df = pd.read_csv(os.path.join(folder_path, filename))
                df['date'] = filename.split('_')[1]  # Extract date from filename
                dfs.append(df)
                file_count += 1
                if file_count >= 5:  # Limit to first 5 files for faster analysis
                    break
            except Exception as e:
                print(f"Error reading {filename}: {e}")
    
    if dfs:
        return pd.concat(dfs, ignore_index=True)
    else:
        return pd.DataFrame()

def calculate_market_impact_vectorized(df, side='buy', max_size=200):
    """
    Vectorized calculation of market impact across all snapshots
    """
    if df.empty:
        return pd.DataFrame()
    
    # Sample every 100th row to make analysis manageable
    df_sample = df.iloc[::100].copy()
    
    results = []
    
    for idx, row in df_sample.iterrows():
        impact_data = simulate_market_order_impact_single(row, side, max_size)
        impact_data['snapshot_id'] = idx
        impact_data['timestamp'] = row.get('ts_event', 'unknown')
        results.append(impact_data)
    
    if results:
        return pd.concat(results, ignore_index=True)
    else:
        return pd.DataFrame()

def simulate_market_order_impact_single(row, side='buy', max_size=200):
    """
    Simulate market order impact for a single order book snapshot
    """
    sizes = np.arange(1, max_size + 1)
    slippages = []
    avg_prices = []
    
    # Calculate mid price
    best_bid = row['bid_px_00']
    best_ask = row['ask_px_00']
    mid_price = (best_bid + best_ask) / 2
    
    for size in sizes:
        shares_remaining = size
        total_cost = 0
        shares_filled = 0
        
        for level in range(LEVELS):
            if side == 'buy':
                px_col = f'ask_px_0{level}'
                sz_col = f'ask_sz_0{level}'
            else:
                px_col = f'bid_px_0{level}'
                sz_col = f'bid_sz_0{level}'
            
            # Check if columns exist and have valid data
            if px_col not in row or sz_col not in row:
                break
                
            px = row[px_col]
            sz = row[sz_col]
            
            # Skip if price or size is NaN or zero
            if pd.isna(px) or pd.isna(sz) or px <= 0 or sz <= 0:
                continue
            
            fill = min(sz, shares_remaining)
            total_cost += fill * px
            shares_filled += fill
            shares_remaining -= fill
            
            if shares_remaining <= 0:
                break
        
        if shares_filled > 0:
            avg_exec_price = total_cost / shares_filled
            if side == 'buy':
                slippage = avg_exec_price - mid_price
            else:
                slippage = mid_price - avg_exec_price
        else:
            avg_exec_price = np.nan
            slippage = np.nan
        
        slippages.append(slippage)
        avg_prices.append(avg_exec_price)
    
    return pd.DataFrame({
        'order_size': sizes,
        'slippage': slippages,
        'avg_exec_price': avg_prices,
        'mid_price': mid_price
    })

def fit_impact_models(impact_df):
    """
    Fit different models to the temporary impact function
    """
    # Remove NaN values
    clean_df = impact_df.dropna()
    if len(clean_df) < 10:
        return {}
    
    x = clean_df['order_size'].values
    y = clean_df['slippage'].values
    
    models = {}
    
    # 1. Linear Model: g(x) = β * x
    try:
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        models['linear'] = {
            'params': {'beta': slope, 'alpha': intercept},
            'r_squared': r_value**2,
            'formula': f'g(x) = {slope:.6f} * x + {intercept:.6f}'
        }
    except:
        models['linear'] = None
    
    # 2. Square Root Model: g(x) = β * √x
    try:
        def sqrt_model(x, beta):
            return beta * np.sqrt(x)
        
        popt, pcov = curve_fit(sqrt_model, x, y)
        y_pred = sqrt_model(x, *popt)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)
        
        models['sqrt'] = {
            'params': {'beta': popt[0]},
            'r_squared': r_squared,
            'formula': f'g(x) = {popt[0]:.6f} * √x'
        }
    except:
        models['sqrt'] = None
    
    # 3. Power Law Model: g(x) = β * x^α
    try:
        # Use log transformation for power law fitting
        log_x = np.log(x[x > 0])
        log_y = np.log(np.abs(y[x > 0]) + 1e-10)  # Add small constant to avoid log(0)
        
        if len(log_x) > 5:
            slope, intercept, r_value, p_value, std_err = stats.linregress(log_x, log_y)
            alpha = slope
            beta = np.exp(intercept)
            
            models['power'] = {
                'params': {'beta': beta, 'alpha': alpha},
                'r_squared': r_value**2,
                'formula': f'g(x) = {beta:.6f} * x^{alpha:.3f}'
            }
    except:
        models['power'] = None
    
    # 4. Quadratic Model: g(x) = β₁ * x + β₂ * x²
    try:
        coeffs = np.polyfit(x, y, 2)
        y_pred = np.polyval(coeffs, x)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)
        
        models['quadratic'] = {
            'params': {'beta2': coeffs[0], 'beta1': coeffs[1], 'alpha': coeffs[2]},
            'r_squared': r_squared,
            'formula': f'g(x) = {coeffs[0]:.8f} * x² + {coeffs[1]:.6f} * x + {coeffs[2]:.6f}'
        }
    except:
        models['quadratic'] = None
    
    return models

def plot_impact_with_models(impact_df, models, ticker, side):
    """
    Plot the impact function with fitted models
    """
    clean_df = impact_df.dropna()
    if len(clean_df) < 10:
        return
    
    plt.figure(figsize=(15, 10))
    
    # Plot raw data
    x = clean_df['order_size'].values
    y = clean_df['slippage'].values
    
    plt.subplot(2, 2, 1)
    plt.scatter(x, y, alpha=0.1, s=1)
    plt.xlabel('Order Size')
    plt.ylabel('Slippage')
    plt.title(f'{ticker} - {side.capitalize()} Side: Raw Data')
    plt.grid(True)
    
    # Plot model fits
    x_smooth = np.linspace(1, max(x), 100)
    
    plt.subplot(2, 2, 2)
    plt.scatter(x, y, alpha=0.1, s=1, label='Data')
    
    colors = ['red', 'blue', 'green', 'orange']
    model_names = ['linear', 'sqrt', 'power', 'quadratic']
    
    for i, model_name in enumerate(model_names):
        if models.get(model_name):
            model = models[model_name]
            params = model['params']
            
            if model_name == 'linear':
                y_smooth = params['beta'] * x_smooth + params['alpha']
            elif model_name == 'sqrt':
                y_smooth = params['beta'] * np.sqrt(x_smooth)
            elif model_name == 'power':
                y_smooth = params['beta'] * (x_smooth ** params['alpha'])
            elif model_name == 'quadratic':
                y_smooth = params['beta2'] * x_smooth**2 + params['beta1'] * x_smooth + params['alpha']
            
            plt.plot(x_smooth, y_smooth, colors[i], 
                    label=f"{model_name.capitalize()} (R²={model['r_squared']:.3f})")
    
    plt.xlabel('Order Size')
    plt.ylabel('Slippage')
    plt.title(f'{ticker} - {side.capitalize()} Side: Model Fits')
    plt.legend()
    plt.grid(True)
    
    # Plot residuals for best model
    best_model_name = max(models.keys(), key=lambda k: models[k]['r_squared'] if models[k] else 0)
    if models.get(best_model_name):
        plt.subplot(2, 2, 3)
        model = models[best_model_name]
        params = model['params']
        
        if best_model_name == 'linear':
            y_pred = params['beta'] * x + params['alpha']
        elif best_model_name == 'sqrt':
            y_pred = params['beta'] * np.sqrt(x)
        elif best_model_name == 'power':
            y_pred = params['beta'] * (x ** params['alpha'])
        elif best_model_name == 'quadratic':
            y_pred = params['beta2'] * x**2 + params['beta1'] * x + params['alpha']
        
        residuals = y - y_pred
        plt.scatter(x, residuals, alpha=0.1, s=1)
        plt.axhline(y=0, color='red', linestyle='--')
        plt.xlabel('Order Size')
        plt.ylabel('Residuals')
        plt.title(f'Residuals - {best_model_name.capitalize()} Model')
        plt.grid(True)
    
    # Summary statistics
    plt.subplot(2, 2, 4)
    plt.axis('off')
    summary_text = f"Summary for {ticker} - {side.capitalize()} Side:\n\n"
    summary_text += f"Total Snapshots Analyzed: {len(impact_df['snapshot_id'].unique())}\n"
    summary_text += f"Order Size Range: 1 - {max(x)}\n"
    summary_text += f"Average Slippage: {np.mean(y):.6f}\n"
    summary_text += f"Slippage Std Dev: {np.std(y):.6f}\n\n"
    
    summary_text += "Model Performance (R²):\n"
    for model_name in model_names:
        if models.get(model_name):
            r_sq = models[model_name]['r_squared']
            summary_text += f"  {model_name.capitalize()}: {r_sq:.4f}\n"
    
    plt.text(0.1, 0.9, summary_text, transform=plt.gca().transAxes, 
             verticalalignment='top', fontsize=10, fontfamily='monospace')
    
    plt.tight_layout()
    plt.savefig(f'{ticker}_{side}_enhanced_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def analyze_impact_characteristics(impact_df, ticker, side):
    """
    Analyze statistical characteristics of the impact function
    """
    clean_df = impact_df.dropna()
    if len(clean_df) < 10:
        return {}
    
    characteristics = {}
    
    # Basic statistics
    characteristics['mean_slippage'] = clean_df['slippage'].mean()
    characteristics['std_slippage'] = clean_df['slippage'].std()
    characteristics['max_slippage'] = clean_df['slippage'].max()
    characteristics['min_slippage'] = clean_df['slippage'].min()
    
    # Analyze slippage per unit size (marginal impact)
    clean_df_sorted = clean_df.sort_values('order_size')
    clean_df_sorted['marginal_impact'] = clean_df_sorted['slippage'] / clean_df_sorted['order_size']
    characteristics['mean_marginal_impact'] = clean_df_sorted['marginal_impact'].mean()
    characteristics['std_marginal_impact'] = clean_df_sorted['marginal_impact'].std()
    
    # Liquidity analysis
    large_orders = clean_df[clean_df['order_size'] > 100]
    small_orders = clean_df[clean_df['order_size'] <= 50]
    
    if len(large_orders) > 0 and len(small_orders) > 0:
        characteristics['large_order_avg_slippage'] = large_orders['slippage'].mean()
        characteristics['small_order_avg_slippage'] = small_orders['slippage'].mean()
        characteristics['liquidity_premium'] = (
            characteristics['large_order_avg_slippage'] - 
            characteristics['small_order_avg_slippage']
        )
    
    return characteristics

# Main Analysis
def main():
    results = {}
    
    print("=== ENHANCED TEMPORARY IMPACT ANALYSIS ===\n")
    
    for ticker in DATASETS:
        print(f"Processing {ticker}...")
        results[ticker] = {}
        
        # Load data
        df = load_all_csvs(ticker)
        if df.empty:
            print(f"  No data found for {ticker}")
            continue
        
        print(f"  Loaded {len(df):,} snapshots")
        
        # Analyze both sides
        for side in ['buy', 'sell']:
            print(f"  Analyzing {side} side...")
            
            # Calculate impact function
            impact_df = calculate_market_impact_vectorized(df, side=side, max_size=200)
            
            if impact_df.empty:
                print(f"    No valid data for {side} side")
                continue
            
            # Fit models
            models = fit_impact_models(impact_df)
            
            # Analyze characteristics
            characteristics = analyze_impact_characteristics(impact_df, ticker, side)
            
            # Store results
            results[ticker][side] = {
                'models': models,
                'characteristics': characteristics,
                'data_points': len(impact_df)
            }
            
            # Create plots
            plot_impact_with_models(impact_df, models, ticker, side)
            
            # Save detailed results
            impact_df.to_csv(f'{ticker}_{side}_detailed_impact.csv', index=False)
            
            print(f"    Completed analysis with {len(impact_df):,} data points")
    
    # Print summary
    print("\n=== SUMMARY OF RESULTS ===")
    for ticker in results:
        print(f"\n{ticker}:")
        for side in results[ticker]:
            data = results[ticker][side]
            models = data['models']
            chars = data['characteristics']
            
            print(f"  {side.capitalize()} Side:")
            print(f"    Data Points: {data['data_points']:,}")
            print(f"    Mean Slippage: {chars.get('mean_slippage', 'N/A'):.6f}")
            print(f"    Std Slippage: {chars.get('std_slippage', 'N/A'):.6f}")
            
            # Best model
            if models:
                best_model = max(models.keys(), key=lambda k: models[k]['r_squared'] if models[k] else 0)
                if models[best_model]:
                    print(f"    Best Model: {best_model} (R² = {models[best_model]['r_squared']:.4f})")
                    print(f"    Formula: {models[best_model]['formula']}")
    
    return results

if __name__ == "__main__":
    results = main()
