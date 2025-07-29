import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.optimize import curve_fit
import warnings
warnings.filterwarnings('ignore')

# Set matplotlib backend for better compatibility
plt.switch_backend('Agg')

DATASETS = ['FROG', 'SOUN', 'CRWV']
BASE_DIR = './'
LEVELS = 10

def load_sample_data(ticker, sample_rate=1000):
    """Load a sample of data for faster analysis"""
    folder_path = os.path.join(BASE_DIR, ticker)
    
    # Just load the first file for quick analysis
    files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
    if not files:
        return pd.DataFrame()
    
    first_file = files[0]
    try:
        df = pd.read_csv(os.path.join(folder_path, first_file))
        # Sample every nth row for speed
        df_sample = df.iloc[::sample_rate].copy()
        df_sample['date'] = first_file.split('_')[1]
        return df_sample
    except Exception as e:
        print(f"Error reading {first_file}: {e}")
        return pd.DataFrame()

def simulate_market_order_impact_single(row, side='buy', max_size=200):
    """Simulate market order impact for a single order book snapshot"""
    sizes = np.arange(1, max_size + 1)
    slippages = []
    
    # Calculate mid price
    try:
        best_bid = float(row['bid_px_00'])
        best_ask = float(row['ask_px_00'])
        mid_price = (best_bid + best_ask) / 2
    except:
        return pd.DataFrame()
    
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
            
            try:
                px = float(row[px_col])
                sz = float(row[sz_col])
            except:
                continue
            
            if px <= 0 or sz <= 0:
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
            slippage = np.nan
        
        slippages.append(slippage)
    
    return pd.DataFrame({
        'order_size': sizes,
        'slippage': slippages,
        'mid_price': mid_price
    })

def calculate_average_impact(df, side='buy', max_size=200):
    """Calculate average impact across multiple snapshots"""
    all_impacts = []
    
    # Sample fewer snapshots for speed
    sample_size = min(50, len(df))
    df_sample = df.sample(n=sample_size).copy()
    
    for idx, row in df_sample.iterrows():
        impact_data = simulate_market_order_impact_single(row, side, max_size)
        if not impact_data.empty:
            all_impacts.append(impact_data)
    
    if not all_impacts:
        return pd.DataFrame()
    
    # Combine all impacts and calculate average
    combined = pd.concat(all_impacts, ignore_index=True)
    avg_impact = combined.groupby('order_size')['slippage'].agg(['mean', 'std', 'count']).reset_index()
    avg_impact.columns = ['order_size', 'slippage', 'slippage_std', 'sample_count']
    
    return avg_impact

def fit_impact_models(impact_df):
    """Fit different models to the temporary impact function"""
    clean_df = impact_df.dropna()
    if len(clean_df) < 5:
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
        pass
    
    # 2. Square Root Model: g(x) = β * √x
    try:
        def sqrt_model(x, beta):
            return beta * np.sqrt(x)
        
        popt, pcov = curve_fit(sqrt_model, x, y, maxfev=1000)
        y_pred = sqrt_model(x, *popt)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        models['sqrt'] = {
            'params': {'beta': popt[0]},
            'r_squared': r_squared,
            'formula': f'g(x) = {popt[0]:.6f} * √x'
        }
    except:
        pass
    
    # 3. Power Law Model: g(x) = β * x^α
    try:
        pos_x = x[x > 0]
        pos_y = y[x > 0]
        pos_y_abs = np.abs(pos_y)
        valid_mask = (pos_y_abs > 0)
        
        if np.sum(valid_mask) > 5:
            log_x = np.log(pos_x[valid_mask])
            log_y = np.log(pos_y_abs[valid_mask])
            
            slope, intercept, r_value, p_value, std_err = stats.linregress(log_x, log_y)
            alpha = slope
            beta = np.exp(intercept)
            
            models['power'] = {
                'params': {'beta': beta, 'alpha': alpha},
                'r_squared': r_value**2,
                'formula': f'g(x) = {beta:.6f} * x^{alpha:.3f}'
            }
    except:
        pass
    
    return models

def plot_analysis(impact_df, models, ticker, side):
    """Create analysis plots"""
    clean_df = impact_df.dropna()
    if len(clean_df) < 5:
        return
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    x = clean_df['order_size'].values
    y = clean_df['slippage'].values
    
    # Raw data plot
    ax1.scatter(x, y, alpha=0.6, s=20)
    ax1.set_xlabel('Order Size')
    ax1.set_ylabel('Slippage')
    ax1.set_title(f'{ticker} - {side.capitalize()}: Raw Impact Data')
    ax1.grid(True)
    
    # Model fits
    x_smooth = np.linspace(1, max(x), 100)
    colors = ['red', 'blue', 'green']
    model_names = ['linear', 'sqrt', 'power']
    
    ax2.scatter(x, y, alpha=0.6, s=20, label='Data')
    
    for i, model_name in enumerate(model_names):
        if model_name in models and models[model_name]:
            model = models[model_name]
            params = model['params']
            
            if model_name == 'linear':
                y_smooth = params['beta'] * x_smooth + params['alpha']
            elif model_name == 'sqrt':
                y_smooth = params['beta'] * np.sqrt(x_smooth)
            elif model_name == 'power':
                y_smooth = params['beta'] * (x_smooth ** params['alpha'])
            
            ax2.plot(x_smooth, y_smooth, colors[i], linewidth=2,
                    label=f"{model_name.capitalize()} (R²={model['r_squared']:.3f})")
    
    ax2.set_xlabel('Order Size')
    ax2.set_ylabel('Slippage')
    ax2.set_title(f'{ticker} - {side.capitalize()}: Model Fits')
    ax2.legend()
    ax2.grid(True)
    
    # Slippage distribution
    ax3.hist(y, bins=30, alpha=0.7, edgecolor='black')
    ax3.set_xlabel('Slippage')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Slippage Distribution')
    ax3.grid(True)
    
    # Summary text
    ax4.axis('off')
    summary_text = f"Analysis Summary - {ticker} ({side.capitalize()})\n\n"
    summary_text += f"Data Points: {len(clean_df)}\n"
    summary_text += f"Mean Slippage: {np.mean(y):.6f}\n"
    summary_text += f"Std Slippage: {np.std(y):.6f}\n"
    summary_text += f"Max Order Size: {max(x)}\n\n"
    
    summary_text += "Model Performance (R²):\n"
    for model_name in model_names:
        if model_name in models and models[model_name]:
            r_sq = models[model_name]['r_squared']
            summary_text += f"{model_name.capitalize()}: {r_sq:.4f}\n"
    
    ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes, 
             verticalalignment='top', fontsize=11, fontfamily='monospace')
    
    plt.tight_layout()
    plt.savefig(f'{ticker}_{side}_impact_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()  # Close to save memory
    print(f"  Plot saved: {ticker}_{side}_impact_analysis.png")

def main():
    print("=== TEMPORARY IMPACT FUNCTION ANALYSIS ===\n")
    
    all_results = {}
    
    for ticker in DATASETS:
        print(f"Processing {ticker}...")
        
        # Load sample data
        df = load_sample_data(ticker, sample_rate=500)  # Sample every 500th row
        if df.empty:
            print(f"  No data available for {ticker}")
            continue
        
        print(f"  Loaded {len(df)} sample snapshots")
        
        ticker_results = {}
        
        for side in ['buy', 'sell']:
            print(f"  Analyzing {side} side...")
            
            # Calculate average impact
            impact_df = calculate_average_impact(df, side=side, max_size=200)
            
            if impact_df.empty:
                print(f"    No valid data for {side} side")
                continue
            
            # Fit models
            models = fit_impact_models(impact_df)
            
            # Create plots
            plot_analysis(impact_df, models, ticker, side)
            
            # Save results
            impact_df.to_csv(f'{ticker}_{side}_impact_results.csv', index=False)
            
            # Store results
            ticker_results[side] = {
                'models': models,
                'data_points': len(impact_df),
                'mean_slippage': impact_df['slippage'].mean(),
                'max_slippage': impact_df['slippage'].max()
            }
            
            print(f"    Completed with {len(impact_df)} data points")
        
        all_results[ticker] = ticker_results
    
    # Print comprehensive summary
    print("\n" + "="*60)
    print("COMPREHENSIVE ANALYSIS SUMMARY")
    print("="*60)
    
    for ticker in all_results:
        print(f"\n{ticker}:")
        for side in all_results[ticker]:
            data = all_results[ticker][side]
            print(f"  {side.capitalize()} Side:")
            print(f"    Data Points: {data['data_points']}")
            print(f"    Mean Slippage: {data['mean_slippage']:.6f}")
            print(f"    Max Slippage: {data['max_slippage']:.6f}")
            
            models = data['models']
            if models:
                print("    Model R² Values:")
                for model_name, model_data in models.items():
                    print(f"      {model_name.capitalize()}: {model_data['r_squared']:.4f}")
                
                # Best model
                best_model = max(models.keys(), key=lambda k: models[k]['r_squared'])
                print(f"    Best Model: {best_model} (R² = {models[best_model]['r_squared']:.4f})")
                print(f"    Formula: {models[best_model]['formula']}")
    
    return all_results

if __name__ == "__main__":
    results = main()
