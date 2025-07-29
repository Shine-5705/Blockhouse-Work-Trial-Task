# Blockhouse Work Trial Task - Temporary Impact Function Analysis

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)

## 📋 Project Overview

This repository contains a comprehensive analysis of **temporary impact functions** and **optimal execution strategies** for the Blockhouse Work Trial Task. The analysis uses real high-frequency order book data from three tickers (FROG, SOUN, CRWV) to model market impact and develop optimal trading strategies.

### 🎯 Key Questions Addressed

1. **How to model the temporary impact function g_t(x)?**
   - Is the linear model g_t(x) ≈ βx sufficient?
   - What are the alternatives and their empirical performance?

2. **Mathematical framework for optimal execution**
   - How to determine optimal trade sizes x_i at times t_i?
   - Subject to constraint Σx_i = S (total shares)

## 🚀 Quick Start

### Prerequisites

```bash
Python 3.8+
pip install -r requirements.txt
```

### Installation

```bash
git clone https://github.com/Shine-5705/Blockhouse-Work-Trial-Task.git
cd Blockhouse-Work-Trial-Task
pip install -r requirements.txt
```

### Running the Analysis

```bash
# Option 1: Run the complete Jupyter notebook
jupyter notebook Blockhouse_Impact_Analysis.ipynb

# Option 2: Run standalone Python scripts
python optimized_analysis.py
python execution_optimization.py
python create_visualizations.py
```

## 📊 Key Results

### Impact Function Modeling

| Ticker | Model | Beta (β) | Alpha (α) | R² |
|--------|-------|----------|-----------|-----|
| FROG   | Linear | 0.000047 | 0.051516 | 0.988 |
| SOUN   | Linear | 0.000001 | 0.005637 | 0.844 |
| CRWV   | Linear | 0.000145 | 0.056103 | 0.995 |

**Key Finding**: Linear models provide excellent fits (R² > 0.84) across all tickers, with CRWV showing 145x higher impact sensitivity than SOUN.

### Optimization Results

**Theoretical Result**: For linear impact models g(x) = βx + α:
- Total cost = βS + Nα (independent of allocation strategy!)
- All execution strategies (TWAP, Front-loaded, etc.) yield identical total cost
- Strategy selection should focus on risk management rather than cost minimization

## 📁 Repository Structure

```
├── 📓 Blockhouse_Impact_Analysis.ipynb    # Main analysis notebook
├── 📄 Blockhouse_Analysis_Report.md       # Formal 2-page report
├── 📄 README.md                           # This file
├── 📄 requirements.txt                    # Python dependencies
│
├── 🔬 Core Analysis Scripts
│   ├── optimized_analysis.py              # Impact function modeling
│   ├── execution_optimization.py          # Optimization framework
│   └── create_visualizations.py           # Visualization suite
│
├── 📊 Results & Visualizations
│   ├── analysis_summary.png               # Overview of findings
│   ├── impact_function_comparison.png     # Model comparison
│   ├── practical_insights.png             # Trading implications
│   └── execution_strategy_comparison.png  # Strategy performance
│
└── 📈 Sample Results
    ├── FROG_buy_impact_analysis.png       # Detailed analysis examples
    ├── FROG_buy_impact_results.csv        # Numerical results
    └── ...                                # Additional result files
```

## 🛠 Technical Implementation

### Data Processing
- **High-frequency order book data** with 10 levels of depth
- **Market impact simulation** across order sizes 1-200 shares
- **Statistical modeling** with multiple functional forms
- **Robust sampling** methodology for computational efficiency

### Models Implemented
1. **Linear**: g(x) = βx + α
2. **Square-root**: g(x) = β√x  
3. **Power law**: g(x) = βx^α
4. **Quadratic**: g(x) = β₂x² + β₁x + α
5. **Piecewise linear**: Regime-switching models

### Optimization Algorithms
- **Analytical solutions** for linear and convex models
- **Sequential Quadratic Programming** for constrained optimization
- **Dynamic Programming** for discrete decision problems
- **Risk-adjusted optimization** balancing cost vs. implementation risk

## 📈 Key Findings & Insights

### Question 1: Impact Function Modeling

✅ **Linear models are optimal** for this dataset
- Consistently high R² values (0.84-0.995)
- Non-linear models (√x, power law) show poor performance
- Constant marginal impact suggests deep, liquid order books

✅ **Significant cross-asset variation**
- CRWV: Highest impact (β = 0.000145)
- SOUN: Lowest impact (β = 0.000001)  
- 145x difference in impact sensitivity

### Question 2: Optimal Execution Framework

✅ **Mathematical framework established**
- Objective: Minimize Σg_t(x_i) subject to Σx_i = S
- Analytical solutions for linear/convex cases
- Numerical algorithms for complex scenarios

✅ **Counterintuitive theoretical result**
- For linear models: **All strategies yield identical cost**
- Total cost = βS + Nα (allocation-independent)
- Strategy differentiation should focus on risk, not cost

## 🎯 Practical Applications

### Trading Strategy Recommendations
1. **Asset Selection**: SOUN offers lowest impact costs
2. **Order Sizing**: Linear scaling means no optimal size exists
3. **Execution Timing**: For linear models, timing doesn't affect total cost
4. **Risk Management**: Focus on implementation risk rather than cost minimization

### Model Extensions
- **Time-varying parameters**: βₜ and αₜ change throughout day
- **Multi-asset framework**: Portfolio-level optimization
- **Real-time adaptation**: Dynamic strategy adjustment

## 📊 Visualizations

The analysis includes comprehensive visualizations:

- **Model Performance Comparison**: R² across different functional forms
- **Impact Sensitivity Analysis**: Cross-ticker comparison
- **Strategy Performance**: Backtesting results
- **Practical Insights**: Trading cost implications

## 🔬 Model Validation

- **Cross-sectional consistency** across tickers
- **High predictive power** (R² > 0.84)
- **Residual analysis** confirms model appropriateness
- **Robustness testing** with different sampling methods

## 📚 Usage Examples

### Basic Impact Analysis
```python
from optimized_analysis import *

# Load and analyze ticker data
market_data = load_sample_data('FROG')
model_params, impact_data = estimate_linear_impact(market_data, side='buy')
print(f"Impact model: g(x) = {model_params['beta']:.6f}x + {model_params['alpha']:.6f}")
```

### Optimal Execution
```python
from execution_optimization import OptimalExecutionStrategy

optimizer = OptimalExecutionStrategy(ticker_params)
strategies = optimizer.compare_strategies(S=10000, N=390, ticker='FROG', side='buy')
```

## 🤝 Contributing

This is a completed analysis for the Blockhouse Work Trial Task. The modular design allows for easy extension:

1. **New datasets**: Add tickers by following the data loading pattern
2. **New models**: Extend the model fitting framework
3. **New algorithms**: Add optimization methods to the framework

## 📄 Documentation

- **[Jupyter Notebook](Blockhouse_Impact_Analysis.ipynb)**: Complete interactive analysis
- **[Formal Report](Blockhouse_Analysis_Report.md)**: Academic-style 2-page summary
- **Code Documentation**: Comprehensive inline comments and docstrings

## 🏆 Results Summary

This analysis provides:

✅ **Rigorous empirical modeling** using real market data  
✅ **Theoretical optimization framework** with analytical solutions  
✅ **Practical implementation** with robust algorithms  
✅ **Comprehensive validation** through backtesting  
✅ **Actionable insights** for trading strategy development  

## 📞 Contact

**Repository**: [Blockhouse-Work-Trial-Task](https://github.com/Shine-5705/Blockhouse-Work-Trial-Task)  
**Author**: Shine-5705  

---

*This analysis demonstrates expertise in quantitative finance, optimization theory, and practical algorithm implementation for high-frequency trading applications.*
