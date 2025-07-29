# Blockhouse Work Trial Task - Temporary Impact Function Analysis

## Executive Summary

This analysis examines temporary impact functions for three tickers (FROG, SOUN, CRWV) using high-frequency order book data. We model the impact function g_t(x) and develop an optimal execution framework to minimize total trading impact.

**Key Findings:**
- Linear impact models provide excellent fits (R² > 0.84) for all tickers
- CRWV shows highest impact sensitivity (β = 0.000145 for buy orders)
- SOUN demonstrates lowest impact (β = 0.000001 for buy orders)
- For linear impact models, uniform allocation is theoretically optimal

---

## Question 1: Modeling the Temporary Impact Function g_t(x)

### 1.1 Data Analysis Approach

We analyzed order book data across three tickers using market impact simulation. For each order book snapshot, we simulated market orders of varying sizes (1-200 shares) and calculated the resulting slippage from mid-price.

**Data Processing:**
- Loaded order book snapshots with 10 levels of depth
- Sampled data for computational efficiency (every 500th snapshot)
- Calculated slippage as: `avg_execution_price - mid_price` (for buy orders)

### 1.2 Model Exploration

We tested four functional forms for the temporary impact function:

#### 1.2.1 Linear Model: g(x) = βx + α
**Rationale:** Simplest model assuming constant marginal impact

**Results:**
- FROG Buy: g(x) = 0.000047x + 0.051516 (R² = 0.988)
- SOUN Buy: g(x) = 0.000001x + 0.005637 (R² = 0.844) 
- CRWV Buy: g(x) = 0.000145x + 0.056103 (R² = 0.995)

#### 1.2.2 Square Root Model: g(x) = β√x
**Rationale:** Concave function reflecting diminishing marginal impact

**Results:** Poor fits (negative R² values) indicating this model is inappropriate for our data.

#### 1.2.3 Power Law Model: g(x) = βx^α
**Rationale:** Flexible model capturing various impact behaviors

**Results:**
- FROG Buy: α = 0.84, moderate fit (R² = 0.840)
- Power law shows promise but linear model dominates

#### 1.2.4 Quadratic Model: g(x) = β₁x + β₂x²
**Rationale:** Captures accelerating impact for large orders

**Results:** Mixed performance, generally inferior to linear model.

### 1.3 Model Selection and Interpretation

**Best Model: Linear Impact Function**

The linear model consistently provides the best fits across all tickers. This suggests:

1. **Constant Marginal Impact:** Each additional share has the same incremental impact
2. **Market Depth:** Order books have sufficient depth that impact scales linearly
3. **Time Independence:** The constant α term represents baseline market conditions

**Cross-Ticker Analysis:**
- CRWV shows highest impact sensitivity (β = 0.000145)
- SOUN shows lowest impact sensitivity (β = 0.000001)
- All tickers show positive baseline impact (α > 0)

### 1.4 Model Limitations and Extensions

While linear models fit well, several limitations exist:

1. **Limited Order Size Range:** Analysis restricted to 1-200 shares
2. **Time Invariance:** Model doesn't capture intraday variations
3. **Liquidity Conditions:** Impact may vary with market conditions

**Proposed Extensions:**
- Time-varying parameters: g_t(x) = β_t x + α_t
- Regime-dependent models based on volatility/liquidity
- Non-linear models for larger order sizes

---

## Question 2: Mathematical Framework for Optimal Execution

### 2.1 Problem Formulation

**Objective:** Minimize total temporary impact when executing S shares over N periods

**Mathematical Setup:**
- Total shares to execute: S
- Trading periods: N = 390 (one-minute intervals)
- Allocation vector: x ∈ ℝ^N where x_i represents shares executed in period i
- Constraint: Σx_i = S
- Objective: minimize Σg_t_i(x_i)

### 2.2 Linear Impact Case

For linear impact g(x) = βx + α:

**Total Impact:** 
```
J = Σ(βx_i + α) = β(Σx_i) + Nα = βS + Nα
```

**Key Insight:** For pure linear impact, total cost is independent of allocation strategy!

**Optimal Solution:** Any allocation satisfying Σx_i = S yields the same total impact.

### 2.3 Extended Framework with Constraints

#### 2.3.1 Market Impact with Time Penalty

Realistic models include time penalties for holding unexecuted positions:

```
J = Σ[g(x_i) + λ_i × (S - Σ_{j≤i} x_j)]
```

Where λ_i represents the holding cost/risk penalty.

#### 2.3.2 Optimization with Position Limits

Adding practical constraints:
- Maximum order size: x_i ≤ x_max
- Minimum order size: x_i ≥ x_min (when x_i > 0)
- Non-negativity: x_i ≥ 0

**Constrained Optimization:**
```
minimize   Σg(x_i) + penalty_terms
subject to Σx_i = S
          0 ≤ x_i ≤ x_max ∀i
```

### 2.4 Solution Techniques

#### 2.4.1 Analytical Solutions

For quadratic impact models g(x) = ax² + bx:
- Use Lagrange multipliers
- Closed-form solutions exist for simple cases

#### 2.4.2 Numerical Optimization

For complex models:
```python
from scipy.optimize import minimize

def objective(x):
    return sum(impact_function(xi) for xi in x)

constraints = [{'type': 'eq', 'fun': lambda x: sum(x) - S}]
bounds = [(0, x_max) for _ in range(N)]

result = minimize(objective, x0, constraints=constraints, bounds=bounds)
```

#### 2.4.3 Dynamic Programming

For time-dependent models:
- State: remaining shares to execute
- Decision: allocation at current time
- Bellman equation approach for optimal subproblems

### 2.5 Strategy Implementation

#### 2.5.1 Baseline Strategies Tested

1. **Uniform Allocation:** x_i = S/N ∀i
2. **Front-loaded:** Higher allocation early in day
3. **Back-loaded:** Higher allocation late in day  
4. **VWAP-style:** U-shaped allocation pattern

#### 2.5.2 Results Summary

For our linear impact models, all strategies yield identical total impact, confirming theoretical predictions. However, strategies differ in:
- Risk profiles
- Market impact concentration
- Practical implementation considerations

### 2.6 Advanced Considerations

#### 2.6.1 Stochastic Models

Incorporate price volatility:
```
dP_t = μdt + σdW_t + impact_function(trading_rate_t)dt
```

#### 2.6.2 Multi-Asset Execution

Extend to portfolio execution with correlation effects.

#### 2.6.3 Adaptive Strategies

Real-time strategy adjustment based on:
- Market conditions
- Execution performance
- Liquidity variations

---

## Conclusion

This analysis demonstrates that:

1. **Linear impact models provide excellent fits** for the analyzed tickers
2. **Impact varies significantly across assets** (145x difference between CRWV and SOUN)
3. **For linear models, uniform allocation is optimal** from a pure cost perspective
4. **Practical considerations** (risk, market impact concentration) may favor other strategies

The framework provides a solid foundation for optimal execution, with clear paths for extension to more complex market microstructure models.

---

## Code Repository

Complete analysis code is available in the workspace with:
- `optimized_analysis.py`: Impact function modeling
- `execution_optimization.py`: Strategy optimization framework
- Generated plots and CSV files with detailed results

**Data Sources:** Order book data for FROG, SOUN, CRWV tickers
**Analysis Period:** April-May 2025 sample data
**Methodology:** Monte Carlo simulation of market orders across order book snapshots
