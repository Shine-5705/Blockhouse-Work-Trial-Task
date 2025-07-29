# Blockhouse Work Trial Task - Formal Analysis Report

**Author:** Shine-5705  
**Date:** July 29, 2025  
**Repository:** [github.com/Shine-5705/Blockhouse-Work-Trial-Task](https://github.com/Shine-5705/Blockhouse-Work-Trial-Task)

---

## 1. Introduction

This report addresses the Blockhouse Work Trial Task by analyzing temporary impact functions and developing optimal execution strategies using real high-frequency order book data. We examine three tickers (FROG, SOUN, CRWV) to model market impact and formulate mathematical frameworks for optimal trade execution.

## 2. Question 1: Modeling the Temporary Impact Function g_t(x)

### 2.1 Methodology

We tested multiple functional forms for the temporary impact function using empirical market data:

1. **Linear Model**: g(x) = βx + α
2. **Square-Root Model**: g(x) = β√x  
3. **Power Law Model**: g(x) = βx^α
4. **Quadratic Model**: g(x) = β₂x² + β₁x + α

For each model, we simulated market orders of varying sizes (1-200 shares) across multiple order book snapshots and calculated the resulting slippage from mid-price. We then fitted regression models to estimate parameters and evaluated performance using R².

### 2.2 Empirical Results

**Linear Model Performance:**

| Ticker | Side | Beta (β) | Alpha (α) | R² |
|--------|------|----------|-----------|-----|
| FROG   | Buy  | 0.000047 | 0.051516  | 0.988 |
| SOUN   | Buy  | 0.000001 | 0.005637  | 0.844 |
| CRWV   | Buy  | 0.000145 | 0.056103  | 0.995 |

**Model Comparison:** Linear models consistently outperformed non-linear alternatives:
- Linear R² range: 0.844-0.995
- Square-root models: Poor fits (negative R² values)
- Power law models: Moderate performance (R² = 0.60-0.85)

### 2.3 Model Selection Conclusion

**The linear impact model g(x) = βx + α is optimal for this dataset.** Key supporting evidence:

1. **Excellent Statistical Fit**: All tickers achieve R² > 0.84, with CRWV reaching 0.995
2. **Cross-Asset Consistency**: Linear relationships hold across different market capitalization and liquidity regimes
3. **Economic Interpretation**: Constant marginal impact (β) suggests deep, liquid order books where each additional share has consistent incremental cost
4. **Practical Superiority**: Non-linear models fail to provide meaningful improvements despite additional complexity

**Cross-Asset Insights:**
- CRWV exhibits highest impact sensitivity (β = 0.000145)
- SOUN shows lowest impact (β = 0.000001)  
- 145x difference in impact sensitivity across assets reflects varying liquidity characteristics

The linear model's success suggests that within the analyzed order size range (1-200 shares), these markets exhibit sufficient depth that impact scales proportionally with order size.

---

## 3. Question 2: Mathematical Framework for Optimal Execution

### 3.1 Problem Formulation

**Objective Function:**
```
Minimize: J(x) = Σᵢ₌₁ᴺ g_tᵢ(xᵢ)
```

**Constraints:**
```
Σᵢ₌₁ᴺ xᵢ = S    (total shares constraint)
xᵢ ≥ 0, ∀i      (non-negativity)
xᵢ ≤ xₘₐₓ, ∀i    (position limits)
```

Where:
- **x** = [x₁, x₂, ..., xₙ]: allocation vector
- **S**: total shares to execute
- **N**: number of time periods (e.g., 390 minutes)

### 3.2 Analytical Solutions

#### 3.2.1 Linear Impact Case

For g(x) = βx + α, the total cost becomes:
```
J(x) = Σᵢ(βᵢxᵢ + αᵢ) = Σᵢβᵢxᵢ + Σᵢαᵢ
```

**Critical Insight**: If βᵢ = β (constant), then:
```
J(x) = βS + Nα
```

This cost is **independent of the allocation vector x**! Any feasible allocation {x₁, x₂, ..., xₙ} satisfying Σxᵢ = S yields identical total cost.

**Implication**: For linear impact models, TWAP, front-loaded, back-loaded, and all other strategies produce identical transaction costs.

#### 3.2.2 Convex Impact Case

For quadratic impact g(x) = βx², the optimization problem becomes:
```
Minimize: Σᵢβᵢxᵢ²
Subject to: Σᵢxᵢ = S
```

Using Lagrange multipliers:
```
L = Σᵢβᵢxᵢ² + λ(S - Σᵢxᵢ)
```

**Optimality conditions:**
```
∂L/∂xᵢ = 2βᵢxᵢ - λ = 0  ⟹  xᵢ = λ/(2βᵢ)
```

**Optimal allocation:**
```
xᵢ* = S × (1/βᵢ) / Σⱼ(1/βⱼ)
```

This yields the **inverse-beta weighted allocation**: allocate more to periods with lower impact sensitivity.

### 3.3 Numerical Algorithms

For complex impact functions requiring numerical optimization:

1. **Sequential Quadratic Programming (SQP)**: For smooth, differentiable objectives
2. **Dynamic Programming**: For discrete state-dependent decisions
3. **Risk-Adjusted Optimization**: Incorporating implementation risk penalties

**Implementation Example:**
```python
def optimize_execution(impact_function, S, N):
    objective = lambda x: sum(impact_function(xi) for xi in x)
    constraints = [{'type': 'eq', 'fun': lambda x: sum(x) - S}]
    bounds = [(0, S) for _ in range(N)]
    return minimize(objective, x0=S/N*ones(N), constraints=constraints, bounds=bounds)
```

### 3.4 Strategy Implementation Results

**Empirical Validation**: Using our fitted linear models, we tested five execution strategies:

1. **Uniform (TWAP)**: xᵢ = S/N ∀i
2. **Front-loaded**: Higher allocation early
3. **Back-loaded**: Higher allocation late  
4. **VWAP-style**: U-shaped volume pattern
5. **Optimized**: Numerically optimized allocation

**Result**: All strategies yielded identical total costs for linear models, confirming our theoretical prediction.

### 3.5 Extensions and Practical Considerations

**Time-Varying Parameters**: Real markets exhibit intraday variation in impact parameters. The framework extends to:
```
J(x) = Σᵢ[βᵢ(t)xᵢ + αᵢ(t)]
```

**Risk Considerations**: Practical implementation requires balancing cost minimization with execution risk:
```
J_risk(x) = Σᵢg(xᵢ) + λ × Risk_penalty(x)
```

**Multi-Asset Extension**: Portfolio execution involves correlation effects and capacity constraints across assets.

## 4. Conclusions

### 4.1 Key Findings

1. **Linear impact models are empirically optimal** for the analyzed datasets, achieving excellent fits (R² > 0.84)
2. **Significant cross-asset heterogeneity exists** in impact sensitivity (145x difference between CRWV and SOUN)
3. **For linear models, all execution strategies yield identical costs**, making strategy selection a risk management rather than cost optimization problem
4. **The mathematical framework provides both analytical and numerical solutions** for various impact function specifications

### 4.2 Practical Implications

- **Asset Selection**: SOUN offers the lowest impact environment for large orders
- **Strategy Focus**: Emphasize risk management and implementation considerations rather than cost minimization
- **Model Extensions**: Time-varying and regime-dependent models warrant investigation for longer time horizons
- **Algorithm Implementation**: Robust numerical methods enable real-time strategy optimization

This analysis provides a rigorous foundation for optimal execution strategy development, demonstrating both theoretical understanding and practical implementation capabilities in quantitative finance applications.

---

**Note**: Complete code implementation, visualizations, and detailed results are available in the accompanying Jupyter notebook and Python scripts in the GitHub repository.
