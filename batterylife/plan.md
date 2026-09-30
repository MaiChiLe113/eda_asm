So your pipeline should be:

```text
Raw .mat battery files
        ↓
Step-level data
        ↓
Find reference capacity measurements Q1, Q2, ...
        ↓
For everything between Qi and Qi+1:
aggregate current / voltage / temperature / usage
        ↓
One row = one battery health transition
        ↓
Target = ΔQ = Qi+1 - Qi
        ↓
EDA
```

This is scientifically defensible and much easier to fit to your rubric.

---

# 0. Build the analytical dataset first

The paper uses **26 usable cells**, grouped into seven operating regimes. Cells 16 and 17 were excluded because of spurious data producing negative-duration cycles. The first five groups were tested at room temperature and groups 6–7 at 40°C. There are 950 reference capacity measurements across the dataset, with capacities spanning roughly 2.10 to 0.80 Ah. 

Therefore your final EDA table will probably contain **slightly fewer than 950 rows**, because each row represents the interval between two capacity measurements. Do not put a final row count in the report until your extraction finishes.

### Recommended final columns

```text
battery_id
group_id
ambient_temperature
load_profile
transition_id

start_capacity_Ah
end_capacity_Ah
delta_capacity_Ah       ← TARGET

delta_t_hours
absolute_age_days
charge_throughput_Ah

mean_abs_current
max_abs_current
current_std

mean_voltage
min_voltage
voltage_std

mean_temperature
max_temperature

time_current_low
time_current_mid
time_current_high

time_temp_low
time_temp_mid
time_temp_high
```

The first three source-backed engineered features should definitely be:

- elapsed time `Δt`
- charge throughput \(Q_{thru}=\int |I|dt\)
- absolute elapsed age

because these are explicitly proposed in the 2018 paper. 

The paper also suggests features representing **time spent in ranges of current, voltage and temperature**.

The mean/min/max/std features above are my recommended **EDA extensions**, not variables explicitly prescribed by the paper.

---

# Notebook Part A — Raw data processing

## Cell 1 — imports

```python
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.io import whosmat, loadmat
```

## Cell 2 — locate `.mat` files

```python
ROOT = Path("dataset")

mat_files = list(ROOT.rglob("Matlab/*.mat"))

print("Number of MAT files:", len(mat_files))

for f in mat_files[:10]:
    print(f)
```

## Cell 3 — inspect without loading the whole file

Since your files are huge, **do this before `loadmat()`**:

```python
for f in mat_files[:3]:
    print("\n", f.name)
    try:
        print(whosmat(f))
    except Exception as e:
        print("Could not inspect:", e)
```

This tells you which variables are stored inside without loading everything into RAM.

Then load **one battery file at a time**, preferably only the `step` object:

```python
def load_one_battery(path):
    mat = loadmat(
        path,
        variable_names=["step"],
        simplify_cells=True
    )
    return mat["step"]
```

You may need to slightly change this after seeing the exact MATLAB structure.

---

# Notebook Part B — Convert raw steps into health transitions

Use the `comment` field you described.

Reference health checkpoints:

```python
reference_mask = (
    steps["comment"]
    .str.lower()
    .eq("reference discharge")
)
```

Random-walk usage:

```python
rw_mask = steps["comment"].str.contains(
    "random walk",
    case=False,
    na=False
)
```

Conceptually:

```python
reference_indices = steps.index[reference_mask].tolist()

records = []

for j in range(len(reference_indices) - 1):

    start_idx = reference_indices[j]
    end_idx   = reference_indices[j + 1]

    Q_start = ...
    Q_end   = ...

    usage = steps.loc[start_idx + 1:end_idx - 1]

    usage = usage[
        usage["comment"].str.contains(
            "random walk",
            case=False,
            na=False
        )
    ]

    record = {
        "transition_id": j,
        "start_capacity_Ah": Q_start,
        "end_capacity_Ah": Q_end,
        "delta_capacity_Ah": Q_end - Q_start,

        # engineered features go here
    }

    records.append(record)

battery_df = pd.DataFrame(records)
```

The capacity value should come from the reference characterisation measurement. The paper describes reference charge/discharge tests as the capacity/health checkpoints.

---

# 1. Dataset Overview

## Report

You can write:

> The NASA Randomized Battery Usage Dataset contains lithium-ion battery ageing experiments conducted under multiple randomized operating conditions. The dataset contains 26 usable LG Chem 18650 lithium cobalt oxide/graphite cells divided into seven experimental groups with different current distributions and temperatures. Periodic reference characterisation tests provide measurements of battery capacity, which serves as an indicator of battery State of Health.
>
> For this analysis, the raw time-series measurements were transformed into a transition-level dataset. Each observation represents battery usage between two consecutive capacity measurements. The target variable is the change in capacity, `delta_capacity_Ah`.

### Target

Use:

```text
delta_capacity_Ah = Q_next - Q_current
```

This matches the regression formulation used by Richardson et al., where the model predicts capacity change between periods of usage.
### Problem type

**Regression.**

For the assignment's class-conversion requirement, keep regression as your actual problem but additionally create:

```python
df["capacity_loss"] = -df["delta_capacity_Ah"]

df["degradation_class"] = pd.qcut(
    df["capacity_loss"],
    q=3,
    labels=["Low", "Moderate", "High"]
)
```

Report:

> The original target is continuous and therefore represents a regression problem. For target-distribution analysis, it was additionally discretised into three equal-frequency degradation classes to obtain reasonably balanced categories. The continuous target was retained for correlation and future modelling.

### Required code evidence

```python
print(df.shape)
print(df.dtypes)
display(df.head())

display(
    df.describe(include="number").T
)
```

---

# 2. Data Quality Assessment

This dataset gives you a strong section.

The paper itself documents one major issue: cells 16 and 17 contained spurious observations producing negative-duration cycles and were excluded from the study.

## Missing values

```python
missing = pd.DataFrame({
    "missing_count": df.isna().sum(),
    "missing_percent": df.isna().mean() * 100
}).sort_values("missing_percent", ascending=False)

display(missing)
```

Heatmap:

```python
sns.heatmap(df.isna(), cbar=False)
plt.title("Missing Values")
plt.show()
```

Do not invent missing percentages in the report; copy the real results.

## Duplicates

Your logical primary key should be approximately:

```python
duplicate_mask = df.duplicated(
    subset=["battery_id", "transition_id"],
    keep=False
)

print("Duplicates:", duplicate_mask.sum())

display(df[duplicate_mask])
```

## Invalid observations

Check:

```python
quality_flags = pd.DataFrame({
    "nonpositive_duration": df["delta_t_hours"] <= 0,
    "missing_target": df["delta_capacity_Ah"].isna(),
    "missing_throughput": df["charge_throughput_Ah"].isna()
})

quality_flags.sum()
```

## Outliers

```python
numeric_cols = df.select_dtypes("number").columns

for col in numeric_cols:
    plt.figure()
    sns.boxplot(x=df[col])
    plt.title(col)
    plt.show()
```

But **do not automatically remove every IQR outlier**.

High current or temperature can represent genuine experimental stress, not an error.

Also, some capacity increases may appear even though the overall trend is downward. The 2018 paper notes occasional positive capacity steps and suggests that some may be measurement artefacts related to reference tests. 

Your report can therefore say:

> Statistical outliers were reviewed using both distributional and engineering criteria. Extreme current and temperature observations were not automatically removed because aggressive operating regimes are intentional parts of the experiment.

That's much stronger than blindly doing `drop(outliers)`.

---

# 3. Key Findings from Univariate Analysis

You need at least six numerical predictor distributions.

I would analyse:

```python
univariate_features = [
    "delta_t_hours",
    "charge_throughput_Ah",
    "absolute_age_days",
    "mean_abs_current",
    "max_abs_current",
    "mean_voltage",
    "min_voltage",
    "mean_temperature",
    "max_temperature"
]
```

Code:

```python
for col in univariate_features:
    plt.figure(figsize=(6, 4))
    sns.histplot(df[col].dropna(), kde=True)
    plt.title(f"Distribution of {col}")
    plt.xlabel(col)
    plt.show()
```

### Three most useful report plots

I would feature:

**1. Target — `delta_capacity_Ah`**

```python
sns.histplot(df["delta_capacity_Ah"], kde=True)
```

Look for:

- mostly negative changes
- skewness
- occasional positive values

**2. `charge_throughput_Ah`**

This is particularly important because the paper identifies charge throughput as a key ageing input.

**3. `delta_t_hours`**

Intervals are not necessarily equal in duration, so investigate skew/range.

Do not write “normal” or “right-skewed” until you've actually plotted them.

---

# 4. Key Findings from Multivariate Analysis

## Correlation matrix

Important: **exclude leakage columns.**

Do not treat this as a predictor:

```text
end_capacity_Ah
```

because:

```text
delta_capacity =
end_capacity - start_capacity
```

Using `end_capacity_Ah` to explain `delta_capacity_Ah` would be partly mechanical leakage.

Use:

```python
TARGET = "delta_capacity_Ah"

predictors = [
    "start_capacity_Ah",
    "delta_t_hours",
    "absolute_age_days",
    "charge_throughput_Ah",
    "mean_abs_current",
    "max_abs_current",
    "current_std",
    "mean_voltage",
    "min_voltage",
    "voltage_std",
    "mean_temperature",
    "max_temperature",
    "time_current_low",
    "time_current_mid",
    "time_current_high",
    "time_temp_low",
    "time_temp_mid",
    "time_temp_high"
]

analysis_cols = predictors + [TARGET]

corr = df[analysis_cols].corr()
```

Heatmap:

```python
plt.figure(figsize=(14, 11))

sns.heatmap(
    corr,
    cmap="coolwarm",
    center=0
)

plt.title("Correlation Matrix")
plt.show()
```

## Top 3 target correlations

```python
target_corr = (
    corr[TARGET]
    .drop(TARGET)
    .sort_values(key=lambda x: x.abs(), ascending=False)
)

top3 = target_corr.head(3)

display(top3)
```

Your report then becomes:

```text
1. [actual feature] — r = [...]
2. [actual feature] — r = [...]
3. [actual feature] — r = [...]
```

Do **not predict these values in advance**.

---

## Multicollinearity

You actually have a very good discussion here.

The paper notes that:

> time spent in all bins for a parameter must sum to total elapsed time.

Therefore:

```text
time_temp_low
+ time_temp_mid
+ time_temp_high
≈ delta_t
```

and similarly:

```text
time_current_low
+ time_current_mid
+ time_current_high
≈ delta_t
```

This means **structural multicollinearity is expected**.

Detect high correlations:

```python
upper = corr.abs().where(
    np.triu(np.ones(corr.shape), k=1).astype(bool)
)

high_corr = (
    upper.stack()
         .sort_values(ascending=False)
)

display(high_corr[high_corr > 0.80])
```

Report:

> Strong correlations between duration-based features are expected because the current- and temperature-bin durations partition the total elapsed time. Consequently, all bin-duration features should not necessarily be used together with total duration in a simple linear model.

That's an excellent multicollinearity interpretation.

---

## Predictor relationships

Pairplot:

```python
key_features = [
    "delta_capacity_Ah",
    "delta_t_hours",
    "charge_throughput_Ah",
    "mean_abs_current",
    "max_temperature",
    "start_capacity_Ah"
]

sns.pairplot(
    df[key_features + ["group_id"]],
    hue="group_id",
    corner=True
)
```

Also useful:

```python
sns.scatterplot(
    data=df,
    x="charge_throughput_Ah",
    y="delta_capacity_Ah",
    hue="group_id"
)
```

This can reveal:

- nonlinear patterns
- group-specific clusters
- different degradation regimes
- temperature effects

---

# 5. Initial Insights for Modelling

Your report can be quite strong because you have published evidence.

The 2018 study found **elapsed time and charge throughput were dominant inputs**, while a simple linear combination of features performed poorly. 

### Promising features

After confirming with your own EDA:

- charge throughput
- elapsed time
- initial capacity
- current stress exposure
- temperature exposure
- battery age
- group/loading regime

### Scaling

Yes for:

- Gaussian Process
- SVR
- linear models
- neural networks

Not essential for tree methods.

### Encoding

Encode:

```text
group_id
load_profile
ambient_temperature
```

using one-hot encoding if necessary.

### Feature selection

Definitely.

You have deliberate multicollinearity among time-bin features.

Consider:

```text
correlation filtering
VIF
regularisation
permutation importance
```

### Candidate models

Write about:

1. **Linear/Ridge regression** — baseline
2. **Gaussian Process Regression with Matérn kernel**
3. **Gradient Boosting**
4. Random Forest as another nonlinear baseline

The paper's results support nonlinear models over a simple linear kernel.

---

# 6. Engineering Interpretation

The 2014 paper gives you your physical explanation.

As batteries age, the main observed effects include **capacity loss and increased internal resistance**. These changes influence voltage response and discharge duration. 

A strong paragraph:

> Increasing cumulative battery usage is expected to reduce available capacity due to electrochemical ageing. Charge throughput represents the total electrical stress placed on the cell, while current and temperature profiles represent operating conditions that may accelerate degradation. As internal resistance increases and available capacity decreases, an aged battery reaches its cutoff voltage earlier under load. Therefore, statistical relationships between usage intensity, thermal exposure and capacity change have direct physical meaning in battery-health monitoring.

You can connect this to:

- EV battery management
- aircraft power systems
- energy storage
- predictive maintenance
- Remaining Useful Life

---

# 7. Challenges Encountered

Your real challenges are actually good portfolio material:

> The original dataset was distributed as large MATLAB files containing nested step-level time-series records rather than a single analysis-ready table. Loading all raw measurements simultaneously exceeded practical memory limits, so files were processed one battery at a time.

Then mention:

- different experimental groups
- variable-length time series
- converting signals to fixed-size features
- capacity reference cycles only occur periodically
- detecting invalid durations
- differentiating experimental extremes from errors
- repeated measurements from the same physical cell

One important insight:

> **Raw dataset size ≠ number of independent batteries.**

You still only have 26 usable physical cells.

---

# 8. Reflection

### Poor data quality

Write:

> Errors in current measurements could directly affect calculated charge throughput and capacity estimates. Timestamp errors could distort elapsed-time features, while temperature sensor errors could incorrectly represent thermal stress. Because the target itself is derived from reference capacity measurements, measurement errors can affect both predictors and labels and therefore reduce model reliability.

### Bias

This is not mainly human/social bias.

Discuss **experimental bias**:

- only one battery chemistry
- one manufacturer/form factor
- only 26 usable cells
- predetermined laboratory load regimes
- limited temperature conditions
- lab operation differs from real EV/device usage

Therefore:

> A model trained on this dataset should not automatically be assumed to generalise to other battery chemistries, manufacturers or real-world environments.

---

# 9. Next Steps

I would write:

> Future work would incorporate temporal history across several previous load patterns, investigate automated feature selection, predict Remaining Useful Life, and evaluate the model on batteries outside the original experimental conditions.

This is well supported by the second paper: using preceding load patterns improved prediction, and the authors note that coverage across different operating conditions is important.

Also propose:

```text
• SOH prediction
• RUL prediction
• internal-resistance estimation
• lagged degradation features
• uncertainty estimation
• additional battery chemistries
```

---

# Recommended notebook order

I would structure the actual submission like this:

```text
0. GenAI declaration

1. Imports + paths

2. Inspect raw .mat files
   └── file count, variable structures

3. Raw data extraction
   └── process one battery at a time

4. Feature engineering
   └── raw step data → load-pattern table

5. Save processed table
   └── battery_eda.parquet

6. Dataset overview
   └── shape / head / dtypes / describe

7. Data quality
   └── missing / duplicates / invalid records / outliers

8. Target preparation
   └── delta_capacity
   └── degradation_class

9. Univariate EDA
   └── ≥6 predictors
   └── target

10. Multivariate EDA
    └── correlations
    └── top 3
    └── multicollinearity
    └── pairplot
    └── group relationships

11. Initial modelling insights

12. Engineering interpretation

13. Challenges

14. Responsible AI reflection

15. Next steps
```

The most important methodological choice is **not to flatten every raw sensor timestamp into an independent row**. The 2018 paper explicitly converts variable-length current, voltage and temperature sequences into fixed-size load-pattern vectors, which is the right structure for your assignment. 

That gives you a final dataset that is manageable for Pandas while still demonstrating that **you personally processed a large, complex engineering dataset into analysis-ready features**.