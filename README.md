# PS1: Food Delivery Time Prediction — Linear Regression

> ACM AI/ML Hackathon · Problem Statement 1

## Status

| Step | Status | Summary |
|------|--------|---------|
| 1. Explore & Prepare | ✅ Done | 1000 → 910 rows; correlation heatmap embedded |
| 2. Build the Model | ✅ Done | LinearRegression, 728 train / 182 test |
| 3. Evaluate | ✅ Done | R²=0.8787, MAE=2.63 min, RMSE=3.86 min |
| Task 1 — Model comparison | ✅ Done | LR CV MAE=2.60, GB CV MAE=3.01 → LR wins |
| Task 2 — Residual check | ✅ Done | RMSE/MAE=1.47; roughly normal, slight heavy tails |
| Task 3 — Leakage check | ✅ Done | Order-time model MAE=5.67 (+3.04 min) |
| Task 4 — Cleaning audit | ✅ Done | 5 neg distances, 90 rows dropped (9.0%) |
| Task 5 — Explanation | ✅ Done | Final markdown cell in notebook |

## Files

| File | Description |
|------|-------------|
| `food_delivery_time.csv` | Raw dataset (1,000 rows × 9 cols) |
| `food_delivery_time_cleaned.csv` | Cleaned dataset (910 rows × 7 cols) |
| `food_delivery_time_prediction.ipynb` | **Main notebook — 26 cells, all outputs pre-rendered** |
| `build_notebook.py` | Script that runs the full pipeline and writes the notebook |

---

## Step 1 — Explore & Prepare

**Null counts in raw data:**

| Column | Nulls |
|--------|-------|
| `distance_km` | 10 |
| `distance_miles` | 10 |
| `courier_experience_years` | 25 |
| `order_items` | 25 |
| `traffic_delay_min` | 30 |

**Cleaning (exact code from brief):**
- Drop `order_id` (identifier) and `distance_miles` (confirmed = `distance_km × 0.6214`, max diff = 0.0049)
- `dropna()` → removed **90 rows (9.0%)**
- `abs()` on `distance_km` → corrected **5 sign-error rows**
- **Result: 910 rows × 7 columns**

---

## Step 2 — Build the Model

80/20 split, `random_state=42` → 728 train / 182 test samples.

| Feature | Coefficient |
|---------|------------|
| `distance_km` | +1.978 min/km |
| `traffic_delay_min` | +0.951 min/min |
| `courier_wait_min` | +0.952 min/min |
| `order_items` | +0.986 min/item |
| `prep_time_min` | +0.644 min/min |
| `courier_experience_years` | −0.210 min/year |
| Intercept | 2.458 min |

---

## Step 3 — Evaluate (unseen test set, 182 samples)

| Metric | Value |
|--------|-------|
| R² | 0.8787 |
| MAE | 2.63 min |
| RMSE | 3.86 min |

---

## Task 1 — Model Comparison

| Model | CV MAE (5-fold) | Test MAE | Test RMSE | Test R² |
|-------|----------------|----------|-----------|---------|
| Guess the average | — | 8.70 | 11.09 | −0.0015 |
| LinearRegression | **2.60** | **2.63** | 3.86 | 0.8787 |
| GradientBoosting | 3.01 | 2.90 | 3.81 | 0.8817 |

**Verdict:** LinearRegression and GradientBoosting are very close on the test set (2.63 vs 2.90 min MAE). LinearRegression actually wins on CV MAE (2.60 vs 3.01). The relationship is effectively linear; LinearRegression is kept for interpretability.

---

## Task 2 — Residual Check

- **RMSE/MAE ratio: 1.47** (Gaussian benchmark ≈ 1.25 → slightly heavy tails)
- Residuals are roughly symmetric around 0 but not perfectly Gaussian
- Worst miss: **24.99 min** (row with `distance_km=15`, `prep_time_min=−10` data anomaly, high traffic)
- Biggest misses share: long distances or anomalous prep times the linear model cannot handle

---

## Task 3 — Order-Time Leakage Check

`traffic_delay_min` and `courier_wait_min` are only known after an order is placed.

| Feature set | MAE | RMSE | R² | ΔMAE |
|-------------|-----|------|-----|------|
| All 6 features (full model) | 2.63 | 3.86 | 0.879 | — |
| 4 order-time features only | 5.67 | 7.49 | 0.543 | +3.04 min |

**Leakage costs ~3 minutes of MAE.** In production: use traffic forecasts or accept the higher error.

---

## Task 4 — Cleaning Justification

- **5** rows had negative `distance_km` (sign errors) → corrected with `abs()`
- After abs(), `distance_km` ranges 0.60–22.00 km (mean 5.32 km) — all plausible
- **90 rows dropped** by `dropna()` = **9.0%** of 1,000
- `distance_miles` confirmed = `distance_km × 0.6214` (max diff = 0.0049) → perfectly collinear
