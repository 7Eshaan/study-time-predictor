"""
Student Study Time Predictor
A linear regression model to estimate required study hours
before an exam based on real student factors.

Author: Eshaan
Course: Fundamentals of AI and ML
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import pandas as pd
import warnings
warnings.filterwarnings("ignore")


# ──────────────────────────────────────────────
# 1. Dataset Generation
# ──────────────────────────────────────────────
np.random.seed(42)
N = 200

days_until_exam     = np.random.randint(1, 15, N)          # 1–14 days
subject_difficulty  = np.random.uniform(1, 10, N)          # 1=easy, 10=hard
past_score          = np.random.uniform(30, 100, N)        # previous exam %
topics_remaining    = np.random.randint(1, 20, N)          # chapters/topics left
daily_free_hours    = np.random.uniform(1, 8, N)           # free time per day

# Ground-truth formula (with noise)
study_hours_needed = (
    0.5 * subject_difficulty
    + 0.3 * topics_remaining
    - 0.15 * (past_score / 10)
    - 0.2 * days_until_exam
    + 0.1 * (10 - daily_free_hours)
    + np.random.normal(0, 0.8, N)
)
study_hours_needed = np.clip(study_hours_needed, 1, 12)  # realistic bounds

df = pd.DataFrame({
    "days_until_exam":    days_until_exam,
    "subject_difficulty": subject_difficulty,
    "past_score":         past_score,
    "topics_remaining":   topics_remaining,
    "daily_free_hours":   daily_free_hours,
    "study_hours_needed": study_hours_needed
})

print("=" * 50)
print("  Student Study Time Predictor")
print("=" * 50)
print(f"\nDataset shape: {df.shape}")
print("\nSample data:")
print(df.head())
print("\nDataset statistics:")
print(df.describe().round(2))


# ──────────────────────────────────────────────
# 2. Preprocessing
# ──────────────────────────────────────────────
features = ["days_until_exam", "subject_difficulty", "past_score",
            "topics_remaining", "daily_free_hours"]

X = df[features].values
y = df["study_hours_needed"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)


# ──────────────────────────────────────────────
# 3. Model Training
# ──────────────────────────────────────────────
model = LinearRegression()
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)


# ──────────────────────────────────────────────
# 4. Evaluation
# ──────────────────────────────────────────────
mse  = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2   = r2_score(y_test, y_pred)

print("\n── Model Performance ──")
print(f"  RMSE : {rmse:.3f} hours")
print(f"  R²   : {r2:.3f}")
print("\n── Feature Coefficients ──")
for feat, coef in zip(features, model.coef_):
    print(f"  {feat:<25} {coef:+.4f}")
print(f"  {'Intercept':<25} {model.intercept_:+.4f}")


# ──────────────────────────────────────────────
# 5. Visualisations
# ──────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Student Study Time Predictor — Results", fontsize=14, fontweight="bold")

# (a) Actual vs Predicted
axes[0].scatter(y_test, y_pred, alpha=0.6, color="#4A90D9", edgecolors="white", s=60)
lims = [min(y_test.min(), y_pred.min()) - 0.5,
        max(y_test.max(), y_pred.max()) + 0.5]
axes[0].plot(lims, lims, "r--", linewidth=1.5, label="Perfect fit")
axes[0].set_xlabel("Actual Study Hours")
axes[0].set_ylabel("Predicted Study Hours")
axes[0].set_title("Actual vs Predicted")
axes[0].legend()
axes[0].set_xlim(lims); axes[0].set_ylim(lims)

# (b) Feature importance (absolute coefficients)
abs_coefs = np.abs(model.coef_)
feat_labels = ["Days to Exam", "Difficulty", "Past Score", "Topics Left", "Free Hours"]
colors = ["#E74C3C" if c == max(abs_coefs) else "#4A90D9" for c in abs_coefs]
axes[1].barh(feat_labels, abs_coefs, color=colors)
axes[1].set_xlabel("|Coefficient|")
axes[1].set_title("Feature Importance")
axes[1].invert_yaxis()

# (c) Residual distribution
residuals = y_test - y_pred
axes[2].hist(residuals, bins=15, color="#2ECC71", edgecolor="white", alpha=0.85)
axes[2].axvline(0, color="red", linestyle="--", linewidth=1.5)
axes[2].set_xlabel("Residual (Actual − Predicted)")
axes[2].set_ylabel("Count")
axes[2].set_title("Residual Distribution")

plt.tight_layout()
plt.savefig("results.png", dpi=150, bbox_inches="tight")
plt.close()
print("\n[✓] Saved results.png")


# ──────────────────────────────────────────────
# 6. Predict for a new student
# ──────────────────────────────────────────────
def predict_study_hours(days, difficulty, past_score_val, topics, free_hours):
    """Predict study hours needed for a single student profile."""
    sample = np.array([[days, difficulty, past_score_val, topics, free_hours]])
    sample_scaled = scaler.transform(sample)
    prediction = model.predict(sample_scaled)[0]
    return round(max(1.0, prediction), 2)


print("\n── Example Predictions ──")
examples = [
    (3,  8.0, 55, 10, 3.0, "Crunch time, tough subject"),
    (10, 4.0, 85, 4,  6.0, "Comfortable pace, easy subject"),
    (1,  9.5, 40, 15, 2.0, "Last day, very hard, unprepared"),
]
for days, diff, ps, topics, fh, label in examples:
    hrs = predict_study_hours(days, diff, ps, topics, fh)
    print(f"  [{label}]  →  {hrs} hrs/day recommended")
