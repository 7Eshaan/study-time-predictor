# 📚 Student Study Time Predictor

A machine learning project that predicts **how many hours per day a student should study** before an exam, based on real factors like subject difficulty, past performance, and available time.

Built using **Linear Regression** as part of the *Fundamentals of AI and ML* course (VIT).

---

## 🔍 Problem Statement

Students often struggle to plan study time effectively — some over-prepare, others under-prepare. This tool takes a set of measurable inputs about a student's situation and outputs a recommended daily study hours figure, helping students allocate their time more strategically.

---

## 🧠 What It Does

- Trains a **Linear Regression** model on a synthetic but realistic student dataset
- Evaluates model performance using **RMSE** and **R²**
- Visualizes actual vs predicted values, feature importance, and residuals
- Accepts a **custom student profile** and returns a study hour recommendation

---

## 📦 Requirements

```
Python 3.8+
numpy
pandas
scikit-learn
matplotlib
```

Install all dependencies with:

```bash
pip install numpy pandas scikit-learn matplotlib
```

---

## 🚀 How to Run

1. Clone the repository:
```bash
git clone https://github.com/YOUR_USERNAME/study-time-predictor.git
cd study-time-predictor
```

2. Run the predictor:
```bash
python predictor.py
```

3. Output includes:
   - Model performance metrics printed to terminal
   - `results.png` — three plots saved to your working directory
   - Example predictions for different student profiles

---

## 📊 Input Features

| Feature | Description |
|---|---|
| `days_until_exam` | Days remaining before the exam (1–14) |
| `subject_difficulty` | Perceived difficulty of the subject (1–10) |
| `past_score` | Score in the previous exam (%) |
| `topics_remaining` | Number of chapters/topics not yet covered |
| `daily_free_hours` | Free hours available per day for studying |

**Target variable:** `study_hours_needed` — recommended hours to study per day

---

## 📈 Results

The model achieves:
- **RMSE ≈ 0.83 hours** — predictions are typically within 1 hour of the actual value
- **R² ≈ 0.87** — the model explains ~87% of variance in study requirements

---

## 🗂️ Project Structure

```
study-time-predictor/
├── predictor.py      # Main ML pipeline (data → train → evaluate → predict)
├── results.png       # Output visualisation (auto-generated)
└── README.md         # This file
```

---

## 💡 How to Use the Predictor Function

You can call `predict_study_hours()` directly for any student profile:

```python
hours = predict_study_hours(
    days=5,
    difficulty=7.5,
    past_score_val=62,
    topics=8,
    free_hours=4.0
)
print(f"Recommended: {hours} hrs/day")
```

---

## 🔄 Extending This Project

Ideas for future work:
- Collect real student data via a survey form
- Add more features (sleep hours, stress level, group study)
- Try Polynomial Regression or Ridge Regression for non-linear patterns
- Build a simple web UI with Flask or Streamlit

---

## 👤 Author

**Eshaan Dogra**
**25BCE10675**
B.Tech Computer Science | VIT Bhopal
Course: Fundamentals of AI and ML
