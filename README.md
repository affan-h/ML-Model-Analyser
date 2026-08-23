# ML Model Comparator

A self-service analytics tool that lets non-technical users upload any tabular dataset and instantly get automated exploratory data analysis plus a side-by-side comparison of classification models — no code required.

Upload a CSV, pick a target column, and the app handles the rest: cleaning the data, training four standard classifiers, and returning a ranked, decision-ready comparison with the metrics needed to pick a model with confidence.

**🚀 Live app:** [ml-model-analyser-6hpvdxscx9wvknt768bjtk.streamlit.app](https://ml-model-analyser-6hpvdxscx9wvknt768bjtk.streamlit.app/)

## Demo

https://github.com/user-attachments/assets/demo.mov

> 📹 A full walkthrough is included at [`assets/demo.mov`](assets/demo.mov), showing the app loaded with the classic Titanic dataset — from upload through to the final model ranking.

## What it does

### 1. Upload & explore
- Drag-and-drop CSV upload — no setup, no environment, no code.
- **Dataset Preview** tab shows the first 15 rows so users can sanity-check their data immediately.
- **Correlation Heatmap** tab auto-generates a Seaborn heatmap across all numeric columns, giving an instant read on feature relationships (skipped gracefully if there aren't enough numeric columns).

### 2. Automated preprocessing pipeline
Before any model sees the data, a Pandas/NumPy pipeline cleans it up automatically, handling common real-world messiness so the user never has to:
- Coerces numeric-looking columns to proper numeric types (`pd.to_numeric`, invalid entries become `NaN` rather than crashing the run).
- Fills missing **categorical** values with the column mode (or `"Unknown"` if the column has no mode at all, e.g. all-null).
- Fills missing **numeric** values with the column mean (or `0` if the entire column is null).
- Label-encodes categorical features and the target column so every model can consume them.
- Runs validation checks before training and stops early with a clear error if:
  - the dataset is empty after cleaning,
  - there are fewer than 10 rows (too small to train/test reliably), or
  - the target column has fewer than 2 distinct classes.

### 3. Multi-model benchmarking
The cleaned data is split 80/20 (train/test, fixed random seed for reproducibility) and run through four scikit-learn classifiers:

| Model | 
|---|
| Logistic Regression |
| Support Vector Machine (SVC) |
| Decision Tree |
| Random Forest |

For each model, the app captures:
- **F1 Score** (weighted) — the primary metric used to rank models
- **Precision** and **Recall** (weighted)
- **Train Accuracy** vs **Test Accuracy**
- **Error Rate**
- **Overfit Gap** (train accuracy − test accuracy) — flags models that memorize rather than generalize
- **Training Time** — useful when speed matters as much as accuracy

Results are shown in a summary table with the best score in each column highlighted automatically, so the strongest model per metric is visible at a glance.

### 4. Detailed analysis, on demand
Four toggleable views dig deeper into the results:
- **Evaluation Metrics** — full precision/recall/F1/accuracy/error-rate breakdown per model.
- **Confusion Matrix** — a heatmap per model showing true vs. predicted classes.
- **Overfitting Graph** — a line chart of train vs. test accuracy across all four models, making generalization gaps easy to spot visually.
- **Feature Importance** — a Random Forest–derived bar chart of the top 10 most predictive features in the dataset.

### 5. Plain-English results summary
The app closes with an automated readout that translates the numbers into a recommendation:
- Names the **recommended model** (highest F1 score) and its accuracy.
- States the **generalization gap** as a plain percentage.
- Gives a **reliability verdict** — flagging high overfitting risk (gap > 10%), calling out excellent stability (gap < 2%), or noting an adequate balance in between.
- A full **model ranking table**, sorted by F1 score, for quick reference.

## Tech stack

- **Frontend / app framework:** [Streamlit](https://streamlit.io/)
- **Data handling:** Pandas, NumPy
- **Modeling:** scikit-learn (`LogisticRegression`, `SVC`, `DecisionTreeClassifier`, `RandomForestClassifier`, `train_test_split`, `LabelEncoder`, metrics module)
- **Visualization:** Matplotlib, Seaborn

## Try it now

No install needed — use the hosted version: **[ml-model-analyser-6hpvdxscx9wvknt768bjtk.streamlit.app](https://ml-model-analyser-6hpvdxscx9wvknt768bjtk.streamlit.app/)**

Upload a CSV, choose a target column, and click **Run Model Comparison**.

## Running locally

```bash
pip install streamlit pandas numpy scikit-learn matplotlib seaborn
streamlit run app.py
```

Then open the local URL Streamlit prints (typically `http://localhost:8501`), upload a CSV, choose a target column, and click **Run Model Comparison**.

## Usage notes

- Works with any classification dataset — the target column just needs at least 2 distinct classes and the dataset needs at least 10 rows.
- No manual data cleaning required; the preprocessing pipeline handles missing values and type coercion automatically.
- Designed for exploration and quick benchmarking, not production model deployment — use it to identify a promising model family before deeper tuning.

## Project structure

```
.
├── app.py              # Streamlit application (UI, preprocessing, model training, visualization)
├── assets/
│   └── demo.mov         # Screen-recorded walkthrough of the app in use
└── README.md
```
