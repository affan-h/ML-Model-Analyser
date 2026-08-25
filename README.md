# ML Model Comparator

A self-service analytics tool that lets non-technical users upload any tabular dataset and instantly get a side-by-side comparison of classification models — no code required.

Upload a CSV, pick a target column, and the app handles the rest: cleaning the data, encoding and scaling it appropriately for each model type, training four classifiers, and returning a ranked, decision-ready comparison with the metrics needed to pick a model with confidence.

**🚀 Live app:** [ml-model-analyser-6hpvdxscx9wvknt768bjtk.streamlit.app](https://ml-model-analyser-6hpvdxscx9wvknt768bjtk.streamlit.app/)

## Demo

https://github.com/user-attachments/assets/demo.mov

> 📹 A full walkthrough of an earlier version of the app, loaded with the classic Titanic dataset — from upload through to the final model ranking. Core flow is unchanged in the current version; see "What's new in v2" below for what's improved since this recording.

## What it does

### 1. Upload & preview
- Drag-and-drop CSV upload — no setup, no environment, no code.
- Shows the first 15 rows so users can sanity-check their data immediately.

### 2. Target selection, with safety checks
- Pick the column to predict from a dropdown.
- **Regression-target guard:** if the selected column looks continuous (e.g. price, age) rather than categorical, the Run button is disabled with a clear explanation, instead of letting the app silently misinterpret every unique number as its own class.
- **Class balance warning:** flags when one class makes up more than 80% of the data, since accuracy becomes misleading in that case — F1 score is used for ranking instead.

### 3. Automated preprocessing pipeline
- Fills missing **numeric** values with the column mean, missing **categorical** values with the column mode (or `"Unknown"` if none exists).
- Coerces numeric-looking text to proper numeric types.
- **Model-appropriate encoding:** categorical columns are one-hot encoded for Logistic Regression and SVM (avoiding a false ordinal relationship between categories), and label-encoded for Decision Tree and Random Forest (which handle it natively).
- **Feature scaling:** numeric features are standardized for Logistic Regression and SVM, which are sensitive to feature scale; tree models skip this since they're scale-invariant.
- Validates the dataset before training and stops with a clear message if it's empty after cleaning, has fewer than 10 rows, or the target has fewer than 2 classes — including a defensive check that catches any class with fewer than 2 rows before the train/test split.

### 4. Multi-model benchmarking
Runs a stratified 80/20 train/test split (preserving class balance in both sets) through four scikit-learn classifiers:

| Model |
|---|
| Logistic Regression |
| Support Vector Machine (SVC) |
| Decision Tree |
| Random Forest |

For each model, the app captures F1 Score (weighted), Precision, Recall, Train/Test Accuracy, Error Rate, Overfit Gap, and Training Time. An optional 5-fold stratified cross-validation toggle is available for a more reliable estimate on smaller datasets.

Results are shown in a summary table with the best score in each column highlighted automatically.

### 5. Detailed analysis, on demand
- **Evaluation Metrics** — full precision/recall/F1/accuracy/error-rate breakdown per model.
- **Confusion Matrix** — a heatmap per model showing true vs. predicted classes.
- **Overfitting Graph** — train vs. test accuracy across all four models, side by side.
- **Feature Importance** — a Random Forest–derived bar chart of the top 10 most predictive features.

Training runs once per click and results are cached for the session, so switching between these views doesn't retrain the models.

### 6. Plain-English results summary
- Names the **recommended model** (highest F1 score) and its accuracy.
- States the **generalization gap** as a plain percentage, with a reliability verdict (flagging overfitting risk above a 10% gap, or excellent stability below 2%).
- A full **model ranking table**, sorted by F1 score.

## What's new in v2

The first version worked end to end but had a few real weaknesses, found through deliberate testing rather than code review alone:

- **Feature scaling added** — Logistic Regression and SVM previously trained on unscaled features, which likely understated their true performance compared to scale-invariant tree models.
- **Model-appropriate encoding** — previously every model used the same label encoding, which is fine for trees but imposes a false ordering on categories for linear/margin-based models. Now Logistic Regression and SVM get one-hot encoding instead.
- **Stratified splitting** — previously a plain random split, which can skew class balance between train and test sets, especially on smaller datasets.
- **Regression-target guard, fixed twice** — the first version of this guard used a ratio-based heuristic that let columns like `Age` (many repeated values, but still too many unique classes) slip through and crash the app during training. Replaced with a simple absolute threshold, plus a second defensive check right before the split as a backstop.
- **A real dtype bug fixed** — the original missing-value logic checked `dtype == "object"` to detect categorical columns, which silently breaks on newer pandas versions where text columns can report a native `str` dtype. Every categorical column was being wiped to zero before training. Fixed by using `pd.api.types.is_numeric_dtype()` instead.
- **Result caching** — training previously re-ran on every button click, including just switching between detail views. Now it runs once per "Run" click.
- **Simplified scope** — a correlation heatmap tab, a second (Logistic Regression) feature-importance view, hyperparameter tuning, and automatic datetime-column detection were all built and then deliberately removed, since they added complexity without a strong enough payoff. The Logistic Regression importance view in particular was actively misleading: high-cardinality identifier columns like `Name` or `Ticket` get one-hot encoded into hundreds of near-meaningless dummy columns, so the chart surfaced individual ticket numbers as "top features."

## Tech stack

- **Frontend / app framework:** [Streamlit](https://streamlit.io/)
- **Data handling:** Pandas, NumPy
- **Modeling:** scikit-learn (`LogisticRegression`, `SVC`, `DecisionTreeClassifier`, `RandomForestClassifier`, `ColumnTransformer`, `Pipeline`, `StandardScaler`, `OneHotEncoder`, `LabelEncoder`, `StratifiedKFold`, metrics module)
- **Visualization:** Matplotlib, Seaborn

## Try it now

No install needed — use the hosted version: **[ml-model-analyser-6hpvdxscx9wvknt768bjtk.streamlit.app](https://ml-model-analyser-6hpvdxscx9wvknt768bjtk.streamlit.app/)**

Upload a CSV, choose a target column, and click **Run Model Comparison**.

## Running locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open the local URL Streamlit prints (typically `http://localhost:8501`), upload a CSV, choose a target column, and click **Run Model Comparison**.

## Usage notes

- Works with any classification dataset — the target column needs a manageable number of repeated categories (not a continuous numeric column like price or age), and the dataset needs at least 10 rows.
- No manual data cleaning required; the preprocessing pipeline handles missing values, type coercion, and encoding automatically.
- Designed for exploration and quick benchmarking, not production model deployment — use it to identify a promising model family before deeper tuning.
- Known limitation: high-cardinality identifier-style columns (e.g. names, ticket numbers) are still one-hot encoded rather than being automatically excluded — this wastes some compute but doesn't affect correctness.

## Project structure

```
.
├── app.py                    # Streamlit application (UI, preprocessing, model training, visualization)
├── requirements.txt          # Python dependencies
├── CHANGES_v1_to_v2.md       # Detailed log of every fix and design decision between v1 and v2
├── assets/
│   └── demo.mov               # Screen-recorded walkthrough of the app in use
└── README.md
```
