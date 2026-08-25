import streamlit as st
import pandas as pd
import numpy as np
import time

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================
# SESSION STATE
# ============================================================
if "results_ready" not in st.session_state:
    st.session_state.results_ready = False
if "section" not in st.session_state:
    st.session_state.section = None
if "trained" not in st.session_state:
    st.session_state.trained = None  # caches everything computed on "Run", so
                                      # switching detail-view tabs doesn't retrain

# ============================================================
# PAGE CONFIG + CSS
# ============================================================
st.set_page_config(page_title="ML Model Comparator", layout="wide")

st.markdown("""
<style>
    .block-container { max-width: 1200px; margin: auto; padding-top: 2rem; }
    h1, h2, h3, h4 { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; }
    .main-title { text-align: center; font-size: 38px; font-weight: 600; margin-bottom: 10px; color: #4AA3DF !important; }
    [data-testid="stFileUploader"] { width: 100%; max-width: 500px; margin: 0 auto; }
    section[data-testid="stFileUploader"] > div { border: 2px dashed #4AA3DF; border-radius: 8px; padding: 20px; background-color: transparent; }
    div.stButton > button { background-color: #2980B9; color: white; border-radius: 6px; font-weight: 500; border: none; width: 100%; transition: all 0.3s ease; }
    div.stButton > button:hover { background-color: #1A5276; color: white; }
    hr { margin-top: 2em; margin-bottom: 2em; }
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='main-title'>Machine Learning Model Comparator</h1>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align: center; color: #7F8C8D; margin-bottom: 30px;'>"
    "Upload a dataset to evaluate and compare the performance of standard classification algorithms.</p>",
    unsafe_allow_html=True
)

col_upload1, col_upload2, col_upload3 = st.columns([1, 2, 1])
with col_upload2:
    uploaded_file = st.file_uploader("Upload Dataset (CSV format)", type=["csv"])

st.markdown("<hr>", unsafe_allow_html=True)


# ============================================================
# TARGET TYPE GUARD
# v1 only checked "at least 2 unique values", so picking a continuous
# numeric column (like Age or Fare) as the target silently treated
# every unique number as its own class, then crashed later inside
# train_test_split(stratify=y) with a confusing raw sklearn error.
# This blocks that up front with a plain-English message.
# ============================================================
def target_is_likely_regression(y, max_classes=20):
    """
    A numeric column with more than `max_classes` distinct values isn't a
    realistic classification target — e.g. Age (~90 unique values) or Fare
    (~250 unique values) in the Titanic dataset. A single absolute threshold
    is used rather than a ratio of unique-to-total-rows: an earlier ratio-based
    version let Age slip through because its uniqueness ratio was low even
    though its absolute number of distinct values was clearly too high for
    classification.
    """
    if pd.api.types.is_numeric_dtype(y):
        if y.nunique() > max_classes:
            return True
    return False


# ============================================================
# PREPROCESSING
# ============================================================
def clean_missing_values(df):
    """
    Fills missing values: mean for numeric columns, most-common value (mode)
    for categorical columns. Simple and fast, appropriate for a tool that has
    to handle any dataset a user uploads without knowing its structure ahead
    of time — the tradeoff is it's not the statistically optimal choice for
    any single dataset (median or model-based imputation would be more
    robust, at the cost of more complexity).

    Uses pd.api.types.is_numeric_dtype() rather than comparing df[col].dtype
    to "object" directly — on newer pandas versions text columns can report
    a native `str` dtype instead of `object`, which silently broke that
    comparison and caused every categorical column to be wiped to zero.
    """
    df = df.copy()
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            if df[col].isna().all():
                df[col] = df[col].fillna(0)
            else:
                df[col] = df[col].fillna(df[col].mean())
        else:
            # handle numbers that were read in as text (e.g. "42" as a string)
            coerced = pd.to_numeric(df[col], errors="coerce")
            if coerced.notna().sum() > 0.9 * len(coerced):
                df[col] = coerced.fillna(coerced.mean() if coerced.notna().any() else 0)
            else:
                mode = df[col].mode()
                df[col] = df[col].fillna(mode[0] if not mode.empty else "Unknown")
    return df.fillna(0)


def build_onehot_scaled_preprocessor(categorical_cols, numeric_cols):
    """
    Preprocessing for Logistic Regression and SVM: one-hot encode categories
    (so we don't invent a false ordinal relationship between them, e.g.
    implying 'Blue' is 'between' 'Red' and 'Green') and scale numeric columns
    (these two models are sensitive to feature scale; tree models are not,
    so trees skip this and use label encoding directly instead).
    """
    transformers = []
    if categorical_cols:
        transformers.append(("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols))
    if numeric_cols:
        transformers.append(("num", StandardScaler(), numeric_cols))
    return ColumnTransformer(transformers, remainder="drop")


# ============================================================
# MAIN
# ============================================================
if uploaded_file:
    raw_df = pd.read_csv(uploaded_file)

    st.subheader("Data Overview")
    st.dataframe(raw_df.head(15), use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col_target1, col_target2 = st.columns([1, 3])
    with col_target1:
        target_col = st.selectbox("Select Target Column", raw_df.columns, key="target_col")

    # Warn immediately if the chosen target looks continuous, before Run is even clicked.
    target_warning = None
    if target_col:
        if target_is_likely_regression(raw_df[target_col]):
            target_warning = (
                f"'{target_col}' looks like a continuous numeric column "
                f"({raw_df[target_col].nunique()} unique values). This tool only supports "
                f"classification (predicting categories), not regression (predicting a number). "
                f"Pick a column with a small number of repeated categories instead."
            )
            st.error(target_warning)

    # Class balance check — shown inline, not as a separate tab, since it's
    # one glance a user needs before running, not something to browse.
    if target_col and not target_warning:
        balance = raw_df[target_col].value_counts(normalize=True) * 100
        if balance.max() > 80:
            st.warning(
                f"'{target_col}' is imbalanced — one class makes up {balance.max():.1f}% of the "
                f"data. Accuracy alone will be misleading here; rely on F1 score instead."
            )

    with col_target1:
        run_disabled = target_warning is not None
        if st.button("Run Model Comparison", disabled=run_disabled):
            st.session_state.results_ready = True
            st.session_state.section = None
            st.session_state.trained = None  # force recompute on a fresh run

    with col_target2:
        use_cv = st.checkbox("Use 5-fold cross-validation (slower, more reliable)", value=False)

    st.markdown("<hr>", unsafe_allow_html=True)

    # ============================================================
    # PIPELINE — only recomputed on "Run", cached afterward so
    # switching between detail-view buttons doesn't retrain.
    # ============================================================
    if st.session_state.results_ready:

        if st.session_state.trained is None:
            with st.spinner("Cleaning data, training models..."):

                df = clean_missing_values(raw_df)

                X = df.drop(columns=[target_col])
                y = df[target_col]

                if not pd.api.types.is_numeric_dtype(y):
                    y = LabelEncoder().fit_transform(y)
                else:
                    y = y.values

                # -------- VALIDATION --------
                if len(X) == 0:
                    st.error("The dataset became empty after preprocessing.")
                    st.stop()
                if len(df) < 10:
                    st.error("The dataset is too small for reliable model training.")
                    st.stop()
                if len(np.unique(y)) < 2:
                    st.error("The target variable must have at least 2 distinct classes.")
                    st.stop()

                # Defensive check: stratified splitting needs every class to have at
                # least 2 rows. If a target ever slips past the guard above, fail here
                # with a clear message instead of a raw sklearn ValueError.
                class_counts = pd.Series(y).value_counts()
                if (class_counts < 2).any():
                    too_small = class_counts[class_counts < 2].index.tolist()
                    st.error(
                        f"Target column '{target_col}' has classes with only 1 row "
                        f"(e.g. {too_small[:5]}). Each class needs at least 2 rows to split "
                        f"into train/test sets — pick a column with fewer, repeated categories."
                    )
                    st.session_state.results_ready = False
                    st.stop()

                categorical_cols = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
                numeric_cols = [c for c in X.columns if c not in categorical_cols]

                # Label-encode categoricals once, for direct use by tree models
                X_label = X.copy()
                for c in categorical_cols:
                    X_label[c] = LabelEncoder().fit_transform(X_label[c].astype(str))

                # -------- SPLIT (stratified, so class balance is preserved) --------
                X_train, X_test, y_train, y_test = train_test_split(
                    X_label, y, test_size=0.2, random_state=42, stratify=y
                )
                X_train_raw = X.loc[X_train.index]  # original (pre-label-encoding) values,
                X_test_raw = X.loc[X_test.index]     # needed for the one-hot pipelines below

                onehot_scaled_pre = build_onehot_scaled_preprocessor(categorical_cols, numeric_cols)

                model_specs = {
                    "Logistic Regression": {
                        "estimator": Pipeline([("prep", onehot_scaled_pre), ("clf", LogisticRegression(max_iter=1000))]),
                        "uses_raw_X": True,
                    },
                    "SVM": {
                        "estimator": Pipeline([("prep", onehot_scaled_pre), ("clf", SVC())]),
                        "uses_raw_X": True,
                    },
                    "Decision Tree": {
                        "estimator": DecisionTreeClassifier(random_state=42),
                        "uses_raw_X": False,
                    },
                    "Random Forest": {
                        "estimator": RandomForestClassifier(random_state=42),
                        "uses_raw_X": False,
                    },
                }

                results = []
                eval_metrics = {}
                confusion_data = {}
                fitted_models = {}
                cv_splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

                for name, spec in model_specs.items():
                    Xtr = X_train_raw if spec["uses_raw_X"] else X_train
                    Xte = X_test_raw if spec["uses_raw_X"] else X_test
                    model = spec["estimator"]

                    start = time.time()
                    model.fit(Xtr, y_train)
                    train_time = time.time() - start

                    y_train_pred = model.predict(Xtr)
                    y_test_pred = model.predict(Xte)

                    train_acc = accuracy_score(y_train, y_train_pred)
                    test_acc = accuracy_score(y_test, y_test_pred)
                    precision = precision_score(y_test, y_test_pred, average="weighted", zero_division=0)
                    recall = recall_score(y_test, y_test_pred, average="weighted", zero_division=0)
                    f1 = f1_score(y_test, y_test_pred, average="weighted", zero_division=0)
                    error_rate = 1 - test_acc
                    overfit_gap = train_acc - test_acc

                    row = {
                        "Model": name,
                        "F1 Score": f1,
                        "Train Accuracy": train_acc,
                        "Test Accuracy": test_acc,
                        "Error Rate": error_rate,
                        "Overfit Gap": overfit_gap,
                        "Training Time (s)": train_time,
                    }

                    if use_cv:
                        Xcv = X_train_raw if spec["uses_raw_X"] else X_train
                        scores = cross_val_score(model, Xcv, y_train, cv=cv_splitter, scoring="f1_weighted")
                        row["CV F1 (mean)"] = scores.mean()
                        row["CV F1 (std)"] = scores.std()

                    results.append(row)
                    eval_metrics[name] = {
                        "F1 Score": f1, "Accuracy": test_acc,
                        "Precision": precision, "Recall": recall, "Error Rate": error_rate,
                    }
                    confusion_data[name] = confusion_matrix(y_test, y_test_pred)
                    fitted_models[name] = model

                st.session_state.trained = {
                    "results_df": pd.DataFrame(results),
                    "eval_metrics": eval_metrics,
                    "confusion_data": confusion_data,
                    "fitted_models": fitted_models,
                    "X_columns": X.columns.tolist(),
                    "use_cv": use_cv,
                }

        # ---- pull cached results out of session_state ----
        T = st.session_state.trained
        results_df = T["results_df"]
        eval_metrics = T["eval_metrics"]
        confusion_data = T["confusion_data"]
        fitted_models = T["fitted_models"]

        # -------- TABLE HIGHLIGHTING --------
        def highlight_best_metrics(s):
            if s.name in ["F1 Score", "Train Accuracy", "Test Accuracy", "CV F1 (mean)"]:
                is_best = s == s.max()
            elif s.name in ["Error Rate", "Training Time (s)", "CV F1 (std)"]:
                is_best = s == s.min()
            elif s.name == "Overfit Gap":
                is_best = s.abs() == s.abs().min()
            else:
                return ["" for _ in s]
            return ["background-color: rgba(46, 204, 113, 0.2); font-weight: bold" if v else "" for v in is_best]

        fmt = {c: "{:.4f}" for c in results_df.columns if c != "Model"}
        styled_results = results_df.style.apply(highlight_best_metrics).format(fmt)

        st.subheader("Model Comparison Summary")
        st.dataframe(styled_results, use_container_width=True)
        st.caption(
            "Logistic Regression and SVM were one-hot encoded and scaled; Decision Tree and "
            "Random Forest used label encoding (tree models are scale-invariant, so scaling "
            "wasn't needed)."
        )

        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown("### Detailed Analysis")
        col_btn1, col_btn2, col_btn3, col_btn4 = st.columns(4)

        if col_btn1.button("Evaluation Metrics"):
            st.session_state.section = "eval"
        if col_btn2.button("Confusion Matrix"):
            st.session_state.section = "confusion"
        if col_btn3.button("Overfitting Graph"):
            st.session_state.section = "overfit"
        if col_btn4.button("Feature Importance"):
            st.session_state.section = "importance"

        st.markdown("<br>", unsafe_allow_html=True)

        if st.session_state.section == "eval":
            st.markdown("#### Evaluation Metrics")
            eval_df = pd.DataFrame(eval_metrics).T
            st.dataframe(eval_df.style.format("{:.4f}"), use_container_width=True)

        elif st.session_state.section == "confusion":
            st.markdown("#### Confusion Matrices")
            st.markdown("Displays the count of true predictions versus false predictions for each class.")
            cols = st.columns(2) + st.columns(2)
            for i, (name, cm) in enumerate(confusion_data.items()):
                with cols[i]:
                    fig, ax = plt.subplots(figsize=(5, 4))
                    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                                linewidths=0.5, linecolor="gray", ax=ax)
                    ax.set_title(name, pad=10, fontweight="bold")
                    ax.set_xlabel("Predicted Class")
                    ax.set_ylabel("Actual Class")
                    st.pyplot(fig, use_container_width=True)

        elif st.session_state.section == "overfit":
            st.markdown("#### Overfitting Analysis")
            st.markdown(
                "Compares training accuracy against testing accuracy. A large gap between the "
                "two lines indicates the model is memorizing the training data rather than "
                "generalizing."
            )
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(results_df["Model"], results_df["Train Accuracy"], marker="o", label="Train Accuracy", linewidth=2)
            ax.plot(results_df["Model"], results_df["Test Accuracy"], marker="o", label="Test Accuracy", linewidth=2)
            ax.set_ylabel("Accuracy")
            ax.grid(True, linestyle="--", alpha=0.7)
            ax.legend()
            st.pyplot(fig)

        elif st.session_state.section == "importance":
            st.markdown("#### Feature Importance")
            st.markdown(
                "Calculated using the Random Forest algorithm, this chart displays which columns "
                "had the most significant impact on predicting the target variable."
            )
            rf_model = fitted_models["Random Forest"]
            importances = rf_model.feature_importances_
            feat_imp_df = pd.DataFrame({
                "Feature": T["X_columns"], "Importance": importances
            }).sort_values(by="Importance", ascending=False)
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(data=feat_imp_df.head(10), x="Importance", y="Feature", palette="Blues_r", ax=ax)
            ax.set_title("Top 10 Most Predictive Features", pad=15)
            ax.set_xlabel("Relative Importance Score")
            ax.set_ylabel("")
            st.pyplot(fig)

        # -------- FINAL RESULT SUMMARY --------
        st.markdown("<hr>", unsafe_allow_html=True)
        st.subheader("Results")

        best_model_row = results_df.loc[results_df["F1 Score"].idxmax()]
        best_model_name = best_model_row["Model"]
        best_f1 = best_model_row["F1 Score"]
        best_accuracy = best_model_row["Test Accuracy"]
        overfit_gap = best_model_row["Overfit Gap"]

        col_res1, col_res2 = st.columns([2, 1])

        with col_res1:
            st.markdown("#### Selection Summary")
            reliability_status = "Adequate balance between learning patterns and generalizing to unseen data."
            if overfit_gap > 0.10:
                reliability_status = "High risk of overfitting. Model may struggle with highly varied future data."
            elif overfit_gap < 0.02:
                reliability_status = "Excellent stability. High confidence when deploying against entirely new data."

            st.markdown(f"""
* **Recommended model:** **{best_model_name}** achieved the highest F1-Score ({best_f1:.4f}).
* **Overall Accuracy:** Successfully predicted {best_accuracy * 100:.1f}% of the validation data.
* **Generalization Gap:** Identified a {overfit_gap * 100:.1f}% difference between training and testing performance.
* **Reliability Check:** {reliability_status}
            """)

        with col_res2:
            st.markdown("#### Model Rank")
            rank_cols = ["Model", "F1 Score", "Test Accuracy"]
            if "CV F1 (mean)" in results_df.columns:
                rank_cols.append("CV F1 (mean)")
            ranking_df = results_df[rank_cols].sort_values(by="F1 Score", ascending=False).reset_index(drop=True)
            ranking_df.index = ranking_df.index + 1
            st.dataframe(ranking_df.style.format({c: "{:.4f}" for c in rank_cols if c != "Model"}), use_container_width=True)
