import streamlit as st
import pandas as pd
import numpy as np
import time

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

import matplotlib.pyplot as plt
import seaborn as sns

# -------------------- SESSION STATE --------------------
if "section" not in st.session_state:
    st.session_state.section = None

if "results_ready" not in st.session_state:
    st.session_state.results_ready = False

# -------------------- PAGE CONFIG --------------------
st.set_page_config(page_title="ML Model Comparator", layout="wide")

# -------------------- CSS --------------------
st.markdown("""
<style>
    .block-container {
        max-width: 1200px;
        margin: auto;
        padding-top: 2rem;
    }
    
    h1, h2, h3, h4 {
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
    }
    
    .main-title {
        text-align: center;
        font-size: 38px;
        font-weight: 600;
        margin-bottom: 10px;
        color: #4AA3DF !important;
    }

    [data-testid="stFileUploader"] {
        width: 100%;
        max-width: 500px;
        margin: 0 auto;
    }
    section[data-testid="stFileUploader"] > div {
        border: 2px dashed #4AA3DF;
        border-radius: 8px;
        padding: 20px;
        background-color: transparent;
    }

    div.stButton > button {
        background-color: #2980B9;
        color: white;
        border-radius: 6px;
        font-weight: 500;
        border: none;
        width: 100%;
        transition: all 0.3s ease;
    }
    div.stButton > button:hover {
        background-color: #1A5276;
        color: white;
    }
    
    hr {
        margin-top: 2em;
        margin-bottom: 2em;
    }
</style>
""", unsafe_allow_html=True)

# -------------------- HEADER --------------------
st.markdown("<h1 class='main-title'>Machine Learning Model Comparator</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #7F8C8D; margin-bottom: 30px;'>Upload a dataset to evaluate and compare the performance of standard classification algorithms.</p>", unsafe_allow_html=True)

col_upload1, col_upload2, col_upload3 = st.columns([1, 2, 1])
with col_upload2:
    uploaded_file = st.file_uploader("Upload Dataset (CSV format)", type=["csv"])

st.markdown("<hr>", unsafe_allow_html=True)

# -------------------- MAIN --------------------
if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.subheader("Data Overview")
    
    # Tabs for organized data presentation
    tab_preview, tab_corr = st.tabs(["Dataset Preview", "Correlation Heatmap"])
    
    with tab_preview:
        st.dataframe(df.head(15), use_container_width=True)
        
    with tab_corr:
        numeric_df = df.select_dtypes(include=[np.number])
        if not numeric_df.empty and len(numeric_df.columns) > 1:
            fig_corr, ax_corr = plt.subplots(figsize=(10, 6))
            sns.heatmap(numeric_df.corr(), annot=True, fmt=".2f", cmap="Blues", ax=ax_corr, cbar_kws={'shrink': .8})
            ax_corr.set_title("Feature Correlation", pad=15)
            st.pyplot(fig_corr)
        else:
            st.info("Not enough numeric columns available to generate a correlation heatmap.")

    st.markdown("<br>", unsafe_allow_html=True)
    
    col_target1, col_target2 = st.columns([1, 3])
    with col_target1:
        target_col = st.selectbox("Select Target Column", df.columns, key='target_col')
    
    with col_target1:
        # -------------------- RUN BUTTON --------------------
        if st.button("Run Model Comparison"):
            st.session_state.results_ready = True
            st.session_state.section = None

    st.markdown("<hr>", unsafe_allow_html=True)

    # -------------------- MAIN LOGIC --------------------
    if st.session_state.results_ready:

        # -------- PREPROCESSING --------
        for col in df.columns:
            if df[col].dtype != 'object':
                df[col] = pd.to_numeric(df[col], errors='coerce')

            if df[col].dtype == 'object':
                if df[col].mode().empty:
                    df[col] = df[col].fillna("Unknown")
                else:
                    df[col] = df[col].fillna(df[col].mode()[0])
            else:
                if df[col].isna().all():
                    df[col] = df[col].fillna(0)
                else:
                    df[col] = df[col].fillna(df[col].mean())

        df = df.fillna(0)

        X = df.drop(columns=[target_col])
        y = df[target_col]

        for col in X.columns:
            if X[col].dtype == 'object':
                X[col] = LabelEncoder().fit_transform(X[col])

        if y.dtype == 'object':
            y = LabelEncoder().fit_transform(y)

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

        # -------- SPLIT --------
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # -------- MODELS --------
        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "SVM": SVC(),
            "Decision Tree": DecisionTreeClassifier(random_state=42),
            "Random Forest": RandomForestClassifier(random_state=42)
        }

        results = []
        eval_metrics = {}
        confusion_data = {}

        for name, model in models.items():
            start = time.time()
            model.fit(X_train, y_train)
            train_time = time.time() - start

            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            train_acc = accuracy_score(y_train, y_train_pred)
            test_acc = accuracy_score(y_test, y_test_pred)

            precision = precision_score(y_test, y_test_pred, average='weighted', zero_division=0)
            recall = recall_score(y_test, y_test_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test, y_test_pred, average='weighted', zero_division=0)

            error_rate = 1 - test_acc
            overfit_gap = train_acc - test_acc

            results.append({
                "Model": name,
                "F1 Score": f1,
                "Train Accuracy": train_acc,
                "Test Accuracy": test_acc,
                "Error Rate": error_rate,
                "Overfit Gap": overfit_gap,
                "Training Time (s)": train_time
            })

            eval_metrics[name] = {
                "F1 Score": f1,
                "Accuracy": test_acc,
                "Precision": precision,
                "Recall": recall,
                "Error Rate": error_rate
            }

            confusion_data[name] = confusion_matrix(y_test, y_test_pred)

        results_df = pd.DataFrame(results)

        # -------- TABLE HIGHLIGHTING --------
        def highlight_best_metrics(s):
            if s.name in ['F1 Score', 'Train Accuracy', 'Test Accuracy']:
                is_best = s == s.max()
                return ['background-color: rgba(46, 204, 113, 0.2); font-weight: bold' if v else '' for v in is_best]
            elif s.name in ['Error Rate', 'Overfit Gap', 'Training Time (s)']:
                if s.name == 'Overfit Gap':
                    is_best = s.abs() == s.abs().min()
                else:
                    is_best = s == s.min()
                return ['background-color: rgba(46, 204, 113, 0.2); font-weight: bold' if v else '' for v in is_best]
            return ['' for _ in s]

        styled_results = results_df.style.apply(highlight_best_metrics).format({
            "F1 Score": "{:.4f}",
            "Train Accuracy": "{:.4f}",
            "Test Accuracy": "{:.4f}",
            "Error Rate": "{:.4f}",
            "Overfit Gap": "{:.4f}",
            "Training Time (s)": "{:.4f}"
        })

        st.subheader("Model Comparison Summary")
        st.dataframe(styled_results, use_container_width=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # -------- VISUALIZATION BUTTONS --------
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

        # -------- DISPLAY --------
        if st.session_state.section == "eval":
            st.markdown("#### Evaluation Metrics")
            eval_df = pd.DataFrame(eval_metrics).T
            st.dataframe(eval_df.style.format("{:.4f}"), use_container_width=True)

        elif st.session_state.section == "confusion":
            st.markdown("#### Confusion Matrices")
            st.markdown("Displays the count of true predictions versus false predictions for each class.")

            model_names = list(confusion_data.keys())

            col_cm1, col_cm2 = st.columns(2)
            col_cm3, col_cm4 = st.columns(2)
            cols = [col_cm1, col_cm2, col_cm3, col_cm4]

            for i, (name, cm) in enumerate(confusion_data.items()):
                with cols[i]:
                    fig, ax = plt.subplots(figsize=(5, 4))
                    sns.heatmap(
                        cm,
                        annot=True,
                        fmt='d',
                        cmap='Blues',
                        cbar=False,
                        linewidths=0.5,
                        linecolor='gray',
                        ax=ax
                    )
                    ax.set_title(name, pad=10, fontweight='bold')
                    ax.set_xlabel("Predicted Class")
                    ax.set_ylabel("Actual Class")
                    st.pyplot(fig, use_container_width=True)

        elif st.session_state.section == "overfit":
            st.markdown("#### Overfitting Analysis")
            st.markdown("Compares training accuracy against testing accuracy. A large gap between the two lines indicates the model is memorizing the training data rather than generalizing.")
            
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(results_df["Model"], results_df["Train Accuracy"], marker='o', label="Train Accuracy", linewidth=2)
            ax.plot(results_df["Model"], results_df["Test Accuracy"], marker='o', label="Test Accuracy", linewidth=2)
            
            ax.set_ylabel("Accuracy")
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.legend()
            st.pyplot(fig)
            
        elif st.session_state.section == "importance":
            st.markdown("#### Feature Importance")
            st.markdown("Calculated using the Random Forest algorithm, this chart displays which columns had the most significant impact on predicting the target variable.")
            
            rf_model = models["Random Forest"]
            importances = rf_model.feature_importances_
            
            feat_imp_df = pd.DataFrame({
                'Feature': X.columns,
                'Importance': importances
            }).sort_values(by='Importance', ascending=False)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(data=feat_imp_df.head(10), x='Importance', y='Feature', palette='Blues_r', ax=ax)
            ax.set_title("Top 10 Most Predictive Features", pad=15)
            ax.set_xlabel("Relative Importance Score")
            ax.set_ylabel("")
            st.pyplot(fig)

        # -------- FINAL RESULT SUMMARY --------
        st.markdown("<hr>", unsafe_allow_html=True)
        st.subheader("Results")

        # Select best model based on F1 Score
        best_model_row = results_df.loc[results_df["F1 Score"].idxmax()]
        best_model_name = best_model_row['Model']
        best_f1 = best_model_row['F1 Score']
        best_accuracy = best_model_row['Test Accuracy']
        overfit_gap = best_model_row['Overfit Gap']

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
            ranking_df = results_df[['Model', 'F1 Score', 'Test Accuracy']].sort_values(by="F1 Score", ascending=False).reset_index(drop=True)
            ranking_df.index = ranking_df.index + 1
            st.dataframe(ranking_df.style.format({"F1 Score": "{:.4f}", "Test Accuracy": "{:.4f}"}), use_container_width=True)