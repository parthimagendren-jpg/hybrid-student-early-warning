"""
SIMPLE HYBRID INTELLIGENT SYSTEM
Student Performance Early-Warning System
Technique: ANN + Simple Fuzzy Logic

By default the script expects StudentPerformanceFactors.csv in the same
folder as this file. You can point it elsewhere with --data.

Install once:
    pip install -r requirements.txt

Run:
    python Hybrid_AI_Simple_Code.py
    python Hybrid_AI_Simple_Code.py --data path/to/StudentPerformanceFactors.csv --out outputs_hybrid

The program will:
1. Show important tables in the terminal/Jupyter output.
2. Show important figures after running.
3. Save all important tables, figures, predictions, and the trained model.
"""

from pathlib import Path
import argparse
import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")

# ============================================================================
# 1. SETTINGS
# ============================================================================

DATA_PATH = Path("StudentPerformanceFactors.csv")
OUTPUT_FOLDER = Path("outputs_hybrid")
RANDOM_STATE = 42
AT_RISK_SCORE = 65
SHOW_FIGURES = True


# ============================================================================
# 2. HELPER FUNCTIONS
# ============================================================================


def show_table(title, table, rows=10):
    """Display a table in Jupyter or print it in a normal Python terminal."""
    print("\n" + "=" * 75)
    print(title)
    print("=" * 75)

    table_to_show = table.head(rows)

    try:
        from IPython.display import display
        display(table_to_show)
    except ImportError:
        print(table_to_show.to_string(index=False))


def save_and_show_figure(filename):
    """Save a figure and display it after running."""
    plt.tight_layout()
    plt.savefig(OUTPUT_FOLDER / filename, dpi=220, bbox_inches="tight")

    if SHOW_FIGURES:
        plt.show()

    plt.close()


def simple_fuzzy_risk(attendance, hours_studied, previous_score):
    """
    Simple fuzzy-style risk score from 0 to 100.
    Higher score means higher academic risk.
    """
    risk = 0
    reasons = []

    # Attendance rules
    if attendance < 75:
        risk += 40
        reasons.append("Low attendance")
    elif attendance < 85:
        risk += 20
        reasons.append("Moderate attendance")
    else:
        risk += 5

    # Study-hour rules
    if hours_studied < 12:
        risk += 30
        reasons.append("Low study hours")
    elif hours_studied < 20:
        risk += 15
        reasons.append("Moderate study hours")
    else:
        risk += 5

    # Previous-score rules
    if previous_score < 65:
        risk += 30
        reasons.append("Low previous score")
    elif previous_score < 75:
        risk += 15
        reasons.append("Moderate previous score")
    else:
        risk += 5

    risk = min(risk, 100)

    if risk >= 65:
        label = "High Risk"
    elif risk >= 40:
        label = "Medium Risk"
    else:
        label = "Low Risk"

    explanation = ", ".join(reasons) if reasons else "Strong learning indicators"
    return risk, label, explanation


def calculate_metrics(model_name, actual, predicted, probability):
    """Return important model metrics as one dictionary."""
    return {
        "Model": model_name,
        "Accuracy": round(accuracy_score(actual, predicted), 4),
        "Precision_At_Risk": round(
            precision_score(actual, predicted, pos_label=1, zero_division=0), 4
        ),
        "Recall_At_Risk": round(
            recall_score(actual, predicted, pos_label=1, zero_division=0), 4
        ),
        "F1_At_Risk": round(
            f1_score(actual, predicted, pos_label=1, zero_division=0), 4
        ),
        "ROC_AUC": round(roc_auc_score(actual, probability), 4),
    }


# ============================================================================
# 3. MAIN PROJECT
# ============================================================================


def main(data_path, output_folder):
    global OUTPUT_FOLDER
    OUTPUT_FOLDER = Path(output_folder)
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------------
    # A. LOAD DATA
    # ------------------------------------------------------------------------
    data_path = Path(data_path)

    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset not found:\n{data_path}\n\n"
            "Check the filename and folder path."
        )

    df = pd.read_csv(data_path)
    print(f"\nDataset loaded successfully: {data_path}")
    print(f"Rows: {df.shape[0]} | Columns: {df.shape[1]}")

    required_columns = [
        "Hours_Studied",
        "Attendance",
        "Previous_Scores",
        "Exam_Score",
    ]

    missing_required = [column for column in required_columns if column not in df.columns]
    if missing_required:
        raise ValueError(f"Required columns missing: {missing_required}")

    # Create classification target
    # 1 = At-Risk, 0 = On-Track
    df["Risk_Target"] = (df["Exam_Score"] < AT_RISK_SCORE).astype(int)
    df["Risk_Label"] = df["Risk_Target"].map({1: "At-Risk", 0: "On-Track"})

    # Save cleaned data
    df.to_csv(OUTPUT_FOLDER / "cleaned_dataset.csv", index=False)

    # ------------------------------------------------------------------------
    # B. IMPORTANT TABLES
    # ------------------------------------------------------------------------
    dataset_summary = pd.DataFrame(
        {
            "Item": [
                "Dataset",
                "Number of rows",
                "Number of original columns",
                "AI problem",
                "Hybrid technique",
                "At-risk definition",
            ],
            "Value": [
                data_path.name,
                len(df),
                df.shape[1] - 2,
                "Student risk classification",
                "ANN + Fuzzy Logic",
                f"Exam_Score < {AT_RISK_SCORE}",
            ],
        }
    )
    dataset_summary.to_csv(OUTPUT_FOLDER / "dataset_summary.csv", index=False)
    show_table("TABLE 1: DATASET SUMMARY", dataset_summary)

    missing_table = df.isna().sum().reset_index()
    missing_table.columns = ["Feature", "Missing_Values"]
    missing_table["Missing_Percentage"] = (
        missing_table["Missing_Values"] / len(df) * 100
    ).round(2)
    missing_table = missing_table.sort_values("Missing_Values", ascending=False)
    missing_table.to_csv(OUTPUT_FOLDER / "missing_values.csv", index=False)
    show_table("TABLE 2: MISSING VALUES", missing_table, rows=20)

    class_table = (
        df["Risk_Label"].value_counts().rename_axis("Class").reset_index(name="Count")
    )
    class_table["Percentage"] = (class_table["Count"] / len(df) * 100).round(2)
    class_table.to_csv(OUTPUT_FOLDER / "class_distribution.csv", index=False)
    show_table("TABLE 3: CLASS DISTRIBUTION", class_table)

    # ------------------------------------------------------------------------
    # C. IMPORTANT EXPLORATORY FIGURES
    # ------------------------------------------------------------------------
    plt.figure(figsize=(8, 5))
    plt.hist(df["Exam_Score"], bins=25, edgecolor="black")
    plt.axvline(
        AT_RISK_SCORE,
        linestyle="--",
        linewidth=2,
        label=f"At-risk threshold = {AT_RISK_SCORE}",
    )
    plt.title("Distribution of Exam Scores")
    plt.xlabel("Exam Score")
    plt.ylabel("Number of Students")
    plt.legend()
    save_and_show_figure("figure_01_exam_score_distribution.png")

    plt.figure(figsize=(7, 5))
    plt.bar(class_table["Class"], class_table["Count"])
    plt.title("Student Risk Class Distribution")
    plt.xlabel("Class")
    plt.ylabel("Number of Students")
    save_and_show_figure("figure_02_class_distribution.png")

    # ------------------------------------------------------------------------
    # D. PREPARE DATA FOR MODELLING
    # ------------------------------------------------------------------------
    X = df.drop(columns=["Exam_Score", "Risk_Target", "Risk_Label"])
    y = df["Risk_Target"]

    numeric_features = X.select_dtypes(include=np.number).columns.tolist()
    categorical_features = [column for column in X.columns if column not in numeric_features]

    numeric_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        [
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    # ------------------------------------------------------------------------
    # E. TRAIN ANN AND SIMPLE BASELINE
    # ------------------------------------------------------------------------
    ann_model = Pipeline(
        [
            ("preprocessor", preprocessor),
            (
                "model",
                MLPClassifier(
                    hidden_layer_sizes=(32, 16),
                    max_iter=300,
                    early_stopping=True,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    baseline_model = Pipeline(
        [
            ("preprocessor", preprocessor),
            (
                "model",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    models = {
        "ANN": ann_model,
        "Logistic Regression": baseline_model,
    }

    metrics_list = []
    predictions = {}
    probabilities = {}

    for model_name, model in models.items():
        print(f"\nTraining {model_name}...")
        model.fit(X_train, y_train)

        predicted = model.predict(X_test)
        probability = model.predict_proba(X_test)[:, 1]

        predictions[model_name] = predicted
        probabilities[model_name] = probability

        metrics_list.append(
            calculate_metrics(model_name, y_test, predicted, probability)
        )

    metrics_table = pd.DataFrame(metrics_list)
    metrics_table.to_csv(OUTPUT_FOLDER / "model_metrics.csv", index=False)
    show_table("TABLE 4: MODEL PERFORMANCE", metrics_table)

    # Save ANN classification report
    ann_report = pd.DataFrame(
        classification_report(
            y_test,
            predictions["ANN"],
            target_names=["On-Track", "At-Risk"],
            output_dict=True,
            zero_division=0,
        )
    ).transpose()
    ann_report.reset_index(inplace=True)
    ann_report.rename(columns={"index": "Class"}, inplace=True)
    ann_report.to_csv(OUTPUT_FOLDER / "ann_classification_report.csv", index=False)
    show_table("TABLE 5: ANN CLASSIFICATION REPORT", ann_report)

    # ------------------------------------------------------------------------
    # F. MODEL FIGURES
    # ------------------------------------------------------------------------
    ann_confusion = confusion_matrix(y_test, predictions["ANN"], labels=[1, 0])

    plt.figure(figsize=(6, 5))
    plt.imshow(ann_confusion, cmap="Blues")
    plt.colorbar()
    plt.title("ANN Confusion Matrix")
    plt.xticks([0, 1], ["At-Risk", "On-Track"])
    plt.yticks([0, 1], ["At-Risk", "On-Track"])
    plt.xlabel("Predicted Class")
    plt.ylabel("Actual Class")

    for row in range(2):
        for column in range(2):
            plt.text(
                column,
                row,
                ann_confusion[row, column],
                ha="center",
                va="center",
                fontsize=12,
            )

    save_and_show_figure("figure_03_ann_confusion_matrix.png")

    plt.figure(figsize=(8, 5))
    x = np.arange(len(metrics_table))
    width = 0.22

    plt.bar(x - width, metrics_table["Accuracy"], width, label="Accuracy")
    plt.bar(x, metrics_table["Recall_At_Risk"], width, label="Recall")
    plt.bar(x + width, metrics_table["F1_At_Risk"], width, label="F1 Score")

    plt.xticks(x, metrics_table["Model"])
    plt.ylim(0, 1.05)
    plt.title("Model Performance Comparison")
    plt.ylabel("Score")
    plt.legend()
    save_and_show_figure("figure_04_model_comparison.png")

    # ANN feature importance
    importance = permutation_importance(
        ann_model,
        X_test,
        y_test,
        scoring="f1",
        n_repeats=5,
        random_state=RANDOM_STATE,
    )

    importance_table = pd.DataFrame(
        {
            "Feature": X.columns,
            "Importance": importance.importances_mean,
        }
    ).sort_values("Importance", ascending=False)

    importance_table.to_csv(OUTPUT_FOLDER / "feature_importance.csv", index=False)
    show_table("TABLE 6: TOP FEATURE IMPORTANCE", importance_table, rows=10)

    top_importance = importance_table.head(10).sort_values("Importance")

    plt.figure(figsize=(8, 6))
    plt.barh(top_importance["Feature"], top_importance["Importance"])
    plt.title("Top 10 ANN Feature Importance")
    plt.xlabel("Permutation Importance")
    save_and_show_figure("figure_05_feature_importance.png")

    # ------------------------------------------------------------------------
    # G. SIMPLE FUZZY + ANN HYBRID OUTPUT
    # ------------------------------------------------------------------------
    test_results = X_test.reset_index(drop=True).copy()
    test_results["Actual_Label"] = y_test.reset_index(drop=True).map(
        {1: "At-Risk", 0: "On-Track"}
    )
    test_results["ANN_Probability_At_Risk"] = probabilities["ANN"]
    test_results["ANN_Predicted_Label"] = pd.Series(predictions["ANN"]).map(
        {1: "At-Risk", 0: "On-Track"}
    )

    fuzzy_scores = []
    fuzzy_labels = []
    fuzzy_explanations = []
    hybrid_scores = []
    hybrid_labels = []
    recommendations = []

    for _, row in test_results.iterrows():
        fuzzy_score, fuzzy_label, explanation = simple_fuzzy_risk(
            attendance=row["Attendance"],
            hours_studied=row["Hours_Studied"],
            previous_score=row["Previous_Scores"],
        )

        # ANN contributes 70%; fuzzy logic contributes 30%.
        hybrid_score = (
            0.70 * row["ANN_Probability_At_Risk"]
            + 0.30 * (fuzzy_score / 100)
        )

        hybrid_label = "At-Risk" if hybrid_score >= 0.50 else "On-Track"

        if hybrid_label == "At-Risk":
            recommendation = "Provide tutoring and monitor the student closely"
        elif hybrid_score >= 0.35:
            recommendation = "Monitor progress and encourage better study habits"
        else:
            recommendation = "Maintain current learning strategy"

        fuzzy_scores.append(fuzzy_score)
        fuzzy_labels.append(fuzzy_label)
        fuzzy_explanations.append(explanation)
        hybrid_scores.append(round(hybrid_score, 4))
        hybrid_labels.append(hybrid_label)
        recommendations.append(recommendation)

    test_results["Fuzzy_Risk_Score"] = fuzzy_scores
    test_results["Fuzzy_Risk_Label"] = fuzzy_labels
    test_results["Fuzzy_Explanation"] = fuzzy_explanations
    test_results["Hybrid_Risk_Score"] = hybrid_scores
    test_results["Hybrid_Final_Label"] = hybrid_labels
    test_results["Recommendation"] = recommendations

    test_results.to_csv(OUTPUT_FOLDER / "prototype_predictions.csv", index=False)
    show_table("TABLE 7: SAMPLE HYBRID PREDICTIONS", test_results, rows=15)

    fuzzy_rules = pd.DataFrame(
        {
            "Rule": ["R1", "R2", "R3", "R4", "R5", "R6"],
            "Condition": [
                "Attendance < 75",
                "Attendance between 75 and 84",
                "Study hours < 12",
                "Study hours between 12 and 19",
                "Previous score < 65",
                "Previous score between 65 and 74",
            ],
            "Risk_Contribution": [40, 20, 30, 15, 30, 15],
        }
    )
    fuzzy_rules.to_csv(OUTPUT_FOLDER / "fuzzy_rule_table.csv", index=False)
    show_table("TABLE 8: SIMPLE FUZZY RULES", fuzzy_rules)

    decision_summary = (
        test_results["Hybrid_Final_Label"]
        .value_counts()
        .rename_axis("Hybrid_Decision")
        .reset_index(name="Count")
    )
    decision_summary.to_csv(OUTPUT_FOLDER / "hybrid_decision_summary.csv", index=False)

    plt.figure(figsize=(7, 5))
    plt.bar(decision_summary["Hybrid_Decision"], decision_summary["Count"])
    plt.title("Hybrid ANN + Fuzzy Decision Summary")
    plt.xlabel("Final Decision")
    plt.ylabel("Number of Test Students")
    save_and_show_figure("figure_06_hybrid_decision_summary.png")

    # ------------------------------------------------------------------------
    # H. SAVE TRAINED MODEL
    # ------------------------------------------------------------------------
    joblib.dump(ann_model, OUTPUT_FOLDER / "ann_student_risk_model.joblib")

    print("\n" + "=" * 75)
    print("PROJECT COMPLETED SUCCESSFULLY")
    print("=" * 75)
    print(f"All tables, figures, predictions, and model saved in:\n{OUTPUT_FOLDER}")
    print("\nImportant saved files:")
    print("- model_metrics.csv")
    print("- ann_classification_report.csv")
    print("- feature_importance.csv")
    print("- prototype_predictions.csv")
    print("- fuzzy_rule_table.csv")
    print("- figure_01 to figure_06 PNG files")
    print("- ann_student_risk_model.joblib")


# ============================================================================
# 4. RUN PROGRAM
# ============================================================================


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simple ANN + Fuzzy AI Project")
    parser.add_argument("--data", default=str(DATA_PATH), help="Dataset CSV path")
    parser.add_argument("--out", default=str(OUTPUT_FOLDER), help="Output folder")
    args = parser.parse_args()

    main(args.data, args.out)
