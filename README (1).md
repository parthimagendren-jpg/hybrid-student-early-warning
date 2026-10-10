# Hybrid Intelligent System: Student Early-Warning

A prototype early-warning system that flags students at risk of a low exam score. It combines an artificial neural network (ANN) with a simple fuzzy-logic risk score.

This is a coursework prototype for TPD 3753 (Group 4). It is for educational use only and should not be used to make decisions about real students without proper validation.

## How it works

1. **Target.** A student is labelled **At-Risk** if `Exam_Score < 65`, otherwise **On-Track**.
2. **ANN.** A multilayer perceptron (hidden layers of 32 and 16 neurons) predicts the probability of being At-Risk. A logistic regression is trained as a baseline for comparison.
3. **Fuzzy rules.** A rule-based score from 0 to 100 uses attendance, study hours and previous scores (see the table below).
4. **Hybrid score.** `0.70 × ANN probability + 0.30 × (fuzzy score / 100)`. A score of 0.50 or more is At-Risk.

### Fuzzy rule table

| Rule | Condition | Risk contribution |
|------|-----------|-------------------|
| R1 | Attendance < 75 | 40 |
| R2 | Attendance between 75 and 84 | 20 |
| R3 | Study hours < 12 | 30 |
| R4 | Study hours between 12 and 19 | 15 |
| R5 | Previous score < 65 | 30 |
| R6 | Previous score between 65 and 74 | 15 |

The weights (70/30), the thresholds and the 65-point cut-off were chosen by the group. They are design decisions, not findings from the data.

## Dataset

The script uses the **Student Performance Factors** dataset (6,607 students). Download it from Kaggle and place it in the project folder, renamed to `StudentPerformanceFactors.csv`. The dataset is not included in this repository. Check its licence on Kaggle before redistributing it.

## Setup

```bash
pip install -r requirements.txt
```

Python 3.9 or newer is recommended.

## Run

From the project folder:

```bash
python Hybrid_AI_Simple_Code.py
```

Or with custom paths:

```bash
python Hybrid_AI_Simple_Code.py --data path/to/StudentPerformanceFactors.csv --out outputs_hybrid
```

The script prints the main tables and shows the figures. It saves everything to the output folder (default `outputs_hybrid/`):

- `model_metrics.csv`, `ann_classification_report.csv`
- `feature_importance.csv`, and figures 01 to 06
- `prototype_predictions.csv`, `fuzzy_rule_table.csv`, `hybrid_decision_summary.csv`
- `cleaned_dataset.csv` and `ann_student_risk_model.joblib`

## Results

On the 20% test split (1,322 students), the hybrid system labelled 1,042 students On-Track and 280 At-Risk. Model metrics (accuracy, precision, recall, F1, ROC AUC) are in `model_metrics.csv` after you run the script.

The most important features by permutation importance were attendance and hours studied, followed by previous scores and parental involvement.

## Limitations

- Labels come from a single exam score threshold, so the model predicts a score cut-off rather than real academic failure.
- The fuzzy weights and thresholds were set by hand and have not been tuned or validated.
- The dataset is a sample and may not reflect any particular school.
- The model is not fair-tested across groups such as gender or income.

## Project structure

```
hybrid-student-early-warning/
├── Hybrid_AI_Simple_Code.py
├── requirements.txt
├── README.md
├── .gitignore
└── StudentPerformanceFactors.csv   (you add this; not included)
```
