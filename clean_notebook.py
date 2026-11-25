import json
import os


SOURCE = r"C:\Users\Azmat\OneDrive\Desktop\Diabetes_AI_Project\pima-indians-diabetes-eda-prediction-0-906.ipynb"
OUTPUT = r"C:\Users\Azmat\OneDrive\Desktop\Diabetes_AI_Project\diabetes_clean.ipynb"


def source(text):
    return text.splitlines(True)


with open(SOURCE, "r", encoding="utf-8") as handle:
    notebook = json.load(handle)

replacements = {
    8: """# Python libraries
import time
import pandas as pd
import numpy as np

# Plots
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
plt.style.use("ggplot")

# Data processing, metrics and modeling
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    cross_val_score,
    cross_val_predict,
    train_test_split,
    KFold,
)
from sklearn.metrics import (
    precision_score,
    recall_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
    accuracy_score,
    roc_auc_score,
    auc,
)
from sklearn.ensemble import VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
import lightgbm as lgbm
from yellowbrick.classifier import DiscriminationThreshold

# Stats
import scipy.stats as ss
from scipy.stats import randint as sp_randint
from scipy.stats import uniform as sp_uniform

# Time
from contextlib import contextmanager

@contextmanager
def timer(title):
    t0 = time.time()
    yield
    print("{} - done in {:.0f}s".format(title, time.time() - t0))

# Ignore warning messages
import warnings
warnings.filterwarnings("ignore")
""",
    11: 'data = pd.read_csv("diabetes.csv")',
    21: """D = data[data["Outcome"] != 0]
H = data[data["Outcome"] == 0]

def target_count():
    counts = data["Outcome"].value_counts().sort_index()
    labels = ["healthy", "diabetic"]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(labels, counts.values, color=["lightskyblue", "gold"], edgecolor="black")
    ax.set_title("Count of Outcome variable")
    plt.tight_layout()
    plt.show()

def target_percent():
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.pie(
        data["Outcome"].value_counts(),
        labels=["healthy", "diabetic"],
        colors=["lightskyblue", "gold"],
        autopct="%1.1f%%",
        wedgeprops={"edgecolor": "black"},
    )
    ax.set_title("Distribution of Outcome variable")
    plt.tight_layout()
    plt.show()
""",
    27: """def missing_plot(dataset, key):
    missing_count = dataset.isnull().sum()
    missing_percent = (missing_count / len(dataset) * 100).round(2)
    plot_data = pd.DataFrame({
        "Count": missing_count,
        "Percent": missing_percent,
    })
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(plot_data.index, plot_data["Count"], color="#7EC0EE", edgecolor="black")
    ax.set_title("Missing Values (count & %)")
    ax.set_ylabel("Missing count")
    ax.tick_params(axis="x", rotation=45)
    for index, (count, percent) in enumerate(zip(plot_data["Count"], plot_data["Percent"])):
        ax.text(index, count, f"{count}\\n({percent:.1f}%)", ha="center", va="bottom")
    plt.tight_layout()
    plt.show()
""",
    33: """def correlation_plot():
    correlation = data.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(correlation, cmap="viridis", annot=False, square=True, ax=ax)
    ax.set_title("Correlation Matrix for variables")
    plt.tight_layout()
    plt.show()
""",
    43: """def plot_distribution(data_select, size_bin):
    plot_data = data[[data_select, "Outcome"]].dropna()
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(
        data=plot_data,
        x=data_select,
        hue="Outcome",
        bins=max(1, int(plot_data[data_select].nunique() / 2)) if size_bin == 0 else size_bin,
        kde=True,
        stat="density",
        common_norm=False,
        palette={0: "lightskyblue", 1: "gold"},
        alpha=0.55,
        ax=ax,
    )
    ax.set_title(data_select)
    plt.tight_layout()
    plt.show()
""",
    76: """def plot_feat1_feat2(feat1, feat2):
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(
        data=data,
        x=feat1,
        y=feat2,
        hue="Outcome",
        palette={0: "lightskyblue", 1: "gold"},
        edgecolor="black",
        ax=ax,
    )
    ax.set_title(f"{feat1} vs {feat2}")
    plt.tight_layout()
    plt.show()
""",
    77: """def barplot(var_select, sub):
    counts = pd.crosstab(data[var_select], data["Outcome"])
    percent_diabetic = counts.get(1, 0) / counts.sum(axis=1) * 100
    fig, ax1 = plt.subplots(figsize=(10, 5))
    counts.plot(kind="bar", ax=ax1, color=["lightskyblue", "gold"], edgecolor="black")
    ax1.set_title(f"{var_select} {sub}")
    ax1.set_ylabel("Count")

    ax2 = ax1.twinx()
    ax2.plot(range(len(percent_diabetic)), percent_diabetic.values, color="black", marker="o")
    ax2.set_ylabel("% diabetic")
    ax2.set_ylim(0, 75)
    ax2.grid(False)
    plt.tight_layout()
    plt.show()
""",
    78: """def plot_pie(var_select, sub):
    diabetic = data[data["Outcome"] != 0][var_select].value_counts()
    healthy = data[data["Outcome"] == 0][var_select].value_counts()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].pie(diabetic.values, labels=diabetic.index, autopct="%1.1f%%",
                colors=sns.color_palette("pastel", len(diabetic)), wedgeprops={"edgecolor": "white"})
    axes[0].set_title(f"Diabetic: {var_select}")
    axes[1].pie(healthy.values, labels=healthy.index, autopct="%1.1f%%",
                colors=sns.color_palette("pastel", len(healthy)), wedgeprops={"edgecolor": "white"})
    axes[1].set_title(f"Healthy: {var_select}")
    fig.suptitle(f"{var_select} distribution by target {sub}")
    plt.tight_layout()
    plt.show()
""",
    150: """def correlation_plot():
    correlation = data.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(correlation, cmap="viridis", annot=False, square=True, ax=ax)
    ax.set_title("Correlation Matrix for variables")
    plt.tight_layout()
    plt.show()
""",
    154: """# Def X and Y
X = data.drop(columns=["Outcome"])
y = data["Outcome"]
""",
    159: """def model_performance(model, subtitle):
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    mean_fpr = np.linspace(0, 1, 100)
    tprs = []
    aucs = []

    y_real = []
    y_proba = []
    for train, test in cv.split(X, y):
        model.fit(X.iloc[train], y.iloc[train])
        pred_proba = model.predict_proba(X.iloc[test])[:, 1]
        y_real.append(y.iloc[test].to_numpy())
        y_proba.append(pred_proba)
        fpr, tpr, _ = roc_curve(y.iloc[test], pred_proba)
        tprs.append(np.interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        aucs.append(auc(fpr, tpr))

    y_pred = cross_val_predict(model, X, y, cv=cv)
    conf_matrix = confusion_matrix(y, y_pred)
    tn, fp, fn, tp = conf_matrix.ravel()
    accuracy = (tp + tn) / conf_matrix.sum()
    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
    metrics = pd.Series({"Accuracy": accuracy, "Precision": precision,
                         "Recall": recall, "F1 score": f1})

    y_real = np.concatenate(y_real)
    y_proba = np.concatenate(y_proba)
    precision_curve, recall_curve, _ = precision_recall_curve(y_real, y_proba)
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="viridis",
                xticklabels=["0 (pred)", "1 (pred)"],
                yticklabels=["0 (true)", "1 (true)"], ax=axes[0, 0])
    axes[0, 0].set_title("Confusion Matrix")

    axes[0, 1].barh(metrics.index, metrics.values, color=["gold", "lightgreen", "lightcoral", "lightskyblue"])
    axes[0, 1].set_xlim(0, 1)
    axes[0, 1].set_title("Metrics")

    axes[1, 0].plot(mean_fpr, mean_tpr, label=f"ROC (AUC = {mean_auc:.3f})")
    axes[1, 0].plot([0, 1], [0, 1], "k--")
    axes[1, 0].set(xlabel="False positive rate", ylabel="True positive rate")
    axes[1, 0].set_title("ROC curve")
    axes[1, 0].legend()

    axes[1, 1].plot(recall_curve, precision_curve)
    axes[1, 1].set(xlabel="Recall", ylabel="Precision", xlim=(0, 1.05), ylim=(0, 1.05))
    axes[1, 1].set_title("Precision - Recall curve")

    fig.suptitle(f"Model performance report (5 folds) - {subtitle}")
    plt.tight_layout()
    plt.show()
""",
    162: """def scores_table(model, subtitle):
    scoring_names = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    rows = {}
    for scoring_name in scoring_names:
        scores = cross_val_score(model, X, y, cv=5, scoring=scoring_name)
        rows[scoring_name] = list(scores) + [scores.mean(), scores.std()]
    df = pd.DataFrame(rows, index=["Fold 1", "Fold 2", "Fold 3", "Fold 4", "Fold 5", "mean", "std"])
    display(df)
""",
    168: """random_state = 42

param_test = {
    "learning_rate": [0.01, 0.02, 0.03, 0.04, 0.05, 0.08, 0.1, 0.2, 0.3, 0.4],
    "n_estimators": [100, 200, 300, 400, 500, 600, 800, 1000],
    "num_leaves": sp_randint(6, 50),
    "min_child_samples": sp_randint(100, 500),
    "min_child_weight": [1e-5, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4],
    "subsample": sp_uniform(loc=0.2, scale=0.8),
    "max_depth": [-1, 1, 2, 3, 4, 5, 6, 7],
    "colsample_bytree": sp_uniform(loc=0.4, scale=0.6),
    "reg_alpha": [0, 1e-1, 1, 2, 5, 7, 10, 50, 100],
    "reg_lambda": [0, 1e-1, 1, 5, 10, 20, 50, 100],
}

n_iter = 50
lgbm_clf = lgbm.LGBMClassifier(random_state=random_state, n_jobs=4, verbosity=-1)
grid_search = RandomizedSearchCV(
    estimator=lgbm_clf,
    param_distributions=param_test,
    n_iter=n_iter,
    scoring="accuracy",
    cv=5,
    refit=True,
    random_state=random_state,
    n_jobs=-1,
)
grid_search.fit(X, y)
opt_parameters = grid_search.best_params_
lgbm_clf = lgbm.LGBMClassifier(**opt_parameters, random_state=random_state, n_jobs=4, verbosity=-1)
""",
    172: """visualizer = DiscriminationThreshold(lgbm_clf)
visualizer.fit(X, y)
visualizer.show()
""",
    181: """visualizer = DiscriminationThreshold(voting_clf)
visualizer.fit(X, y)
visualizer.show()
""",
}

for index, text in replacements.items():
    notebook["cells"][index]["source"] = source(text)
    if notebook["cells"][index]["cell_type"] == "code":
        notebook["cells"][index]["outputs"] = []
        notebook["cells"][index]["execution_count"] = None

# Clear outputs and execution counts from every code cell.
for cell in notebook["cells"]:
    if cell["cell_type"] == "code":
        cell["outputs"] = []
        cell["execution_count"] = None

with open(OUTPUT, "w", encoding="utf-8") as handle:
    json.dump(notebook, handle, ensure_ascii=False, indent=1)

print(OUTPUT)
