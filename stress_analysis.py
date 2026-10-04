import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score


# ============================================================
# 1. LOAD DATASET
# ============================================================

print("=" * 60)
print("1. LOADING DATASET")
print("=" * 60)

try:
    df = pd.read_csv("Stress-Lysis.csv")
    print("Data successfully loaded!")

except FileNotFoundError:
    print("Error: 'Stress-Lysis.csv' not found.")
    print("Please keep the CSV file in the same folder as this Python file.")
    exit()


print("\nDataset Shape:", df.shape)

print("\nFirst 5 Rows:")
print(df.head())

print("\nMissing Values Count:")
print(df.isnull().sum())

print("\nSummary Statistics:")
print(df.describe())

print("\nTarget Variable (Stress Level) Counts:")
print(df["Stress Level"].value_counts().sort_index())


# ============================================================
# 2. DATA VISUALIZATION
# ============================================================

print("\n" + "=" * 60)
print("2. DATA VISUALIZATION")
print("=" * 60)


fig, axes = plt.subplots(2, 2, figsize=(12, 9))

fig.suptitle(
    "Stress-Lysis Exploratory Data Analysis",
    fontsize=14,
    fontweight="bold"
)


# ------------------------------------------------------------
# Graph 1: Stress Level Distribution
# ------------------------------------------------------------

sns.countplot(
    x="Stress Level",
    hue="Stress Level",
    data=df,
    palette="Set2",
    legend=False,
    ax=axes[0, 0]
)

axes[0, 0].set_title("Distribution of Stress Levels")
axes[0, 0].set_xticks([0, 1, 2])
axes[0, 0].set_xticklabels(
    ["Low (0)", "Normal (1)", "High (2)"]
)
axes[0, 0].set_ylabel("Count")


# ------------------------------------------------------------
# Graph 2: Correlation Matrix
# ------------------------------------------------------------

sns.heatmap(
    df.corr(),
    annot=True,
    cmap="Blues",
    fmt=".2f",
    ax=axes[0, 1]
)

axes[0, 1].set_title("Correlation Matrix")


# ------------------------------------------------------------
# Graph 3: Humidity vs Temperature
# ------------------------------------------------------------

sns.scatterplot(
    x="Humidity",
    y="Temperature",
    hue="Stress Level",
    data=df,
    palette="viridis",
    style="Stress Level",
    s=50,
    ax=axes[1, 0]
)

axes[1, 0].set_title(
    "Humidity vs Temperature by Stress Level"
)

axes[1, 0].set_xlabel("Humidity (mg/min)")
axes[1, 0].set_ylabel("Temperature (°F)")


# ------------------------------------------------------------
# Graph 4: Step Count Distribution
# ------------------------------------------------------------

sns.boxplot(
    x="Stress Level",
    y="Step count",
    hue="Stress Level",
    data=df,
    palette="Set2",
    legend=False,
    ax=axes[1, 1]
)

axes[1, 1].set_title(
    "Step Count Distribution Across Stress Levels"
)

axes[1, 1].set_xticks([0, 1, 2])
axes[1, 1].set_xticklabels(
    ["Low (0)", "Normal (1)", "High (2)"]
)

axes[1, 1].set_ylabel("Step Count (steps/min)")


plt.tight_layout()

plot_filename = "stress_lysis_analysis.png"

plt.savefig(
    plot_filename,
    dpi=300,
    bbox_inches="tight"
)

print(
    "\nVisualizations saved locally as "
    "'stress_lysis_analysis.png'!"
)

plt.show()


# ============================================================
# 3. MACHINE LEARNING MODEL
# ============================================================

print("\n" + "=" * 60)
print("3. MACHINE LEARNING MODEL TRAINING & EVALUATION")
print("=" * 60)


# Input features
X = df[
    [
        "Humidity",
        "Temperature",
        "Step count"
    ]
]


# Target variable
y = df["Stress Level"]


# ------------------------------------------------------------
# Train-Test Split
# ------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)


print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# ------------------------------------------------------------
# Feature Scaling
# ------------------------------------------------------------

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)


# ------------------------------------------------------------
# Random Forest Classifier
# ------------------------------------------------------------

clf = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)


clf.fit(
    X_train_scaled,
    y_train
)


# ------------------------------------------------------------
# Prediction on Test Data
# ------------------------------------------------------------

y_pred = clf.predict(X_test_scaled)


# ------------------------------------------------------------
# Accuracy
# ------------------------------------------------------------

accuracy = accuracy_score(
    y_test,
    y_pred
)


print(
    "\nModel Accuracy:",
    f"{accuracy * 100:.2f}%"
)


# ------------------------------------------------------------
# Classification Report
# ------------------------------------------------------------

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=[
            "Low (0)",
            "Normal (1)",
            "High (2)"
        ]
    )
)


# ============================================================
# 4. FEATURE IMPORTANCE
# ============================================================

print("Feature Importance Breakdown:")


feature_importances = pd.Series(
    clf.feature_importances_,
    index=X.columns
).sort_values(
    ascending=False
)


for feature, importance in feature_importances.items():

    print(
        f" - {feature}: "
        f"{importance * 100:.2f}%"
    )


# ============================================================
# 5. NEW INPUT STRESS PREDICTION
# ============================================================

print("\n" + "=" * 60)
print("4. NEW INPUT STRESS PREDICTION")
print("=" * 60)

print("\nEnter new sensor values below.")


# ------------------------------------------------------------
# Ask user for new values
# ------------------------------------------------------------

humidity = float(
    input("Enter Humidity (mg/min): ")
)

temperature = float(
    input("Enter Body Temperature (°F): ")
)

step_count = float(
    input("Enter Step Count (steps/min): ")
)


# ------------------------------------------------------------
# Create new input DataFrame
# ------------------------------------------------------------

new_input = pd.DataFrame({

    "Humidity": [humidity],

    "Temperature": [temperature],

    "Step count": [step_count]

})


# ------------------------------------------------------------
# Scale new input
# ------------------------------------------------------------

new_input_scaled = scaler.transform(
    new_input
)


# ------------------------------------------------------------
# Predict stress level
# ------------------------------------------------------------

prediction = clf.predict(
    new_input_scaled
)[0]


# ------------------------------------------------------------
# Convert prediction number to name
# ------------------------------------------------------------

if prediction == 0:

    stress_name = "Low Stress"

elif prediction == 1:

    stress_name = "Normal Stress"

else:

    stress_name = "High Stress"


# ------------------------------------------------------------
# Display result
# ------------------------------------------------------------

print("\n" + "-" * 60)

print("NEW INPUT:")

print(
    "Humidity:",
    humidity,
    "mg/min"
)

print(
    "Temperature:",
    temperature,
    "°F"
)

print(
    "Step Count:",
    step_count,
    "steps/min"
)


print("\nPREDICTED STRESS LEVEL:")

print(
    stress_name
)

print(
    "Stress Class:",
    prediction
)

print("-" * 60)


print("\nProgram completed successfully!")
