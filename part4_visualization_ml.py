# ============================================================
# Assignment Part 4 — Data Visualization & Machine Learning
# Theme: Student Performance Analysis & Prediction
# ============================================================

# =========================
# Import Required Libraries
# =========================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# =========================
# Task 1 — Data Exploration
# =========================

# Load dataset
df = pd.read_csv("students.csv")

print("\n========== DATA PREVIEW ==========")
print(df.head())

# Shape and data types
print("\n========== DATA INFO ==========")
print("Shape:", df.shape)
print("\nData Types:\n", df.dtypes)

# Summary statistics
print("\n========== SUMMARY STATISTICS ==========")
print(df.describe())

# Count pass vs fail
print("\n========== PASS / FAIL COUNT ==========")
print(df['passed'].value_counts())

# Average score comparison (Pass vs Fail)
subject_cols = ['math', 'science', 'english', 'history', 'pe']

print("\n========== AVERAGE SCORES (PASS) ==========")
print(df[df['passed'] == 1][subject_cols].mean())

print("\n========== AVERAGE SCORES (FAIL) ==========")
print(df[df['passed'] == 0][subject_cols].mean())

# Identify student with highest average score
df['avg_score'] = df[subject_cols].mean(axis=1)
top_student = df.loc[df['avg_score'].idxmax()]

print("\n========== TOP PERFORMER ==========")
print(f"{top_student['name']} with average score {top_student['avg_score']:.2f}")

# =========================
# Task 2 — Matplotlib Plots
# =========================

# 1. Bar Chart — Average score per subject
plt.figure()
df[subject_cols].mean().plot(kind='bar')
plt.title("Average Score per Subject")
plt.xlabel("Subjects")
plt.ylabel("Average Score")
plt.savefig("plot1_bar.png")
plt.show()

# 2. Histogram — Math score distribution
plt.figure()
plt.hist(df['math'], bins=5)
mean_math = df['math'].mean()
plt.axvline(mean_math, linestyle='dashed', label=f"Mean = {mean_math:.2f}")
plt.legend()
plt.title("Distribution of Math Scores")
plt.savefig("plot2_hist.png")
plt.show()

# 3. Scatter Plot — Study hours vs average score
plt.figure()

pass_df = df[df['passed'] == 1]
fail_df = df[df['passed'] == 0]

plt.scatter(pass_df['study_hours_per_day'], pass_df['avg_score'], label='Pass')
plt.scatter(fail_df['study_hours_per_day'], fail_df['avg_score'], label='Fail')

plt.xlabel("Study Hours per Day")
plt.ylabel("Average Score")
plt.title("Study Hours vs Average Score")
plt.legend()
plt.savefig("plot3_scatter.png")
plt.show()

# 4. Box Plot — Attendance comparison
plt.figure()

pass_attendance = pass_df['attendance_pct']
fail_attendance = fail_df['attendance_pct']

plt.boxplot([pass_attendance, fail_attendance], labels=['Pass', 'Fail'])
plt.title("Attendance Distribution (Pass vs Fail)")
plt.savefig("plot4_box.png")
plt.show()

# 5. Line Plot — Math vs Science scores
plt.figure()

plt.plot(df['name'], df['math'], marker='o', label='Math')
plt.plot(df['name'], df['science'], marker='x', label='Science')

plt.xticks(rotation=45)
plt.xlabel("Student Name")
plt.ylabel("Score")
plt.title("Math vs Science Scores")
plt.legend()
plt.savefig("plot5_line.png")
plt.show()

# =========================
# Task 3 — Seaborn Plots
# =========================

# Bar plots for Math and Science grouped by pass/fail
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

sns.barplot(data=df, x='passed', y='math', ax=axes[0])
axes[0].set_title("Average Math Score by Result")

sns.barplot(data=df, x='passed', y='science', ax=axes[1])
axes[1].set_title("Average Science Score by Result")

plt.savefig("plot6_seaborn_bar.png")
plt.show()

# Scatter + Regression plot
plt.figure()

sns.scatterplot(data=df, x='attendance_pct', y='avg_score', hue='passed')

# Regression lines for each group
sns.regplot(data=df[df['passed'] == 1], x='attendance_pct', y='avg_score', scatter=False, label='Pass')
sns.regplot(data=df[df['passed'] == 0], x='attendance_pct', y='avg_score', scatter=False, label='Fail')

plt.title("Attendance vs Average Score")
plt.savefig("plot7_seaborn_scatter.png")
plt.show()

# COMMENT:
# Seaborn simplifies grouped visualizations and regression plots with minimal code.
# Matplotlib provides more control but requires more manual setup.

# =========================
# Task 4 — Machine Learning
# =========================

# Define features (X) and target (y)
X = df[['math', 'science', 'english', 'history', 'pe', 'attendance_pct', 'study_hours_per_day']]
y = df['passed']

# Split dataset into training and testing sets (80-20 split)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale features (important for logistic regression performance)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Logistic Regression model
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# Training accuracy
train_acc = model.score(X_train_scaled, y_train)
print("\n========== TRAINING ACCURACY ==========")
print(train_acc)

# Test accuracy
y_pred = model.predict(X_test_scaled)
test_acc = accuracy_score(y_test, y_pred)

print("\n========== TEST ACCURACY ==========")
print(test_acc)

# Display predictions with student names
print("\n========== PREDICTIONS ==========")
names = df.loc[X_test.index, 'name']

for name, actual, pred in zip(names, y_test, y_pred):
    result = "✅" if actual == pred else "❌"
    print(f"{name}: Actual={actual}, Predicted={pred} {result}")

# =========================
# Feature Importance
# =========================

coefficients = model.coef_[0]
features = X.columns

# Sort features by importance
importance = sorted(zip(features, coefficients), key=lambda x: abs(x[1]), reverse=True)

print("\n========== FEATURE IMPORTANCE ==========")
for feature, coef in importance:
    print(f"{feature}: {coef:.3f}")

# Plot feature importance
plt.figure()

colors = ['green' if c > 0 else 'red' for c in coefficients]
plt.barh(features, coefficients, color=colors)

plt.title("Feature Importance (Logistic Regression)")
plt.xlabel("Coefficient Value")
plt.ylabel("Features")
plt.show()

# =========================
# Bonus — New Student Prediction
# =========================

new_student = [[75, 70, 68, 65, 80, 82, 3.2]]

new_scaled = scaler.transform(new_student)
prediction = model.predict(new_scaled)
probability = model.predict_proba(new_scaled)

print("\n========== NEW STUDENT PREDICTION ==========")
print("Prediction:", "Pass" if prediction[0] == 1 else "Fail")
print("Probability:", probability)
