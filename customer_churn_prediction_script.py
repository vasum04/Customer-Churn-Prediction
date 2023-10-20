import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pickle
import numpy as np
import matplotlib.pyplot as plt

# Load the dataset
churn_data = pd.read_excel("./customer_churn_large_dataset.xlsx")

churn_data.dropna(subset=["Gender", "Location"], inplace=True)
for column in ["Gender", "Location"]:
    le = LabelEncoder()
    churn_data[column] = le.fit_transform(churn_data[column])

# Split the data into training and testing sets
X = churn_data.drop(columns=["CustomerID", "Name", "Churn"])
y = churn_data["Churn"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature Engineering
bins = [18, 30, 40, 50, 60, 70]
labels = ["18-30", "31-40", "41-50", "51-60", "61-70"]
X_train["Usage_per_Dollar"] = X_train["Total_Usage_GB"] / X_train["Monthly_Bill"]
X_test["Usage_per_Dollar"] = X_test["Total_Usage_GB"] / X_test["Monthly_Bill"]
X_train["Age_Group"] = pd.cut(X_train["Age"], bins=bins, labels=labels, right=False)
X_test["Age_Group"] = pd.cut(X_test["Age"], bins=bins, labels=labels, right=False)
X_train = pd.get_dummies(X_train, columns=["Age_Group"], drop_first=True)
X_test = pd.get_dummies(X_test, columns=["Age_Group"], drop_first=True)

# Apply Min-Max scaling
features_to_scale = ["Age", "Subscription_Length_Months", "Monthly_Bill", "Total_Usage_GB", "Usage_per_Dollar"]
scaler = MinMaxScaler()
X_train[features_to_scale] = scaler.fit_transform(X_train[features_to_scale])
X_test[features_to_scale] = scaler.transform(X_test[features_to_scale])

# Model Building with Random Forest
rf_model = RandomForestClassifier(
    n_estimators=150,
    min_samples_split=2,
    min_samples_leaf=5,
    max_features='log2',
    max_depth=10,
    random_state=42
)
rf_model.fit(X_train, y_train)
y_test_pred_rf = rf_model.predict(X_test)
accuracy_test_rf = accuracy_score(y_test, y_test_pred_rf)
precision_test_rf = precision_score(y_test, y_test_pred_rf)
recall_test_rf = recall_score(y_test, y_test_pred_rf)
f1_test_rf = f1_score(y_test, y_test_pred_rf)

# Model Building with Optimized Logistic Regression
optimized_logreg_model = LogisticRegression(
    solver='newton-cg',
    penalty='l2',
    C=0.0001,
    max_iter=5000,
    random_state=42
)
optimized_logreg_model.fit(X_train, y_train)
y_test_pred_optimized_logreg = optimized_logreg_model.predict(X_test)
accuracy_test_optimized_logreg = accuracy_score(y_test, y_test_pred_optimized_logreg)
precision_test_optimized_logreg = precision_score(y_test, y_test_pred_optimized_logreg)
recall_test_optimized_logreg = recall_score(y_test, y_test_pred_optimized_logreg)
f1_test_optimized_logreg = f1_score(y_test, y_test_pred_optimized_logreg)

# Model Evaluation
y_test_pred_rf = rf_model.predict(X_test)
y_test_pred_optimized_logreg = optimized_logreg_model.predict(X_test)

# Model Deployment (Serialization)
model_filename = "optimized_logreg_model.pkl"
with open(model_filename, "wb") as model_file:
    pickle.dump(optimized_logreg_model, model_file)

# Load the model and make a prediction
with open(model_filename, "rb") as model_file:
    loaded_model = pickle.load(model_file)

sample_data = X_test.sample(1, random_state=42)
predicted_churn = loaded_model.predict(sample_data)
predicted_probability = loaded_model.predict_proba(sample_data)

print(predicted_churn[0], predicted_probability[0][1])

# Metrics for Random Forest and Optimized Logistic Regression
rf_metrics = [accuracy_test_rf, precision_test_rf, recall_test_rf, f1_test_rf]
logreg_metrics = [accuracy_test_optimized_logreg, precision_test_optimized_logreg, recall_test_optimized_logreg, f1_test_optimized_logreg]

# Visualization setup
labels = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
index = np.arange(len(labels))
bar_width = 0.35

# Generate the visualization for model comparison
fig, ax = plt.subplots(figsize=(10, 6))
bar1 = ax.bar(index, rf_metrics, bar_width, label='Random Forest', color='b', alpha=0.7)
bar2 = ax.bar(index + bar_width, logreg_metrics, bar_width, label='Optimized Logistic Regression', color='r', alpha=0.7)
ax.set_xlabel('Metrics')
ax.set_ylabel('Scores')
ax.set_title('Model Performance Comparison')
ax.set_xticks(index + bar_width / 2)
ax.set_xticklabels(labels)
ax.legend()
plt.tight_layout()
plt.show()