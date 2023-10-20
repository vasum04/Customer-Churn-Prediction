import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split

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