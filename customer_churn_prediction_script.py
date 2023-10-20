import pandas as pd
from sklearn.preprocessing import LabelEncoder
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