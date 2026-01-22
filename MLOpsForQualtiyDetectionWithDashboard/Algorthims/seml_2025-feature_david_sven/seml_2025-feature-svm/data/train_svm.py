import pandas as pd
import os
import pickle
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from pathlib import Path


# Reading out the data from the dataset path
project_path = Path(__file__).resolve().parents[2]
dataset_path = os.path.join(project_path, "data", "processed", "dataset.csv")
data = pd.read_csv(dataset_path)
dataset_val_path = os.path.join(project_path, "data", "processed", "dataset_val.csv")
data_val = pd.read_csv(dataset_val_path)

# Splitting the data into subsets
X = data.drop(columns="Label")
y = data["Label"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_val = data_val.drop(columns="Label")
y_val = data_val["Label"]



# Scaling the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)
X_val = scaler.fit_transform(X_val)


# Creating an instance of SVC and train the model
svm_model = SVC(kernel="rbf", C=1.0, random_state=42, class_weight="balanced")
svm_model.fit(X_train, y_train)


# Computing the predictions on X_train and X_test
y_pred_train = svm_model.predict(X_train)
y_pred_test = svm_model.predict(X_test)


# Printing out the results
print("Training Accuracy:", accuracy_score(y_train, y_pred_train))
print("Test Accuracy:", accuracy_score(y_test, y_pred_test))
print("\nClassification Report:\n", classification_report(y_test, y_pred_test))


# Saving the trained model as pickle
with open(os.path.join(project_path, "models", "svm_model.pkl"), "wb") as model:
    pickle.dump(svm_model, model)


y_pred_val = svm_model.predict(X_val)


# Printing out the results
print("Validation Accuracy:", accuracy_score(y_val, y_pred_val))
print("\nClassification Report:\n", classification_report(y_val, y_pred_val))
