import os
from azureml.core import Workspace, Experiment
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from joblib import dump
import pandas as pd

# Load the Azure ML workspace
ws = Workspace.from_config()

# Create an experiment
experiment = Experiment(workspace=ws, name="diabetic-detection-experiment")
run = experiment.start_logging()

# Load the dataset
data_path = "diabetes.csv"  # Specify the correct path to your dataset
data = pd.read_csv(data_path)

# Define features (X) and target (y)
X = data.drop(columns=['Outcome'])
y = data['Outcome']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a logistic regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Make predictions and calculate accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Log accuracy to Azure ML
run.log("accuracy", accuracy)

# Save the model
os.makedirs("outputs", exist_ok=True)
dump(model, "outputs/diabetic_detection_model.joblib")

# Complete the run
run.complete()
