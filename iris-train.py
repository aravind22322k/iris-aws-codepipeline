import os
import subprocess
import sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
import pandas as pd

# Ensure necessary directories exist
os.makedirs('data', exist_ok=True)
os.makedirs('model', exist_ok=True)

# Load the Iris dataset
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)

# Save the train and test datasets to CSV files
train_data = pd.DataFrame(X_train, columns=iris.feature_names)
train_data['target'] = y_train

test_data = pd.DataFrame(X_test, columns=iris.feature_names)
test_data['target'] = y_test

train_file_path = 'data/iris_train.csv'
test_file_path = 'data/iris_test.csv'

# Save to local files
train_data.to_csv(train_file_path, index=False)
test_data.to_csv(test_file_path, index=False)

# Train the RandomForest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save the trained model
model_file_path = 'model/iris_model.joblib'
joblib.dump(model, model_file_path)

# GitHub repo details
github_repo_url = "git@github.com:aravind2232k/iris-aws-codepipeline.git"  # Change to your GitHub repo
git_branch = "main"  # Change if needed

try:
    # Initialize Git repo if not already a repo
    if not os.path.exists(".git"):
        subprocess.run(["git", "init"], check=True)
        subprocess.run(["git", "remote", "add", "origin", github_repo_url], check=True)
    
    # Add files to Git
    subprocess.run(["git", "add", "data/iris_train.csv", "data/iris_test.csv", "model/iris_model.joblib"], check=True)
    
    # Check if there are changes before committing
    status_output = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True)
    
    if status_output.stdout.strip():  # If there are changes
        subprocess.run(["git", "commit", "-m", "Updated Iris dataset and trained model"], check=True)
        print("Changes committed successfully.")
    else:
        print("No changes to commit.")

    # Push to GitHub
    subprocess.run(["git", "push", "origin", git_branch], check=True)
    print("Files successfully pushed to GitHub repository.")
except subprocess.CalledProcessError as e:
    print(f"Error executing Git command: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
