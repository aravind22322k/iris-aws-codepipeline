import os
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
import pandas as pd
import boto3

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

# Upload the train and test CSV files to S3
s3 = boto3.client('s3')
bucket_name = 'iris-airflow-bucket'

try:
    s3.upload_file(train_file_path, bucket_name, 'iris_train.csv')
    s3.upload_file(test_file_path, bucket_name, 'iris_test.csv')
    print(f"Train and test datasets uploaded to S3 bucket: {bucket_name}")
except Exception as e:
    print(f"Error uploading datasets to S3: {e}")

# Train the RandomForest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save the trained model
model_file_path = 'model/iris_model.joblib'
joblib.dump(model, model_file_path)

try:
    # Upload the trained model to S3
    s3.upload_file(model_file_path, bucket_name, 'iris_model.joblib')
    print(f"Model trained and uploaded to S3 as iris_model.joblib")
except Exception as e:
    print(f"Error uploading model to S3: {e}")
import os
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
import pandas as pd
import boto3

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

# Upload the train and test CSV files to S3
s3 = boto3.client('s3')
bucket_name = 'iris-airflow-bucket'

try:
    s3.upload_file(train_file_path, bucket_name, 'iris_train.csv')
    s3.upload_file(test_file_path, bucket_name, 'iris_test.csv')
    print(f"Train and test datasets uploaded to S3 bucket: {bucket_name}")
except Exception as e:
    print(f"Error uploading datasets to S3: {e}")

# Train the RandomForest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save the trained model
model_file_path = 'model/iris_model.joblib'
joblib.dump(model, model_file_path)

try:
    # Upload the trained model to S3
    s3.upload_file(model_file_path, bucket_name, 'iris_model.joblib')
    print(f"Model trained and uploaded to S3 as iris_model.joblib")
except Exception as e:
    print(f"Error uploading model to S3: {e}")
