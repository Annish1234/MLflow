from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
import mlflow
import numpy as np
# Load dataset
data = load_iris()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42)

input_example = np.array([X_test[0]])
# Train the model
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)
accuracy = model.score(X_test, y_test)
mlflow.start_run()
mlflow.log_metric("accuracy", accuracy)
mlflow.sklearn.log_model(model, "model", input_example=input_example)
mlflow.end_run()
<<<<<<< HEAD
#this is pushed to github
=======
>>>>>>> 7406396a4ff708543244d85f087a3cc86f1fc22a
# Save the model
joblib.dump(model, "model/model.pkl")