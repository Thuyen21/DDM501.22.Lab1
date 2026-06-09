# train.py - train an Iris classifier and save it to a file
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

# 1. Load data (X = features, y = labels)
X, y = load_iris(return_X_y=True)

# 2. Split into train/test (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. Train a Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 4. Quick evaluation on the test set
accuracy = model.score(X_test, y_test)
print(f"Test accuracy: {accuracy:.3f}")

# 5. Save the model to a file for reuse
joblib.dump(model, "model.joblib")
print("Saved model to model.joblib")
