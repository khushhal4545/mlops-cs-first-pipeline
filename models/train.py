# from pathlib import Path

# import joblib
# from sklearn.datasets import load_iris
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score


# # 1. Load the Iris dataset
# iris = load_iris()

# X = iris.data
# y = iris.target


# # 2. Split data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(
#     X,
#     y,
#     test_size=0.2,
#     random_state=42
# )


# # 3. Create and train the model
# model = RandomForestClassifier(random_state=42)

# model.fit(X_train, y_train)


# # 4. Predict on test data
# y_pred = model.predict(X_test)


# # 5. Calculate accuracy
# accuracy = accuracy_score(y_test, y_pred)

# print(f"Accuracy: {accuracy}")


# # 6. Save model in the same folder as train.py
# model_path = Path(__file__).resolve().parent / "iris_model.joblib"

# joblib.dump(model, model_path)

# print(f"Model saved to: {model_path}")


from pathlib import Path
import joblib
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, 
                                                    y, 
                                                    test_size=0.2, 
                                                    random_state=42)


model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

# Save the trained model as reusable artifact
model_dir = Path("models")
model_dir.mkdir(parents=True, exist_ok=True)
joblib.dump(model, model_dir / "iris_model.joblib")
print(f"Model saved to {model_dir / 'iris_model.joblib'}")