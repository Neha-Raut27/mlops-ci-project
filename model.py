
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

def train_model():

    iris = load_iris()
    X = iris.data
    y = iris.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    joblib.dump(model, "iris_model.pkl")

    os.makedirs("reports", exist_ok=True)

    with open("reports/build_report.txt", "w") as f:
        f.write("MLOps Build Report\n")
        f.write("==================\n")
        f.write(f"Model: Decision Tree\n")
        f.write(f"Accuracy: {accuracy:.2f}\n")
        f.write("\nClassification Report:\n")
        f.write(classification_report(y_test, predictions))

    print("Model Training Completed!")
    print("Model Accuracy:", accuracy)

    return model, accuracy


if __name__ == "__main__":
    train_model()
