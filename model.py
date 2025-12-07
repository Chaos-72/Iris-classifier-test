import pickle
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

def train_and_save(path="iris_model.pkl"):
    iris = load_iris()
    X, y = iris.data, iris.target
    # simple pipeline: standardize -> logistic regression
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=200))
    model.fit(X, y)

    # Save model + target names to the pickle for easy mapping
    payload = {
        "model": model,
        "target_names": iris.target_names.tolist()
    }

    with open(path, "wb") as f:
        pickle.dump(payload, f)
    print(f"Saved model to {path}")

if __name__ == "__main__":
    train_and_save()
