import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib

# This is a script to load our model without going
# through the Notebook, just run this file and
# it would be ready to use!

PATH = "data/processed/dataset.csv"
SAVE_PATH = "models/logistic_regression_cv.pkl"

df = pd.read_csv(PATH)

X = df.copy().drop("Target", axis=1)
y = df["Target"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=47,
)

model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegressionCV(
        max_iter=1000,
        n_jobs=-1,
        random_state=42,
        cv=5,
        solver="lbfgs",
        class_weight="balanced",
        scoring="roc_auc"
    ))
])

model.fit(X_train, y_train)

joblib.dump(model, SAVE_PATH)