import pandas as pd
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import GradientBoostingRegressor

# CSV-Datei laden
df = pd.read_csv("data.csv")

# Neue Feature erstellen
df["apartment_age"] = 2026 - df["year_built"]

# Zielvariable festlegen
target = "rent"

# Eingabevariablen definieren
numeric_features = [
    "living_space",
    "rooms",
    "floor",
    "year_built",
    "apartment_age"
]

categorical_features = [
    "zip_code",
    "balcony",
    "parking"
]

# X und y erstellen
X = df[numeric_features + categorical_features]
y = df[target]

# Pipeline für numerische Daten
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# Pipeline für kategoriale Daten
categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

# Gesamtes Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

# Finales Modell definieren
model = GradientBoostingRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=3,
    random_state=42
)

# Gesamte Pipeline erstellen
pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", model)
])

# Modell trainieren
pipeline.fit(X, y)

# Modell speichern
joblib.dump(pipeline, "best_model.joblib")

print("Modell erfolgreich trainiert und als best_model.joblib gespeichert.")
