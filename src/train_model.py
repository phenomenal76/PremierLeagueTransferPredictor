
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

print("Premier League Transfer Fee Prediction Model")
print("=" * 50)

# Find the project folder
project_folder = Path(__file__).resolve().parent.parent

# Load cleaned dataset
file_path = project_folder / "data" / "cleaned_transfers.csv"
df = pd.read_csv(file_path)

print("\nCleaned dataset loaded!")
print("Rows:", len(df))

# Select features
features = [
    "age",
    "position",
    "club_name",
    "club_involved_name",
    "transfer_period",
    "year",
    "country"
]

target = "fee_cleaned"

# Remove rows with missing values in selected columns
df = df.dropna(subset=features + [target])

# Input and output
X = df[features]
y = df[target]

print("\nFeatures selected:")
print(features)

print("\nTarget column:", target)

# Separate numerical and categorical columns
numerical_features = ["age", "year"]

categorical_features = [
    "position",
    "club_name",
    "club_involved_name",
    "transfer_period",
    "country"
]

# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        ),
        (
            "numerical",
            "passthrough",
            numerical_features
        )
    ]
)

# Create machine learning model
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

# Create pipeline
pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ]
)

# Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining rows:", len(X_train))
print("Testing rows:", len(X_test))

# Train the model
print("\nTraining model...")
pipeline.fit(X_train, y_train)

print("Model training completed!")

# Make predictions
predictions = pipeline.predict(X_test)

# Evaluate model
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("\nModel Evaluation")
print("-" * 30)
print("Mean Absolute Error:", round(mae, 2))
print("R2 Score:", round(r2, 2))

print("\nFirst 10 predictions:")
for actual, predicted in zip(y_test.head(10), predictions[:10]):
    print(
        "Actual Fee:",
        round(actual, 2),
        "| Predicted Fee:",
        round(predicted, 2)
    )

print("\nStep 6 completed successfully!") 
import joblib

# Create models folder
model_folder = project_folder / "models"
model_folder.mkdir(exist_ok=True)

# Save trained model
model_path = model_folder / "transfer_fee_model.pkl"

joblib.dump(pipeline, model_path)

print("Trained model saved successfully!")
print("Saved file:", model_path)