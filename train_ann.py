import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib

# Load dataset
df = pd.read_excel("dataset.xlsx")

# Show dataset info
print("\nFirst 5 Rows:\n", df.head())
print("\nColumns:\n", df.columns)
print("\nShape:", df.shape)

# Features and target
X = df[['attendance', 'assignment', 'quiz', 'mid', 'study_hours']]
y = df['result']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Build ANN model
model = MLPClassifier(
    hidden_layer_sizes=(10, 5),
    activation='relu',
    max_iter=500,
    random_state=42
)

# Train model
model.fit(X_train, y_train)

print("\nTraining Complete")
print("Iterations:", model.n_iter_)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Save model and scaler
joblib.dump(model, "model.joblib")
joblib.dump(scaler, "scaler.joblib")

print("\nModel and scaler saved successfully!")