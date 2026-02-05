import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder

print("Starting program...")

# ---------- Load Dataset ----------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(BASE_DIR, "..", "dataset", "Sleep_health_and_lifestyle_dataset.csv")

df = pd.read_csv(data_path)

print("\nDataset loaded successfully!")
print(df.head())

print("\nOriginal Dataset Info:")
df.info()

# ---------- Drop Unnecessary Columns ----------
df = df.drop(columns=[
    "Person ID",
    "Occupation",
    "Blood Pressure",
    "Sleep Disorder"
])

print("\nColumns after dropping:")
print(df.columns)

# ---------- Check Missing Values ----------
print("\nMissing values before handling:")
print(df.isnull().sum())

# ---------- Handle Missing Values ----------
df["BMI Category"].fillna(df["BMI Category"].mode()[0], inplace=True)

# ---------- Encode Categorical Columns ----------
le = LabelEncoder()

df["Gender"] = le.fit_transform(df["Gender"])
df["BMI Category"] = le.fit_transform(df["BMI Category"])

print("\nFinal Dataset Info after cleaning:")
df.info()

print("\nData cleaning and preprocessing DONE ✅")
# ---------- Train-Test Split ----------
from sklearn.model_selection import train_test_split

X = df.drop("Quality of Sleep", axis=1)
y = df["Quality of Sleep"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nTrain-test split done!")
print("Training data size:", X_train.shape)
print("Testing data size:", X_test.shape)

# ---------- Train Machine Learning Model ----------
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

print("\nModel training completed!")

# ---------- Model Evaluation ----------
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("\nModel Accuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))
# ---------- Save the Trained Model ----------
import pickle

model_path = os.path.join(BASE_DIR, "..", "model", "model.pkl")

with open(model_path, "wb") as file:
    pickle.dump(model, file)

print("\nModel saved successfully as model.pkl")

# ---------- Test Prediction with Sample Input ----------
# Sample input order:
# Gender, Age, Sleep Duration, Physical Activity Level,
# Stress Level, BMI Category, Heart Rate, Daily Steps

sample_input = [[
    1,      # Gender (Male=1, Female=0)
    25,     # Age
    7.0,    # Sleep Duration
    60,     # Physical Activity Level
    5,      # Stress Level
    1,      # BMI Category
    72,     # Heart Rate
    8000    # Daily Steps
]]

prediction = model.predict(sample_input)

# Convert prediction to label
if prediction[0] >= 8:
    result = "Good Sleep"
elif prediction[0] >= 5:
    result = "Average Sleep"
else:
    result = "Poor Sleep"

print("\nSample Prediction Result:", result)


