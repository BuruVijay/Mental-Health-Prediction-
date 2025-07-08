import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# Load dataset
df = pd.read_csv("Mental Health Dataset.csv")  # Replace with your actual CSV filename

# Drop irrelevant columns
df_cleaned = df.drop(columns=["Timestamp"])

# Fill missing values in 'self_employed' with most frequent value
# Fill missing values in 'self_employed' with most frequent value
imputer = SimpleImputer(strategy="most_frequent")
df_cleaned["self_employed"] = imputer.fit_transform(df_cleaned[["self_employed"]])[:, 0]



# Label encode all categorical columns
label_encoders = {}
for col in df_cleaned.columns:
    le = LabelEncoder()
    df_cleaned[col] = le.fit_transform(df_cleaned[col])
    label_encoders[col] = le

# Split features and target
X = df_cleaned.drop(columns=["treatment"])
y = df_cleaned["treatment"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Predict function for new inputs
def predict_new(input_dict):
    input_df = pd.DataFrame([input_dict])
    for col in input_df.columns:
        if col in label_encoders:
            input_df[col] = label_encoders[col].transform(input_df[col])
    prediction = model.predict(input_df)[0]
    return "Treatment Recommended" if prediction == 1 else "No Treatment Needed"
