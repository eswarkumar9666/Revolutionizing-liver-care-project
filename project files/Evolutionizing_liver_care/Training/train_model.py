import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

# Load and preprocess data
df = pd.read_excel(r"C:\Users\Evolutionizing_liver_care\dataset\HealthCareData.xlsx")

# Rename for consistency (optional)
df.rename(columns={'SGOT (AST)': 'SGOT', 'SGPT (ALT)': 'SGPT'}, inplace=True)

df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})
df['Cirrhosis_Status'] = df['Cirrhosis_Status'].map({'No': 0, 'Yes': 1})

X = df.drop(columns=['Patient_ID', 'Cirrhosis_Status'])
y = df['Cirrhosis_Status']

X = X.fillna(X.mean())
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Balance the dataset
smote = SMOTE(sampling_strategy=0.8, random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Train XGBoost model
model = XGBClassifier(n_estimators=200, learning_rate=0.1, max_depth=4, use_label_encoder=False, eval_metric='logloss')
model.fit(X_train, y_train)

# Predict with threshold adjustment
y_proba = model.predict_proba(X_test)[:, 1]
y_pred = (y_proba >= 0.6).astype(int)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
