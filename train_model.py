import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

# Load your dataset
df = pd.read_csv(r"C:\Users\devar\Downloads\heart_attack\data\Cardiovascular_Disease_Dataset.csv")

# Drop ID column if present
if 'patientid' in df.columns:
    df = df.drop(columns=['patientid'])

# Features & target
X = df.drop(columns=['target'])
y = df['target']

# Define categorical & numeric columns
cat_cols = ['chestpain', 'restingrelectro', 'slope']
num_cols = [c for c in X.columns if c not in cat_cols]

# Preprocessing
numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])
categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, num_cols),
        ('cat', categorical_transformer, cat_cols)
    ],
    remainder='passthrough'
)

# Model pipeline
model = Pipeline(steps=[('preprocessor', preprocessor),
                        ('classifier', LogisticRegression(max_iter=1000))])

# Train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model.fit(X_train, y_train)

# Save the model
joblib.dump(model, "best_heart_model.joblib")
print("Model saved as best_heart_model.joblib")
