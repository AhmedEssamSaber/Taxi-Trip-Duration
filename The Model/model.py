import numpy as np
import os
import pickle
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# Define the model path in the same folder as the script
MODEL_PATH = os.path.join(os.getcwd(), 'saved_models', 'model.pkl')

def log_transform(x):
    return np.log1p(np.maximum(x, 0))

def make_pipeline(numeric_columns, categorical_columns):
    LogFeatures = FunctionTransformer(log_transform)
    
    # Numeric transformation pipeline
    numeric_transform = Pipeline(steps=[ 
        ('scaler', StandardScaler()), 
        ('poly', PolynomialFeatures(degree=3, include_bias=False)),  # Remove sparse argument
        ('log', LogFeatures)
    ])
    
    # Categorical transformation pipeline
    categorical_transform = Pipeline(steps=[ 
        ('ohe', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    # Column transformer
    transformer = ColumnTransformer(
        transformers=[
            ('num', numeric_transform, numeric_columns),
            ('cat', categorical_transform, categorical_columns)
        ]
    )
    
    # Final pipeline with Ridge regressor
    model = Pipeline(steps=[
        ('preprocessor', transformer),
        ('regressor', Ridge())
    ])
    
    return model

def save_model(model):
    """Save the trained model using joblib."""
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)  # Ensure the folder exists
    joblib.dump(model, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"The model file does not exist at {MODEL_PATH}")
    model = joblib.load(MODEL_PATH)
    print(f"Model loaded from {MODEL_PATH}")
    return model

def eval_model(model, x, target):
    y_predict = model.predict(x)
    rmse = np.sqrt(mean_squared_error(target, y_predict))
    r2 = r2_score(target, y_predict)
    print(f'RMSE = {rmse:.4f} and R² score = {r2:.4f}')
    return rmse, r2
