from model import load_model
from preprocess import prepare
from config import TEST_PATH, NUMERIC_COLUMNS, CATEGORICAL_COLUMNS
import pandas as pd

df_test = prepare(pd.read_csv(TEST_PATH))
model = load_model()

y_predict = model.predict(df_test[NUMERIC_COLUMNS + CATEGORICAL_COLUMNS])
print(f"Predictions: {y_predict[:5]}")
