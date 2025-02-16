import pandas as pd
from preprocess import prepare
from model import make_pipeline, save_model, eval_model
from config import TRAIN_PATH, VAL_PATH, TEST_PATH, NUMERIC_COLUMNS, CATEGORICAL_COLUMNS

# Load datasets
df_train = pd.read_csv(TRAIN_PATH)
df_val = pd.read_csv(VAL_PATH)
df_test = pd.read_csv(TEST_PATH)

# Prepare data
df_train = prepare(df_train)
df_val = prepare(df_val)
df_test = prepare(df_test)

# Create and train the model
model = make_pipeline(NUMERIC_COLUMNS, CATEGORICAL_COLUMNS)
model.fit(df_train[NUMERIC_COLUMNS + CATEGORICAL_COLUMNS], df_train['trip_duration'])

# Evaluate the model
print("Test Data Evaluation:")
eval_model(model, df_test[NUMERIC_COLUMNS + CATEGORICAL_COLUMNS], df_test['trip_duration'])

print("Train Data Evaluation:")
eval_model(model, df_train[NUMERIC_COLUMNS + CATEGORICAL_COLUMNS], df_train['trip_duration'])

print("Validation Data Evaluation:")
eval_model(model, df_val[NUMERIC_COLUMNS + CATEGORICAL_COLUMNS], df_val['trip_duration'])

# Save model
save_model(model)
