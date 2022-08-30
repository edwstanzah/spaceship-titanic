# TODO: treat test.csv as EDA
# TODO: treat df as feature engineering
# TODO: make predictions based on test.csv
# TODO: save .csv file
# TODO: submit to Kaggle

import pandas as pd
from custom_functions import ProcessTestData
from joblib import load


df = pd.read_csv('data/test.csv')

df = ProcessTestData.full_processing(df)

X = df.drop('PassengerId', axis=1)

model = load('models/simple_model.joblib')

pred = (model.predict_proba(X)[:,1] >= 0.52).astype(bool)

print(pred)

df_to_save = pd.DataFrame(data=pred, index=df.PassengerId, columns=['Transported'])

# df_to_save['Transported'] = df_to_save['Transported'].map({0:False, 1:True})

df_to_save.to_csv('data/sixth_submission.csv')

# best result so far, simple LR with 0.52 threshold