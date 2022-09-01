import pandas as pd
from custom_functions import ProcessTestData
from joblib import load
import subprocess

df = pd.read_csv('data/test.csv')

df = ProcessTestData.full_processing(df)

X = df.drop('PassengerId', axis=1)

model = load('models/simple_model.joblib')

pred = (model.predict_proba(X)[:,1] >= 0.52).astype(bool)


df_to_save = pd.DataFrame(data=pred, index=df.PassengerId, columns=['Transported'])

# df_to_save['Transported'] = df_to_save['Transported'].map({0:False, 1:True})

df_to_save.to_csv('data/hello_again.csv')

# best result so far, simple LR with 0.52 threshold

# submit the prediction to Kaggle
subprocess.run('kaggle competitions submit -c spaceship-titanic -f data/shell_submission2.csv -m "from shell"')