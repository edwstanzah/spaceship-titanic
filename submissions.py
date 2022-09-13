import subprocess

# df = pd.read_csv('data/test.csv')

# # df = ProcessTestData.full_processing(df)
# # df = ProcessTestData.fill_missing_values(df)


# X = df.drop('PassengerId', axis=1)

# model = load('models/default_catboost.joblib')

# pred = (model.predict_proba(X)[:,1] >= 0.5).astype(bool)


# df_to_save = pd.DataFrame(data=pred, index=df.PassengerId, columns=['Transported'])

# # df_to_save['Transported'] = df_to_save['Transported'].map({0:False, 1:True})

# df_to_save.to_csv('submission-files/base_catboost.csv')

# # best result so far, simple LR with 0.52 threshold

# submit the prediction to Kaggle
subprocess.run('kaggle competitions submit -c spaceship-titanic -f submission-files/final_final_catboost.csv -m "final catboost"')