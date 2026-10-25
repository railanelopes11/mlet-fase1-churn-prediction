import joblib

churn_model = joblib.load('models/churn_model.pkl')
print(churn_model)