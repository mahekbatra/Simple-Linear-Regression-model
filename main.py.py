#loading dataset
import pandas
db=pandas.read_csv('salary_data.csv')
print("Dataset has been loaded")

#Creating features and target.
Feature=db['YearsExperience'].values.reshape(-1,1)
Target=db['Salary']

#loading LinearRegression
from sklearn.linear_model import LinearRegression
model=LinearRegression()
model.fit(Feature,Target)
print("Model has been trained")

print(model.coef_)

#Saving Model
import joblib
joblib.dump(model,'salary_model.pk1')
print("Model has been saved in workspace")
