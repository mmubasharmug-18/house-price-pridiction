#importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

#loading dataset
df = pd.read_csv("Housing.csv")

#data exploration
print("First 5 rows of dataset:")
print(df.head())

print("\nDataset Info:")
print(df.info())

print("\nStatistical Summary:")
print(df.describe())

#viualization

#creating scatter plot
plt.scatter(df["area"], df["price"])
plt.xlabel("Area")
plt.ylabel("Price")
plt.title("Area vs House Price")
plt.show()

#creating histogram
plt.hist(df["price"], bins=20)
plt.title("House Price Distribution")
plt.xlabel("Price")
plt.ylabel("Frequency")
plt.show()

#features and target
X = df[["area","bedrooms","bathrooms","stories","parking"]]
y = df["price"]

#test train split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

#creating object and train the model
model = LinearRegression()
model.fit(X_train,y_train)

#making predictions
y_pred = model.predict(X_test)
print("Predicted Prices:",y_pred)

#actual prices vs pridicted prices
plt.scatter(X_test["area"], y_test)
plt.scatter(X_test["area"], y_pred)

plt.xlabel("Area")
plt.ylabel("Price")
plt.title("Actual vs Predicted House Prices")

plt.legend(["Actual Price","Predicted Price"])
plt.show()

#evaluation metrics
mae = mean_absolute_error(y_test,y_pred)
mse = mean_squared_error(y_test,y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test,y_pred)

print("MAE:",mae)
print("MSE:",mse)
print("RMSE:",rmse)
print("R2 Score:",r2)

#predicting new house price
area = float(input("Enter the area of house: "))
bedrooms = int(input("Enter number of bedrooms: "))
bathrooms = int(input("Enter bathrooms: "))
stories = int(input("Enter number of stories: "))
parking = int(input("Enter parking spaces: "))

prediction = model.predict([[area,bedrooms,bathrooms,stories,parking]])
print("Predicted price:",prediction)