House Price Prediction using Linear Regression
This project implements a Multiple Linear Regression model to predict house prices based on physical attributes like area, number of bedrooms, bathrooms, and parking spaces.

Project Overview
The goal of this project is to analyze a housing dataset and build a predictive model that estimates the market value of a property. It includes data visualization, model training, evaluation, and a simple interactive command-line interface for manual predictions.

Features Used
The model uses the following features for prediction:
Area: Total square footage of the house.
Bedrooms: Number of bedrooms.
Bathrooms: Number of bathrooms.
Stories: Number of floors.
Parking: Number of parking spaces available.

Tech Stack
Language: Python
Libraries: * pandas & numpy (Data manipulation)
matplotlib (Data visualization)
scikit-learn (Machine learning & Evaluation)

Model Performance
The script evaluates the model using standard regression metrics:
Mean Absolute Error (MAE): Average magnitude of errors.
Root Mean Squared Error (RMSE): Standard deviation of the prediction errors.
R² Score: Indicates how well the independent variables explain the variance in price.

Future Improvements
Implement Feature Scaling (StandardScaler/MinMaxScaler) to improve accuracy.
Handle categorical variables (e.g., "Mainroad", "Basement", "Airconditioning") using One-Hot Encoding.

Experiment with more complex models like Random Forest or Gradient Boosting.
