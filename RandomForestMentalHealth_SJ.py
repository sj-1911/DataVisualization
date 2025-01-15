import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

def predict_next_month_from_file(file_path):
    # Read the provided CSV file
    data = pd.read_csv(file_path)
    
    required_columns = ['Time Period', 'Value']
    if not all(col in data.columns for col in required_columns):
        print(f"Error: Dataset must contain the columns {required_columns}.")
        return

    data['Time Period'] = pd.to_numeric(data['Time Period'], errors='coerce')
    data = data.dropna(subset=['Time Period', 'Value'])

    data = data.sort_values('Time Period')
    
    X = data[['Time Period']]
    y = data['Value']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Mean Squared Error: {mse:.2f}")
    print(f"R^2 Score: {r2:.2f}")
    
    next_month = X['Time Period'].max() + 1
    next_month_pred = model.predict([[next_month]])[0]
    print(f"Predicted value for next month (Time Period {next_month}): {next_month_pred:.2f}")
    
    plt.scatter(X_test, y_test, label='Actual', alpha=0.7)
    plt.scatter(X_test, y_pred, label='Predicted', alpha=0.7)
    plt.xlabel('Time Period')
    plt.ylabel('Percentage of Mental Health Care Usage')  
    plt.legend()
    plt.title('Actual vs Predicted Values')
    plt.show()

file_path = r"C:\Users\spenc\Downloads\Mental_Health_Care_in_the_Last_4_Weeks.csv"
predict_next_month_from_file(file_path)
