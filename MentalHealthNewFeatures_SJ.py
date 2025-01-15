import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score

def predict_next_month_from_file(file_path):
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
    
    models = {
        "Random Forest Regressor": RandomForestRegressor(random_state=42),
        "Linear Regression": LinearRegression(),
        "Decision Tree Regressor": DecisionTreeRegressor(random_state=42),
        "K-Nearest Neighbors Regressor": KNeighborsRegressor(n_neighbors=5)
    }
    
    model_performance = {}
    
    for model_name, model in models.items():
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        model_performance[model_name] = {
            "MSE": mse,
            "R²": r2,
            "Next Month Prediction": model.predict([[X['Time Period'].max() + 1]])[0]
        }
        
        print(f"Model: {model_name}")
        print(f"Mean Squared Error: {mse:.2f}")
        print(f"R² Score: {r2:.2f}")
        print(f"Predicted value for next month (Time Period {X['Time Period'].max() + 1}): {model_performance[model_name]['Next Month Prediction']:.2f}")
        print("-" * 50)
    
    plt.figure(figsize=(12, 8))
    for model_name, model in models.items():
        y_pred_plot = model.predict(X_test)
        plt.scatter(X_test, y_test, label='Actual', alpha=0.7, color='gray')
        plt.scatter(X_test, y_pred_plot, label=f'Predicted - {model_name}', alpha=0.7)
    
    plt.xlabel('Time Period')
    plt.ylabel('Percentage of Mental Health Care Usage')
    plt.title('Actual vs Predicted Values (Multiple Models)')
    plt.legend()
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

# Replace file path to use
file_path = r"C:\Users\spenc\Downloads\Mental_Health_Care_in_the_Last_4_Weeks.csv"
predict_next_month_from_file(file_path)
