# 🌟 Data Analysis and Visualization Projects  

Welcome to my repository showcasing data analysis and visualization projects, developed as part of my academic journey in **Computer Science** and **Data Analytics**. These projects reflect my passion for leveraging data-driven insights to solve real-world problems and highlight my proficiency in key tools and techniques.  

---

## 🚦 Florida Department of Transportation (FDOT) Data Project  

This project focuses on analyzing traffic data from the Florida Department of Transportation to uncover insights into **traffic patterns**, **vehicle types**, and **roadway conditions**. It showcases the **end-to-end data analysis process**, from cleaning and sorting raw data to creating meaningful visualizations.  

### ✨ Key Features  
- **Data Sorting and Cleaning**:  
  Organized datasets by date, location, and traffic volume while addressing missing and inconsistent data.  
- **Visualizations**:  
  - Line charts to show traffic trends over time.  
  - Bar charts for location-based comparisons.  
  - Heatmaps to identify peak traffic periods.  
- **Tools and Libraries**:  
  - 🐍 **Python**: Core language for data analysis.  
  - 📊 **Pandas**: For preprocessing and sorting data.  
  - 📈 **Matplotlib & Seaborn**: To craft insightful visualizations.  

---

## 🔮 Predicting Mental Health Care Trends  

This project aims to forecast **mental health care usage trends** using a **Random Forest Regressor**. It demonstrates predictive modeling and visualization based on historical data.  

### ✨ Key Steps in the Code  
- **Data Preparation**:  
  - Imported and cleaned a CSV dataset containing "Time Period" and "Value" columns.  
  - Converted time-series data into numeric format and handled missing values.  
- **Model Training**:  
  - Used a **Random Forest Regressor** to train and test the model.  
  - Evaluated performance with metrics like **Mean Squared Error (MSE)** and **R² Score**.  
- **Visualization**:  
  - Compared actual vs. predicted values through **scatter plots** for better interpretability.  
- **Outcome**:  
  - Forecasted future trends and visualized data insights to aid in decision-making.  

---

### 📂 Example Code Usage  
```python
file_path = r"C:\Users\YourPath\Mental_Health_Care_in_the_Last_4_Weeks.csv"  
predict_next_month_from_file(file_path)
