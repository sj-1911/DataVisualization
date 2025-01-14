# -*- coding: utf-8 -*-
"""
Spencer Jackson
Data Visualization Project - FDOT Crash Data Analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate
from scipy.stats import ttest_ind, zscore

# Load dataset
url = "https://raw.githubusercontent.com/cam-alvarez/Personal-Portfolio-and-Blog/main/public/2020-tristatecounty-crashdata.csv"
df = pd.read_csv(url)

# Filter the dataset for accidents with speed limits under 50 mph
speed_limit_under_50 = df[df['SPEED_LIMIT'] < 50]
speed_limit_50_or_higher = df[df['SPEED_LIMIT'] >= 50]

# Calculate statistics for speed limits under 50 mph
total_accidents_under_50 = len(speed_limit_under_50)
avg_injuries_under_50 = speed_limit_under_50['NUMBER_OF_INJURED'].mean()
avg_fatalities_under_50 = speed_limit_under_50['NUMBER_OF_KILLED'].mean()
avg_serious_injuries_under_50 = speed_limit_under_50['NUMBER_OF_SERIOUS_INJURIES'].mean()

# Calculate statistics for speed limits of 50 mph or higher
total_accidents_50_or_higher = len(speed_limit_50_or_higher)
avg_injuries_50_or_higher = speed_limit_50_or_higher['NUMBER_OF_INJURED'].mean()
avg_fatalities_50_or_higher = speed_limit_50_or_higher['NUMBER_OF_KILLED'].mean()
avg_serious_injuries_50_or_higher = speed_limit_50_or_higher['NUMBER_OF_SERIOUS_INJURIES'].mean()

# Display statistics
print("Statistics for accidents with speed limits under 50 mph:")
print(f"Total Accidents: {total_accidents_under_50}")
print(f"Average Number of Injured: {avg_injuries_under_50:.2f}")
print(f"Average Number of Fatalities: {avg_fatalities_under_50:.2f}")
print(f"Average Number of Serious Injuries: {avg_serious_injuries_under_50:.2f}\n")

print("Statistics for accidents with speed limits of 50 mph or higher:")
print(f"Total Accidents: {total_accidents_50_or_higher}")
print(f"Average Number of Injured: {avg_injuries_50_or_higher:.2f}")
print(f"Average Number of Fatalities: {avg_fatalities_50_or_higher:.2f}")
print(f"Average Number of Serious Injuries: {avg_serious_injuries_50_or_higher:.2f}\n")

# Bar chart visualization
categories = ['Avg. Injuries', 'Avg. Fatalities', 'Avg. Serious Injuries']
speed_limit_under_50_stats = [avg_injuries_under_50, avg_fatalities_under_50, avg_serious_injuries_under_50]
speed_limit_50_or_higher_stats = [avg_injuries_50_or_higher, avg_fatalities_50_or_higher, avg_serious_injuries_50_or_higher]

plt.figure(figsize=(10, 6))
bar_width = 0.35
index = range(len(categories))
bar1 = plt.bar(index, speed_limit_under_50_stats, bar_width, label='Speed Limit < 50 mph')
bar2 = plt.bar([i + bar_width for i in index], speed_limit_50_or_higher_stats, bar_width, label='Speed Limit ≥ 50 mph')

# Add value labels on top of bars
for bars in [bar1, bar2]:
    for bar in bars:
        height = bar.get_height()
        plt.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width() / 2, height), 
                     xytext=(0, 3), textcoords='offset points', ha='center', va='bottom')

# Customize plot
plt.xlabel('Categories', fontsize=12)
plt.ylabel('Average Statistics', fontsize=12)
plt.title('Average Accident Statistics by Speed Limit', fontsize=14)
plt.xticks([i + bar_width / 2 for i in index], categories)
plt.legend()
plt.grid(True, axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# Tabulated data for accidents by speed limit
data = [
    [25.0, 0.465, 0.023, 0.105],
    [30.0, 0.299, 0.005, 0.021],
    [35.0, 0.391, 0.010, 0.040],
    [40.0, 0.387, 0.009, 0.030],
    [45.0, 0.499, 0.015, 0.050],
    [50.0, 0.604, 0.014, 0.085],
    [55.0, 0.400, 0.008, 0.038],
    [60.0, 0.372, 0.009, 0.032],
    [65.0, 0.539, 0.019, 0.088],
    [70.0, 0.463, 0.018, 0.059]
]
headers = ["Speed Limit", "Avg. Number of Injured", "Avg. Number of Killed", "Avg. Number of Serious Injuries"]
print(tabulate(data, headers=headers, tablefmt="pretty"))

# T-test and Z-score analysis
injuries_under_50 = speed_limit_under_50['NUMBER_OF_INJURED']
injuries_50_or_higher = speed_limit_50_or_higher['NUMBER_OF_INJURED']

t_stat, p_value = ttest_ind(injuries_under_50, injuries_50_or_higher)
z_under_50 = zscore(injuries_under_50)
z_50_or_higher = zscore(injuries_50_or_higher)

print("T-test results:")
print(f"T-statistic: {t_stat:.4f}")
print(f"P-value: {p_value:.4f}\n")

print("Z-scores for accidents with speed limits under 50 mph:")
print(z_under_50)
print("\nZ-scores for accidents with speed limits of 50 mph or higher:")
print(z_50_or_higher)
