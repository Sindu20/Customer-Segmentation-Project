# Customer Segmentation Project

## Overview
Segmented mall customers based on Annual Income and Spending Score
using KMeans Clustering.

## Tools Used
- Python
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn

## Files
- `Customer_Segmentation.py` — Main code
- `Mall_Customers.csv` — Dataset
- `outputs/` — All generated plots and segmented CSV

## Results
- Used Elbow Method to find optimal K = 5
- 5 Customer Segments identified:
  - Cluster 0 → High Income, High Spending (VIP Customers)
  - Cluster 1 → Low Income, Low Spending (Budget Customers)
  - Cluster 2 → Medium Income, Medium Spending (Moderate Customers)
  - Cluster 3 → High Income, Low Spending (Careful Spenders)
  - Cluster 4 → Low Income, High Spending (Target Customers)
