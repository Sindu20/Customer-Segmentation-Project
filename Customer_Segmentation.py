import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')   # saves plots as files (works in CMD without GUI)
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import warnings
import os
warnings.filterwarnings('ignore')
os.makedirs("outputs", exist_ok=True)
print("=" * 55)
print("   CUSTOMER SEGMENTATION PROJECT")
print("=" * 55)
data = pd.read_csv("Mall_Customers.csv")
print("\n[1] Dataset Loaded Successfully")
print(f"    Rows: {data.shape[0]}  |  Columns: {data.shape[1]}")
print("\nFirst 5 rows:")
print(data.head())
print("\nDataset Info:")
print(data.describe())
print("\n[2] Running Elbow Method (K = 1 to 10)...")
X = data[['Annual Income (k$)', 'Spending Score (1-100)']].values
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42, n_init=10)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), wcss, marker='o', color='steelblue', linewidth=2, markersize=8)
plt.title('Elbow Method - Finding Optimal Clusters', fontsize=14, fontweight='bold')
plt.xlabel('Number of Clusters', fontsize=12)
plt.ylabel('WCSS (Within Cluster Sum of Squares)', fontsize=12)
plt.xticks(range(1, 11))
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig("outputs/1_elbow_method.png", dpi=150)
plt.close()
print("    Saved: outputs/1_elbow_method.png")
print("\n[3] Training KMeans with 5 Clusters...")
kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
data['Cluster'] = kmeans.fit_predict(X)
print("\n[4] Cluster Statistics (Mean Values):")
print("-" * 55)
stats = data.groupby('Cluster')[
    ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
].mean().round(1)
stats['Count'] = data.groupby('Cluster').size()
print(stats)
print("\nCluster Interpretations:")
print("-" * 55)
interpretations = {
    0: "High Income, High Spending  → VIP / Premium Customers",
    1: "Low Income,  Low Spending   → Budget Customers",
    2: "Medium Income, Medium Spend → Average / Moderate Customers",
    3: "High Income,  Low Spending  → Careful / Conservative Spenders",
    4: "Low Income,  High Spending  → Impulsive / Target Customers",
}
for k, v in interpretations.items():
    print(f"  Cluster {k} → {v}")
print("\n[5] Generating Cluster Scatter Plot...")
colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6']
plt.figure(figsize=(10, 7))
for i in range(5):
    cluster_data = data[data['Cluster'] == i]
    plt.scatter(
        cluster_data['Annual Income (k$)'],
        cluster_data['Spending Score (1-100)'],
        s=80, c=colors[i], label=f'Cluster {i}', alpha=0.8
    )
plt.scatter(
    kmeans.cluster_centers_[:, 0],
    kmeans.cluster_centers_[:, 1],
    s=300, c='black', marker='X', label='Centroids', zorder=5
)
plt.title('Customer Segmentation - Annual Income vs Spending Score',
          fontsize=13, fontweight='bold')
plt.xlabel('Annual Income (k$)', fontsize=12)
plt.ylabel('Spending Score (1-100)', fontsize=12)
plt.legend(fontsize=10)
plt.grid(True, linestyle='--', alpha=0.4)
plt.tight_layout()
plt.savefig("outputs/2_cluster_scatter.png", dpi=150)
plt.close()
print("    Saved: outputs/2_cluster_scatter.png")
print("\n[6] Generating Demographics Plots...")
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
data.boxplot(column='Age', by='Cluster', ax=axes[0],
             patch_artist=True,
             boxprops=dict(facecolor='lightblue', color='navy'),
             medianprops=dict(color='red', linewidth=2))
axes[0].set_title('Age Distribution by Cluster', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Cluster', fontsize=11)
axes[0].set_ylabel('Age', fontsize=11)
axes[0].grid(True, linestyle='--', alpha=0.4)
gender_cluster = data.groupby(['Cluster', 'Gender']).size().unstack()
gender_cluster.plot(kind='bar', ax=axes[1], color=['#ff9999', '#66b3ff'],
                    edgecolor='black', width=0.6)
axes[1].set_title('Gender Distribution by Cluster', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Cluster', fontsize=11)
axes[1].set_ylabel('Number of Customers', fontsize=11)
axes[1].legend(title='Gender', fontsize=10)
axes[1].tick_params(axis='x', rotation=0)
axes[1].grid(True, linestyle='--', alpha=0.4, axis='y')
plt.suptitle('')   # remove auto suptitle from boxplot
plt.tight_layout()
plt.savefig("outputs/3_demographics.png", dpi=150)
plt.close()
print("    Saved: outputs/3_demographics.png")
plt.figure(figsize=(10, 6))
for i in range(5):
    cluster_data = data[data['Cluster'] == i]['Spending Score (1-100)']
    cluster_data.plot(kind='kde', label=f'Cluster {i}',
                      color=colors[i], linewidth=2)
plt.title('Spending Score Distribution by Cluster',
          fontsize=13, fontweight='bold')
plt.xlabel('Spending Score (1-100)', fontsize=12)
plt.ylabel('Density', fontsize=12)
plt.legend(fontsize=10)
plt.grid(True, linestyle='--', alpha=0.4)
plt.tight_layout()
plt.savefig("outputs/4_spending_distribution.png", dpi=150)
plt.close()
print("    Saved: outputs/4_spending_distribution.png")
data.to_csv("outputs/Mall_Customers_Segmented.csv", index=False)
print("\n[7] Segmented data saved: outputs/Mall_Customers_Segmented.csv")
print("\n" + "=" * 55)
print("   ALL DONE! Check the 'outputs' folder for:")
print("   1_elbow_method.png")
print("   2_cluster_scatter.png")
print("   3_demographics.png")
print("   4_spending_distribution.png")
print("   Mall_Customers_Segmented.csv")
print("=" * 55)