import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("creditcard.csv")

print(df['Class'].value_counts())

fraud_df = df[df['Class'] == 1]
nonfraud_df = df[df['Class'] == 0]

# Transaction Class Distribution Graph
plt.figure(figsize = (6,4))
sns.countplot(data=df, x='Class', hue='Class', palette='Set2')
plt.title("Transaction Class Distribution")
plt.xlabel("Class (0 = Non-Fraud, 1 = Fraud)")
plt.ylabel("Count")
plt.savefig("transaction_class_distribution.png")
plt.show()

#Fraud vs Non-Fraud Transaction Amounts
plt.figure(figsize=(14,6))

# Fraud transactions distribution
plt.subplot(1, 2, 1)
sns.kdeplot(data=fraud_df, x="Amount", fill=True, color="red", alpha=0.6)
plt.xlim(0, 1000)  # Focus on lower amounts for visibility
plt.title("Fraud Transactions - Distribution")
plt.xlabel("Transaction Amount ($)")
plt.ylabel("Density")

# Non-fraud transactions distribution
plt.subplot(1, 2, 2)
sns.kdeplot(data=nonfraud_df, x="Amount", fill=True, color="blue", alpha=0.6)
plt.xlim(0, 1000)  # Keep same scale for comparison
plt.title("Non-Fraud Transactions - Distribution")
plt.xlabel("Transaction Amount ($)")
plt.ylabel("Density")

plt.tight_layout()
plt.savefig("Fraud VS Non Fraud.png")
plt.show()

# Time vs Fraud Histogram
#Fraud
plt.figure(figsize=(12,6))
sns.histplot(data=fraud_df, x="Time", hue="Class", bins=50, palette={0: "blue", 1: "red"}, alpha=0.6)
plt.title("Fraud vs Non-Fraud Transactions Over Time")
plt.xlabel("Time (seconds since first transaction)")
plt.ylabel("Count")
plt.savefig("Time vs Fraud (Fraud)")
plt.show()

#Non Fraud
plt.figure(figsize=(12,6))
sns.histplot(data=nonfraud_df, x="Time", hue="Class", bins=50, palette={0: "blue", 1: "red"}, alpha=0.6)
plt.title("Fraud vs Non-Fraud Transactions Over Time")
plt.xlabel("Time (seconds since first transaction)")
plt.ylabel("Count")
plt.savefig("Time vs Fraud (Non Fraud)")
plt.show()

# Correlation Heatmap
# Compute correlation matrix
corr = df.corr()

# Focus only on correlation with Class
class_corr = corr[['Class']].drop('Class')  # drop Class itself for clarity

# Plot heatmap
plt.figure(figsize=(6,10))
sns.heatmap(class_corr, annot=True, cmap="coolwarm", center=0)
plt.title("Feature Correlation with Class (Fraud)")
plt.xlabel("Correlation with Class")
plt.ylabel("Features")
plt.savefig("Correlation Heatmap")
plt.show()

# Pairplot
top_features = df.corr()['Class'].abs().sort_values(ascending=False).index[1:4]

sns.pairplot(df, vars=top_features, hue="Class", palette={0: "blue", 1: "red"}, height=2)
plt.savefig("Pairplot")
plt.show()
