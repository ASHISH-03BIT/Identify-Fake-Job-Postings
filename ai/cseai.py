import pandas as pd
import numpy as np


import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.cluster import KMeans

# Load dataset
df = pd.read_csv("C:\\Users\\Ashish\\Desktop\\ai\\fake_jobs.csv")

# Convert 'is_fake' column to numeric (yes = 1, no = 0)
label_encoder = LabelEncoder()
df["is_fake"] = label_encoder.fit_transform(df["is_fake"])

# Feature selection
X = df[["title_length", "description_length", "has_company_profile"]]
y = df["is_fake"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize numerical features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train classification model (Random Forest)
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# Calculate evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")

# Generate confusion matrix heatmap
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix Heatmap")
plt.show()

# Perform clustering using K-Means (segmentation)
kmeans = KMeans(n_clusters=2, random_state=42)
df["cluster"] = kmeans.fit_predict(X)

# Visualize clusters
plt.figure(figsize=(6, 4))
sns.scatterplot(x=df["title_length"], y=df["description_length"], hue=df["cluster"], palette="Set1")
plt.xlabel("Title Length")
plt.ylabel("Description Length")
plt.title("Job Posting Clustering")
plt.show()