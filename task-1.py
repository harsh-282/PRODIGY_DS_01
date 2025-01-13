# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Step 1: Load the dataset
# Replace 'path_to_file' with the actual path to the CSV file
file_path = "iris.csv"  # Ensure the dataset is saved as iris.csv in the same directory as this script
df = pd.read_csv('F:\HARSHINI\Prodigy Infotech\iris.csv'
, header=None, names=["SepalLength", "SepalWidth", "PetalLength", "PetalWidth", "Species"])

# Step 2: Display dataset information
print("Dataset Head:")
print(df.head())
print("\nDataset Description:")
print(df.describe())
print("\nDataset Info:")
print(df.info())

# Step 3: Visualize the data
sns.pairplot(df, hue="Species")
plt.show()

# Correlation heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()

# Step 4: Prepare the data for training
X = df.iloc[:, :-1]  # Features: SepalLength, SepalWidth, PetalLength, PetalWidth
y = df["Species"]    # Target: Species

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 5: Train a Random Forest Classifier
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Step 6: Make predictions and evaluate the model
y_pred = model.predict(X_test)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Feature importance visualization
importances = model.feature_importances_
features = X.columns

plt.figure(figsize=(8, 6))
sns.barplot(x=importances, y=features, palette="viridis")
plt.title("Feature Importance")
plt.xlabel("Importance Score")
plt.ylabel("Feature")
plt.show()