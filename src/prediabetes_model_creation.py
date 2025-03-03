import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import os

# Load dataset
df = pd.read_csv('../datasets/diabetes_012_health_indicators_BRFSS2015.csv')

# Assuming the first column is the target variable
y = df.iloc[:, 0]
X = df.iloc[:, 1:]

# Apply SMOTE to balance the classes
smote = SMOTE(sampling_strategy='auto', random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Create a new DataFrame with resampled data
resampled_df = pd.DataFrame(X_resampled, columns=X.columns)
resampled_df['Diabetes_012'] = y_resampled

# Save the resampled dataset to CSV
resampled_df.to_csv('../datasets/resampled_dataset.csv', index=False)
print("Resampled dataset saved as 'resampled_dataset.csv'")

# Plot original class distribution
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
sns.countplot(x=y, palette='Set2')
plt.title('Original Class Distribution')
plt.xlabel('Target Class')
plt.ylabel('Count')

# Plot resampled class distribution
plt.subplot(1, 2, 2)
sns.countplot(x=y_resampled, palette='Set2')
plt.title('Resampled Class Distribution')
plt.xlabel('Target Class')
plt.ylabel('Count')

plt.tight_layout()
plt.show()

# Load resampled dataset
data1 = pd.read_csv('../datasets/resampled_dataset.csv')

# Drop unnecessary columns
data_modeling = data1.drop(columns=['CholCheck', 'Fruits', 'Veggies', 'NoDocbcCost', 'MentHlth', 'CholCheck', 'Smoker'])
print(data1.head())

# Prepare features and target variable
X = data1.drop('Diabetes_012', axis=1)
y = data1['Diabetes_012']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Build a Random Forest Classifier
model_rf = RandomForestClassifier(n_estimators=100, random_state=42)
model_rf.fit(X_train, y_train)

# Make predictions
y_pred = model_rf.predict(X_test)

# Evaluate the Random Forest Model
accuracy_rf = accuracy_score(y_test, y_pred)
conf_matrix_rf = confusion_matrix(y_test, y_pred)
conf_matrix_rf_df = pd.DataFrame(conf_matrix_rf, index=['Actual 0 (No Diabetes)', 'Actual 1 (PreDiabetes)', 'Actual 2 (Diabetes)'], columns=['Predicted 0 (No Diabetes)', 'Predicted 1 (PreDiabetes)', 'Predicted 2 (Diabetes)'])
class_report_rf = classification_report(y_test, y_pred)

print(f'Accuracy score of Random Forest Classifier: {accuracy_rf:.4f}')
print('Confusion matrix:\n', conf_matrix_rf_df)
print('Classification report:\n', class_report_rf)

# Plot the confusion matrix as a heatmap
plt.figure(figsize=(12, 12))
sns.heatmap(conf_matrix_rf_df, annot=True, fmt='d', cmap='Blues', linewidths=0.5)

# Set plot labels and title
plt.xlabel("Predicted Labels")
plt.ylabel("Actual Labels")
plt.title("Confusion Matrix - Random Forest")

# Show the plot
plt.show()

# Save the trained model using pickle
models_dir = "models"
os.makedirs(models_dir, exist_ok=True)  # Create 'models' directory if it doesn't exist

model_path = os.path.join(models_dir, r"../models/random_forest_prediabetes.pkl")
with open(model_path, "wb") as model_file:
    pickle.dump(model_rf, model_file)

print(f"Model saved to {model_path}")
