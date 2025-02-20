# Import necessary libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

#Load the dataset
df_012 = pd.read_csv('datasets/diabetes_012_health_indicators_BRFSS2015.csv')

# Display the first few rows of dataset
df_012.head()
df_012.info()
df_012.describe()

# Correlation heatmap for df_012
numeric_df_012 = df_012.select_dtypes(include=[np.number])
plt.figure(figsize=(12, 8))
seaborn.heatmap(numeric_df_012.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap for df_012')
plt.show()

data1=df_012.drop(columns=['CholCheck','Fruits', 'Veggies','NoDocbcCost','MentHlth','CholCheck','Smoker'])
print(data1.head())
# Prepare the data for modeling
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
conf_matrix_rf_df = pd.DataFrame(conf_matrix_rf, index=['Actual 0 (No Diabetes)', 'Actual 1 (PreDiabetes)','Actual 2(Diabetes)'],columns=['Predicted 0 (No Diabetes)', 'Predicted 1 (PreDiabetes)','Predicted 2(Diabetes)'])
class_report_rf = classification_report(y_test, y_pred)

print(f'Accuracy score of Random Forest Classifier: {accuracy_rf:.4f}')

print('Confusion matrix:\n',conf_matrix_rf_df)

print('Classification report:\n',class_report_rf)



# Plot the confusion matrix as a heatmap
plt.figure(figsize=(12, 12))
seaborn.heatmap(conf_matrix_rf_df, annot=True, fmt='d', cmap='Blues', linewidths=0.5)

# Set plot labels and title
plt.xlabel("Predicted Labels")
plt.ylabel("Actual Labels")
plt.title("Confusion Matrix - Random Forest")

# Show the plot
plt.show()
