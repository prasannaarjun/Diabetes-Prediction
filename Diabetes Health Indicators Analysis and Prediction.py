# Import necessary libraries
import matplotlib.pyplot as plt
import pickle
import numpy as np
import pandas as pd
import seaborn
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

#making sure models folder exists
if not os.path.exists('models'):
    os.makedirs('models')

#Load the datasets
df_012 = pd.read_csv('datasets/diabetes_012_health_indicators_BRFSS2015.csv')
df_binary = pd.read_csv('datasets/diabetes_binary_5050split_health_indicators_BRFSS2015.csv')
df_5050 = pd.read_csv('datasets/diabetes_binary_health_indicators_BRFSS2015.csv')

# Display the first few rows of each dataset
df_012.head(),df_012.info(), df_binary.head(),df_012.describe()
df_binary.head(),df_binary.describe(),df_binary.info()
df_5050.head(),df_5050.info(), df_5050.describe()

# Correlation heatmap for df_012
numeric_df_012 = df_012.select_dtypes(include=[np.number])
plt.figure(figsize=(12, 8))
seaborn.heatmap(numeric_df_012.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap for df_012')
plt.show()

# Correlation heatmap for df_binary
numeric_df_binary = df_binary.select_dtypes(include=[np.number])
plt.figure(figsize=(12, 8))
seaborn.heatmap(numeric_df_binary.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap for df_binary')
plt.show()

# Correlation heatmap for df_5050
numeric_df_5050 = df_5050.select_dtypes(include=[np.number])
plt.figure(figsize=(12, 8))
seaborn.heatmap(numeric_df_5050.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap for df_5050')
plt.show()

#Removing the unnecessary columns after EDA
data1=df_binary.drop(columns=['CholCheck','Fruits', 'Veggies','NoDocbcCost','MentHlth','CholCheck','Smoker'])
data1.head()
# Prepare the data for modeling
X = data1.drop('Diabetes_binary', axis=1)
y = data1['Diabetes_binary']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build a Random Forest Classifier
model_rf = RandomForestClassifier(n_estimators=100, random_state=42)
model_rf.fit(X_train, y_train)


#saving the rf model as pickle file
with open('models/random_forest.pkl', 'wb') as f:
  pickle.dump(model_rf, f)
  print('RF Model saved')

# Make predictions
y_pred = model_rf.predict(X_test)

# Evaluate the Random Forest Model
accuracy_rf = accuracy_score(y_test, y_pred)
conf_matrix_rf = confusion_matrix(y_test, y_pred)
conf_matrix_rf_df = pd.DataFrame(conf_matrix_rf, index=['Actual 0 (No Diabetes)', 'Actual 1 (Diabetes)'],columns=['Predicted 0 (No Diabetes)', 'Predicted 1 (Diabetes)'])
class_report_rf = classification_report(y_test, y_pred)

print(f'Accuracy score of Random Forest Classifier: {accuracy_rf:.4f}')

print('Confusion matrix:\n',conf_matrix_rf_df)

print('Classification report:\n',class_report_rf)

# Build a XGB Classifier
model_XGB = XGBClassifier()
model_XGB.fit(X_train, y_train)

#save the XGB model as a pickle file
with open('models/xgb_classifier.pkl', 'wb') as f:
  pickle.dump(model_XGB, f)
  print('XGB Model saved')

# Make predictions
xgb_pred = model_XGB.predict(X_test)


# Evaluate the XGBClassifier
accuracy_XGB = accuracy_score(y_test, y_pred)
conf_matrix_XGB = confusion_matrix(y_test, y_pred)
conf_matrix_XGB_df = pd.DataFrame(conf_matrix_XGB, index=['Actual 0 (No Diabetes)', 'Actual 1 (Diabetes)'],columns=['Predicted 0 (No Diabetes)', 'Predicted 1 (Diabetes)'])
class_report_XGB = classification_report(y_test, y_pred)

print(f'Accuracy score of XGBClassifier : {accuracy_XGB:.4f}')

#Printing the confusion matrix of XGB
print(conf_matrix_XGB_df)

#Printing the class report of XGB
print(class_report_XGB)
