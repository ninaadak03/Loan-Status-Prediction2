import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('loanapprovaldataset.csv')

# Encoding Technique : Label Encoding, One Hot Encoding

from sklearn.preprocessing import LabelEncoder
cols = [' education',' self_employed',' loan_status']
le =  LabelEncoder()
for col in cols:
  df[col] =  le.fit_transform(df[col])

#This is just to assign 1 to graduate and 0 to not graduate because it was done ulta in the encoding above
#Same done with loan status, assigning 1 to approved and 0 to rejected (self employed was fine, so didnt change)

mapping = {0: 1, 1: 0}
df[' education'] = df[' education'].replace(mapping)

mapping = {0: 1, 1: 0}
df[' loan_status'] = df[' loan_status'].replace(mapping)

df.head(10)

# Split Independent and dependent features

X = df.drop(columns = ['loan_id',' loan_status'],axis = 1)
y = df[' loan_status']

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.ensemble import RandomForestClassifier

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.3,random_state = 42)

# Initialize the RandomForestClassifier
RFmodel = RandomForestClassifier()

# Perform k-fold cross-validation (e.g., k=5)
k = 10
cv_scores = cross_val_score(RFmodel, X, y, cv=k)

# Print the cross-validated scores and mean accuracy
print(f'Cross-validated scores (k={k}): {cv_scores}')
print(f'Mean cross-validated accuracy: {cv_scores.mean() * 100:.2f}%')

# Train the RandomForestClassifier on the training data
RFmodel.fit(X_train, y_train)

# Predict on the test data
y_pred_RFmodel = RFmodel.predict(X_test)

# Calculate the accuracy
accuracy = accuracy_score(y_test, y_pred_RFmodel)
print(f'Accuracy score of Random Forest: {accuracy * 100:.2f}%')

# Assuming you have true labels (y_test) and predicted labels (y_pred)
y_true = y_test
y_pred = y_pred_RFmodel  # Replace with your predicted labels

# Calculate the confusion matrix
cm = confusion_matrix(y_true, y_pred)

# Print the confusion matrix
print("Confusion Matrix:\n", cm)

pickle.dump(RFmodel, open("rfmodel.pkl", "wb"))