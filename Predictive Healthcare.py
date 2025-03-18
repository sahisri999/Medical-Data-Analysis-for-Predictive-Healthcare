#!/usr/bin/env python
# coding: utf-8

# In[3]:


get_ipython().system('pip install pandas')
get_ipython().system('pip install numpy')
get_ipython().system('pip install matplotlib')
get_ipython().system('pip install seaborn')
get_ipython().system('pip install scikit-learn')


# In[30]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# In[8]:


dbt = pd.read_csv("C:/Users/user/OneDrive/Documents/diabetes.csv")


# In[10]:


df = pd.DataFrame(dbt)


# In[11]:


df.head()


# In[14]:


df.isnull().sum()


# In[12]:


df.info()


# In[13]:


df.describe()


# In[20]:


X = df.drop(columns=['Outcome'])
y = df['Outcome']

X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# In[26]:


log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)

y_pred_log = log_reg.predict(X_test)
print("\nLogistic Regression Accuracy:", accuracy_score(y_test, y_pred_log))
print("\Classification Report (Logistic Regression): \n", classification_report(y_test, y_pred_log))


# In[27]:


rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train,y_train)

y_pred_rf = rf_model.predict(X_test)
print("\nRandom Forest Accuracy:", accuracy_score(y_test,y_pred_rf))
print("\nClassification Report (Random Forest):\n", classification_report(y_test,y_pred_rf))


# In[31]:


# visualizing results

plt.figure(figsize=(6,4))
sns.heatmap(confusion_matrix(y_test, y_pred_rf), annot=True, fmt='d', cmap='Blues')
plt.xlabel('Prdeicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix - Random Forest')
plt.show()


# In[33]:


feature_importance = rf_model.feature_importances_
features = df.drop(columns=['Outcome']).columns


plt.figure(figsize=(8, 5))
sns.barplot(x=feature_importance, y=features)
plt.xlabel("Feature Importance Score")
plt.ylabel("Features")
plt.title("Feature Importance in Predicting Diabetes")
plt.show()


# In[35]:


import joblib

joblib.dump(log_reg, 'logistic_regression_diabetes.pk1')
joblib.dump(rf_model, 'random_forest_diabetes.pk1')
joblib.dump(scaler, 'scaler_diabetes.pk1')

print("Models Saved Successfully")


# In[ ]:




