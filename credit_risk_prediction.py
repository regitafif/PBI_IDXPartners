#!/usr/bin/env python
# coding: utf-8

# # Project-Based Internship : ID/X Partners x Rakamin

# # Credit Risk Prediction Model

# #### Import Libraries

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

import warnings
warnings.filterwarnings('ignore')


# ## Data Understanding
# ---

# In[2]:


df = pd.read_csv('loan_data_2007_2014.csv')
df.head()


# In[3]:


df.shape


# In[4]:


df.info()


# In[5]:


nums = df.select_dtypes(include=['int64', 'float64']).columns
cats = df.select_dtypes(include=['object']).columns


# In[6]:


df[nums].describe()


# In[7]:


df[cats].describe()


# **Data Understanding:**
# 
# - There are **466285 rows** and **75 columns** in the dataset.
# - There are **53 numerical** columns and **22 categorical** columns in the dataset.
# - There are **many null values** in the dataset.
# - Some features have no values and filled with null values.
# - Some features have unique values as much as rows.
# - There is indication of skewed data distribution in some features because of the mean and median are very different, like in feature 'annual_inc', 'total_pymnt', and 'total_pymnt_inv', etc.

# ### Handling Missing Values

# Before EDA, some of the missing values needs to be handled to reduce the number of features. So, the **features that has no values** and the **features that only have unique values** needs to be dropped.

# In[8]:


df1 = df.copy()


# **Dropping features with no values**

# In[9]:


null_only = df1.columns[df1.isnull().all()]
null_only


# In[10]:


df1.drop(columns=null_only, axis=1, inplace=True)


# **Recheck null values**

# In[11]:


df1.isnull().mean().sort_values(ascending=False)


# There are still some features that have almost or more than 50% of missing values, so it's better for these features to be dropped.

# In[12]:


df1.drop(['desc','mths_since_last_delinq','mths_since_last_record','mths_since_last_major_derog','next_pymnt_d'], axis=1, inplace=True)


# **Dropping features with only unique values**

# In[13]:


unique_only = [col for col in df1.columns if df1[col].is_unique]
unique_only


# In[14]:


df1.drop(columns=unique_only, axis=1, inplace=True)


# **Dropping features with unique values**

# In[15]:


nums1 = df1.select_dtypes(include=['int64', 'float64']).columns
cats1 = df1.select_dtypes(include=['object']).columns


# In[16]:


# Categorical Features
df1[cats1].nunique()


# In[17]:


# Numerical Features
df1[nums1].nunique()


# Dropped because too many unique values:
# - emp_title
# - title
# - zip_code
# - earliest_cr_line
# - last_credit_pull_d
# 
# Dropped because only have one unique value:
# - application_type
# - policy_code

# In[18]:


df1.drop(['emp_title',
    'title','zip_code',
    'earliest_cr_line',
    'last_credit_pull_d',
    'application_type',
    'policy_code'], axis=1, inplace=True)


# ## Exploratory Data Analysis (EDA)

# **Defining target variable**

# In[19]:


df1['loan_status'].value_counts()


# The values in 'loan_status' would be the target variable, so encoding is needed to convert the values into binary values of 0 for bad credit and 1 for good credit. 
# The values would be classified as such:
# - Current: good credit (1)
# - Fully Paid: good credit (1)
# - Charged Off: bad credit (0)
# - Late (31-120 days): bad credit (0)
# - In Grace Period: bad credit (0)
# - Does not meet the credit policy. Status:Fully Paid: good credit (1)
# - Late (16-30 days): bad credit (0)
# - Default: bad credit (0)
# - Does not meet the credit policy. Status:Charged Off: bad credit (0)

# In[20]:


# Mapping loan_status to bad credit (0) and good credit (1)
loan_status_map = {
    'Fully Paid': 0,
    'Does not meet the credit policy. Status:Fully Paid': 0,
    'Charged Off': 1,
    'Default': 1,
    'Late (31-120 days)': 1,
    'Late (16-30 days)': 1,
    'In Grace Period': 1,
    'Does not meet the credit policy. Status:Charged Off': 1
}

df1['loan_status'] = df1['loan_status'].map(loan_status_map)


# In[21]:


# Check the distribution of good and bad credit
df1['loan_status'].value_counts()


# **Data Distribution**

# In[22]:


df2 = df1.copy()


# In[23]:


nums2 = df2.select_dtypes(include=['int64', 'float64']).columns
cats2 = df2.select_dtypes(include=['object']).columns


# In[24]:


df2[nums2].describe()


# In[25]:


# Boxplots
plt.figure(figsize=(20, 25))
for i, var in enumerate(nums2):
    plt.subplot(5, 6, i + 1)
    sns.boxplot(y=df2[var])
    plt.title(f'{var}')
plt.tight_layout()
plt.show()


# - Most features have outliers.
# - Features that don't have outliers are:
#     - loan_amnt
#     - funded_amnt
#     - funded_amnt_inv
#     - dti

# In[26]:


# Histograms
plt.figure(figsize=(18, 25))
for i, var in enumerate(nums2):
    plt.subplot(6, 5, i + 1)
    sns.histplot(df2[var])
    plt.title(f'Distribution of {var}')
plt.tight_layout()
plt.show()


# - Most features have skewed distribution.
# - Features that have almost normal distribution are:
#     - loan_amnt
#     - funded_amnt
#     - funded_amnt_inv
#     - int_rate
#     - dti

# In[28]:


plt.figure(figsize=(15, 25))

# Loop through categorical features and create a subplot for each
for i, var in enumerate(cats2):
    plt.subplot(5, 3, i + 1)
    
    # Create a count plot with hue
    sns.countplot(x=var, data=df2, hue='loan_status', order=df2[var].value_counts().index)
    
    # Set title for each subplot
    plt.title(f'{var}')
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=90)

# Adjust layout to ensure no overlap
plt.tight_layout()
plt.show()


# - Loans with a 36-month term have a higher proportion of good credit (1.0) compared to 60-month term loans.
# - Higher grades (A, B) are associated with a greater proportion of good credit, while lower grades (F, G) have more bad credit loans.
# - The proportion of good credit increases with longer employment lengths (10+ years).
# - Customer who have ownership of their home or have mortgages show higher proportion of good credit.
# - Loan purposes like debt consolidation and credit card refinancing, have a relatively higher proportion of bad credit loans.

# In[29]:


plt.figure(figsize=(12, 8))
heatmap = sns.heatmap(df2[nums2].corr(), annot=True, cmap='coolwarm', fmt='.2f', annot_kws={"size": 7})
plt.title('Correlation Matrix')
plt.xticks(rotation=90, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()


# Some features that show high correlation are:
# - loan_amnt
# - funded_amnt
# - funded_amnt_inv
# - total_pymnt
# - total_pymnt_inv
# - recoveries
# - revol_bal
# - out_prncp
# - out_prncp_inv
# - total_rec_prncp
# - total_rec_int
# - last_pymnt_amnt

# ## Data Preprocessing

# **Handling missing values**

# In[30]:


df3 = df2.copy()


# In[31]:


df3.isnull().sum().sort_values(ascending=False)


# For the features `total_rev_hi_lim`,`tot_cur_bal`,`tot_coll_amt`,`emp_length`, imputation is needed since the are a lot of null values.
# - fill null values with mean for numerical features.
# - fill null values with mode for categorical features

# In[32]:


# For numerical columns, fill with mean
df3['total_rev_hi_lim'].fillna(df3['total_rev_hi_lim'].mean(), inplace=True)
df3['tot_cur_bal'].fillna(df3['tot_cur_bal'].mean(), inplace=True)
df3['tot_coll_amt'].fillna(df3['tot_coll_amt'].mean(), inplace=True)

# For categorical columns, fill with mode
df3['emp_length'].fillna(df3['emp_length'].mode()[0], inplace=True)


# Drop the rest of the null values because there are less than 1%.

# In[33]:


# Drop the rest of the null values
df3.dropna(inplace=True)


# In[34]:


df3.isnull().sum()


# In[35]:


df3.head()


# **Duplicated data**

# In[36]:


df3.duplicated().sum()


# There is no duplicated data

# **Removing Outliers**

# In[37]:


nums3 = df3.select_dtypes(include=['int64', 'float64']).columns
cats3 = df3.select_dtypes(include=['object']).columns


# In[38]:


# Ensure nums3 is a DataFrame
if isinstance(nums3, pd.Index):
    nums3 = df3[nums3]


# In[39]:


from scipy.stats import zscore

# Calculate Z-scores for each numerical column
z_scores = nums3.apply(zscore)

# Define a threshold for severe outliers
threshold = 3

# Find severe outliers in each column
severe_outliers = (abs(z_scores) > threshold).sum()

# Print columns with severe outliers directly
column_outliers = severe_outliers[severe_outliers > 0]
column_outliers


# In[40]:


# Define a function to remove outliers based on Z-scores
def remove_outliers(df3, threshold=3):
    z_scores = nums3.apply(zscore)
    abs_z_scores = abs(z_scores)
    filtered_entries = (abs_z_scores < threshold).all(axis=1)
    return df3[filtered_entries]

# Apply the function
df4 = remove_outliers(df3)


# In[41]:


nums4 = df4.select_dtypes(include=['int64', 'float64']).columns
cats4 = df4.select_dtypes(include=['object']).columns


# In[42]:


plt.figure(figsize=(18, 25))
for i, var in enumerate(nums4):
    plt.subplot(6, 5, i + 1)
    sns.histplot(df4[var])
    plt.title(f'Distribution of {var}')
plt.tight_layout()
plt.show()


# The distribution of the data is better after the outliers are removed.

# **Standardization**

# In[43]:


dfs = df4.copy()


# In[44]:


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
dfs[nums4] = scaler.fit_transform(dfs[nums4])


# In[45]:


dfs.describe()


# **Feature Encoding**

# Changing data types:
# - `issue_d`: to datetime
# - `last_pymnt_d`: to datetime
# 
# Make a new feature `repayment_months` to calculate the repayment duration in months that get extracted from 'issue_d' and 'last_pymnt_d', and then both features would later be dropped.

# In[46]:


dfs[cats4].describe()


# In[82]:


dfe = dfs.copy()


# In[83]:


dfe['issue_d'] = pd.to_datetime(dfe['issue_d'], format='%b-%y')
dfe['last_pymnt_d'] = pd.to_datetime(dfe['last_pymnt_d'], format='%b-%y')


# In[84]:


from dateutil.relativedelta import relativedelta

def calculate_months(start_date, end_date):
    delta = relativedelta(end_date, start_date)
    return delta.years * 12 + delta.months

dfe['repayment_months'] = dfe.apply(lambda row: calculate_months(row['issue_d'], row['last_pymnt_d']), axis=1)


# Label Encoding

# In[85]:


dfe['grade'].unique()


# In[86]:


# Mapping grade to numeric values
grade_map = {
    'A': 0,
    'B': 1,
    'C': 2,
    'D': 3,
    'E': 4,
    'F': 5,
    'G': 6,
}

dfe['grade'] = dfe['grade'].map(grade_map)


# In[87]:


dfe['emp_length'].unique()


# In[88]:


# Mapping emp_length to numeric values
emp_length_map = {
    '< 1 year': 0,
    '1 year': 1,
    '2 years': 2,
    '3 years': 3,
    '4 years': 4,
    '5 years': 5,
    '6 years': 6,
    '7 years': 7,
    '8 years': 8,
    '9 years': 9,
    '10+ years': 6,
}

dfe['emp_length'] = dfe['emp_length'].map(emp_length_map)


# In[89]:


dfe['verification_status'].unique()


# In[90]:


# Mapping verification_status to numeric values
verification_map = {
    'Verified': 1,
    'Source Verified': 1,
    'Not Verified': 0
}

dfe['verification_status'] = dfe['verification_status'].map(verification_map)


# In[91]:


dfe['pymnt_plan'].unique()


# In[92]:


# Mapping pymnt_plan to numeric values
pymnt_plan_map = {
    'y': 0,
    'n': 1
}

dfe['pymnt_plan'] = dfe['pymnt_plan'].map(pymnt_plan_map)


# In[93]:


dfe['home_ownership'].unique()


# In[94]:


# Mapping home_ownership to numeric values
home_ownership_map = {
    'OWN': 1,
    'MORTGAGE': 1,
    'RENT': 0,
    'OTHER': 0,
    'NONE': 0,
    'ANY': 0
}

dfe['home_ownership'] = dfe['home_ownership'].map(home_ownership_map)


# In[95]:


dfe['initial_list_status'].unique()


# In[96]:


# Mapping initial_list_status to numeric values
initial_list_status_map = {
    'w': 0,
    'f': 1
}

dfe['initial_list_status'] = dfe['initial_list_status'].map(initial_list_status_map)


# In[97]:


dfe['term'].unique()


# In[98]:


# Mapping term to numeric values
term_map = {
    ' 60 months': 0,
    ' 36 months': 1
}

dfe['term'] = dfe['term'].map(term_map)


# One hot Encoding

# In[99]:


dfe['purpose'].unique()


# In[100]:


from sklearn.preprocessing import OneHotEncoder

dfe = pd.get_dummies(dfe, columns=['purpose'], drop_first=True)


# In[101]:


# Mengubah isi True dan False menjadi 0 dan 1
for col in dfe.columns:
    if dfe[col].dtype == 'bool':
        dfe[col] = dfe[col].astype(int)


# In[102]:


dfe.drop(['sub_grade','addr_state','issue_d','last_pymnt_d'], axis=1, inplace=True)


# In[103]:


dfe.head()


# In[104]:


dfe.shape


# **Train test split**

# In[105]:


from sklearn.model_selection import train_test_split

X = dfe.drop('loan_status', axis=1)
y = dfe['loan_status']

# Split data train dan test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# In[106]:


threshold = 0.0
y_train = (y_train > 0).astype(int)
y_test = (y_test > 0).astype(int)


# In[107]:


y_train.value_counts()


# **Class imbalance**<br>
# Since the class imbalance is quite high, resampling is requires. SMOTE will be used for oversampling and make the '1' class to be 60% of the '0' class.
# - 0 : 112182
# - 1 : 60% x '0' class = 0.6 x 112182 = 67309

# In[108]:


from imblearn.over_sampling import SMOTE

# Set the sampling strategy to achieve the desired 60% ratio
smote = SMOTE(sampling_strategy=67309/112182, random_state=42)

X_over_SMOTE, y_over_SMOTE = smote.fit_resample(X_train, y_train)

# Check the new class distribution
y_over_SMOTE.value_counts()


# ## Modeling

# Metric that will be used for the modeling is ROC-AUC because it has better performance especially with imbalanced dataset.

# In[109]:


from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import xgboost as xgb

# Define the models to be evaluated
models = {
    'Logistic Regression': LogisticRegression(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42),
    'XGBoost': xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42)
}

# Loop through each model, fit, predict, and evaluate
for name, model in models.items():
    model.fit(X_over_SMOTE, y_over_SMOTE)
    
    y_pred = model.predict(X_test)
    y_pred_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else y_pred
    
    print(f"Results for {name}:")
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("ROC AUC Score:", roc_auc_score(y_test, y_pred_prob))
    print("="*50)


# **Hyperparameter Tuning**

# In[110]:


from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
import xgboost as xgb

# Define a smaller hyperparameter grid without `min_child_weight`
param_dist = {
    'classifier__learning_rate': [0.01, 0.1],
    'classifier__n_estimators': [100, 150],
    'classifier__max_depth': [3, 4],
    'classifier__subsample': [0.8, 1.0],
    'classifier__colsample_bytree': [0.8, 1.0],
}

# Define the model
xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)

# Create a pipeline
pipeline = Pipeline([
    ('classifier', xgb_model)
])

# Set up RandomizedSearchCV
random_search = RandomizedSearchCV(pipeline, param_distributions=param_dist, 
                                    n_iter=10, cv=3, scoring='roc_auc', n_jobs=-1, random_state=42)

# Fit RandomizedSearchCV
random_search.fit(X_over_SMOTE, y_over_SMOTE)

# Print best parameters and best score
print("Best Parameters:", random_search.best_params_)
print("Best Score:", random_search.best_score_)


# ## Evaluation

# In[111]:


from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report
import joblib

xgb_model = XGBClassifier()
xgb_model.fit(X_over_SMOTE, y_over_SMOTE)

# Make predictions
y_pred_prob = xgb_model.predict_proba(X_test)[:, 1]
y_pred = xgb_model.predict(X_test)

# Calculate metrics
roc_auc = roc_auc_score(y_test, y_pred_prob)
cm = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f'ROC AUC Score: {roc_auc}')
print('Confusion Matrix:')
print(cm)
print('Classification Report:')
print(report)

# Optionally, save the model
joblib.dump(xgb_model, 'xgb_model.pkl')


# The ROC AUC score is very close to 1, indicating that the model has an excellent ability to distinguish between classes. An AUC score of 0.9998 suggests near-perfect performance in distinguishing between class 0 and class 1.

# In[112]:


from sklearn.metrics import roc_curve
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)

import matplotlib.pyplot as plt
plt.figure()
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc='lower right')
plt.show()


# In[113]:


import seaborn as sns
import matplotlib.pyplot as plt

plt.figure()
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Negative', 'Positive'], 
            yticklabels=['Negative', 'Positive'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()


# - True Negatives (TN): The model correctly predicted 48,253 instances of class 0.
# - False Positives (FP): The model incorrectly predicted 1 instance as class 1 when it was actually class 0.
# - False Negatives (FN): The model missed 55 instances of class 1, predicting them as class 0.
# - True Positives (TP): The model correctly predicted 10,692 instances of class 1.

# In[114]:


# Get feature importance
importance = xgb_model.feature_importances_

# Create a DataFrame for better visualization
import pandas as pd
feature_names = X_test.columns
feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importance})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# Plot feature importance
plt.figure(figsize=(10, 8))
plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'])
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Feature Importance')
plt.gca().invert_yaxis()
plt.show()


# The top 5 feature importance are:
# - `last_pymnt_amnt`: the amount of the last payment made is a key factor in predicting the target variable.
# - `repayment_months`: longer repayment periods could influence the risk associated with the loan or credit.
# - `recoveries`: the amount recovered from the loan or credit could be indicative of the financial health or recovery capacity of the borrower.
# - `out_prncp`: higher outstanding principal amounts could imply higher risk or a greater potential loss.
# - `total_rec_prncp`: the total amount of principal that has been recovered may reflect the borrower’s payment history and recovery trend.
