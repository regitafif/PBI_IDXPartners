### ID/X Partners – Data Scientist

# Credit Risk Prediction Model

## 1. Data Understanding
- There are **466285 rows** and **75 columns** in the dataset.
- There are **53 numerical** columns and **22 categorical** columns in the dataset.
- There are **many null values** in the dataset.
- Some features **have no values** and filled with null values.
- Some features have **unique values as much as rows.**
- There is indication of **skewed data distribution** in some features because of the mean and median are very different, like in feature 'annual_inc', 'total_pymnt', etc.

## 2. Exploratory Data Analysis
Some insights from the analysis:
- Most features have outliers.
- Most features have skewed distribution.
- Loans with a 36-month term have a higher proportion of good credit (1.0) compared to 60-month term loans.
- Higher grades (A, B) are associated with a greater proportion of good credit, while lower grades (F, G) have more bad credit loans.
- The proportion of good credit increases with longer employment lengths (10+ years).
- Customer who have ownership of their home or have mortgages show higher proportion of good credit.
- Loan purposes like debt consolidation and credit card refinancing, have a relatively higher proportion of bad credit loans.

## 3. Data Preprocessing
1. **Handling missing values**
	- Drop features with only null values
	- Drop features with only unique values
	- Drop features that only have one unique value
	- Drop features with more than 50% unique values
2. **Removing outliers**
3. **Standardization**
4. **Feature Encoding**
	- Label Encoding
	- One hot encoding
5. **Test & train split** (70:30)

## 4. Modeling
![image](https://github.com/user-attachments/assets/f98a1485-17c9-434f-9674-00b8e25f44b1)
- XGBoost Classifier model is chosen with the highest ROC-AUC score of 0.9998.
- XGBoost Classifier has better result on classifying false positive and false negative than Random Forest.
- ROC-AUC is choosen as the metric because it is robust to imbalance data.

## 5. Evaluation
1. **Confusion matrix**<br>
   ![image](https://github.com/user-attachments/assets/ae6d2f85-e117-4b52-b0a2-0a6f61534a5e)
  - `True Negatives (TN)`: The model correctly predicted 48,253 instances of class 0.
  - `False Positives (FP)`: The model incorrectly predicted 1 instance as class 1 when it was actually class 0.
  - `False Negatives (FN)`: The model missed 55 instances of class 1, predicting them as class 0.
  - `True Positives (TP)`: The model correctly predicted 10,692 instances of class 1.

2. **Feature importance**<br>
   ![image](https://github.com/user-attachments/assets/7ab859df-4140-41a5-9fa5-4a51337b0f58) <br>
  The top 5 feature importance are:
  - `last_pymnt_amnt`: the amount of the last payment made is a key factor in predicting the target variable.
  - `repayment_months`: longer repayment periods could influence the risk associated with the loan or credit.
  - `recoveries`: the amount recovered from the loan or credit could be indicative of the financial health or recovery capacity of the borrower.
  - `out_prncp`: higher outstanding principal amounts could imply higher risk or a greater potential loss.
  - `total_rec_prncp`: the total amount of principal that has been recovered may reflect the borrower’s payment history and recovery trend.

