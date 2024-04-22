#importing all packages
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics  import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

#reading csv file
df=pd.read_csv("train_u6lujuX_CVtuZ9i.csv")

#seeing the top values of the data set
sns.set_style('darkgrid')
df.head()

#seeing the  tail values of the data set
df.tail()

#seeing the size of the data set
df.shape

#there are 614 rows and 13 columns

# seeing the size of the data set
df.size

#there are 7982 values in the data set

#seeing the data types of the data set
df.dtypes

#seeing the info of the data set
df.info()

#count of the null values in the data set according to column
df.isna().sum()

df['Gender'] = df["Gender"].fillna(df['Gender'].mode()[0])
df['Married'] = df["Married"].fillna(df['Married'].mode()[0])
df['Self_Employed'] = df["Self_Employed"].fillna(df['Self_Employed'].mode()[0])
df['Dependents'] = df["Dependents"].fillna(df['Dependents'].mode()[0])
df['LoanAmount'] = df["LoanAmount"].fillna(df['LoanAmount'].mode()[0])
df['Loan_Amount_Term'] = df["Loan_Amount_Term"].fillna(df['Loan_Amount_Term'].mode()[0])
df['Credit_History'] = df["Credit_History"].fillna(df['Credit_History'].mode()[0])

null_cols = ['Credit_History', 'Self_Employed', 'LoanAmount','Dependents', 'Loan_Amount_Term', 'Gender', 'Married']
for col in null_cols:
    print(f"\n{col}:\n{df[col].value_counts()}\n","-"*50)

sns.countplot(x ='Dependents', data = df)

sns.countplot(x ='Credit_History', data = df)

sns.boxplot(x="LoanAmount", data=df)

Q1 = df['LoanAmount'].quantile(0.25)
Q3 = df['LoanAmount'].quantile(0.75)
IQR = Q3 - Q1

low_lim = Q1 - 1.5 * IQR
up_lim = Q3 + 1.5 * IQR
print('low_limit is', low_lim)
print('up_limit is', up_lim)

df.drop('Loan_ID',axis=1,inplace=True)

df.head()



from sklearn.preprocessing import LabelEncoder
cols=['Gender', 'Married', 'Education', 'Self_Employed', 'Property_Area', 'Loan_Status', 'Dependents']
le=LabelEncoder()
for col in cols:
    df[col]=le.fit_transform(df[col])

df.head()

X = df.drop(columns=['Loan_Status'], axis=1)
y = df['Loan_Status']

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

sc = StandardScaler()

X_train_scaled = sc.fit_transform(x_train)
X_test_scaled = sc.fit_transform(x_test)

from sklearn.metrics import accuracy_score, confusion_matrix, f1_score,classification_report, roc_curve

from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier(criterion= 'gini',max_depth=11 , random_state=42)
dt.fit(X_train_scaled,y_train)
dt_pred = dt.predict(X_test_scaled)
print(classification_report(y_test, dt_pred))
DT_SC = accuracy_score(dt_pred,y_test)
print('Accuracy_Score of Decision Tree: ', accuracy_score(y_test, dt_pred))
matrix=confusion_matrix(y_test, dt_pred)

plt.figure(figsize = (8,4))
sns.heatmap(matrix , annot = True, cmap="YlOrBr")

#logistic regression

from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()
logreg.fit(X_train_scaled, y_train)
logreg_pred = logreg.predict(X_test_scaled)
print(classification_report(y_test, logreg_pred))
LR_SC = accuracy_score(logreg_pred,y_test)
print('Accuracy Score of Logistic Regression: ', accuracy_score(y_test, logreg_pred))
matrix2=confusion_matrix(y_test, logreg_pred)

plt.figure(figsize = (8,4))
sns.heatmap(matrix2 , annot = True, cmap="rocket")

from sklearn.ensemble import RandomForestClassifier

RF = RandomForestClassifier()
RF.fit(X_train_scaled, y_train)
RF_pred = RF.predict(X_test_scaled)
print(classification_report(y_test, RF_pred))
RF_SC = accuracy_score(RF_pred,y_test)
print('Accuracy Score of Random Forest Classifier: ', accuracy_score(y_test, RF_pred))
matrix4=confusion_matrix(y_test, RF_pred)

plt.figure(figsize = (8,4))
sns.heatmap(matrix4 , annot = True, cmap="magma")

score = [DT_SC,RF_SC,LR_SC ]
Models = pd.DataFrame({
    'Models': ["Decision Tree","Random Forest", "Logistic Regression" ],
    'Score': score})
Models.sort_values(by='Score', ascending=False)
