```
pip install imbalanced-learn
```
### Importing necessary libraries

```
import numpy as np
import pandas as pd
import sklearn
import scipy
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report,accuracy_score
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from pylab import rcParams
```
```
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix,accuracy_score,precision_score
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
```

### Load the dataset
```
df = pd.read_csv('credit card.csv',sep=',')
df.head()
```

### Exploratory Data Analysis

```
df.info()
```
```
rcParams['figure.figsize'] = 14, 8
RANDOM_SEED = 42
LABELS = ["Normal", "Fraud"]
```
```
df.isnull()
```
```
df.isnull().values.any()
```
```
count_class = pd.value_counts(df['Class'], sort = True)

count_class.plot(kind = 'bar', rot=0)

plt.title("Transaction Class Distribution")

plt.xticks(range(2), LABELS)

plt.xlabel("Class")

plt.ylabel("Frequency")
``` 
```
#Get the Fraud and the normal dataset
fraud = df[df['Class']==1]
normal = df[df['Class']==0]
```

```
print(fraud.shape,normal.shape)
```

```
fraud.Amount.describe()
```
```
normal.Amount.describe()
```
```
f, (axis1, axis2) = plt.subplots(2, 1, sharex=True)
f.suptitle('Amount per transaction by class')
bins = 50
axis1.hist(fraud.Amount, bins = bins)
axis1.set_title('Fraud')
axis2.hist(normal.Amount, bins = bins)
axis2.set_title('Normal')
plt.xlabel('Amount ($)')
plt.ylabel('Number of Transactions')
plt.xlim((0, 20000))
plt.yscale('log')
plt.show()
```
```
f, (axis1, axis2) = plt.subplots(2, 1, sharex=True)
f.suptitle('Time of transaction vs Amount by class')
axis1.scatter(fraud.Time, fraud.Amount)
axis1.set_title('Fraud')
axis2.scatter(normal.Time, normal.Amount)
axis2.set_title('Normal')
plt.xlabel('Time (in Seconds)')
plt.ylabel('Amount')
plt.show()
```
### Correlation
```
corrmatrix = df.corr()
top_corr_features = corrmatrix.index
plt.figure(figsize=(20,20))
#plot heat map
g=sns.heatmap(df[top_corr_features].corr(),annot=True,cmap="RdYlGn")
```
### Data Preprocessing
```
X = df.drop('Class', axis=1)
y = df['Class']
```
```
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```
```
nan_indices = y_train.index[y_train.isnull()]
X_train_cleaned = X_train.drop(index=nan_indices)
y_train_cleaned = y_train.drop(index=nan_indices)
```
### Balancing the Dataset
```
print("Class distribution before SMOTE:")
print(y_train.value_counts())
```
```
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train_cleaned, y_train_cleaned)
```
```
print("\nClass distribution after SMOTE:")
print(pd.Series(y_train_balanced).value_counts())
```
### Training model using logistic regression
```
logreg = LogisticRegression(max_iter=1000, random_state=42)
logreg.fit(X_train_balanced, y_train_balanced)
```
```
y_pred = logreg.predict(X_test)
```
```
logistic_accuracy = accuracy_score(y_test, y_pred)
print("Logistic Regression Accuracy:", logistic_accuracy)
```
```
logistic_precision = precision_score(y_test, y_pred)
print("Logistic Regression Precision:", logistic_precision)
```
```
#Cofusion Matrix
cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
cnf_matrix

logistic_conf_matrix = confusion_matrix(y_test, y_pred)
print("Logistic Regression Confusion Matrix:")
print(logistic_conf_matrix)

%matplotlib inline
class_names=[0,1] # name  of classes
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
# create heatmap
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
```
#### Classification report of Log_regression
```
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
```














