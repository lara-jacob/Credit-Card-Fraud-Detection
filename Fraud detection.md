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
```

### Load the dataset
```
data = pd.read_csv('credit card.csv',sep=',')
data.head()
```

## Exploratory Data Analysis

```
data.info()
```
```
rcParams['figure.figsize'] = 14, 8
RANDOM_SEED = 42
LABELS = ["Normal", "Fraud"]
```
```
data.isnull()
```
```
data.isnull().values.any()
```
```
count_class = pd.value_counts(data['Class'], sort = True)

count_class.plot(kind = 'bar', rot=0)

plt.title("Transaction Class Distribution")

plt.xticks(range(2), LABELS)

plt.xlabel("Class")

plt.ylabel("Frequency")
``` 
```
#Get the Fraud and the normal dataset
fraud = data[data['Class']==1]
normal = data[data['Class']==0]
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
import seaborn as sns
corrmatrix = data.corr()
top_corr_features = corrmatrix.index
plt.figure(figsize=(20,20))
g=sns.heatmap(data[top_corr_features].corr(),annot=True,cmap="RdYlGn")
```










