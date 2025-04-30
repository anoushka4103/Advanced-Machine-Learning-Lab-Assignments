# %% [markdown]
# #### 1. Gaussian NB

# %%
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix

# %%
data = pd.read_csv('shopping_data.csv')
data.head()

# %%
data.isnull().sum()

# %%
data_clean = data.drop(['CustomerID'], axis=1)
data_clean.head()

# %% [markdown]
# No missing values so we can proceed with label encoding correlation matrix

# %%
data_clean['Genre'] = data_clean['Genre'].map({'Female': 0, 'Male': 1})

# %%
num_cols = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)', 'Genre'] 

plt.figure(figsize=(8,6))
sns.heatmap(data_clean[num_cols].corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Heatmap (Only Numerical Features)")
plt.show()


# %%
X = data[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']]
y = data['Genre'] 

# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# %%
gnb = GaussianNB()
gnb.fit(X_train_scaled, y_train)

# %%
y_pred = gnb.predict(X_test_scaled)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# %% [markdown]
# #### 2. Bernoulli's NB

# %%
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

# %%
breast_data = pd.read_csv('breastcancer.csv')
breast_data.head()

# %%
breast_data.drop(columns=['id'], inplace=True)
breast_data['diagnosis'] = breast_data['diagnosis'].map({'M': 0, 'B': 1})
X = breast_data.drop(columns='diagnosis')
y = breast_data['diagnosis']

# %%
# binarize the features
X_bin = X.apply(lambda col: (col > col.mean()).astype(int))

# %%
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_bin, y, test_size=0.2, random_state=42)

# %%
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import accuracy_score, classification_report

bnb = BernoulliNB()
bnb.fit(X_train, y_train)

y_pred = bnb.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# %% [markdown]
# #### 3. Multinational NB

# %%
spam_data = pd.read_csv('spam.csv', encoding='latin1')
spam_data.head()

# %%
spam_data = spam_data.drop(columns=['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'])
spam_data.head()

# %%
X = spam_data['v2']
y = spam_data['v1']

# %%
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# %%
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# %%
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix

mnb = MultinomialNB()
mnb.fit(X_train_vec, y_train)

y_pred = mnb.predict(X_test_vec)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))



