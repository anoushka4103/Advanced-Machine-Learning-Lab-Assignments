# %% [markdown]
# ### Decision Tree Classification

# %%
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# %%
data = pd.read_csv('drug.csv')
data.head()

# %%
data.isnull().sum()

# %% [markdown]
# No missing values so we can continue with splitting

# %%
label_encoder = {}
for column in ['Sex', 'BP', 'Cholesterol', 'Drug']:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    label_encoder[column] = le

# %%
X = data.drop('Drug', axis=1)
y = data['Drug']

# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# %%
clf =  DecisionTreeClassifier(criterion='entropy', random_state=42)
clf.fit(X_train, y_train)

# %%
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# %%
plt.figure(figsize=(12, 8))
plot_tree(clf, feature_names=X.columns, class_names=label_encoder['Drug'].classes_, filled=True)
plt.show()
