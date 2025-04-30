# %%
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import svm
from sklearn.model_selection import train_test_split

# %%
data = pd.read_csv('breastcancer.csv')
data.tail(10)

# %% [markdown]
# ### Cleaning the Dataset

# %%
data.dropna()
data.isnull().sum()

# %% [markdown]
# ### Making the correlation heatmap

# %%
data_clean = data.drop(['id','Unnamed: 32'], axis=1) # drop unnecessary columns
data_clean['diagnosis'] = data_clean['diagnosis'].map({'B': 0, 'M': 1})
corr_matrix = data_clean.corr()
plt.figure(figsize=(20, 16))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Heatmap')
plt.show()

# %% [markdown]
# Conclusion: The features that we will be using for diagnosis are:
# 
# * radius_mean - 0.73
# * concavity_mean - 0.7
# * texture_mean - 0.42
# * compactness_mean - 0.6
# * smoothness_mean - 0.36
# * symmetry_mean - 0.33

# %%
x = data[["radius_mean","texture_mean","smoothness_mean","compactness_mean","concavity_mean", "symmetry_mean"]]
y = data["diagnosis"]

# %% [markdown]
# ### Splitting for testing and training

# %%
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# %%
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(x_train)
X_test_scaled = scaler.transform(x_test)

# %%
from sklearn.svm import SVC

model = SVC(kernel='linear')
model.fit(X_train_scaled, y_train)

# %%
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

y_pred = model.predict(X_test_scaled)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# %% [markdown]
# ## we get a 95% accuracy

# %% [markdown]
# ### Plotting the hyperplane

# %%
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC

#Feature selection
features = ['radius_mean', 'concavity_mean', 'texture_mean', 'compactness_mean', 'smoothness_mean', 'symmetry_mean']
X = data[features]
y = data['diagnosis'].map({'B': 0, 'M': 1})  # Encode labels

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#PCA for 2D visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Train Linear SVM
clf = SVC(kernel='linear')
clf.fit(X_pca, y)

# Plot
plt.figure(figsize=(10, 7))

# Plot decision regions
ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()

# Create grid to evaluate model
xx = np.linspace(X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1, 500)
yy = np.linspace(X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1, 500)
YY, XX = np.meshgrid(yy, xx)
xy = np.vstack([XX.ravel(), YY.ravel()]).T
Z = clf.decision_function(xy).reshape(XX.shape)

# Plot decision boundary and margins
contour = plt.contour(XX, YY, Z, colors='k',
                      levels=[-1, 0, 1], alpha=0.7,
                      linestyles=['--', '-', '--'])

# Plot support vectors
plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1],
            s=120, facecolors='none', edgecolors='k', label='Support Vectors')

# Plot the data points
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=y, palette='coolwarm', edgecolor='k')
plt.title("Linear SVM with Hyperplane, Margins & Support Vectors (2D PCA)")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.legend(title='Diagnosis', labels=['Benign', 'Malignant'])
plt.show()

# %% [markdown]
# ### Non Linear SVM using RBF kernel

# %%
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

#Select features & label
features = ['radius_mean', 'concavity_mean', 'texture_mean', 'compactness_mean', 'smoothness_mean', 'symmetry_mean']
X = data[features]
y = data['diagnosis'].map({'B': 0, 'M': 1})

#Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#Train non-linear SVM with RBF kernel
clf_rbf = SVC(kernel='rbf', C=1, gamma='scale')
clf_rbf.fit(X_train_scaled, y_train)

#Evaluate
y_pred = clf_rbf.predict(X_test_scaled)

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# %% [markdown]
# ## We get 97% accuracy 


