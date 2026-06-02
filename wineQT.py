import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap

from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.model_selection import (
    train_test_split, StratifiedKFold, KFold,
    cross_val_score, GridSearchCV
)
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    ConfusionMatrixDisplay, f1_score, accuracy_score,
    mean_squared_error, mean_absolute_error, r2_score
)
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression

import warnings
warnings.filterwarnings('ignore')

SEED = 42
np.random.seed(SEED)

plt.rcParams.update({
    'figure.figsize': (12, 6),
    'axes.spines.top': False,
    'axes.spines.right': False,
    'font.size': 11,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
})

COLORS = {'bad': '#F44336', 'good': '#4CAF50', 'accent': '#FF9800',
          'blue': '#2196F3', 'purple': '#9C27B0'}

print('Ready.')




df = pd.read_csv('./WineQT.csv')
print(f'Shape: {df.shape}')
df.head()


# Quick checks
print('Missing values:', df.isnull().sum().sum())
print(f'Duplicates: {df.duplicated().sum()}')



df = df.drop(['Id','pH','residual sugar','free sulfur dioxide','chlorides'], axis=1)
print('Dropped Id, pH, residual sugar, chlorides, and free sulfur dioxide columns.')

print(df)



# --- GRAPHIQUE 4 : RAPPORT entre variables et quality ---
# --- 14. Scatter Plots with Trend Lines (The Ultimate Proof) ---
# We will plot every single feature against Quality.
# We add a "Regression Line" to prove the direction of the relationship.
# x_jitter: Shakes the dots slightly so they don't overlap perfectly.

features_all = [
    'fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
    'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
    'pH', 'sulphates', 'alcohol'
]
# Create a 4x3 grid (12 slots for 11 features)
fig, axes = plt.subplots(4, 3, figsize=(20, 20))
axes = axes.flatten() # Flatten the grid for easy looping

for i, col in enumerate(features_all):
    # sns.regplot is powerful: It shows the dots AND the best-fitting line.
    # scatter_kws={'alpha':0.3}: Makes dots transparent so we see density.
    sns.regplot(x='quality', y=col, data=df, ax=axes[i], x_jitter=0.2, 
                line_kws={'color':'red'}, scatter_kws={'alpha':0.3, 'color':'teal'})
    
    axes[i].set_title(f'{col} vs Quality', fontsize=12, fontweight='bold')
    axes[i].set_xlabel('Quality')
    axes[i].set_ylabel(col)

# Remove the empty 12th plot
fig.delaxes(axes[11])

plt.tight_layout()
plt.show()


