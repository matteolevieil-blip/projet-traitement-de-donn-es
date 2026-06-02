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



df = df.drop(['Id','pH','residual sugar','free sulfur dioxide'], axis=1)
print('Dropped Id, pH, residual sugar, and free sulfur dioxide columns.')

print(df)


