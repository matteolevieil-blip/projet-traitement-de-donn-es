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
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor
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


df = pd.read_csv('./winequality-red.csv') 
print(f'Shape initial : {df.shape}')

# Suppression des doublons (Data Cleaning)
df.drop_duplicates(inplace=True)
print(f'Shape après suppression des doublons : {df.shape}')



# Variable 1 : Le ratio de dioxyde de soufre (Soufre actif relatif)
df['sulfur_dioxide_ratio'] = np.where(
    df['total sulfur dioxide'] > 0, 
    df['free sulfur dioxide'] / df['total sulfur dioxide'], 
    0
)

# Variable 2 : L'indice d'impact pH / Chlorures (Indicateur d'altération)
df['pH_chlorides_impact'] = df['pH'] * df['chlorides']

# Variable 3 : Somme des "bonnes" acidités d'origine fruitée
df['total_good_acidity'] = df['fixed acidity'] + df['citric acid']

print('Les 3 variables handcrafted ont été ajoutées.')


df = df.drop([
    'free sulfur dioxide', 
    'total sulfur dioxide', 
    'chlorides',
    'residual sugar',
    'pH',
    'fixed acidity',
    'citric acid'
], axis=1, errors='ignore')

print('Dropped free sulfur dioxide, total sulfur dioxide, and chlorides columns.')
print(f'Shape final du DataFrame : {df.shape}\n')

print(df)


# =========================================================================
# MATRICE DE CORRÉLATION DU NOUVEAU JEU DE DONNÉES
# =========================================================================

# Calcul de la matrice de corrélation (Pearson par défaut)
nouveau_corr = df.corr()

print("--- VALEURS DE LA NOUVELLE MATRICE DE CORRÉLATION ---")
print(nouveau_corr.round(2)) # Arrondi à 2 décimales pour une lecture plus propre

# Visualisation graphique de la matrice
plt.figure(figsize=(10, 8))

# Masque pour masquer la moitié supérieure diagonale (optionnel, pour alléger le visuel)
mask = np.triu(np.ones_like(nouveau_corr, dtype=bool))

# Création de la heatmap
sns.heatmap(
    nouveau_corr, 
    mask=mask,
    annot=True,             # Affiche les valeurs numériques dans les cases
    fmt=".2f",              # Format à 2 décimales
    cmap="coolwarm",        # Palette de couleurs (bleu=négatif, rouge=positif)
    vmax=1, vmin=-1,        # Bornes de l'échelle des couleurs
    center=0,               # Le blanc représente l'absence de corrélation
    square=True,            # Cases carrées
    linewidths=.5,          # Ligne de séparation entre les cases
    cbar_kws={"shrink": .8} # Réduction de la taille de la barre de légende
)

plt.title("Nouvelle Matrice de Corrélation\n(Après Feature Engineering et Nettoyage)", fontweight='bold', pad=20)
plt.tight_layout()
plt.show()
