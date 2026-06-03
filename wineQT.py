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

# Variable 3 : L'indice de pureté et de fraîcheur aromatique
# On ajoute +0.001 au dénominateur pour éviter une division par zéro si un vin a 0 en volatile
df['citric_volatile_ratio'] = df['citric acid'] / (df['volatile acidity'] + 0.001)

print('Les 3 variables handcrafted ont été ajoutées.')


df = df.drop([
    'free sulfur dioxide', 
    'total sulfur dioxide', 
    'chlorides',
    'residual sugar',
    'pH',
    'volatile acidity',
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




# =========================================================================
# 7. VISUALISATION : IMPACT DE CHAQUE VARIABLE SUR LA QUALITÉ
# =========================================================================

# On liste toutes les variables explicatives (on exclut juste la cible 'quality')
variables_a_tracer = [col for col in df.columns if col != 'quality']

# Calcul du nombre de lignes nécessaires pour les graphiques (3 graphiques par ligne)
nb_colonnes_grid = 3
nb_lignes_grid = (len(variables_a_tracer) + nb_colonnes_grid - 1) // nb_colonnes_grid

# Création de la figure globale
fig, axes = plt.subplots(nb_lignes_grid, nb_colonnes_grid, figsize=(18, 5 * nb_lignes_grid))
axes = axes.flatten() # Aplatit la matrice d'axes en tableau 1D pour boucler facilement

print("Génération des boxplots par rapport à la qualité...")

# Boucle pour tracer chaque variable
for i, var in enumerate(variables_a_tracer):
    sns.boxplot(
        x='quality', 
        y=var, 
        data=df, 
        ax=axes[i],
        palette='viridis', # Palette de couleurs dégradées du violet au jaune
        width=0.6,
        fliersize=3        # Taille des points aberrants (outliers)
    )
    
    axes[i].set_title(f'{var} vs Quality', fontweight='bold', fontsize=12)
    axes[i].set_xlabel('Quality (3 à 8)')
    axes[i].set_ylabel(var)
    axes[i].grid(axis='y', linestyle='--', alpha=0.5)

# Si le nombre de variables est impair, on cache les graphiques vides restants
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()



