import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# 1. CHARGEMENT DU FICHIER CSV
df = pd.read_csv('C:/Users/mehdi/Documents/introduction_traitement_données/bodyPerformance.csv')

# 2. NETTOYAGE ET GESTION DES DONNÉES 
df = df.dropna() # Suppression des valeurs manquantes

# --- NOUVEAU : ENCODAGE CATÉGORIEL (Concept de la Lecture 2) ---
# On transforme la colonne texte "gender" en valeurs numériques (Binaire)
# "M" devient 1, "F" devient 0
df['gender'] = df['gender'].replace({'M': 1, 'F': 0})

# 3. SÉPARATION DES VARIABLES (X = données physiques, y = classe de performance)
# La colonne à prédire s'appelle "class"
X = df.drop(columns=['class'])  # Toutes les variables physiques (age, taille, pompes...)
y = df['class']                 # La classe finale (A, B, C ou D)

# 4. DÉCOUPAGE EN JEU D'ENTRAÎNEMENT ET DE TEST (Holdout Method - Lecture 4)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. STANDARDISATION DES CARACTÉRISTIQUES (Lecture 2 & 4)
# Indispensable car le poids (kg) et la force de préhension (gripForce) 
# n'ont pas du tout la même échelle !
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 6. ENTRAÎNEMENT DE LA RÉGRESSION SOFTMAX
model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)

# 7. ÉVALUATION ET VÉRIFICATION DE LA PRÉCISION
y_pred = model.predict(X_test_scaled)
precision = accuracy_score(y_test, y_pred)

print(f"Précision globale du modèle Softmax : {precision * 100:.2f}%\n")
print("Rapport détaillé par Classe de Performance (A, B, C, D) :")
print(classification_report(y_test, y_pred))


# --- 8. TESTER UN NOUVEAU PROFIL (CAS PRATIQUE) ---

# Étape 1 : Récupérer l'ordre exact des colonnes utilisées lors de l'entraînement
colonnes_attendues = X.columns.tolist()
print("\nLes colonnes attendues sont :", colonnes_attendues)

# Étape 2 : Créer le profil du nouveau client
# (Exemple de valeurs : 25 ans, Homme (1), 180cm, 75kg, 15% graisse, etc...)
# /!\ Il faut mettre des valeurs dans le MÊME ORDRE que la liste ci-dessus !
nouveau_client = [25, 1, 180.0, 75.0, 15.0, 80.0, 120.0, 50.0, 20.0, 55.0, 220.0]

# Étape 3 : Créer un DataFrame (pour éviter le message d'erreur UserWarning)
profil_df = pd.DataFrame([nouveau_client], columns=colonnes_attendues)

# Étape 4 : Standardiser le profil (avec le scaler déjà entraîné !)
profil_scaled = scaler.transform(profil_df)

# Étape 5 : La magie du Softmax (Prédiction)
classe_predite = model.predict(profil_scaled)
probabilites = model.predict_proba(profil_scaled)[0] # Récupère les pourcentages

print("\n--- RÉSULTAT DU DIAGNOSTIC ---")
print(f"La classe prédite pour ce client est : {classe_predite[0]}")
print(f"Détail des probabilités Softmax :")
print(f"Classe A : {probabilites[0]*100:.1f} %")
print(f"Classe B : {probabilites[1]*100:.1f} %")
print(f"Classe C : {probabilites[2]*100:.1f} %")
print(f"Classe D : {probabilites[3]*100:.1f} %")


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Configuration du style pour respecter le "Data-ink ratio" (Lecture 3)
sns.set_style("white") # Fond blanc épuré, pas de quadrillage lourd

# 1. CHARGEMENT DES DONNÉES
df = pd.read_csv('C:/Users/mehdi/Documents/introduction_traitement_données/bodyPerformance.csv')
df = df.dropna()

# =====================================================================
# GRAPHIQUE 1 : Le Graphique de la Confusion (Boxplot)
# Pourquoi le modèle a du mal à séparer les classes B et C ?
# =====================================================================
plt.figure(figsize=(10, 6))

# Création du boxplot
sns.boxplot(data=df, x='class', y='sit-ups counts', order=['A', 'B', 'C', 'D'], palette='viridis')

# Storytelling : Le titre donne la conclusion
plt.title("L'endurance musculaire sépare nettement les extrêmes (A et D),\nmais les classes moyennes (B et C) se confondent fortement", 
          fontsize=14, fontweight='bold', pad=20)
plt.xlabel("Classe de Performance", fontsize=12)
plt.ylabel("Nombre de pompes (sit-ups)", fontsize=12)

# Optimisation du Data-ink ratio (On enlève le cadre du haut et de droite)
sns.despine()
plt.tight_layout()
plt.show()

# =====================================================================
# GRAPHIQUE 2 : Le Radar de l'Athlète (Spider Chart)
# Comparaison multidimensionnelle entre le profil A et D
# =====================================================================
# On sélectionne 4 compétences "Positives" (où un gros score = bon)
features = ['gripForce', 'sit and bend forward_cm', 'sit-ups counts', 'broad jump_cm']

# On calcule la moyenne de ces stats pour les classes A et D
df_mean = df.groupby('class')[features].mean().reset_index()

# Pour comparer sur un même radar, on transforme les valeurs en pourcentages par rapport au maximum
for f in features:
    df_mean[f] = (df_mean[f] / df_mean[f].max()) * 100

# Préparation des angles pour le radar
categories = ['Force (Grip)', 'Souplesse', 'Endurance (Pompes)', 'Explosivité (Saut)']
N = len(categories)
angles = [n / float(N) * 2 * np.pi for n in range(N)]
angles += angles[:1] # Fermer le cercle

# Initialisation du graphique circulaire
fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

# Tracé pour la Classe A
val_A = df_mean[df_mean['class'] == 'A'][features].values.flatten().tolist()
val_A += val_A[:1]
ax.plot(angles, val_A, linewidth=2, linestyle='solid', label='Classe A (Excellente)')
ax.fill(angles, val_A, alpha=0.25)

# Tracé pour la Classe D
val_D = df_mean[df_mean['class'] == 'D'][features].values.flatten().tolist()
val_D += val_D[:1]
ax.plot(angles, val_D, linewidth=2, linestyle='solid', label='Classe D (Faible)')
ax.fill(angles, val_D, alpha=0.25)

# Nettoyage visuel du radar
plt.xticks(angles[:-1], categories, fontsize=12)
ax.set_yticklabels([]) # On cache les chiffres de l'axe Y pour alléger l'encre

plt.title("Profil A vs D : La vraie différence se fait sur l'endurance et la souplesse,\nbeaucoup moins sur la force pure des bras", 
          fontsize=14, fontweight='bold', pad=30)
plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1))
plt.tight_layout()
plt.show()

# =====================================================================
# GRAPHIQUE 3 : L'impact de la Graisse (Scatter Plot)
# Démontrer que le poids seul ne veut rien dire
# =====================================================================
plt.figure(figsize=(10, 6))

# Pour rendre le graphique plus lisible, on peut comparer seulement les A et D (ou tout afficher avec un peu de transparence)
sns.scatterplot(data=df, x='weight_kg', y='body fat_%', hue='class', 
                hue_order=['A', 'B', 'C', 'D'], palette='RdYlGn_r', alpha=0.6, s=50)

# Storytelling
plt.title("Le poids seul est trompeur : c'est le taux de graisse corporelle\nqui condamne presque à coup sûr à la classe D", 
          fontsize=14, fontweight='bold', pad=20)
plt.xlabel("Poids (en kg)", fontsize=12)
plt.ylabel("Graisse Corporelle (%)", fontsize=12)

# Optimisation Data-ink
sns.despine()
plt.legend(title="Classe", frameon=False) # Légende sans bordure
plt.tight_layout()
plt.show()
