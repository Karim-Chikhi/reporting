import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE

# ==========================================
# 0. CHARGEMENT ET PRÉPARATION DES DONNÉES
# ==========================================
print("Chargement des données...")
df = pd.read_csv("BankChurners.csv")

# Nettoyage des colonnes inutiles
cols_to_drop = ['CLIENTNUM']
naive_bayes_cols = [col for col in df.columns if col.startswith('Naive_Bayes_')]
df = df.drop(columns=cols_to_drop + naive_bayes_cols, errors='ignore')

print(f"Dimensions du dataset : {df.shape}\n")
print("-" * 50)


# ==========================================
# SUJET A : CLUSTERING - TYPOLOGIE COMPORTEMENTALE
# ==========================================
print("SUJET A : Exécution du Clustering Comportemental...")

# 1. Sélection des variables comportementales
features_A = ['Total_Trans_Amt', 'Total_Trans_Ct', 'Avg_Utilization_Ratio', 'Total_Revolving_Bal']
X_A = df[features_A]

# 2. Standardisation des données (Crucial pour le K-Means)
scaler_A = StandardScaler()
X_A_scaled = scaler_A.fit_transform(X_A)

# 3. Entraînement du modèle K-Means (On choisit 4 clusters arbitrairement)
kmeans_A = KMeans(n_clusters=4, random_state=42, n_init=10)
df['Cluster_Comportement'] = kmeans_A.fit_predict(X_A_scaled)

print("Centres des clusters comportementaux (données standardisées) :")
print(pd.DataFrame(kmeans_A.cluster_centers_, columns=features_A))
print("-" * 50)


# ==========================================
# SUJET B : CLUSTERING - PARADOXE PREMIUM VS STANDARD
# ==========================================
print("SUJET B : Exécution du Clustering Premium vs Standard...")

# 1. Sélection des variables
features_B_num = ['Credit_Limit', 'Months_on_book', 'Total_Relationship_Count']
features_B_cat = ['Income_Category'] # Variable catégorielle
X_B = df[features_B_num + features_B_cat]

# 2. Pipeline de transformation (Standardisation + Encodage One-Hot)
preprocessor_B = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), features_B_num),
        ('cat', OneHotEncoder(drop='first'), features_B_cat) # drop='first' évite la colinéarité
    ])

X_B_processed = preprocessor_B.fit_transform(X_B)

# 3. Entraînement du modèle K-Means (On teste avec 3 clusters)
kmeans_B = KMeans(n_clusters=3, random_state=42, n_init=10)
df['Cluster_Premium'] = kmeans_B.fit_predict(X_B_processed)

print(f"Répartition des clients dans les clusters Premium/Standard : \n{df['Cluster_Premium'].value_counts()}")
print("-" * 50)


# ==========================================
# SUJET C : CLASSIFICATION - PRÉDICTION DU CHURN
# ==========================================
print("SUJET C : Exécution du Modèle de Prédiction du Churn...")

# 1. Préparation de la cible (Y) et des features (X)
# On convertit la cible en binaire : 1 pour Attrited (Départ), 0 pour Existing (Reste)
y = df['Attrition_Flag'].map({'Attrited Customer': 1, 'Existing Customer': 0})

# On enlève la cible et les colonnes de clusters générées précédemment
X_C = df.drop(columns=['Attrition_Flag', 'Cluster_Comportement', 'Cluster_Premium'], errors='ignore')

# 2. Encodage de toutes les variables catégorielles restantes
X_C_encoded = pd.get_dummies(X_C, drop_first=True)

# 3. Séparation Train / Test (80% entraînement, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X_C_encoded, y, test_size=0.2, random_state=42, stratify=y)

# 4. Gestion du déséquilibre avec SMOTE (uniquement sur le Train set !)
print(f"Distribution de la cible AVANT SMOTE :\n{y_train.value_counts()}")
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
print(f"Distribution de la cible APRÈS SMOTE :\n{y_train_smote.value_counts()}")

# 5. Entraînement du modèle Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
rf_model.fit(X_train_smote, y_train_smote)

# 6. Évaluation sur le jeu de Test
y_pred = rf_model.predict(X_test)

print("\n--- RAPPORT DE CLASSIFICATION ---")
print(classification_report(y_test, y_pred, target_names=['Existing (0)', 'Attrited (1)']))

print("\n--- MATRICE DE CONFUSION ---")
cm = confusion_matrix(y_test, y_pred)
print(pd.DataFrame(cm, 
                   index=['Vrai Existing', 'Vrai Attrited'], 
                   columns=['Prédit Existing', 'Prédit Attrited']))
                   
# 7. Importance des variables (Bonus)
feature_importances = pd.DataFrame(
    rf_model.feature_importances_,
    index=X_train_smote.columns,
    columns=['Importance']
).sort_values('Importance', ascending=False)

print("\n--- TOP 5 DES VARIABLES LES PLUS IMPORTANTES ---")
print(feature_importances.head(5))