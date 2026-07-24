import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import squareform
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
)

RANDOM_STATE = 42

# ==========================================
# ÉTAPE 1 — PRÉPARATION
# ==========================================
print("=" * 60)
print("ÉTAPE 1 — PRÉPARATION")
print("=" * 60)

df = pd.read_csv("BankChurners.csv")

naive_bayes_cols = [c for c in df.columns if c.startswith("Naive_Bayes_")]
df = df.drop(columns=["CLIENTNUM"] + naive_bayes_cols, errors="ignore")
print(f"Dimensions après nettoyage : {df.shape}")

# Variables comportementales numériques utilisées pour la typologie
behavior_features = [
    "Customer_Age",
    "Months_on_book",
    "Total_Relationship_Count",
    "Months_Inactive_12_mon",
    "Contacts_Count_12_mon",
    "Credit_Limit",
    "Total_Revolving_Bal",
    "Avg_Open_To_Buy",
    "Total_Amt_Chng_Q4_Q1",
    "Total_Trans_Amt",
    "Total_Trans_Ct",
    "Total_Ct_Chng_Q4_Q1",
    "Avg_Utilization_Ratio",
]

# CAH réalisée sur la totalité du dataset (peut être plus lent que sur un échantillon,
# la complexité de la CAH étant O(n^2) à O(n^3) contrairement au K-Means)
df_sample = df.reset_index(drop=True)
print(f"Clients retenus pour la CAH : {df_sample.shape[0]} (dataset complet)")

X = df_sample[behavior_features]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled_df = pd.DataFrame(X_scaled, columns=behavior_features)
print("-" * 60)


# ==========================================
# ÉTAPE 2 — CAH SUR LES VARIABLES (hors-d'œuvre)
# ==========================================
print("=" * 60)
print("ÉTAPE 2 — CAH SUR LES VARIABLES")
print("=" * 60)

corr = X_scaled_df.corr()
dist_vars = 1 - corr.abs()
condensed_vars = squareform(dist_vars.values, checks=False)
Z_vars = linkage(condensed_vars, method="average")

plt.figure(figsize=(12, 6))
dendrogram(Z_vars, labels=corr.columns.tolist(), leaf_rotation=90)
plt.title("Dendrogramme des variables (distance = 1 - |corrélation|)")
plt.ylabel("Distance (1 - |corrélation|)")
plt.tight_layout()
plt.savefig("dendrogramme_variables.png", dpi=150)
plt.close()
print("Dendrogramme des variables sauvegardé : dendrogramme_variables.png")

# Paires les plus corrélées (fusion immédiate attendue)
corr_pairs = (
    corr.abs()
    .where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    .stack()
    .sort_values(ascending=False)
)
print("\nTop 5 des paires de variables les plus corrélées (redondance) :")
print(corr_pairs.head(5))



# ==========================================
# ÉTAPE 3 — PCA
# ==========================================
print("=" * 60)
print("ÉTAPE 3 — ACP (PCA)")
print("=" * 60)

N_COMPONENTS = 4
pca = PCA(n_components=N_COMPONENTS, random_state=RANDOM_STATE)
X_pca = pca.fit_transform(X_scaled)

explained = pca.explained_variance_ratio_
print("Variance expliquée par composante :")
for i, v in enumerate(explained, start=1):
    print(f"  PC{i} : {v * 100:.1f}%")
print(f"Variance cumulée (4 axes) : {explained.sum() * 100:.1f}%")

loadings = pd.DataFrame(
    pca.components_.T,
    index=behavior_features,
    columns=[f"PC{i}" for i in range(1, N_COMPONENTS + 1)],
)
print("\nContributions des variables aux axes (loadings) :")
print(loadings.round(2))




# ==========================================
# ÉTAPE 4 — CAH SUR LES INDIVIDUS (le cœur)
# ==========================================
print("=" * 60)
print("ÉTAPE 4 — CAH SUR LES INDIVIDUS (Ward, sur composantes PCA)")
print("=" * 60)

Z_ind = linkage(X_pca, method="ward")

plt.figure(figsize=(12, 6))
dendrogram(Z_ind, truncate_mode="lastp", p=30, show_leaf_counts=True)
plt.title("Dendrogramme des individus (Ward sur composantes PCA, tronqué)")
plt.ylabel("Distance")
plt.tight_layout()
plt.savefig("dendrogramme_individus.png", dpi=150)
plt.close()
print("Dendrogramme des individus sauvegardé : dendrogramme_individus.png")

# Choix du nombre de clusters : plus grand saut de distance dans les dernières fusions
last_merges = Z_ind[-10:, 2]
jumps = np.diff(last_merges)
n_clusters = len(last_merges) - np.argmax(jumps) - 1
n_clusters = max(2, min(6, n_clusters))
print(f"\nPlus grand saut de distance détecté → k = {n_clusters} segments retenus")

df_sample["Segment"] = fcluster(Z_ind, t=n_clusters, criterion="maxclust")

# Validation visuelle : projection des clusters sur le plan factoriel PC1-PC2
plt.figure(figsize=(9, 7))
scatter = plt.scatter(
    X_pca[:, 0], X_pca[:, 1], c=df_sample["Segment"], cmap="tab10", alpha=0.6, s=15
)
plt.xlabel(f"PC1 ({explained[0] * 100:.1f}%)")
plt.ylabel(f"PC2 ({explained[1] * 100:.1f}%)")
plt.title("Projection des segments sur le plan factoriel (PC1, PC2)")
plt.legend(*scatter.legend_elements(), title="Segment")
plt.tight_layout()
plt.savefig("plan_factoriel_segments.png", dpi=150)
plt.close()
print("Projection des segments sauvegardée : plan_factoriel_segments.png")



# ==========================================
# ÉTAPE 5 — PROFILAGE ET CROISEMENT AU CHURN
# ==========================================
print("=" * 60)
print("ÉTAPE 5 — PROFILAGE DES SEGMENTS ET CROISEMENT AU CHURN")
print("=" * 60)

df_sample["Churn"] = (df_sample["Attrition_Flag"] == "Attrited Customer").astype(int)

# Moyennes des variables clés par segment
profile_means = df_sample.groupby("Segment")[behavior_features].mean().round(1)
print("\nProfil moyen (variables clés) par segment :")
print(profile_means)

# Composition démographique par segment
demographic_cols = ["Gender", "Income_Category", "Card_Category", "Education_Level"]
print("\nComposition démographique dominante par segment :")
for col in demographic_cols:
    dominant = df_sample.groupby("Segment")[col].agg(lambda s: s.value_counts().idxmax())
    print(f"  {col} (modalité dominante) :\n{dominant.to_string()}\n")

# Taux de churn et taille par segment — la table qui porte le projet
segment_size = df_sample.groupby("Segment").size().rename("Nb_clients")
segment_churn = (df_sample.groupby("Segment")["Churn"].mean() * 100).round(1).rename("Taux_churn_%")

resultats = pd.concat([segment_size, segment_churn, profile_means], axis=1)
resultats = resultats.sort_values("Taux_churn_%", ascending=False)

print("\n" + "=" * 60)
print("TABLE FINALE — SEGMENTS x TAUX DE CHURN")
print("=" * 60)
print(resultats)

resultats.to_csv("segments_churn_resultats.csv")
print("\nTable exportée : segments_churn_resultats.csv")



# ==========================================
# ÉTAPE 6 — KNN : PRÉDICTION SUPERVISÉE DU CHURN
# ==========================================
print("\n" + "=" * 60)
print("ÉTAPE 6 — KNN : PRÉDICTION SUPERVISÉE DU CHURN")
print("=" * 60)

y = df_sample["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.25, stratify=y, random_state=RANDOM_STATE
)
print(f"Train : {X_train.shape[0]} clients | Test : {X_test.shape[0]} clients")
print(f"Taux de churn train : {y_train.mean() * 100:.1f}% | test : {y_test.mean() * 100:.1f}%")

# Recherche du meilleur k par validation croisée (F1, car classes déséquilibrées)
param_grid = {"n_neighbors": list(range(3, 26, 2)), "weights": ["uniform", "distance"]}
grid = GridSearchCV(
    KNeighborsClassifier(), param_grid, scoring="f1", cv=5, n_jobs=-1
)
grid.fit(X_train, y_train)
print(f"\nMeilleurs paramètres (CV, score F1) : {grid.best_params_}")
print(f"Score F1 en validation croisée : {grid.best_score_:.3f}")

knn = grid.best_estimator_
y_pred = knn.predict(X_test)

print(f"\nAccuracy (test) : {accuracy_score(y_test, y_pred):.3f}")
print(f"F1-score churn (test) : {f1_score(y_test, y_pred):.3f}")
print("\nRapport de classification :")
print(classification_report(y_test, y_pred, target_names=["Non-churn", "Churn"]))

cm = confusion_matrix(y_test, y_pred)
print("Matrice de confusion :")
print(cm)

plt.figure(figsize=(6, 5))
ConfusionMatrixDisplay(cm, display_labels=["Non-churn", "Churn"]).plot(
    cmap="Blues", colorbar=False
)
plt.title("KNN — Matrice de confusion (prédiction du churn)")
plt.tight_layout()
plt.savefig("knn_confusion_matrix.png", dpi=150)
plt.close()
print("Matrice de confusion sauvegardée : knn_confusion_matrix.png")
