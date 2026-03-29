import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = {
    "age":     [25, 30, 35, 40, 45, 50, 55, 60],
    "salaire": [2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500],
    "score":   [60, 65, 70, 75, 80, 85, 90, 95]
}

df = pd.DataFrame(data)
print("Les données :")
print(df)

print("\nValeurs manquantes :", df.isnull().sum().sum())

print("\nStatistiques :")
print(df.describe())

plt.figure(figsize=(8, 4))
plt.bar(df["age"], df["salaire"], color="steelblue")
plt.title("Salaire par âge")
plt.xlabel("Âge")
plt.ylabel("Salaire")
plt.tight_layout()
plt.savefig("graphique.png")
print("\nGraphique sauvegardé !")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# -----------------------------------------------
# ÉTAPE 1 : Télécharger un vrai dataset (891 passagers)
# -----------------------------------------------
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)

print(f"Données chargées : {df.shape[0]} lignes, {df.shape[1]} colonnes")
print("\nAperçu :")
print(df.head())

# -----------------------------------------------
# ÉTAPE 2 : Nettoyage
# -----------------------------------------------
# On garde seulement les colonnes utiles
df = df[["Survived", "Pclass", "Sex", "Age", "Fare"]]

# Remplir les âges manquants par la moyenne
df["Age"] = df["Age"].fillna(df["Age"].mean())

# Convertir homme/femme en 0/1
df["Sex"] = df["Sex"].map({"male": 0, "female": 1})

print(f"\nValeurs manquantes restantes : {df.isnull().sum().sum()}")

# -----------------------------------------------
# ÉTAPE 3 : Entraîner le modèle
# -----------------------------------------------
X = df[["Pclass", "Sex", "Age", "Fare"]]
y = df["Survived"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

modele = LogisticRegression(max_iter=200)
modele.fit(X_train, y_train)

score = accuracy_score(y_test, modele.predict(X_test))
print(f"\nPrécision du modèle : {score * 100:.0f}%")

# -----------------------------------------------
# ÉTAPE 4 : Prédire un nouveau passager
# -----------------------------------------------
# Classe 3, homme, 25 ans, billet 10€
passager = pd.DataFrame({
    "Pclass": [3], "Sex": [0], "Age": [25], "Fare": [10]
})
resultat = modele.predict(passager)
print(f"\nPassager classe 3, homme, 25 ans → {'Survécu ✅' if resultat[0] == 1 else 'Pas survécu ❌'}")

# -----------------------------------------------
# ÉTAPE 5 : Graphique
# -----------------------------------------------
survie = df.groupby("Sex")["Survived"].mean() * 100
plt.figure(figsize=(6, 4))
plt.bar(["Homme", "Femme"], survie.values, color=["steelblue", "salmon"])
plt.title("Taux de survie par sexe (%)")
plt.ylabel("%")
plt.savefig("graphique.png")
print("\nGraphique sauvegardé !")
import sqlite3

# Créer une base de données locale
conn = sqlite3.connect("titanic.db")

# Sauvegarder les données dedans
df.to_sql("passagers", conn, if_exists="replace", index=False)
print("Données sauvegardées dans la base SQL !")

# Relire avec une requête SQL
df_sql = pd.read_sql("SELECT * FROM passagers WHERE Age > 18", conn)
print(f"\nPassagers adultes via SQL : {len(df_sql)} personnes")

conn.close()
# Exporter les résultats pour le dashboard
resultats = X_test.copy()
resultats["reel"] = y_test.values
resultats["predit"] = modele.predict(X_test)
resultats["correct"] = (resultats["reel"] == resultats["predit"]).astype(int)
resultats.to_csv("resultats.csv", index=False)
print("Résultats exportés dans resultats.csv !")