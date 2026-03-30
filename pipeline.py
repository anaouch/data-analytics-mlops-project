import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import sqlite3

url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)

df = df[["Survived", "Pclass", "Sex", "Age", "Fare"]]
df["Age"] = df["Age"].fillna(df["Age"].mean())
df["Sex"] = df["Sex"].map({"male": 0, "female": 1})

X = df[["Pclass", "Sex", "Age", "Fare"]]
y = df["Survived"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

modele = LogisticRegression(max_iter=200)
modele.fit(X_train, y_train)

score = accuracy_score(y_test, modele.predict(X_test))
print(f"Précision du modèle : {score * 100:.0f}%")

passager = pd.DataFrame({"Pclass": [3], "Sex": [0], "Age": [25], "Fare": [10]})
resultat = modele.predict(passager)
print(f"Passager classe 3, homme, 25 ans → {'Survécu ✅' if resultat[0] == 1 else 'Pas survécu ❌'}")

conn = sqlite3.connect("titanic.db")
df.to_sql("passagers", conn, if_exists="replace", index=False)
df_sql = pd.read_sql("SELECT * FROM passagers WHERE Age > 18", conn)
print(f"Passagers adultes via SQL : {len(df_sql)} personnes")
conn.close()

resultats = X_test.copy()
resultats["reel"] = y_test.values
resultats["predit"] = modele.predict(X_test)
resultats["correct"] = (resultats["reel"] == resultats["predit"]).astype(int)
resultats.to_csv("resultats.csv", index=False)

survie = df.groupby("Sex")["Survived"].mean() * 100
plt.figure(figsize=(6, 4))
plt.bar(["Homme", "Femme"], survie.values, color=["steelblue", "salmon"])
plt.title("Taux de survie par sexe (%)")
plt.ylabel("%")
plt.savefig("graphique.png")
