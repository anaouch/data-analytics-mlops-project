from fastapi import FastAPI
import pandas as pd
from pydantic import BaseModel
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

app = FastAPI()

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

class Passager(BaseModel):
    pclass: int
    sex: int
    age: float
    fare: float

@app.get("/")
def accueil():
    return {"message": "API Titanic opérationnelle !"}

@app.post("/predire")
def predire(passager: Passager):
    data = pd.DataFrame([{
        "Pclass": passager.pclass,
        "Sex": passager.sex,
        "Age": passager.age,
        "Fare": passager.fare
    }])
    resultat = modele.predict(data)[0]
    return {
        "survie": int(resultat),
        "prediction": "Survécu ✅" if resultat == 1 else "Pas survécu ❌"
    }
