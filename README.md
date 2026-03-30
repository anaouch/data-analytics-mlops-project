# 🚀 Data Analytics & MLOps Project

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Docker](https://img.shields.io/badge/Docker-containerized-2496ED)
![FastAPI](https://img.shields.io/badge/FastAPI-REST%20API-009688)
![Power BI](https://img.shields.io/badge/Power%20BI-Dashboard-F2C811)

## 📌 Overview
End-to-end data analytics pipeline built from scratch using Python and SQL, containerized with Docker, served via a FastAPI REST API, and visualized with an interactive Power BI dashboard.

## 🛠️ Technologies
- **Python** — pandas, scikit-learn, matplotlib, FastAPI
- **SQL** — SQLite database
- **Docker** — containerization & reproducibility
- **Power BI** — interactive KPI dashboard
- **GitHub** — version control

## 📊 Results
- 891 real passengers analyzed (Titanic dataset)
- **80% model accuracy** (Logistic Regression)
- REST API with real-time predictions
- 3 interactive dashboard visualizations

## ⚡ How to run

**With Docker :**
```bash
docker build -t mon-pipeline .
docker run -p 8000:8000 mon-pipeline
```

**API endpoint :**
```bash
POST http://localhost:8000/predire
{
  "pclass": 3,
  "sex": 0,
  "age": 25,
  "fare": 10
}
```

**API docs :** http://localhost:8000/docs

## 📁 Project Structure
```
├── pipeline.py        # ML pipeline (ingestion, cleaning, training)
├── api.py             # FastAPI REST API
├── requirements.txt   # Dependencies
├── Dockerfile         # Docker configuration
├── resultats.csv      # Model results
└── titanic.db         # SQLite database
```

## 📈 Dashboard Power BI

![Précision du modèle](dashboard1.png)
![Survie par classe](dashboard2.png)
![Survie par sexe](dashboard3.png)
