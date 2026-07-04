# 🤖 AI Job Market – Prédiction des Salaires avec le Machine Learning

## 📖 Description

Ce projet a été réalisé dans le cadre de mon apprentissage en **Data Science** et **Machine Learning**.

L'objectif est de développer une application capable d'estimer le **salaire annuel (USD)** d'un professionnel du domaine **Data & Intelligence Artificielle** en fonction de plusieurs caractéristiques telles que le poste occupé, le niveau d'expérience, le pays, le type d'emploi, le mode de travail et la taille de l'entreprise.

Le projet couvre l'ensemble du cycle de vie d'un projet de Data Science, depuis le nettoyage des données jusqu'a la réalisation d'une interface interactive avec **Streamlit**.

---

# 🚀 Fonctionnalités

## 🧹 Nettoyage et préparation des données

Les données ont été nettoyées et préparées afin d'améliorer leur qualité avant l'entraînement des modèles :

- Suppression des valeurs incohérentes
- Traitement des valeurs manquantes
- Encodage des variables catégorielles
- Préparation des variables explicatives
- Séparation des jeux d'entraînement et de test

---

## 📊 Analyse Exploratoire des Données (EDA)

Une analyse exploratoire complète a été réalisée afin de mieux comprendre le marché de l'emploi dans les métiers de la Data et de l'IA.

Les analyses portent notamment sur :

- Répartition des salaires
- Salaire moyen par métier
- Salaire selon le niveau d'expérience
- Salaire par pays
- Salaire selon le type de contrat
- Salaire selon la taille de l'entreprise
- Corrélations entre les variables

---

## 🤖 Machine Learning

Plusieurs modèles de régression ont été entraînés et comparés afin de prédire le salaire.

### Modèles testés

- ✅ Régression Linéaire
- 🌳 Decision Tree Regressor
- 🌲 Random Forest Regressor

Après comparaison des performances, le modèle de **Régression Linéaire** a été retenu comme modèle final.

---

## 💻 Application Web

Une interface interactive a été développée avec **Streamlit**.

Elle permet à l'utilisateur de :

- saisir les caractéristiques d'un emploi ;
- prédire instantanément le salaire estimé ;
- visualiser une interface simple et intuitive.

---

# 📂 Structure du projet

```text
AI-Job-Market/
│
├── app/
│   ├── app.py
│   └── .streamlit/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── figures/
│
├── models/
│   └── model_salary.pkl
│
├── notebooks/
│   ├── 01_Cleaning.ipynb
│   ├── 02_visualisation.ipynb
│   └── 03_machine_learning.ipynb
│
├── requirements.txt
└── README.md
```

---

# 🛠️ Technologies utilisées

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Streamlit

---

# 📈 Dataset

Le projet utilise un dataset contenant plus de **10 000 offres d'emploi** dans le domaine de la **Data & Intelligence Artificielle**.

Les données couvrent :

- plusieurs pays ;
- différents niveaux d'expérience ;
- plusieurs métiers de la Data ;
- différents types de contrats ;
- différentes tailles d'entreprise.

---

# ⚙️ Installation

## Cloner le projet

```bash
git clone https://github.com/FATIMAZOHRAENNEJJARI/AI-Job-Market.git
```

## Installer les dépendances

```bash
pip install -r requirements.txt
```

## Lancer l'application

```bash
streamlit run app/app.py
```

---

# 📷 Aperçu de l'application

<img width="1878" height="876" alt="image" src="https://github.com/user-attachments/assets/108c409a-2929-4c06-bfad-3a059860dc6f" />
<img width="1915" height="917" alt="image" src="https://github.com/user-attachments/assets/3e5d245c-27cc-4040-a528-a8c7ecb45009" />
<img width="1913" height="908" alt="image" src="https://github.com/user-attachments/assets/68bc2e3a-79c7-4e9a-8a76-0a4ddb22e70c" />
<img width="1912" height="908" alt="image" src="https://github.com/user-attachments/assets/1c6a9831-b476-420a-addc-5b988144feb2" />
<img width="1913" height="911" alt="image" src="https://github.com/user-attachments/assets/1140d906-1837-4f5e-848d-b78fdd04d2a2" />






# 🎯 Compétences développées

À travers ce projet, j'ai renforcé mes compétences en :

- Préparation et nettoyage des données
- Analyse exploratoire des données (EDA)
- Data Visualization
- Machine Learning
- Évaluation de modèles
- Développement d'applications avec Streamlit


---

# 👩‍💻 Auteur

**Fatima Zohra Ennejjari**

Étudiante en Ingénierie des Sciences des Données et Intelligence Artificielle.

GitHub : https://github.com/FATIMAZOHRAENNEJJARI
