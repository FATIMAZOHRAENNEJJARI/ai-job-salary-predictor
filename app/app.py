"""
╔══════════════════════════════════════════════════════════════════╗
║  04 — Interface Graphique : Prédiction de Salaire                ║
║  Projet : AI Data Job Market                                     ║
║  Bibliothèque : Streamlit                                        ║
║  Usage : streamlit run app.py                                    ║
╚══════════════════════════════════════════════════════════════════╝
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os


st.set_page_config(
    page_title="Data & IA -> Estime le Salaire que tu Mérites",
    page_icon="👩🏻‍💻",
    layout="centered",
    initial_sidebar_state="collapsed"
)

import base64
import os

def set_background(image_file):
    with open(image_file, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()

    css = f"""
    <style>
    .stApp {{
        background-image: url("data:image/jpg;base64,{encoded}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}

    /* ecriture dehors les cases : noire et gras */
    h1, h2, h3, h4, h5, h6 {{
        color: black !important;
        font-weight: 800 !important;
    }}

    p, label, span {{
        color: black !important;
        font-weight: 700 !important;
    }}

    .stCheckbox label {{
        color: black !important;
        font-weight: 700 !important;
    }}

    /* cases selectbox : fond noir, ecriture blanche */
    .stSelectbox > div > div {{
        background-color: #1a1a1a !important;
        color: white !important;
        border-radius: 8px !important;
    }}

    /* liste deroulante : fond noir opaque, ecriture blanche */
    ul[data-testid="stSelectboxVirtualDropdown"],
    [data-testid="stSelectboxVirtualDropdown"] li,
    [role="listbox"],
    [role="option"] {{
        background-color: #1a1a1a !important;
        color: white !important;
    }}

    [role="option"]:hover {{
        background-color: #333333 !important;
        color: white !important;
    }}

    /* bouton predire : fond noir, ecriture blanche */
    .stButton > button {{
        background-color: #1a1a1a !important;
        color: white !important;
        font-weight: 700 !important;
        border-radius: 8px !important;
        border: none !important;
    }}

    .stButton > button p,
    .stButton > button span,
    .stButton > button div {{
        color: white !important;
        -webkit-text-fill-color: white !important;
    }}

    /* message info en bas : fond noir, ecriture blanche */
    .stAlert {{
        background-color: #1a1a1a !important;
        color: white !important;
        border: none !important;
    }}

    .stAlert p, .stAlert span {{
        color: white !important;
        -webkit-text-fill-color: white !important;
    }}

    /* menu deploy et 3 points */
    [data-testid="stToolbar"],
    [data-testid="stToolbarActions"] {{
        background-color: transparent !important;
    }}

    /* popup deploy et 3 points : fond semi-transparent, ecriture blanche */
    div[role="dialog"],
    div[role="menu"],
    div[role="menuitem"] {{
        background-color: rgba(0, 0, 0, 0.75) !important;
        color: white !important;
        border-radius: 8px !important;
    }}

    div[role="dialog"] p,
    div[role="dialog"] span,
    div[role="dialog"] h1,
    div[role="dialog"] h2,
    div[role="dialog"] h3,
    div[role="dialog"] label,
    div[role="menu"] span,
    div[role="menu"] p,
    div[role="menuitem"] span {{
        color: white !important;
        -webkit-text-fill-color: white !important;
        font-weight: 600 !important;
    }}

    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


# chemin vers image (IMPORTANT : data.jpg)
base_dir = os.path.dirname(os.path.abspath(__file__))
image_path = os.path.join(base_dir, "b4.png")

set_background(image_path)



@st.cache_resource
def charger_modele():
    base_dir   = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, '..', 'models', 'model_salary.pkl')
    if not os.path.exists(model_path):
        return None
    with open(model_path, 'rb') as f:
        bundle = pickle.load(f)
    return bundle

bundle = charger_modele()

def construire_vecteur(
    job_title, company_size, company_industry, country,
    remote_type, education_level, hiring_urgency,
    years_experience, job_openings, job_posting_month,
    job_posting_year, skills_python, skills_sql, skills_ml,
    skills_deep_learning, skills_cloud, colonnes_model
):
    ligne = {col: 0 for col in colonnes_model}

    ligne['years_experience']  = years_experience
    ligne['job_openings']      = job_openings
    ligne['job_posting_month'] = job_posting_month
    ligne['job_posting_year']  = job_posting_year

    urgency_map = {'Low': 1, 'Medium': 2, 'High': 3}
    ligne['hiring_urgency'] = urgency_map[hiring_urgency]

    ligne['skills_python']        = 1 if skills_python        else 0
    ligne['skills_sql']           = 1 if skills_sql           else 0
    ligne['skills_ml']            = 1 if skills_ml            else 0
    ligne['skills_deep_learning'] = 1 if skills_deep_learning else 0
    ligne['skills_cloud']         = 1 if skills_cloud         else 0

    ligne['total_skills'] = (
        ligne['skills_python'] + ligne['skills_sql'] +
        ligne['skills_ml'] + ligne['skills_deep_learning'] +
        ligne['skills_cloud']
    )

    job_title_col = f'job_title_{job_title.lower()}'
    if job_title_col in ligne:
        ligne[job_title_col] = 1

    size_map = {'Startup': 'small', 'Medium': 'medium', 'MNC': 'large', 'Enterprise': 'large'}
    size_converti = size_map[company_size]
    size_col = f'company_size_{size_converti}'
    if size_col in ligne:
        ligne[size_col] = 1

    industry_col = f'company_industry_{company_industry.lower()}'
    if industry_col in ligne:
        ligne[industry_col] = 1

    country_col = f'country_{country.lower()}'
    if country_col in ligne:
        ligne[country_col] = 1

    remote_col = f'remote_type_{remote_type.lower()}'
    if remote_col in ligne:
        ligne[remote_col] = 1

    edu_col = f'education_level_{education_level.lower()}'
    if edu_col in ligne:
        ligne[edu_col] = 1

    df_input = pd.DataFrame([ligne])[colonnes_model]
    return df_input


# ═══════════════════════════════════════════════════════════════════
# INTERFACE — FORMULAIRE CENTRÉ
# ═══════════════════════════════════════════════════════════════════

st.title("Vous êtes Ingénieur Data ou IA? Estimez le Salaire que vous Méritez:")
st.markdown("Remplissez les caractéristiques du poste ci-dessous pour obtenir une estimation de salaire :")
st.divider()

if bundle is None:
    st.error(
        "⚠️ **Modèle introuvable.**  \n"
        "Le fichier `models/model_salary.pkl` n'existe pas encore.  \n"
        "Exécute d'abord le notebook `03_machine_learning.ipynb` pour générer ce fichier."
    )
    st.stop()

model             = bundle['model']
scaler            = bundle['scaler']
colonnes_model    = bundle['colonnes']
cols_a_normaliser = bundle['cols_a_normaliser']

# ── Section : Le poste ───────────────────────────────────────────────────────
st.subheader("🏢 Le poste :")

job_title = st.selectbox(
    "Intitulé du poste",
    options=['AI Engineer', 'Business Analyst', 'Data Analyst',
             'Data Engineer', 'Data Scientist', 'Machine Learning Engineer'],
    help="Le titre officiel du poste"
)

remote_type = st.selectbox(
    "Mode de travail",
    options=['Hybrid', 'Onsite', 'Remote'],
    help="Hybrid = mix présentiel/télétravail"
)

hiring_urgency = st.selectbox(
    "Urgence du recrutement",
    options=['Low', 'Medium', 'High'],
    index=1,
    help="High = poste à pourvoir très rapidement"
)

job_openings = st.slider(
    "Nombre de postes ouverts",
    min_value=1, max_value=9, value=3,
    help="Combien de personnes sont recrutées pour ce poste"
)

st.divider()

# ── Section : Profil candidat ─────────────────────────────────────────────────
st.subheader("👨🏻‍💼​👩🏻‍💼​Profil du candidat :")

years_experience = st.slider(
    "Années d'expérience",
    min_value=0, max_value=14, value=5
)

education_level = st.selectbox(
    "Niveau d'éducation requis",
    options=['Bachelor', 'Master', 'PhD'],
    index=1
)

st.markdown("**Compétences requises**")
skills_python        = st.checkbox("Python",         value=True)
skills_sql           = st.checkbox("SQL",            value=True)
skills_ml            = st.checkbox("Machine Learning")
skills_deep_learning = st.checkbox("Deep Learning")
skills_cloud         = st.checkbox("Cloud")

st.divider()

# ── Section : L'entreprise ───────────────────────────────────────────────────
st.subheader("🏭 L'entreprise :")

company_size = st.selectbox(
    "Taille de l'entreprise",
    options=['Startup', 'Medium', 'MNC', 'Enterprise'],
    help="MNC = Multinationale | Enterprise = Grande entreprise"
)

company_industry = st.selectbox(
    "Secteur d'activité",
    options=['E-commerce', 'Education', 'Finance',
             'Healthcare', 'Retail', 'Technology'],
    index=5
)

country = st.selectbox(
    "Pays du poste",
    options=['Australia', 'Canada', 'Germany',
             'India', 'Singapore', 'UK', 'USA'],
    index=6
)

st.divider()

# ── Section : Date de publication ─────────────────────────────────────────────
st.subheader("📅 Publication :")

job_posting_month = st.selectbox(
    "Mois de publication",
    options=list(range(1, 13)),
    index=0,
    format_func=lambda m: [
        'Janvier','Février','Mars','Avril','Mai','Juin',
        'Juillet','Août','Septembre','Octobre','Novembre','Décembre'
    ][m-1]
)

job_posting_year = st.selectbox(
    "Année de publication",
    options=[2020, 2021, 2022, 2023, 2024, 2025, 2026],
    index=4
)

st.divider()

predict_btn = st.button(
    "🔮 Prédire le salaire",
    type="primary",
    use_container_width=True
)


# ═══════════════════════════════════════════════════════════════════
# RÉSULTATS — RÉCAPITULATIF PUIS PRÉDICTION
# ═══════════════════════════════════════════════════════════════════

if predict_btn:

    # ── Récapitulatif des saisies ─────────────────────────────────────────────
    st.divider()
    st.subheader("📍 Récapitulatif de ta saisie")

    st.markdown("**🏢 Poste**")
    st.write(f"• Intitulé : **{job_title}**")
    st.write(f"• Mode de travail : {remote_type}")
    st.write(f"• Urgence : {hiring_urgency}")
    st.write(f"• Postes ouverts : {job_openings}")

    st.markdown("**👨‍💻 Profil**")
    st.write(f"• Expérience : **{years_experience} ans**")
    st.write(f"• Éducation : {education_level}")
    skills_actives = []
    if skills_python:        skills_actives.append("Python")
    if skills_sql:           skills_actives.append("SQL")
    if skills_ml:            skills_actives.append("ML")
    if skills_deep_learning: skills_actives.append("Deep Learning")
    if skills_cloud:         skills_actives.append("Cloud")
    st.write(f"• Compétences ({len(skills_actives)}) : {', '.join(skills_actives) if skills_actives else 'Aucune'}")

    st.markdown("**🏭 Entreprise**")
    st.write(f"• Taille : {company_size}")
    st.write(f"• Secteur : {company_industry}")
    st.write(f"• Pays : **{country}**")
    mois_noms = ['Jan','Fév','Mar','Avr','Mai','Juin','Juil','Août','Sep','Oct','Nov','Déc']
    st.write(f"• Publication : {mois_noms[job_posting_month-1]} {job_posting_year}")

    st.divider()

    # 1. Construire le vecteur d'entrée
    df_input = construire_vecteur(
        job_title         = job_title,
        company_size      = company_size,
        company_industry  = company_industry,
        country           = country,
        remote_type       = remote_type,
        education_level   = education_level,
        hiring_urgency    = hiring_urgency,
        years_experience  = years_experience,
        job_openings      = job_openings,
        job_posting_month = job_posting_month,
        job_posting_year  = job_posting_year,
        skills_python     = skills_python,
        skills_sql        = skills_sql,
        skills_ml         = skills_ml,
        skills_deep_learning = skills_deep_learning,
        skills_cloud      = skills_cloud,
        colonnes_model    = colonnes_model
    )

    # 2. Normaliser
    df_input_scaled = df_input.copy()
    df_input_scaled[cols_a_normaliser] = scaler.transform(df_input[cols_a_normaliser])

    # 3. Prédire
    salary_predit = model.predict(df_input_scaled)[0]

    # 4. Intervalle de confiance
    mae_modele  = 16_210
    salary_bas  = max(0, salary_predit - mae_modele)
    salary_haut = salary_predit + mae_modele

    # 5. Résultats
    st.subheader("🎯 Résultat de la prédiction")

    st.metric(
        label="💰 Salaire prédit",
        value=f"${salary_predit:,.0f}",
        help="Prédiction centrale du modèle Random Forest"
    )
    st.metric(
        label="📉 Borne basse (±MAE)",
        value=f"${salary_bas:,.0f}",
        help=f"Prédiction − {mae_modele:,} USD (MAE du modèle)"
    )
    st.metric(
        label="📈 Borne haute (±MAE)",
        value=f"${salary_haut:,.0f}",
        help=f"Prédiction + {mae_modele:,} USD (MAE du modèle)"
    )

    # 6. Interprétation
    st.info(
        f"**Interprétation :** Le modèle estime que ce poste serait rémunéré entre "
        f"**${salary_bas:,.0f}** et **${salary_haut:,.0f}** USD par an, "
        f"avec une valeur centrale de **${salary_predit:,.0f}** USD.  \n"
        f"L'erreur moyenne du modèle est de ±{mae_modele:,} USD (MAE sur les données de test)."
    )

    # 7. Jauge visuelle
    st.subheader("📊 Position dans la distribution des salaires")

    salary_min_dataset = 45_083
    salary_max_dataset = 202_523
    salary_moy_dataset = 113_403

    pct = (salary_predit - salary_min_dataset) / (salary_max_dataset - salary_min_dataset) * 100
    pct = max(0, min(100, pct))

    st.markdown(f"Le salaire prédit se situe à **{pct:.0f}%** de la plage des salaires du dataset.")
    st.progress(int(pct))
    st.caption(
        f"Min : ${salary_min_dataset/1000:.0f}k  |  "
        f"Moyenne : ${salary_moy_dataset:,}  |  "
        f"Max : ${salary_max_dataset/1000:.0f}k  |  "
        f"Ton estimation : **${salary_predit:,.0f} USD**"
    )

    # 8. Debug
    with st.expander("🔍 Voir le vecteur d'entrée envoyé au modèle (debug)"):
        st.dataframe(df_input_scaled.T.rename(columns={0: 'Valeur'}))

else:
    st.info("👆 Remplissez le formulaire ci-dessus et cliquez sur **'Prédire le salaire'** pour obtenir une estimation.")