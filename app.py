import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# --- CONFIGURATION DE LA PAGE ---
st.set_page_config(page_title="Le vrai visage de l'attrition bancaire", layout="wide")

# --- STYLE CSS PERSONNALISÉ (Palette Bleue) ---
st.markdown("""
<style>
    [data-testid="stHeader"] { display: none; }
    [data-testid="stDecoration"] { display: none; }
    [data-testid="stToolbar"] { display: none; }

    .stApp { background-color: #f4f7fb; }

    .main .block-container,
    [data-testid="stAppViewBlockContainer"] {
        padding-top: 1.5rem !important;
        padding-bottom: 2rem;
    }

    div[data-testid="stAppViewContainer"] { padding-top: 0 !important; }

    h1 {
        color: #0f172a;
        font-family: 'Arial', sans-serif;
        font-weight: bold !important;
        font-size: 2.5rem !important;
        margin-bottom: 0px !important;
        padding-bottom: 0px !important;
    }

    .subtitle {
        color: #64748b;
        font-size: 1.1rem;
        margin-top: 0px;
        margin-bottom: 20px;
    }

    div[data-testid="stMetric"], .stPlotlyChart {
        background-color: #ffffff;
        border-radius: 8px;
        border: 1px solid #dbe4f0;
        padding: 15px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }

    div[data-testid="stMetric"] {
        border-top: 4px solid #1d4ed8;
        text-align: center;
    }

    div[data-testid="stMetricValue"] {
        color: #1d4ed8;
        font-size: 2.2rem !important;
        font-weight: bold;
    }

    div[data-testid="stMetricLabel"] {
        color: #64748b;
        font-size: 0.9rem !important;
        margin-bottom: 5px;
    }

    h3 {
        color: #0f172a;
        font-size: 1.2rem !important;
        font-weight: 600 !important;
        margin-top: 0px !important;
        padding-top: 0px !important;
        margin-bottom: 5px !important;
    }

    .chart-subtitle {
        color: #64748b;
        font-size: 0.85rem;
        margin-bottom: 10px;
    }

    /* Onglets */
    .stTabs [data-baseweb="tab-list"] { gap: 4px; }
    .stTabs [data-baseweb="tab"] {
        background-color: #ffffff;
        border-radius: 8px 8px 0 0;
        padding: 10px 18px;
        border: 1px solid #dbe4f0;
        border-bottom: none;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1d4ed8 !important;
        color: white !important;
    }

    /* --- FILTRES --- */
    .filters-title {
        color: #0f172a;
        font-size: 0.95rem;
        font-weight: 700;
        letter-spacing: 0.03em;
        text-transform: uppercase;
        margin-bottom: 8px;
    }

    /* Cadre blanc autour de chaque multiselect, comme les autres cartes */
    div[data-testid="stMultiSelect"] {
        background-color: #ffffff;
        border: 1px solid #dbe4f0;
        border-radius: 8px;
        padding: 10px 12px 6px 12px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    div[data-testid="stMultiSelect"] label p {
        color: #0f172a !important;
        font-weight: 600 !important;
        font-size: 0.85rem !important;
    }
    /* Le champ de sélection lui-même */
    div[data-testid="stMultiSelect"] [data-baseweb="select"] > div {
        background-color: #f8fafc;
        border-color: #dbe4f0 !important;
        border-radius: 6px;
    }
    /* Les pastilles bleues au lieu du rouge par défaut */
    span[data-baseweb="tag"] {
        background-color: #1d4ed8 !important;
        border-radius: 6px !important;
    }
    span[data-baseweb="tag"] span {
        color: #ffffff !important;
    }
    span[data-baseweb="tag"] svg {
        fill: #ffffff !important;
    }
    span[data-baseweb="tag"]:hover {
        background-color: #1e3a8a !important;
    }
</style>
""", unsafe_allow_html=True)

# --- PALETTE DE COULEURS ---
COLOR_BLUE_DARK = '#1d4ed8'   # Bleu roi dominant (churners / signal fort)
COLOR_BLUE_MID = '#3b82f6'
COLOR_BLUE_LIGHT = '#93c5fd'  # Bleu clair (clients actifs / neutre)
COLOR_GREY_LIGHT = '#cbd5e1'
COLOR_GREY_TEXT = '#64748b'
BLUE_SCALE = ['#dbeafe', '#93c5fd', '#3b82f6', '#1d4ed8', '#1e3a8a']

CHART_LAYOUT = dict(
    plot_bgcolor='white',
    paper_bgcolor='white',
    font=dict(color='#0f172a'),
)

# --- CHARGEMENT ET NETTOYAGE DES DONNÉES ---
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("BankChurners.csv")
    except FileNotFoundError:
        st.error("Fichier 'BankChurners.csv' introuvable. Veuillez vous assurer qu'il est dans le même dossier.")
        st.stop()

    cols_to_drop = ['CLIENTNUM']
    naive_bayes_cols = [col for col in df.columns if col.startswith('Naive_Bayes_')]
    df.drop(columns=cols_to_drop + naive_bayes_cols, inplace=True, errors='ignore')

    # Tranches d'âge pour des graphes plus lisibles
    df['Age_Group'] = pd.cut(
        df['Customer_Age'],
        bins=[25, 35, 45, 55, 65, 75],
        labels=['26-35', '36-45', '46-55', '56-65', '66-75']
    )

    return df

df = load_data()

# --- EN-TÊTE DU DASHBOARD ---
st.markdown("<h1>Le vrai visage de l'attrition bancaire</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Qui, quand et pourquoi les clients quittent la banque — Portfolio de Cartes de Crédit</p>", unsafe_allow_html=True)

# --- FILTRES (EN HAUT) ---
def multiselect_avec_tout(label, options, key):
    """Multiselect avec une case 'Tout' pratique au lieu de devoir tout cocher à la main."""
    options = list(options)
    choix = st.multiselect(label, options=["Tout"] + options, default=["Tout"], key=key)
    if not choix or "Tout" in choix:
        return options
    return choix

st.markdown("<p class='filters-title'>🔍 Filtres</p>", unsafe_allow_html=True)
filt_col1, filt_col2, filt_col3 = st.columns(3)

with filt_col1:
    gender_filter = multiselect_avec_tout("Genre", df["Gender"].unique(), key="filter_gender")
with filt_col2:
    card_filter = multiselect_avec_tout("Catégorie de carte", df["Card_Category"].unique(), key="filter_card")
with filt_col3:
    education_filter = multiselect_avec_tout("Niveau d'éducation", df["Education_Level"].unique(), key="filter_education")

df_filtered = df.query(
    "Gender == @gender_filter & Card_Category == @card_filter & "
    "Education_Level == @education_filter"
)

st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)

# --- LIGNE 1 : KPIs GLOBAUX ---
total_clients = len(df_filtered)
churned_clients = len(df_filtered[df_filtered['Attrition_Flag'] == 'Attrited Customer'])
churn_rate = (churned_clients / total_clients) * 100 if total_clients > 0 else 0

avg_relationship_active = df_filtered[df_filtered['Attrition_Flag'] == 'Existing Customer']['Total_Relationship_Count'].mean()
avg_relationship_churn = df_filtered[df_filtered['Attrition_Flag'] == 'Attrited Customer']['Total_Relationship_Count'].mean()

avg_util_active = df_filtered[df_filtered['Attrition_Flag'] == 'Existing Customer']['Avg_Utilization_Ratio'].mean()
avg_util_churn = df_filtered[df_filtered['Attrition_Flag'] == 'Attrited Customer']['Avg_Utilization_Ratio'].mean()

avg_trans_active = df_filtered[df_filtered['Attrition_Flag'] == 'Existing Customer']['Total_Trans_Ct'].mean()
avg_trans_churn = df_filtered[df_filtered['Attrition_Flag'] == 'Attrited Customer']['Total_Trans_Ct'].mean()
trans_ratio = avg_trans_active / avg_trans_churn if avg_trans_churn > 0 else 0

col1, col2, col3, col4 = st.columns(4)
col1.metric(label="Taux d'attrition global", value=f"{churn_rate:.1f} %")
col2.metric(label="Produits détenus (partants vs actifs)", value=f"{avg_relationship_churn:.1f} vs {avg_relationship_active:.1f}")
col3.metric(label="Taux d'utilisation crédit (partants)", value=f"{avg_util_churn*100:.1f} %", delta=f"{(avg_util_churn-avg_util_active)*100:.1f} pts vs actifs", delta_color="inverse")
col4.metric(label="Fois plus de transactions chez les actifs", value=f"x {trans_ratio:.1f}")

st.markdown("<br>", unsafe_allow_html=True)

# --- ONGLETS THÉMATIQUES ---
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Vue d'ensemble",
    "👤 Profil client",
    "💳 Comportement financier",
    "🤝 Relation & engagement"
])

# ==================================================================
# TAB 1 : VUE D'ENSEMBLE
# ==================================================================
with tab1:
    top_col1, top_col2 = st.columns([1.3, 1])

    with top_col1:
        st.markdown("<h3>La chute des transactions, signal d'alarme</h3>", unsafe_allow_html=True)
        st.markdown("<p class='chart-subtitle'>Montant total vs Nombre total de transactions</p>", unsafe_allow_html=True)

        fig_scatter = px.scatter(
            df_filtered,
            x="Total_Trans_Ct",
            y="Total_Trans_Amt",
            color="Attrition_Flag",
            opacity=0.65,
            color_discrete_map={'Existing Customer': COLOR_BLUE_LIGHT, 'Attrited Customer': COLOR_BLUE_DARK},
            labels={"Total_Trans_Ct": "Nombre de transactions", "Total_Trans_Amt": "Montant des transactions ($)", "Attrition_Flag": "Statut"}
        )
        fig_scatter.update_layout(**CHART_LAYOUT, margin=dict(l=0, r=0, t=10, b=0),
                                   legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, title=None))
        fig_scatter.add_annotation(x=30, y=8500, text="Zone de haut risque", showarrow=False, font=dict(color=COLOR_BLUE_DARK, size=13))
        st.plotly_chart(fig_scatter, use_container_width=True, config={'displayModeBar': False})

    with top_col2:
        st.markdown("<h3>L'attrition frappe surtout entre 46 et 55 ans</h3>", unsafe_allow_html=True)
        st.markdown("<p class='chart-subtitle'>Taux d'attrition par tranche d'âge</p>", unsafe_allow_html=True)

        churn_by_age = df_filtered.groupby('Age_Group', observed=True)['Attrition_Flag'].apply(
            lambda x: (x == 'Attrited Customer').mean() * 100
        ).reset_index()
        churn_by_age.columns = ['Tranche', 'Taux']

        fig_age = px.bar(
            churn_by_age, x='Tranche', y='Taux',
            text=churn_by_age['Taux'].apply(lambda x: f"{x:.1f}%"),
            color='Taux', color_continuous_scale=BLUE_SCALE
        )
        fig_age.update_traces(textposition='outside')
        fig_age.update_layout(**CHART_LAYOUT, margin=dict(l=0, r=0, t=10, b=0),
                               xaxis_title=None, yaxis_title=None,
                               yaxis=dict(showticklabels=False, showgrid=False),
                               coloraxis_showscale=False, height=310)
        st.plotly_chart(fig_age, use_container_width=True, config={'displayModeBar': False})

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<h3>Répartition des clients par statut</h3>", unsafe_allow_html=True)
    st.markdown("<p class='chart-subtitle'>Vue globale du portefeuille filtré</p>", unsafe_allow_html=True)

    status_counts = df_filtered['Attrition_Flag'].value_counts().reset_index()
    status_counts.columns = ['Statut', 'Nombre']
    status_counts['Statut'] = status_counts['Statut'].replace({'Existing Customer': 'Client actif', 'Attrited Customer': 'Client parti'})

    fig_donut = px.pie(
        status_counts, names='Statut', values='Nombre', hole=0.6,
        color='Statut', color_discrete_map={'Client actif': COLOR_BLUE_LIGHT, 'Client parti': COLOR_BLUE_DARK}
    )
    fig_donut.update_traces(textinfo='percent+label')
    fig_donut.update_layout(**CHART_LAYOUT, margin=dict(l=0, r=0, t=10, b=0), showlegend=False, height=300)
    st.plotly_chart(fig_donut, use_container_width=True, config={'displayModeBar': False})

# ==================================================================
# TAB 2 : PROFIL CLIENT (Gender, Education, Marital, Income, Dependents)
# ==================================================================
with tab2:
    p_col1, p_col2 = st.columns(2)

    with p_col1:
        st.markdown("<h3>Le churn frappe plus les femmes</h3>", unsafe_allow_html=True)
        st.markdown("<p class='chart-subtitle'>Taux d'attrition selon le genre</p>", unsafe_allow_html=True)

        churn_by_gender = df_filtered.groupby('Gender')['Attrition_Flag'].apply(lambda x: (x == 'Attrited Customer').mean() * 100).reset_index()
        churn_by_gender.columns = ['Gender', 'Taux']
        churn_by_gender['Gender'] = churn_by_gender['Gender'].replace({'F': 'Femme', 'M': 'Homme'})

        fig_gender = px.bar(
            churn_by_gender, x='Gender', y='Taux',
            text=churn_by_gender['Taux'].apply(lambda x: f"{x:.1f}%"),
            color='Taux', color_continuous_scale=BLUE_SCALE
        )
        fig_gender.update_traces(textposition='outside')
        fig_gender.update_layout(**CHART_LAYOUT, margin=dict(l=0, r=0, t=10, b=0),
                                  xaxis_title=None, yaxis_title=None,
                                  yaxis=dict(showticklabels=False, showgrid=False),
                                  coloraxis_showscale=False, height=280)
        st.plotly_chart(fig_gender, use_container_width=True, config={'displayModeBar': False})

        st.markdown("<h3>Le statut marital influence peu le départ</h3>", unsafe_allow_html=True)
        st.markdown("<p class='chart-subtitle'>Taux d'attrition selon la situation familiale</p>", unsafe_allow_html=True)

        churn_by_marital = df_filtered.groupby('Marital_Status')['Attrition_Flag'].apply(lambda x: (x == 'Attrited Customer').mean() * 100).reset_index()
        churn_by_marital.columns = ['Statut', 'Taux']
        churn_by_marital = churn_by_marital.sort_values('Taux', ascending=True)

        fig_marital = px.bar(
            churn_by_marital, x='Taux', y='Statut', orientation='h',
            text=churn_by_marital['Taux'].apply(lambda x: f"{x:.1f}%"),
            color='Taux', color_continuous_scale=BLUE_SCALE
        )
        fig_marital.update_traces(textposition='outside')
        fig_marital.update_layout(**CHART_LAYOUT, margin=dict(l=0, r=20, t=10, b=0),
                                   xaxis=dict(showticklabels=False, showgrid=False, title=None),
                                   yaxis_title=None, coloraxis_showscale=False, height=280)
        st.plotly_chart(fig_marital, use_container_width=True, config={'displayModeBar': False})

    with p_col2:
        st.markdown("<h3>Les hauts revenus résistent mieux</h3>", unsafe_allow_html=True)
        st.markdown("<p class='chart-subtitle'>Taux d'attrition selon la tranche de revenu</p>", unsafe_allow_html=True)

        income_order = ['Less than $40K', '$40K - $60K', '$60K - $80K', '$80K - $120K', '$120K +', 'Unknown']
        churn_by_income = df_filtered.groupby('Income_Category')['Attrition_Flag'].apply(lambda x: (x == 'Attrited Customer').mean() * 100).reset_index()
        churn_by_income.columns = ['Revenu', 'Taux']
        churn_by_income['Revenu'] = pd.Categorical(churn_by_income['Revenu'], categories=[c for c in income_order if c in churn_by_income['Revenu'].values], ordered=True)
        churn_by_income = churn_by_income.sort_values('Revenu')

        fig_income = px.bar(
            churn_by_income, x='Revenu', y='Taux',
            text=churn_by_income['Taux'].apply(lambda x: f"{x:.1f}%"),
            color='Taux', color_continuous_scale=BLUE_SCALE
        )
        fig_income.update_traces(textposition='outside')
        fig_income.update_layout(**CHART_LAYOUT, margin=dict(l=0, r=0, t=10, b=0),
                                  xaxis_title=None, yaxis_title=None,
                                  yaxis=dict(showticklabels=False, showgrid=False),
                                  coloraxis_showscale=False, height=280)
        st.plotly_chart(fig_income, use_container_width=True, config={'displayModeBar': False})

        st.markdown("<h3>Le niveau d'éducation a peu d'impact</h3>", unsafe_allow_html=True)
        st.markdown("<p class='chart-subtitle'>Taux d'attrition selon le niveau d'études</p>", unsafe_allow_html=True)

        churn_by_edu = df_filtered.groupby('Education_Level')['Attrition_Flag'].apply(lambda x: (x == 'Attrited Customer').mean() * 100).reset_index()
        churn_by_edu.columns = ['Niveau', 'Taux']
        churn_by_edu = churn_by_edu.sort_values('Taux', ascending=True)

        fig_edu = px.bar(
            churn_by_edu, x='Taux', y='Niveau', orientation='h',
            text=churn_by_edu['Taux'].apply(lambda x: f"{x:.1f}%"),
            color='Taux', color_continuous_scale=BLUE_SCALE
        )
        fig_edu.update_traces(textposition='outside')
        fig_edu.update_layout(**CHART_LAYOUT, margin=dict(l=0, r=20, t=10, b=0),
                               xaxis=dict(showticklabels=False, showgrid=False, title=None),
                               yaxis_title=None, coloraxis_showscale=False, height=280)
        st.plotly_chart(fig_edu, use_container_width=True, config={'displayModeBar': False})

    st.markdown("<h3>Les foyers avec plus d'enfants partent-ils davantage ?</h3>", unsafe_allow_html=True)
    st.markdown("<p class='chart-subtitle'>Taux d'attrition selon le nombre de personnes à charge</p>", unsafe_allow_html=True)

    churn_by_dep = df_filtered.groupby('Dependent_count')['Attrition_Flag'].apply(lambda x: (x == 'Attrited Customer').mean() * 100).reset_index()
    churn_by_dep.columns = ['Personnes à charge', 'Taux']

    fig_dep = px.line(
        churn_by_dep, x='Personnes à charge', y='Taux', markers=True,
        text=churn_by_dep['Taux'].apply(lambda x: f"{x:.1f}%")
    )
    fig_dep.update_traces(line_color=COLOR_BLUE_DARK, marker=dict(size=10, color=COLOR_BLUE_DARK), textposition='top center')
    fig_dep.update_layout(**CHART_LAYOUT, margin=dict(l=0, r=0, t=10, b=0),
                           yaxis_title="Taux d'attrition (%)", xaxis_title="Nombre de personnes à charge", height=280)
    st.plotly_chart(fig_dep, use_container_width=True, config={'displayModeBar': False})

# ==================================================================
# TAB 3 : COMPORTEMENT FINANCIER
# ==================================================================
with tab3:
    f_col1, f_col2 = st.columns(2)

    with f_col1:
        st.markdown("<h3>Les partants n'utilisent presque plus leur crédit</h3>", unsafe_allow_html=True)
        st.markdown("<p class='chart-subtitle'>Distribution du taux d'utilisation du crédit</p>", unsafe_allow_html=True)

        fig_util = px.box(
            df_filtered, x='Attrition_Flag', y='Avg_Utilization_Ratio', color='Attrition_Flag',
            color_discrete_map={'Existing Customer': COLOR_BLUE_LIGHT, 'Attrited Customer': COLOR_BLUE_DARK},
            labels={'Attrition_Flag': 'Statut', 'Avg_Utilization_Ratio': "Taux d'utilisation"}
        )
        fig_util.update_layout(**CHART_LAYOUT, margin=dict(l=0, r=0, t=10, b=0), showlegend=False, height=320)
        st.plotly_chart(fig_util, use_container_width=True, config={'displayModeBar': False})

        st.markdown("<h3>La baisse d'activité précède le départ</h3>", unsafe_allow_html=True)
        st.markdown("<p class='chart-subtitle'>Changement du montant de transactions (Q4 vs Q1)</p>", unsafe_allow_html=True)

        avg_chng_active = df_filtered[df_filtered['Attrition_Flag'] == 'Existing Customer']['Total_Amt_Chng_Q4_Q1'].mean()
        avg_chng_churn = df_filtered[df_filtered['Attrition_Flag'] == 'Attrited Customer']['Total_Amt_Chng_Q4_Q1'].mean()
        avg_ct_chng_active = df_filtered[df_filtered['Attrition_Flag'] == 'Existing Customer']['Total_Ct_Chng_Q4_Q1'].mean()
        avg_ct_chng_churn = df_filtered[df_filtered['Attrition_Flag'] == 'Attrited Customer']['Total_Ct_Chng_Q4_Q1'].mean()

        df_chng = pd.DataFrame({
            'Statut': ['Actif', 'Partant', 'Actif', 'Partant'],
            'Indicateur': ['Montant', 'Montant', 'Nb. transactions', 'Nb. transactions'],
            'Changement': [avg_chng_active, avg_chng_churn, avg_ct_chng_active, avg_ct_chng_churn]
        })

        fig_chng = px.bar(
            df_chng, x='Indicateur', y='Changement', color='Statut', barmode='group',
            text=df_chng['Changement'].apply(lambda x: f"{x:.2f}"),
            color_discrete_map={'Partant': COLOR_BLUE_DARK, 'Actif': COLOR_BLUE_LIGHT}
        )
        fig_chng.update_traces(textposition='outside')
        fig_chng.update_layout(**CHART_LAYOUT, margin=dict(l=0, r=0, t=10, b=0),
                                xaxis_title=None, yaxis_title="Variation Q4/Q1", height=280,
                                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, title=None))
        st.plotly_chart(fig_chng, use_container_width=True, config={'displayModeBar': False})

    with f_col2:
        st.markdown("<h3>Plafond de crédit vs solde renouvelable</h3>", unsafe_allow_html=True)
        st.markdown("<p class='chart-subtitle'>Les partants gardent un solde revolving quasi nul</p>", unsafe_allow_html=True)

        fig_credit = px.scatter(
            df_filtered, x='Credit_Limit', y='Total_Revolving_Bal', color='Attrition_Flag',
            opacity=0.6,
            color_discrete_map={'Existing Customer': COLOR_BLUE_LIGHT, 'Attrited Customer': COLOR_BLUE_DARK},
            labels={'Credit_Limit': 'Plafond de crédit ($)', 'Total_Revolving_Bal': 'Solde renouvelable ($)', 'Attrition_Flag': 'Statut'}
        )
        fig_credit.update_layout(**CHART_LAYOUT, margin=dict(l=0, r=0, t=10, b=0), height=320,
                                  legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, title=None))
        st.plotly_chart(fig_credit, use_container_width=True, config={'displayModeBar': False})

        st.markdown("<h3>Crédit disponible non utilisé</h3>", unsafe_allow_html=True)
        st.markdown("<p class='chart-subtitle'>Moyenne du crédit disponible (Avg Open To Buy)</p>", unsafe_allow_html=True)

        avg_open_active = df_filtered[df_filtered['Attrition_Flag'] == 'Existing Customer']['Avg_Open_To_Buy'].mean()
        avg_open_churn = df_filtered[df_filtered['Attrition_Flag'] == 'Attrited Customer']['Avg_Open_To_Buy'].mean()
        df_open = pd.DataFrame({'Statut': ['Actif', 'Partant'], 'Crédit disponible': [avg_open_active, avg_open_churn]})

        fig_open = px.bar(
            df_open, x='Statut', y='Crédit disponible',
            text=df_open['Crédit disponible'].apply(lambda x: f"${x:,.0f}"),
            color='Statut', color_discrete_map={'Partant': COLOR_BLUE_DARK, 'Actif': COLOR_BLUE_LIGHT}
        )
        fig_open.update_traces(textposition='outside')
        fig_open.update_layout(**CHART_LAYOUT, margin=dict(l=0, r=0, t=10, b=0),
                                xaxis_title=None, yaxis_title=None,
                                yaxis=dict(showticklabels=False, showgrid=False),
                                showlegend=False, height=280)
        st.plotly_chart(fig_open, use_container_width=True, config={'displayModeBar': False})

# ==================================================================
# TAB 4 : RELATION & ENGAGEMENT
# ==================================================================
with tab4:
    r_col1, r_col2 = st.columns(2)

    with r_col1:
        st.markdown("<h3>Moins de produits, plus de risque de départ</h3>", unsafe_allow_html=True)
        st.markdown("<p class='chart-subtitle'>Taux d'attrition selon le nombre de produits détenus</p>", unsafe_allow_html=True)

        churn_by_rel = df_filtered.groupby('Total_Relationship_Count')['Attrition_Flag'].apply(lambda x: (x == 'Attrited Customer').mean() * 100).reset_index()
        churn_by_rel.columns = ['Produits', 'Taux']

        fig_rel = px.bar(
            churn_by_rel, x='Produits', y='Taux',
            text=churn_by_rel['Taux'].apply(lambda x: f"{x:.1f}%"),
            color='Taux', color_continuous_scale=BLUE_SCALE
        )
        fig_rel.update_traces(textposition='outside')
        fig_rel.update_layout(**CHART_LAYOUT, margin=dict(l=0, r=0, t=10, b=0),
                               xaxis_title="Nombre de produits", yaxis_title=None,
                               yaxis=dict(showticklabels=False, showgrid=False),
                               coloraxis_showscale=False, height=300)
        st.plotly_chart(fig_rel, use_container_width=True, config={'displayModeBar': False})

        st.markdown("<h3>L'ancienneté ne protège pas du départ</h3>", unsafe_allow_html=True)
        st.markdown("<p class='chart-subtitle'>Ancienneté moyenne (mois) selon le statut</p>", unsafe_allow_html=True)

        avg_tenure_active = df_filtered[df_filtered['Attrition_Flag'] == 'Existing Customer']['Months_on_book'].mean()
        avg_tenure_churn = df_filtered[df_filtered['Attrition_Flag'] == 'Attrited Customer']['Months_on_book'].mean()
        df_tenure = pd.DataFrame({'Statut': ['Actif', 'Partant'], 'Ancienneté': [avg_tenure_active, avg_tenure_churn]})

        fig_tenure = px.bar(
            df_tenure, x='Statut', y='Ancienneté',
            text=df_tenure['Ancienneté'].apply(lambda x: f"{x:.1f} mois"),
            color='Statut', color_discrete_map={'Partant': COLOR_BLUE_DARK, 'Actif': COLOR_BLUE_LIGHT}
        )
        fig_tenure.update_traces(textposition='outside')
        fig_tenure.update_layout(**CHART_LAYOUT, margin=dict(l=0, r=0, t=10, b=0),
                                  xaxis_title=None, yaxis_title=None,
                                  yaxis=dict(showticklabels=False, showgrid=False),
                                  showlegend=False, height=280)
        st.plotly_chart(fig_tenure, use_container_width=True, config={'displayModeBar': False})

    with r_col2:
        st.markdown("<h3>Plus ils contactent, plus ils partent</h3>", unsafe_allow_html=True)
        st.markdown("<p class='chart-subtitle'>Taux d'attrition selon le nombre de contacts (12 mois)</p>", unsafe_allow_html=True)

        churn_by_contact = df_filtered.groupby('Contacts_Count_12_mon')['Attrition_Flag'].apply(lambda x: (x == 'Attrited Customer').mean() * 100).reset_index()
        churn_by_contact.columns = ['Contacts', 'Taux']

        fig_contact = go.Figure()
        colors = [COLOR_BLUE_DARK if c >= 3 else COLOR_BLUE_LIGHT for c in churn_by_contact['Contacts']]
        fig_contact.add_trace(go.Bar(
            x=churn_by_contact['Taux'], y=churn_by_contact['Contacts'].astype(str), orientation='h',
            marker_color=colors, text=churn_by_contact['Taux'].apply(lambda x: f"{x:.1f}%"), textposition='outside'
        ))
        fig_contact.update_layout(**CHART_LAYOUT, margin=dict(l=0, r=20, t=10, b=0),
                                   xaxis=dict(showticklabels=False, showgrid=False, title=None),
                                   yaxis=dict(title="Nb de contacts", categoryorder='category descending'), height=300)
        st.plotly_chart(fig_contact, use_container_width=True, config={'displayModeBar': False})

        st.markdown("<h3>L'inactivité, un signal précoce du départ</h3>", unsafe_allow_html=True)
        st.markdown("<p class='chart-subtitle'>Taux d'attrition selon les mois d'inactivité (12 mois)</p>", unsafe_allow_html=True)

        churn_by_inactive = df_filtered.groupby('Months_Inactive_12_mon')['Attrition_Flag'].apply(lambda x: (x == 'Attrited Customer').mean() * 100).reset_index()
        churn_by_inactive.columns = ['Mois inactifs', 'Taux']

        fig_inactive = px.line(
            churn_by_inactive, x='Mois inactifs', y='Taux', markers=True,
            text=churn_by_inactive['Taux'].apply(lambda x: f"{x:.1f}%")
        )
        fig_inactive.update_traces(line_color=COLOR_BLUE_DARK, marker=dict(size=10, color=COLOR_BLUE_DARK), textposition='top center')
        fig_inactive.update_layout(**CHART_LAYOUT, margin=dict(l=0, r=0, t=10, b=0),
                                    yaxis_title="Taux d'attrition (%)", height=280)
        st.plotly_chart(fig_inactive, use_container_width=True, config={'displayModeBar': False})