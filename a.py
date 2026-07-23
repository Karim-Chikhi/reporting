import streamlit as st
import pandas as pd
import plotly.express as px

# 1. Configuration de la page (DOIT ÊTRE LA PREMIÈRE LIGNE)
st.set_page_config(page_title="Dashboard Attrition", page_icon="📉", layout="wide")

# 2. Nettoyage extrême du CSS (plus de conflits de blocs)
st.markdown("""
    <style>
    header {visibility: hidden;}
    #MainMenu {visibility: hidden;}
    .block-container { padding-top: 1rem !important; }
    .stApp { background-color: #f4f6f9; font-family: 'Arial', sans-serif; }
    
    /* Titres */
    .main-title { font-size: 2.2rem; font-weight: 800; color: #1e293b; margin-bottom: 0; line-height: 1.2; }
    .sub-title { font-size: 1rem; color: #64748b; margin-bottom: 1rem; }
    
    /* Style des KPIs */
    div[data-testid="metric-container"] {
        background-color: white;
        border-top: 4px solid #b91d1d;
        border-radius: 5px;
        padding: 1rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        text-align: center;
    }
    div[data-testid="metric-container"] label { color: #64748b !important; justify-content: center; font-size: 0.95rem !important; }
    div[data-testid="metric-container"] div[data-testid="stMetricValue"] { font-size: 2rem !important; font-weight: 800 !important; color: #b91d1d !important; justify-content: center; }
    
    /* Titres des graphiques */
    .chart-title { font-size: 1.1rem; font-weight: bold; color: #1e293b; margin-top: 1rem; margin-bottom: 0px;}
    .chart-subtitle { font-size: 0.85rem; color: #64748b; margin-bottom: 5px;}
    </style>
""", unsafe_allow_html=True)

# 3. Chargement et nettoyage des données
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("BankChurners.csv")
    except FileNotFoundError:
        st.error("Fichier 'BankChurners.csv' introuvable.")
        st.stop()

    cols_to_drop = ['CLIENTNUM']
    naive_bayes_cols = [col for col in df.columns if col.startswith('Naive_Bayes_')]
    df.drop(columns=cols_to_drop + naive_bayes_cols, inplace=True, errors='ignore')

    df['Age_Group'] = pd.cut(
        df['Customer_Age'],
        bins=[20, 35, 45, 55, 65, 80],
        labels=['20-35', '36-45', '46-55', '56-65', '66+']
    )
    return df

df = load_data()

# Palette et design commun des graphiques
color_map = {'Existing Customer': '#cbd5e0', 'Attrited Customer': '#b91d1d'}
layout_updates = dict(
    height=350, # HAUTEUR FIXE POUR ALIGNEMENT PARFAIT
    paper_bgcolor='white', # Fond blanc géré par Plotly (pas de bloc HTML vide)
    plot_bgcolor='white',
    margin=dict(l=20, r=20, t=30, b=20)
)

# 4. EN-TÊTE
st.markdown('<div class="main-title">Profil et Comportement d\'Attrition</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Analyse des facteurs clés liés au départ des clients</div>', unsafe_allow_html=True)

# 5. FILTRES HORIZONTAUX
col_f1, col_f2, col_f3 = st.columns(3)
with col_f1:
    selected_genders = st.multiselect("Genre :", options=['Tout'] + list(df['Gender'].unique()), default=['Tout'])
with col_f2:
    selected_ages = st.multiselect("Tranche d'âge :", options=['Tout'] + list(df['Age_Group'].dropna().unique()), default=['Tout'])
with col_f3:
    card_options = ['Tout'] + list(df['Card_Category'].unique()) if 'Card_Category' in df.columns else ['Tout']
    selected_cards = st.multiselect("Type de carte :", options=card_options, default=['Tout'])

# Filtrage
filtered_df = df.copy()
if 'Tout' not in selected_genders and selected_genders:
    filtered_df = filtered_df[filtered_df['Gender'].isin(selected_genders)]
if 'Tout' not in selected_ages and selected_ages:
    filtered_df = filtered_df[filtered_df['Age_Group'].isin(selected_ages)]
if 'Card_Category' in df.columns and 'Tout' not in selected_cards and selected_cards:
    filtered_df = filtered_df[filtered_df['Card_Category'].isin(selected_cards)]

if filtered_df.empty:
    st.warning("Aucune donnée ne correspond à ces filtres.")
    st.stop()

# 6. KPIs
churn_count = len(filtered_df[filtered_df['Attrition_Flag'] == 'Attrited Customer'])
total_count = len(filtered_df)
churn_rate = (churn_count / total_count * 100) if total_count > 0 else 0

avg_inactive_churn = filtered_df[filtered_df['Attrition_Flag'] == 'Attrited Customer']['Months_Inactive_12_mon'].mean()
avg_trans_churn = filtered_df[filtered_df['Attrition_Flag'] == 'Attrited Customer']['Total_Trans_Ct'].mean()
avg_trans_exist = filtered_df[filtered_df['Attrition_Flag'] == 'Existing Customer']['Total_Trans_Ct'].mean()
ratio_trans = (avg_trans_exist / avg_trans_churn) if avg_trans_churn and avg_trans_churn > 0 else 0
avg_util_churn = filtered_df[filtered_df['Attrition_Flag'] == 'Attrited Customer']['Avg_Utilization_Ratio'].mean() * 100

col1, col2, col3, col4 = st.columns(4)
col1.metric("Taux de résiliation", f"{churn_rate:.1f} %")
col2.metric("Inactivité avant départ (Mois)", f"{avg_inactive_churn:.1f}" if pd.notna(avg_inactive_churn) else "N/A")
col3.metric("Baisse du volume (Rapport)", f"÷ {ratio_trans:.1f}" if ratio_trans > 0 else "N/A")
col4.metric("Utilisation crédit (Partants)", f"{avg_util_churn:.1f} %" if pd.notna(avg_util_churn) else "N/A")

st.write("---") # Séparateur propre

# 7. GRILLE DE GRAPHIQUES (Alignement strict)
col_left, col_right = st.columns(2)

with col_left:
    # Ligne 1 - Gauche
    st.markdown('<div class="chart-title">Volume et Montant des transactions</div>', unsafe_allow_html=True)
    st.markdown('<div class="chart-subtitle">Analyse sur les 12 derniers mois</div>', unsafe_allow_html=True)
    fig_scatter = px.scatter(
        filtered_df, x="Total_Trans_Ct", y="Total_Trans_Amt", color="Attrition_Flag",
        color_discrete_map=color_map, opacity=0.6,
        labels={"Total_Trans_Ct": "Nombre de transactions", "Total_Trans_Amt": "Montant cumulé ($)"}
    )
    fig_scatter.update_layout(**layout_updates, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, title=""))
    fig_scatter.update_xaxes(showgrid=True, gridcolor='rgba(200,200,200,0.2)')
    fig_scatter.update_yaxes(showgrid=True, gridcolor='rgba(200,200,200,0.2)')
    st.plotly_chart(fig_scatter, use_container_width=True)

    # Ligne 2 - Gauche
    st.markdown('<div class="chart-title">Taux d\'utilisation du crédit</div>', unsafe_allow_html=True)
    st.markdown('<div class="chart-subtitle">Distribution du ratio d\'utilisation</div>', unsafe_allow_html=True)
    fig_box_util = px.box(
        filtered_df, x="Avg_Utilization_Ratio", y="Attrition_Flag", color="Attrition_Flag",
        color_discrete_map=color_map,
        labels={"Avg_Utilization_Ratio": "Taux d'utilisation", "Attrition_Flag": ""}
    )
    fig_box_util.update_layout(**layout_updates, showlegend=False)
    fig_box_util.update_xaxes(showgrid=True, gridcolor='rgba(200,200,200,0.2)')
    st.plotly_chart(fig_box_util, use_container_width=True)

with col_right:
    # Ligne 1 - Droite
    st.markdown('<div class="chart-title">Mois d\'inactivité</div>', unsafe_allow_html=True)
    st.markdown('<div class="chart-subtitle">Répartition des clients (12 derniers mois)</div>', unsafe_allow_html=True)
    inactivity_counts = filtered_df.groupby(['Months_Inactive_12_mon', 'Attrition_Flag']).size().reset_index(name='Count')
    fig_bar_inactive = px.bar(
        inactivity_counts, x="Months_Inactive_12_mon", y="Count", color="Attrition_Flag",
        barmode="group", color_discrete_map=color_map,
        labels={"Months_Inactive_12_mon": "Mois inactifs", "Count": "Nb. clients"}
    )
    fig_bar_inactive.update_layout(**layout_updates, showlegend=False)
    fig_bar_inactive.update_yaxes(showgrid=True, gridcolor='rgba(200,200,200,0.2)')
    st.plotly_chart(fig_bar_inactive, use_container_width=True)
    
    # Ligne 2 - Droite
    st.markdown('<div class="chart-title">Fréquence de contact</div>', unsafe_allow_html=True)
    st.markdown('<div class="chart-subtitle">Nombre de sollicitations avec la banque</div>', unsafe_allow_html=True)
    contact_counts = filtered_df.groupby(['Contacts_Count_12_mon', 'Attrition_Flag']).size().reset_index(name='Count')
    fig_bar_contacts = px.bar(
        contact_counts, x="Contacts_Count_12_mon", y="Count", color="Attrition_Flag",
        barmode="group", color_discrete_map=color_map,
        labels={"Contacts_Count_12_mon": "Nombre de contacts", "Count": "Nb. clients"}
    )
    fig_bar_contacts.update_layout(**layout_updates, showlegend=False)
    fig_bar_contacts.update_yaxes(showgrid=True, gridcolor='rgba(200,200,200,0.2)')
    st.plotly_chart(fig_bar_contacts, use_container_width=True)