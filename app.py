import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy import stats

st.set_page_config(page_title="Validation des hypothèses — BankChurners", layout="wide")

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

    df['Age_Group'] = pd.cut(
        df['Customer_Age'],
        bins=[25, 35, 45, 55, 65, 75],
        labels=['26-35', '36-45', '46-55', '56-65', '66-75']
    )

    return df


def lerp_color(c1, c2, t):
    a = tuple(int(c1[i:i + 2], 16) for i in (1, 3, 5))
    b = tuple(int(c2[i:i + 2], 16) for i in (1, 3, 5))
    r = [round(a[j] + (b[j] - a[j]) * t) for j in range(3)]
    return f'rgb({r[0]},{r[1]},{r[2]})'


ACCENT = '#E2001A'
ACCENT_SOFT = '#FF4D5E'
NEUTRAL_GREY = '#d9d9d9'
NEUTRAL_GREY_DARK = '#3d3d3d'
ED_TEXT_DARK = '#0a0a0a'
ED_TEXT_GREY = '#595959'

EDITORIAL_CHART_LAYOUT = dict(
    plot_bgcolor='white',
    paper_bgcolor='white',
    font=dict(color=ED_TEXT_DARK, family='Arial, Helvetica, sans-serif'),
)


def inject_editorial_css():
    st.markdown("""
    <style>
        [data-testid="stHeader"] { display: none; }
        [data-testid="stDecoration"] { display: none; }
        [data-testid="stToolbar"] { display: none; }

        .stApp { background-color: #0a0a0a; }

        section[data-testid="stSidebar"] {
            background-color: #0a0a0a;
            border-right: 1px solid #262626;
        }
        section[data-testid="stSidebar"] * {
            color: #e5e5e5 !important;
        }
        section[data-testid="stSidebar"] [aria-selected="true"] {
            background-color: rgba(226,0,26,0.18) !important;
        }

        .main .block-container,
        [data-testid="stAppViewBlockContainer"] {
            padding-top: 1.8rem !important;
            padding-bottom: 3rem;
            max-width: 1300px;
        }
        div[data-testid="stAppViewContainer"] { padding-top: 0 !important; }

        h1.ed-title {
            color: #ffffff;
            font-family: -apple-system, 'Helvetica Neue', Arial, sans-serif;
            font-weight: 800 !important;
            font-size: 2.4rem !important;
            letter-spacing: -0.01em;
            margin: 0 0 2px 0 !important;
        }
        .ed-subtitle {
            color: #b3b3b3;
            font-size: 1rem;
            margin: 0 0 18px 0;
        }
        .ed-meta {
            color: #808080;
            font-size: 0.78rem;
            text-align: right;
            margin-top: 6px;
        }
        .ed-rule {
            height: 4px;
            background-color: #E2001A;
            border-radius: 2px;
            margin: 4px 0 18px 0;
        }
        .ed-divider {
            height: 1px;
            background-color: #262626;
            margin: 8px 0 26px 0;
        }

        .kpi-card {
            background-color: #ffffff;
            border: 1px solid #e7e5e4;
            border-top: 3px solid #E2001A;
            border-radius: 8px;
            padding: 18px 14px;
            text-align: center;
            box-shadow: 0 1px 2px rgba(0,0,0,0.03), 0 6px 16px rgba(17,24,39,0.04);
        }
        .kpi-value {
            color: #E2001A;
            font-size: 1.9rem;
            font-weight: 800;
            line-height: 1.2;
        }
        .kpi-caption {
            color: #6b7280;
            font-size: 0.85rem;
            margin-top: 6px;
        }

        div[class*="st-key-edcard_"] {
            background-color: #ffffff;
            border: 1px solid #e7e5e4;
            border-radius: 12px;
            padding: 20px 22px 14px 22px;
            margin-bottom: 8px;
            box-shadow: 0 1px 2px rgba(0,0,0,0.03), 0 8px 20px rgba(17,24,39,0.045);
        }
        .ed-kicker {
            color: #E2001A;
            font-size: 0.72rem;
            font-weight: 700;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            margin-bottom: 4px;
        }
        .ed-statement {
            color: #6b7280;
            font-size: 0.88rem;
            font-style: italic;
            margin-bottom: 16px;
        }
        .ed-card-title {
            color: #111827;
            font-size: 1.08rem;
            font-weight: 700;
            line-height: 1.3;
            margin-bottom: 2px;
        }
        .ed-card-title::before {
            content: '';
            display: inline-block;
            width: 14px;
            height: 3px;
            border-radius: 2px;
            background-color: #E2001A;
            margin-right: 8px;
            margin-bottom: 3px;
        }
        .ed-card-subtitle {
            color: #6b7280;
            font-size: 0.8rem;
            margin-bottom: 6px;
        }
        .ed-verdict {
            color: #b3b3b3;
            font-size: 0.85rem;
            padding: 4px 2px 22px 2px;
        }
        .ed-verdict b { color: #ffffff; }
        .ed-verdict .accent { color: #E2001A; }
    </style>
    """, unsafe_allow_html=True)


def bar2(labels, values, highlight_first, value_fmt="{:,.0f}", height=260):
    colors = [ACCENT if highlight_first else NEUTRAL_GREY,
              NEUTRAL_GREY if highlight_first else ACCENT]
    fig = go.Figure(go.Bar(
        x=labels, y=values, marker_color=colors, width=0.48,
        marker_line=dict(width=0),
        text=[value_fmt.format(v) for v in values],
        textposition='outside',
        textfont=dict(size=17, color=ED_TEXT_DARK),
    ))
    y_max = max(values) * 1.32
    fig.update_layout(
        **EDITORIAL_CHART_LAYOUT,
        margin=dict(l=0, r=0, t=10, b=0), height=height, showlegend=False,
        xaxis=dict(showgrid=False, showline=False, tickfont=dict(size=13, color=ED_TEXT_GREY)),
        yaxis=dict(showgrid=False, showticklabels=False, zeroline=False, range=[0, y_max]),
    )
    return fig


def hbar2(labels, values, highlight_first, value_fmt="{:,.0f}", height=230):
    colors = [ACCENT if highlight_first else NEUTRAL_GREY,
              NEUTRAL_GREY if highlight_first else ACCENT]
    fig = go.Figure(go.Bar(
        y=labels, x=values, orientation='h', marker_color=colors, width=0.55,
        marker_line=dict(width=0),
        text=[value_fmt.format(v) for v in values],
        textposition='outside',
        textfont=dict(size=18, color=ED_TEXT_DARK),
    ))
    fig.update_layout(
        **EDITORIAL_CHART_LAYOUT,
        margin=dict(l=0, r=60, t=10, b=0), height=height, showlegend=False,
        yaxis=dict(showgrid=False, tickfont=dict(size=15, color=ED_TEXT_DARK)),
        xaxis=dict(showgrid=False, showticklabels=False, zeroline=False, range=[0, max(values) * 1.3]),
    )
    return fig


def dumbbell(label_a, val_a, label_b, val_b, value_fmt="{:.1f}", height=220):
    a_wins = val_a >= val_b
    color_a = ACCENT if a_wins else NEUTRAL_GREY_DARK
    color_b = NEUTRAL_GREY_DARK if a_wins else ACCENT
    span = abs(val_a - val_b) or max(val_a, val_b) * 0.1 or 1
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[val_a, val_b], y=[0, 0], mode='lines',
        line=dict(color='#d8d3ce', width=3), showlegend=False, hoverinfo='skip',
    ))
    fig.add_trace(go.Scatter(
        x=[val_a], y=[0], mode='markers+text',
        marker=dict(size=24, color=color_a, line=dict(width=3, color='white')),
        text=[f"<b>{label_a}</b><br>{value_fmt.format(val_a)}"], textposition='top center',
        textfont=dict(size=14, color=ED_TEXT_DARK), showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        x=[val_b], y=[0], mode='markers+text',
        marker=dict(size=24, color=color_b, line=dict(width=3, color='white')),
        text=[f"<b>{label_b}</b><br>{value_fmt.format(val_b)}"], textposition='bottom center',
        textfont=dict(size=14, color=ED_TEXT_DARK), showlegend=False,
    ))
    fig.update_layout(
        **EDITORIAL_CHART_LAYOUT,
        margin=dict(l=50, r=50, t=45, b=45), height=height, showlegend=False,
        xaxis=dict(showgrid=False, showticklabels=False, zeroline=False,
                    range=[min(val_a, val_b) - span * 0.9, max(val_a, val_b) + span * 0.9]),
        yaxis=dict(showgrid=False, showticklabels=False, zeroline=False, range=[-1, 1]),
    )
    return fig


def age_strip(series_a, series_b, name_a, name_b, y_title, height=340, sample_size=280):
    rng = np.random.default_rng(42)

    def sample(s):
        arr = s.to_numpy()
        if len(arr) > sample_size:
            idx = rng.choice(len(arr), sample_size, replace=False)
            return arr[idx]
        return arr

    vals_a, vals_b = sample(series_a), sample(series_b)
    jitter_a = rng.uniform(-0.18, 0.18, size=len(vals_a))
    jitter_b = rng.uniform(-0.18, 0.18, size=len(vals_b))

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=jitter_b, y=vals_b, mode='markers',
        marker=dict(size=6, color=NEUTRAL_GREY_DARK, opacity=0.45),
        showlegend=False, hoverinfo='skip',
    ))
    fig.add_trace(go.Scatter(
        x=1 + jitter_a, y=vals_a, mode='markers',
        marker=dict(size=6, color=ACCENT, opacity=0.45),
        showlegend=False, hoverinfo='skip',
    ))
    fig.add_trace(go.Scatter(
        x=[0], y=[series_b.mean()], mode='markers+text',
        marker=dict(size=16, color=NEUTRAL_GREY_DARK, symbol='diamond', line=dict(width=2, color='white')),
        text=[f"{series_b.mean():.1f}"], textposition='top center',
        textfont=dict(size=13, color=ED_TEXT_DARK), showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        x=[1], y=[series_a.mean()], mode='markers+text',
        marker=dict(size=16, color=ACCENT, symbol='diamond', line=dict(width=2, color='white')),
        text=[f"{series_a.mean():.1f}"], textposition='top center',
        textfont=dict(size=13, color=ED_TEXT_DARK), showlegend=False,
    ))
    fig.update_layout(
        **EDITORIAL_CHART_LAYOUT,
        margin=dict(l=0, r=0, t=30, b=0), height=height, showlegend=False,
        xaxis=dict(showgrid=False, tickvals=[0, 1], ticktext=[name_b, name_a], range=[-0.5, 1.5],
                    tickfont=dict(size=14, color=ED_TEXT_DARK), zeroline=False),
        yaxis=dict(showgrid=False, title=y_title, tickfont=dict(size=13, color=ED_TEXT_GREY)),
    )
    return fig


def signal_line(x_vals, y_vals, value_fmt="{:.1f}%", height=250):
    fig = go.Figure(go.Scatter(
        x=x_vals, y=y_vals, mode='lines+markers+text',
        line=dict(color=ACCENT, width=3),
        marker=dict(size=10, color=ACCENT),
        text=[value_fmt.format(v) for v in y_vals], textposition='top center',
        textfont=dict(size=13, color=ED_TEXT_DARK),
        cliponaxis=False,
    ))
    fig.update_layout(
        **EDITORIAL_CHART_LAYOUT,
        margin=dict(l=0, r=10, t=35, b=0), height=height, showlegend=False,
        xaxis=dict(showgrid=False, tickfont=dict(size=13, color=ED_TEXT_GREY), title=None),
        yaxis=dict(showgrid=False, showticklabels=False, zeroline=False, range=[0, max(y_vals) * 1.35]),
    )
    return fig


def ranked_lollipop(categories, values, value_fmt="{:.1f}%", height=300):
    order = list(categories)[::-1]
    vals = list(values)[::-1]
    n = len(vals)
    ranks = sorted(range(n), key=lambda i: vals[i])
    rank_of = {i: r for r, i in enumerate(ranks)}
    colors = [lerp_color('#e0e0e0', ACCENT, rank_of[i] / max(1, n - 1)) for i in range(n)]
    sizes = [12 + 8 * (rank_of[i] / max(1, n - 1)) for i in range(n)]
    fig = go.Figure()
    for cat, val in zip(order, vals):
        fig.add_trace(go.Scatter(
            x=[0, val], y=[cat, cat], mode='lines',
            line=dict(color='#e7e5e4', width=2), showlegend=False, hoverinfo='skip',
        ))
    fig.add_trace(go.Scatter(
        x=vals, y=order, mode='markers+text',
        marker=dict(size=sizes, color=colors, line=dict(width=2, color='white')),
        text=[value_fmt.format(v) for v in vals],
        textposition='middle right',
        textfont=dict(size=13, color=ED_TEXT_DARK),
        showlegend=False,
    ))
    fig.update_layout(
        **EDITORIAL_CHART_LAYOUT,
        margin=dict(l=0, r=40, t=10, b=0), height=height,
        xaxis=dict(showgrid=False, showticklabels=False, zeroline=False, range=[0, max(vals) * 1.35]),
        yaxis=dict(showgrid=False, tickfont=dict(size=13, color=ED_TEXT_DARK)),
    )
    return fig


def contact_hbar(categories, values, threshold=3, value_fmt="{:.1f}%", height=320):
    colors = [ACCENT if c >= threshold else NEUTRAL_GREY for c in categories]
    order = sorted(categories, reverse=True)
    fig = go.Figure(go.Bar(
        y=[str(c) for c in categories], x=values, orientation='h',
        marker_color=colors, width=0.6,
        text=[value_fmt.format(v) for v in values], textposition='outside',
        textfont=dict(size=13, color=ED_TEXT_DARK),
    ))
    fig.update_layout(
        **EDITORIAL_CHART_LAYOUT,
        margin=dict(l=0, r=40, t=10, b=0), height=height, showlegend=False,
        xaxis=dict(showgrid=False, showticklabels=False, zeroline=False, range=[0, max(values) * 1.3]),
        yaxis=dict(showgrid=False, tickfont=dict(size=13, color=ED_TEXT_DARK),
                    categoryorder='array', categoryarray=[str(c) for c in order]),
    )
    return fig


def predictor_hbar(labels, values, height=340):
    order = sorted(range(len(values)), key=lambda i: values[i], reverse=True)
    labels_sorted = [labels[i] for i in order]
    values_sorted = [values[i] for i in order]
    fig = go.Figure(go.Bar(
        y=labels_sorted, x=values_sorted, orientation='h',
        marker_color=ACCENT, width=0.6,
    ))
    fig.update_layout(
        **EDITORIAL_CHART_LAYOUT,
        margin=dict(l=0, r=10, t=10, b=0), height=height, showlegend=False,
        xaxis=dict(showgrid=False, showticklabels=False, zeroline=False, range=[0, max(values_sorted) * 1.1]),
        yaxis=dict(showgrid=False, tickfont=dict(size=13, color=ED_TEXT_DARK),
                    categoryorder='array', categoryarray=list(reversed(labels_sorted))),
    )
    return fig


def density_curve(series_a, series_b, name_a, name_b, x_title, height=340):
    lo = min(series_a.min(), series_b.min())
    hi = max(series_a.max(), series_b.max())
    xs = np.linspace(lo, hi, 200)
    kde_a = stats.gaussian_kde(series_a)(xs)
    kde_b = stats.gaussian_kde(series_b)(xs)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=xs, y=kde_b, mode='lines', name=name_b,
        line=dict(color=NEUTRAL_GREY_DARK, width=2),
        fill='tozeroy', fillcolor='rgba(61,61,61,0.15)',
    ))
    fig.add_trace(go.Scatter(
        x=xs, y=kde_a, mode='lines', name=name_a,
        line=dict(color=ACCENT, width=2.5),
        fill='tozeroy', fillcolor='rgba(226,0,26,0.20)',
    ))
    fig.update_layout(
        **EDITORIAL_CHART_LAYOUT,
        margin=dict(l=0, r=0, t=10, b=0), height=height,
        legend=dict(orientation='h', yanchor='bottom', y=1.03, xanchor='left', x=0, font=dict(size=13, color=ED_TEXT_GREY)),
        xaxis=dict(showgrid=False, title=x_title, tickfont=dict(size=13, color=ED_TEXT_GREY), zeroline=False),
        yaxis=dict(showgrid=False, showticklabels=False, title=None, zeroline=False),
    )
    return fig


_card_counter = {"n": 0}


def card(kicker="", statement=""):
    _card_counter["n"] += 1
    box = st.container(key=f"edcard_{_card_counter['n']}")
    with box:
        if kicker or statement:
            parts = []
            if kicker:
                parts.append(f"<p class='ed-kicker'>{kicker}</p>")
            if statement:
                parts.append(f"<p class='ed-statement'>{statement}</p>")
            st.markdown("".join(parts), unsafe_allow_html=True)
    return box


def card_chart_header(title, subtitle):
    st.markdown(
        f"<p class='ed-card-title'>{title}</p><p class='ed-card-subtitle'>{subtitle}</p>",
        unsafe_allow_html=True,
    )


def kpi_card(value, caption):
    st.markdown(
        f"<div class='kpi-card'><div class='kpi-value'>{value}</div>"
        f"<div class='kpi-caption'>{caption}</div></div>",
        unsafe_allow_html=True,
    )


def render_hypotheses(df):
    inject_editorial_css()

    head_col, meta_col = st.columns([3, 1])
    with head_col:
        st.markdown("<h1 class='ed-title'>Le vrai visage des hypothèses</h1>", unsafe_allow_html=True)
        st.markdown(
            "<p class='ed-subtitle'>Genre, inactivité et âge — ce que les données du portefeuille cartes de crédit confirment vraiment</p>",
            unsafe_allow_html=True,
        )
    with meta_col:
        st.markdown(f"<p class='ed-meta'>Base : {len(df):,} clients · source Kaggle BankChurners</p>", unsafe_allow_html=True)

    st.markdown("<div class='ed-rule'></div>", unsafe_allow_html=True)

    spend_m = df.loc[df["Gender"] == "M", "Total_Trans_Amt"]
    spend_f = df.loc[df["Gender"] == "F", "Total_Trans_Amt"]
    ct_m = df.loc[df["Gender"] == "M", "Total_Trans_Ct"]
    ct_f = df.loc[df["Gender"] == "F", "Total_Trans_Ct"]
    mean_f, mean_m = spend_f.mean(), spend_m.mean()
    ecart_h1 = (mean_f - mean_m) / mean_m * 100
    h1_femmes_plus = bool(mean_f > mean_m)
    kpi_h1_val = f"{abs(ecart_h1):.0f}%"
    kpi_h1_cap = "de dépenses en plus chez les femmes" if h1_femmes_plus else "de dépenses en plus chez les hommes"

    h1_effet_faible = bool(abs(ecart_h1) < 10)

    inact_att = df.loc[df["Attrition_Flag"] == "Attrited Customer", "Months_Inactive_12_mon"]
    inact_exist = df.loc[df["Attrition_Flag"] == "Existing Customer", "Months_Inactive_12_mon"]
    mean_att, mean_exist = inact_att.mean(), inact_exist.mean()
    h2_att_plus_inactifs = bool(mean_att > mean_exist)
    ratio_h2 = (mean_att / mean_exist) if h2_att_plus_inactifs else (mean_exist / mean_att)
    kpi_h2_val = f"×{ratio_h2:.2f}"
    kpi_h2_cap = "de mois d'inactivité en plus chez les partants" if h2_att_plus_inactifs else "de mois d'inactivité en plus chez les actifs"

    inactive_grp = df.groupby("Months_Inactive_12_mon")["Attrition_Flag"]
    churn_by_inactive = (inactive_grp.apply(lambda x: (x == "Attrited Customer").mean() * 100)).sort_index()

    churn_by_contact = df.groupby("Contacts_Count_12_mon")["Attrition_Flag"].apply(
        lambda x: (x == "Attrited Customer").mean() * 100
    ).sort_index()

    avg_contacts_active = df.loc[df["Attrition_Flag"] == "Existing Customer", "Contacts_Count_12_mon"].mean()
    avg_contacts_churn = df.loc[df["Attrition_Flag"] == "Attrited Customer", "Contacts_Count_12_mon"].mean()
    contacts_ratio = avg_contacts_churn / avg_contacts_active if avg_contacts_active > 0 else 0
    kpi_h4_val = f"×{contacts_ratio:.2f}"
    kpi_h4_cap = "plus de contacts (12 mois) chez les partants"

    df_h3 = df.dropna(subset=["Age_Group"]).copy()
    df_h3["Churn"] = (df_h3["Attrition_Flag"] == "Attrited Customer").astype(int)
    churn_by_age = df_h3.groupby("Age_Group", observed=True)["Churn"].mean() * 100

    plus_jeune_groupe = churn_by_age.index[0]
    plus_jeune_taux = churn_by_age.iloc[0]
    groupe_max_taux = churn_by_age.idxmax()
    jeunes_ont_le_taux_max = bool(groupe_max_taux == plus_jeune_groupe)
    kpi_h3_val = f"{plus_jeune_taux:.0f}%"
    kpi_h3_cap = f"de départs chez les {plus_jeune_groupe} ans"

    age_att = df_h3.loc[df_h3["Attrition_Flag"] == "Attrited Customer", "Customer_Age"]
    age_exist = df_h3.loc[df_h3["Attrition_Flag"] == "Existing Customer", "Customer_Age"]
    age_att_plus_ages = bool(age_att.mean() > age_exist.mean())

    predictors = {
        "Âge": "Customer_Age",
        "Produits détenus": "Total_Relationship_Count",
        "Mois inactifs": "Months_Inactive_12_mon",
        "Contacts (12 mois)": "Contacts_Count_12_mon",
        "Solde renouvelable": "Total_Revolving_Bal",
        "Nb. transactions": "Total_Trans_Ct",
    }
    corr_labels, corr_values = [], []
    for label, col in predictors.items():
        r, _ = stats.pointbiserialr(df_h3["Churn"], df_h3[col])
        corr_labels.append(label)
        corr_values.append(abs(r))
    corr_order = sorted(range(len(corr_values)), key=lambda i: corr_values[i])
    corr_labels = [corr_labels[i] for i in corr_order]
    corr_values = [corr_values[i] for i in corr_order]
    age_rank_last = corr_labels[0] == "Âge"

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        kpi_card(kpi_h1_val, kpi_h1_cap)
    with k2:
        kpi_card(kpi_h2_val, kpi_h2_cap)
    with k3:
        kpi_card(kpi_h3_val, kpi_h3_cap)
    with k4:
        kpi_card(kpi_h4_val, kpi_h4_cap)

    st.markdown("<div class='ed-divider'></div>", unsafe_allow_html=True)

    titre_h1a = "Les femmes dépensent plus que les hommes" if h1_femmes_plus else "Les hommes dépensent plus que les femmes"
    titre_h1b = "Elles réalisent aussi plus de transactions" if ct_f.mean() > ct_m.mean() else "Ils réalisent aussi plus de transactions"

    c1, c2 = st.columns(2)
    with c1:
        with card("Hypothèse H1", "« Les femmes dépensent plus »"):
            card_chart_header(titre_h1a, "Montant total moyen des transactions ($)")
            fig = hbar2(["Femme", "Homme"], [mean_f, mean_m], highlight_first=h1_femmes_plus, value_fmt="${:,.0f}")
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
    with c2:
        with card():
            card_chart_header(titre_h1b, "Nombre moyen de transactions")
            fig2 = dumbbell("Femme", ct_f.mean(), "Homme", ct_m.mean(), value_fmt="{:.1f}")
            st.plotly_chart(fig2, use_container_width=True, config={'displayModeBar': False})

    titre_h1c = "Un écart statistiquement significatif, mais ténu à l'œil nu" if h1_effet_faible else "Un écart net, visible à l'œil nu"
    with card():
        card_chart_header(titre_h1c, "Densité de clients par montant de transactions (courbes lissées)")
        fig3 = density_curve(spend_f, spend_m, "Femme", "Homme", x_title="Montant total des transactions ($)")
        st.plotly_chart(fig3, use_container_width=True, config={'displayModeBar': False})

    st.markdown("<div class='ed-divider'></div>", unsafe_allow_html=True)

    titre_h2a = "Les clients partis étaient inactifs plus longtemps" if h2_att_plus_inactifs else "Les clients partis étaient en réalité plus actifs"

    c1, c2 = st.columns(2)
    with c1:
        with card("Hypothèse H2", "« L'inactivité prolongée précède le départ »"):
            card_chart_header(titre_h2a, "Mois d'inactivité moyens (sur 12 mois)")
            fig = bar2(["Partants", "Actifs"], [mean_att, mean_exist], highlight_first=h2_att_plus_inactifs, value_fmt="{:.2f}")
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
    with c2:
        with card():
            card_chart_header("L'inactivité, un signal précoce du départ", "Taux d'attrition (%) par mois inactifs")
            fig2 = signal_line(churn_by_inactive.index.astype(str), churn_by_inactive.values)
            st.plotly_chart(fig2, use_container_width=True, config={'displayModeBar': False})

    with card():
        card_chart_header("Plus ils contactent, plus ils partent", "Taux d'attrition (%) selon le nombre de contacts")
        fig3 = contact_hbar(churn_by_contact.index.tolist(), churn_by_contact.values.tolist())
        st.plotly_chart(fig3, use_container_width=True, config={'displayModeBar': False})

    st.markdown("<div class='ed-divider'></div>", unsafe_allow_html=True)

    titre_h3a = f"Les {plus_jeune_groupe} ans partent le plus" if jeunes_ont_le_taux_max else f"Les {groupe_max_taux} ans partent le plus, pas les plus jeunes"
    titre_h3b = "Les clients partis sont en moyenne plus âgés" if age_att_plus_ages else "Les clients partis sont en moyenne plus jeunes"

    c1, c2 = st.columns(2)
    with c1:
        with card("Hypothèse H3", "« Le changement de banque touche surtout les jeunes »"):
            card_chart_header(titre_h3a, "Taux de départ (%) par tranche d'âge, classé")
            churn_sorted = churn_by_age.sort_values(ascending=False)
            fig = ranked_lollipop(churn_sorted.index.tolist(), churn_sorted.values.tolist())
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
    with c2:
        with card():
            card_chart_header(titre_h3b, "Âge individuel, partants vs actifs")
            fig2 = age_strip(age_att, age_exist, "Partants", "Actifs", y_title="Âge")
            st.plotly_chart(fig2, use_container_width=True, config={'displayModeBar': False})

    titre_h3c = ("L'âge est le plus mauvais prédicteur du départ" if age_rank_last
                 else f"{corr_labels[0]} est le plus mauvais prédicteur du départ, l'âge fait mieux")
    with card():
        card_chart_header(titre_h3c, "Corrélation absolue avec le départ, par variable")
        fig3 = predictor_hbar(corr_labels, corr_values)
        st.plotly_chart(fig3, use_container_width=True, config={'displayModeBar': False})


df = load_data()
render_hypotheses(df)
