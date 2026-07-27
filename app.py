import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import streamlit.components.v1 as components
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

        html, body { height: 100%; overflow: hidden !important; }
        .stApp {
            background-color: #0a0a0a;
            height: 100vh;
            overflow: hidden;
        }
        [data-testid="stAppViewContainer"],
        [data-testid="stAppViewBlockContainer"],
        [data-testid="stMain"] {
            overflow: hidden !important;
            height: 100vh;
        }

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

        .block-container,
        [data-testid="stMainBlockContainer"],
        [data-testid="stAppViewBlockContainer"] {
            padding-top: 0.6rem !important;
            padding-bottom: 0.3rem !important;
            padding-left: 1.5rem !important;
            padding-right: 1.5rem !important;
            max-width: 1650px;
        }
        div[data-testid="stAppViewContainer"] { padding-top: 0 !important; }
        div[data-testid="stVerticalBlock"] { gap: 0.3rem !important; }
        div[data-testid="stVerticalBlockBorderWrapper"] { gap: 0.3rem !important; }
        div[data-testid="stElementContainer"],
        div[data-testid="stElementContainer"] > div,
        div[data-testid="stMarkdown"],
        div[data-testid="stMarkdown"] > div,
        div[data-testid="stMarkdownContainer"] {
            height: auto !important;
            min-height: 0 !important;
            transition: none !important;
        }
        /* Streamlit's flex-row markdown wrapper miscomputes cross-size (height) for
           wrapped/multi-line text; forcing block layout sidesteps that entirely. */
        div[data-testid="stMarkdown"] > div {
            display: block !important;
        }

        h1.ed-title {
            color: #ffffff;
            font-family: -apple-system, 'Helvetica Neue', Arial, sans-serif;
            font-weight: 800 !important;
            font-size: 1.5rem !important;
            letter-spacing: -0.01em;
            margin: 0 0 1px 0 !important;
        }
        .ed-subtitle {
            color: #b3b3b3 !important;
            font-size: 0.72rem !important;
            line-height: 1.3 !important;
            margin: 0 0 6px 0 !important;
        }
        .ed-meta {
            color: #808080 !important;
            font-size: 0.62rem !important;
            line-height: 1.3 !important;
            text-align: right;
            margin-top: 4px !important;
        }
        .ed-rule {
            height: 3px;
            background-color: #E2001A;
            border-radius: 2px;
            margin: 2px 0 8px 0;
        }
        .ed-divider {
            height: 1px;
            background-color: #262626;
            margin: 2px 0 4px 0;
        }

        .kpi-card {
            background-color: #ffffff;
            border: 1px solid #e7e5e4;
            border-top: 3px solid #E2001A;
            border-radius: 8px;
            padding: 8px 10px;
            text-align: center;
            box-shadow: 0 1px 2px rgba(0,0,0,0.03), 0 6px 16px rgba(17,24,39,0.04);
        }
        .kpi-value {
            color: #E2001A;
            font-size: 1.25rem;
            font-weight: 800;
            line-height: 1.15;
        }
        .kpi-caption {
            color: #6b7280;
            font-size: 0.68rem;
            margin-top: 2px;
        }

        div[class*="st-key-edcard_"] {
            background-color: #ffffff;
            border: 1px solid #e7e5e4;
            border-radius: 10px;
            padding: 8px 14px 4px 14px;
            margin-bottom: 3px;
            box-shadow: 0 1px 2px rgba(0,0,0,0.03), 0 8px 20px rgba(17,24,39,0.045);
        }
        .ed-kicker {
            color: #E2001A !important;
            font-size: 0.65rem !important;
            line-height: 1.3 !important;
            font-weight: 700;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            margin-bottom: 4px !important;
        }
        .ed-statement {
            color: #6b7280 !important;
            font-size: 0.7rem !important;
            line-height: 1.3 !important;
            font-style: italic;
            margin-bottom: 16px !important;
        }
        .ed-card-title {
            color: #111827 !important;
            font-size: 0.78rem !important;
            line-height: 1.3 !important;
            font-weight: 700;
            margin-bottom: 0 !important;
        }
        .ed-card-title::before {
            content: '';
            display: inline-block;
            width: 11px;
            height: 2px;
            border-radius: 2px;
            background-color: #E2001A;
            margin-right: 6px;
            margin-bottom: 2px;
        }
        .ed-card-subtitle {
            color: #6b7280 !important;
            font-size: 0.6rem !important;
            line-height: 1.3 !important;
            margin-bottom: 2px !important;
        }
        .ed-verdict {
            color: #b3b3b3 !important;
            font-size: 0.7rem !important;
            line-height: 1.3 !important;
            padding: 4px 2px 22px 2px;
        }
        .ed-verdict b { color: #ffffff; }
        .ed-verdict .accent { color: #E2001A; }
    </style>
    """, unsafe_allow_html=True)


def hbar2(labels, values, highlight_first, value_fmt="{:,.0f}", height=150):
    colors = [ACCENT if highlight_first else NEUTRAL_GREY,
              NEUTRAL_GREY if highlight_first else ACCENT]
    fig = go.Figure(go.Bar(
        y=labels, x=values, orientation='h', marker_color=colors, width=0.55,
        marker_line=dict(width=0),
        text=[value_fmt.format(v) for v in values],
        textposition='outside',
        textfont=dict(size=13, color=ED_TEXT_DARK),
    ))
    fig.update_layout(
        **EDITORIAL_CHART_LAYOUT,
        margin=dict(l=0, r=50, t=6, b=0), height=height, showlegend=False,
        yaxis=dict(showgrid=False, tickfont=dict(size=11, color=ED_TEXT_DARK)),
        xaxis=dict(showgrid=False, showticklabels=False, zeroline=False, range=[0, max(values) * 1.3]),
    )
    return fig


def pie2(labels, values, highlight_first, value_fmt="{:.1f}", height=150):
    colors = [ACCENT if highlight_first else NEUTRAL_GREY,
              NEUTRAL_GREY if highlight_first else ACCENT]
    fig = go.Figure(go.Pie(
        labels=labels, values=values, hole=0.45, sort=False,
        marker=dict(colors=colors, line=dict(color='white', width=2)),
        text=[value_fmt.format(v) for v in values],
        textinfo='label+text', textposition='outside',
        textfont=dict(size=11, color=ED_TEXT_DARK),
    ))
    fig.update_layout(
        **EDITORIAL_CHART_LAYOUT,
        margin=dict(l=10, r=10, t=8, b=8), height=height, showlegend=False,
    )
    return fig


def signal_line(x_vals, y_vals, value_fmt="{:.1f}%", height=150):
    fig = go.Figure(go.Scatter(
        x=x_vals, y=y_vals, mode='lines+markers+text',
        line=dict(color=ACCENT, width=3),
        marker=dict(size=8, color=ACCENT),
        text=[value_fmt.format(v) for v in y_vals], textposition='top center',
        textfont=dict(size=10, color=ED_TEXT_DARK),
        cliponaxis=False,
    ))
    fig.update_layout(
        **EDITORIAL_CHART_LAYOUT,
        margin=dict(l=0, r=10, t=18, b=0), height=height, showlegend=False,
        xaxis=dict(showgrid=False, tickfont=dict(size=11, color=ED_TEXT_GREY), title=None),
        yaxis=dict(showgrid=False, showticklabels=False, zeroline=False, range=[0, max(y_vals) * 1.35]),
    )
    return fig


def ranked_lollipop(categories, values, value_fmt="{:.1f}%", height=150):
    order = list(categories)[::-1]
    vals = list(values)[::-1]
    n = len(vals)
    ranks = sorted(range(n), key=lambda i: vals[i])
    rank_of = {i: r for r, i in enumerate(ranks)}
    colors = [lerp_color('#e0e0e0', ACCENT, rank_of[i] / max(1, n - 1)) for i in range(n)]
    sizes = [10 + 6 * (rank_of[i] / max(1, n - 1)) for i in range(n)]
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
        textfont=dict(size=11, color=ED_TEXT_DARK),
        showlegend=False,
    ))
    fig.update_layout(
        **EDITORIAL_CHART_LAYOUT,
        margin=dict(l=0, r=40, t=6, b=0), height=height,
        xaxis=dict(showgrid=False, showticklabels=False, zeroline=False, range=[0, max(vals) * 1.35]),
        yaxis=dict(showgrid=False, tickfont=dict(size=11, color=ED_TEXT_DARK)),
    )
    return fig


def contact_hbar(categories, values, threshold=3, value_fmt="{:.1f}%", height=150):
    colors = [ACCENT if c >= threshold else NEUTRAL_GREY for c in categories]
    order = sorted(categories, reverse=True)
    fig = go.Figure(go.Bar(
        y=[str(c) for c in categories], x=values, orientation='h',
        marker_color=colors, width=0.65,
        text=[value_fmt.format(v) for v in values], textposition='outside',
        textfont=dict(size=10, color=ED_TEXT_DARK),
    ))
    fig.update_layout(
        **EDITORIAL_CHART_LAYOUT,
        margin=dict(l=0, r=40, t=6, b=0), height=height, showlegend=False,
        xaxis=dict(showgrid=False, showticklabels=False, zeroline=False, range=[0, max(values) * 1.3]),
        yaxis=dict(showgrid=False, tickfont=dict(size=10, color=ED_TEXT_DARK),
                    categoryorder='array', categoryarray=[str(c) for c in order]),
    )
    return fig


def predictor_hbar(labels, values, height=150):
    order = sorted(range(len(values)), key=lambda i: values[i], reverse=True)
    labels_sorted = [labels[i] for i in order]
    values_sorted = [values[i] for i in order]
    fig = go.Figure(go.Bar(
        y=labels_sorted, x=values_sorted, orientation='h',
        marker_color=ACCENT, width=0.65,
    ))
    fig.update_layout(
        **EDITORIAL_CHART_LAYOUT,
        margin=dict(l=0, r=10, t=6, b=0), height=height, showlegend=False,
        xaxis=dict(showgrid=False, showticklabels=False, zeroline=False, range=[0, max(values_sorted) * 1.1]),
        yaxis=dict(showgrid=False, tickfont=dict(size=11, color=ED_TEXT_DARK),
                    categoryorder='array', categoryarray=list(reversed(labels_sorted))),
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
        f"<div><p class='ed-card-title'>{title}</p><p class='ed-card-subtitle'>{subtitle}</p></div>",
        unsafe_allow_html=True,
    )


def inject_autoscale_js():
    components.html(
        """
        <script>
        function fixHeaderHeights() {
            const doc = window.parent.document;
            const containers = doc.querySelectorAll('[data-testid="stElementContainer"]');
            containers.forEach(c => {
                if (c.querySelector('.ed-card-title, .ed-card-subtitle, .kpi-value, .kpi-caption, .ed-title, .ed-subtitle, .ed-meta')) {
                    const child = c.firstElementChild;
                    if (child && child.scrollHeight > 0) {
                        c.style.setProperty('height', child.scrollHeight + 'px', 'important');
                    }
                }
            });
        }
        function fitSlide() {
            try {
                fixHeaderHeights();
                const el = window.parent.document.querySelector('.block-container');
                if (!el) return;
                el.style.transform = 'none';
                const natural = el.scrollHeight;
                const vh = window.parent.innerHeight;
                const scale = Math.min(1, (vh - 6) / natural);
                el.style.transformOrigin = 'top center';
                el.style.transform = 'scale(' + scale + ')';
            } catch (e) {}
        }
        fitSlide();
        window.parent.addEventListener('resize', fitSlide);
        setTimeout(fitSlide, 150);
        setTimeout(fitSlide, 500);
        setTimeout(fitSlide, 1200);
        setInterval(fixHeaderHeights, 400);
        </script>
        """,
        height=0,
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
        st.markdown(
            f"<p class='ed-meta'>Base : {len(df):,} clients · source Kaggle BankChurners"
            f"<br>Analyse sur 12 mois d'activité</p>",
            unsafe_allow_html=True,
        )

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
    ct_f_plus = bool(ct_f.mean() > ct_m.mean())

    c1, c2 = st.columns(2)
    with c1:
        with card():
            card_chart_header(titre_h1a, "Montant total moyen des transactions ($)")
            fig = hbar2(["Femme", "Homme"], [mean_f, mean_m], highlight_first=h1_femmes_plus, value_fmt="${:,.0f}")
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
    with c2:
        with card():
            card_chart_header(titre_h1b, "Nombre moyen de transactions")
            fig2 = pie2(["Femme", "Homme"], [ct_f.mean(), ct_m.mean()], highlight_first=ct_f_plus, value_fmt="{:.1f}")
            st.plotly_chart(fig2, use_container_width=True, config={'displayModeBar': False})

    st.markdown("<div class='ed-divider'></div>", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        with card():
            card_chart_header("L'inactivité, un signal précoce du départ", "Taux d'attrition (%) par mois inactifs")
            fig = signal_line(churn_by_inactive.index.astype(str), churn_by_inactive.values)
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
    with c2:
        with card():
            card_chart_header("Plus ils contactent, plus ils partent", "Taux d'attrition (%) selon le nombre de contacts")
            fig2 = contact_hbar(churn_by_contact.index.tolist(), churn_by_contact.values.tolist())
            st.plotly_chart(fig2, use_container_width=True, config={'displayModeBar': False})

    st.markdown("<div class='ed-divider'></div>", unsafe_allow_html=True)

    titre_h3a = f"Les {plus_jeune_groupe} ans partent le plus" if jeunes_ont_le_taux_max else f"Les {groupe_max_taux} ans partent le plus, pas les plus jeunes"
    titre_h3c = ("L'âge est le plus mauvais prédicteur du départ" if age_rank_last
                 else f"{corr_labels[0]} est le plus mauvais prédicteur du départ, l'âge fait mieux")

    c1, c2 = st.columns(2)
    with c1:
        with card():
            card_chart_header(titre_h3a, "Taux de départ (%) par tranche d'âge, classé")
            churn_sorted = churn_by_age.sort_values(ascending=False)
            fig = ranked_lollipop(churn_sorted.index.tolist(), churn_sorted.values.tolist())
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
    with c2:
        with card():
            card_chart_header(titre_h3c, "Corrélation absolue avec le départ, par variable")
            fig2 = predictor_hbar(corr_labels, corr_values)
            st.plotly_chart(fig2, use_container_width=True, config={'displayModeBar': False})

    inject_autoscale_js()


df = load_data()
render_hypotheses(df)
