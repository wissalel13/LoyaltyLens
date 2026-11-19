import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = next((p for p in (HERE, HERE.parent) if (p / 'models').exists() and (p / 'data').exists()), HERE)
for _p in (HERE, HERE / 'src', ROOT, ROOT / 'src'):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.metrics import average_precision_score, roc_auc_score

import churn_pipeline as cp

st.set_page_config(page_title='LoyaltyLens', page_icon='🔎', layout='wide')

# ----------------------------------------------------------------------------- palette
NAVY, TEAL, CORAL, AMBER, GREY = '#1B4965', '#2A9D8F', '#E76F51', '#F4A261', '#98A2B3'
TIER_ORDER = ['Élevé', 'Moyen', 'Faible']
TIER_COLORS = {'Élevé': CORAL, 'Moyen': AMBER, 'Faible': TEAL}
YES_NO = lambda x: {'Yes': 'oui', 'No': 'non'}.get(x, x)
FR_OPTION = lambda x: cp.FR_MODALITY.get(x, x)

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, .stApp, .stMarkdown { font-family: 'Inter', 'Segoe UI', sans-serif; }
.block-container { padding-top: 1.4rem; max-width: 1280px; }
header[data-testid="stHeader"] { background: transparent; }
.stDeployButton, #MainMenu, footer { display: none; }
.hero { background: linear-gradient(120deg, #1B4965 0%, #2A9D8F 60%, #62B6CB 100%); border-radius: 20px; padding: 30px 36px; color: #fff; margin-bottom: 18px; box-shadow: 0 10px 30px rgba(27,73,101,.25); }
.hero-badge { display: inline-block; background: rgba(255,255,255,.18); border-radius: 999px; padding: 4px 14px; font-size: .78rem; font-weight: 600; letter-spacing: .06em; text-transform: uppercase; }
.hero h1 { color: #fff; font-size: 3rem; font-weight: 800; margin: 10px 0 4px 0; padding: 0; line-height: 1.1; }
.hero p { font-size: 1.15rem; margin: 0 0 14px 0; opacity: .95; }
.hero-chips span { display: inline-block; background: rgba(255,255,255,.16); border-radius: 10px; padding: 5px 12px; margin: 0 8px 6px 0; font-size: .85rem; font-weight: 500; }
.stTabs [data-baseweb="tab-list"] { gap: 6px; }
.stTabs [data-baseweb="tab"] { height: 46px; padding: 0 18px; font-weight: 600; font-size: 1rem; }
.section-title { font-size: 1.25rem; font-weight: 700; margin: 1.1rem 0 .15rem 0; color: inherit; }
.section-sub { opacity: .7; margin: 0 0 .8rem 0; font-size: .95rem; color: inherit; }
.kpi { background: #fff; border-radius: 16px; padding: 16px 18px; box-shadow: 0 2px 12px rgba(16,24,40,.08); border-left: 6px solid var(--c); height: 100%; }
.kpi-label { font-size: .74rem; font-weight: 700; color: #667085; text-transform: uppercase; letter-spacing: .05em; }
.kpi-value { font-size: 2.1rem; font-weight: 800; color: #101828; line-height: 1.25; }
.kpi-sub { font-size: .84rem; color: #475467; }
.insight { background: #ECFDF3; border: 1px solid #ABEFC6; border-radius: 14px; padding: 14px 18px; color: #067647; font-weight: 500; margin: 16px 0 6px 0; }
.card { background: #fff; border-radius: 16px; padding: 18px 22px; box-shadow: 0 2px 12px rgba(16,24,40,.08); color: #101828; }
.card h4 { margin: 0 0 8px 0; color: #1B4965; }
.step { display: flex; align-items: flex-start; margin: 10px 0; color: #344054; }
.step b { background: #2A9D8F; color: #fff; border-radius: 50%; min-width: 26px; height: 26px; display: inline-flex; align-items: center; justify-content: center; margin-right: 12px; font-size: .85rem; }
.pill { display: inline-block; color: #fff; border-radius: 999px; padding: 6px 18px; font-weight: 700; font-size: 1rem; }
.bigp { font-size: 3.4rem; font-weight: 800; color: #101828; line-height: 1; }
.bigp-label { color: #667085; font-size: .85rem; font-weight: 600; text-transform: uppercase; letter-spacing: .05em; }
.meter { position: relative; height: 14px; border-radius: 999px; background: linear-gradient(90deg, #2A9D8F 0%, #2A9D8F 25%, #F4A261 25%, #F4A261 50%, #E76F51 50%, #E76F51 100%); margin: 14px 0 4px 0; }
.meter-marker { position: absolute; top: -6px; width: 6px; height: 26px; background: #101828; border-radius: 4px; transform: translateX(-3px); box-shadow: 0 0 0 3px rgba(255,255,255,.9); }
.meter-scale { display: flex; justify-content: space-between; font-size: .75rem; color: #667085; }
.action { border-left: 6px solid var(--c); background: #fff; border-radius: 12px; padding: 14px 18px; margin: 14px 0; box-shadow: 0 2px 10px rgba(16,24,40,.07); color: #101828; font-weight: 600; }
.chip { display: inline-block; border-radius: 999px; padding: 5px 12px; margin: 3px 6px 3px 0; font-size: .85rem; font-weight: 500; }
.chip-up { background: #FDE8E2; color: #B93815; }
.chip-down { background: #D9F2EE; color: #146C61; }
.footer { text-align: center; opacity: .6; font-size: .82rem; margin-top: 30px; color: inherit; }
.stButton > button, .stDownloadButton > button, [data-testid="stFormSubmitButton"] > button { border-radius: 10px; font-weight: 600; }
"""


def render(html):
    # Le HTML est mis sur une seule ligne : sinon le Markdown de Streamlit prend les lignes indentées pour du code.
    st.markdown(' '.join(line.strip() for line in html.splitlines()), unsafe_allow_html=True)


render(f'<style>{CSS}</style>')


# ----------------------------------------------------------------------------- petits composants HTML
def kpi(label, value, sub, color):
    return f'<div class="kpi" style="--c:{color}"><div class="kpi-label">{label}</div><div class="kpi-value">{value}</div><div class="kpi-sub">{sub}</div></div>'


def kpi_row(cards):
    for col, html in zip(st.columns(len(cards)), cards):
        col.markdown(html, unsafe_allow_html=True)


def section(title, sub=''):
    render(f'<div class="section-title">{title}</div>' + (f'<div class="section-sub">{sub}</div>' if sub else ''))


def pill(tier):
    return f'<span class="pill" style="background:{TIER_COLORS[tier]}">Risque {tier.lower()}</span>'


def risk_meter(p):
    return (f'<div class="meter"><div class="meter-marker" style="left:{min(max(p, 0), 1) * 100:.1f}%"></div></div>'
            '<div class="meter-scale"><span>0 %</span><span>25 %</span><span>50 %</span><span>100 %</span></div>')


def chips(text, kind):
    items = [t for t in str(text).split(' | ') if t]
    if not items:
        return '<span style="opacity:.6">Aucun</span>'
    return ''.join(f'<span class="chip chip-{kind}">{t}</span>' for t in items)


# ----------------------------------------------------------------------------- graphiques (matplotlib, style commun)
def style_ax(ax):
    ax.set_facecolor('white')
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color('#D0D5DD')
    ax.tick_params(colors='#475467', labelsize=9)
    ax.yaxis.grid(True, color='#EAECF0', lw=.8)
    ax.set_axisbelow(True)


def new_fig(w, h):
    fig, ax = plt.subplots(figsize=(w, h))
    style_ax(ax)
    return fig, ax


def chart_title(ax, text):
    ax.set_title(text, loc='left', fontsize=12, fontweight='bold', color=NAVY, pad=12)


def show(fig):
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def pct_axis(ax, axis='y'):
    (ax.yaxis if axis == 'y' else ax.xaxis).set_major_formatter(lambda v, _: f'{v:.0%}')


def donut_fig(counts, center_label='clients'):
    total = int(counts.sum())
    fig, ax = plt.subplots(figsize=(4.4, 3.9))
    wedges, _ = ax.pie(counts.values, colors=[TIER_COLORS[t] for t in counts.index], startangle=90, counterclock=False,
                       wedgeprops=dict(width=.36, edgecolor='white', linewidth=2))
    ax.text(0, .08, f'{total:,}', ha='center', va='center', fontsize=22, fontweight='bold', color=NAVY)
    ax.text(0, -.2, center_label, ha='center', va='center', fontsize=10, color='#667085')
    ax.legend(wedges, [f'{t} · {int(c):,} ({c / max(total, 1):.0%})' for t, c in counts.items()], loc='upper center',
              bbox_to_anchor=(.5, .02), frameon=False, fontsize=9, ncol=1)
    ax.set_title('Répartition par niveau de risque', loc='left', fontsize=12, fontweight='bold', color=NAVY, pad=8)
    return fig


def gains_fig(p, y):
    n = len(p)
    order = np.argsort(-p)
    cum = np.cumsum(y[order]) / max(y.sum(), 1)
    share = np.arange(1, n + 1) / n
    fig, ax = new_fig(5.2, 3.9)
    ax.fill_between(share, cum, share, color=TEAL, alpha=.12)
    ax.plot(share, cum, color=TEAL, lw=2.8, label='LoyaltyLens')
    ax.plot([0, 1], [0, 1], ls='--', color=GREY, label='Au hasard')
    k = max(int(.2 * n), 1)
    ax.scatter([.2], [cum[k - 1]], color=CORAL, zorder=5, s=70)
    ax.annotate(f'{cum[k - 1]:.0%} des départs couverts\nen contactant 20 % des clients', xy=(.2, cum[k - 1]),
                xytext=(.33, max(cum[k - 1] - .27, .05)), fontsize=9, fontweight='bold', color=NAVY,
                arrowprops=dict(arrowstyle='-', color=CORAL))
    pct_axis(ax); pct_axis(ax, 'x'); ax.legend(frameon=False, fontsize=9, loc='lower right')
    ax.set_xlabel('Part des clients contactés (du plus au moins à risque)', fontsize=9, color='#475467')
    chart_title(ax, 'Combien de départs anticipe-t-on ?')
    return fig, float(cum[k - 1])


def reliability_fig(tiers):
    fig, ax = new_fig(4.6, 3.9)
    x = np.arange(len(tiers)); w = .38
    b1 = ax.bar(x - w / 2, tiers['predit'], w, color=GREY, label='Probabilité prédite')
    b2 = ax.bar(x + w / 2, tiers['reel'], w, color=CORAL, label='Départs réels')
    for bars in (b1, b2):
        for r in bars:
            ax.text(r.get_x() + r.get_width() / 2, r.get_height() + .01, f'{r.get_height():.0%}', ha='center', fontsize=9, color='#344054')
    ax.set_xticks(x); ax.set_xticklabels(tiers.index); pct_axis(ax); ax.set_ylim(0, 1)
    ax.legend(frameon=False, fontsize=9, loc='upper right')
    chart_title(ax, 'Les niveaux sont-ils fiables ?')
    return fig


def drivers_fig(drivers):
    top = drivers.head(8).iloc[::-1]
    fig, ax = new_fig(5.4, 3.9)
    ax.barh(top.index, top.values, color=NAVY, height=.62)
    ax.yaxis.grid(False); ax.xaxis.grid(True, color='#EAECF0', lw=.8)
    ax.set_xlabel('Poids moyen dans le score', fontsize=9, color='#475467')
    chart_title(ax, 'Ce qui pèse le plus dans le risque de départ')
    return fig


def segments_fig(board):
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.8), gridspec_kw={'width_ratios': [3, 5]})
    specs = [('Contract', ['Month-to-month', 'One year', 'Two year'], lambda v: cp.FR_MODALITY.get(v, v), 'Départs selon le contrat'),
             ('tenure_group', cp.TENURE_LABELS, lambda v: f'{v} mois', "Départs selon l'ancienneté")]
    for ax, (col, order, lab, ttl) in zip(axes, specs):
        style_ax(ax)
        g = board.groupby(col).agg(reel=('churn_reel', 'mean'), pred=('churn_probability', 'mean')).reindex(order)
        x = np.arange(len(g))
        ax.bar(x, g['reel'], color=CORAL, width=.58)
        ax.scatter(x, g['pred'], color=NAVY, s=48, zorder=5)
        for xi, v in zip(x, g['reel']):
            ax.text(xi, v + .02, f'{v:.0%}', ha='center', fontsize=9, fontweight='bold', color='#344054')
        ax.set_xticks(x); ax.set_xticklabels([lab(o) for o in order]); pct_axis(ax); ax.set_ylim(0, max(g['reel'].max() * 1.25, .1))
        chart_title(ax, ttl)
    handles = [plt.Rectangle((0, 0), 1, 1, color=CORAL), plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=NAVY, markersize=8)]
    axes[1].legend(handles, ['Départs réels', 'Probabilité prédite'], frameon=False, fontsize=9, loc='upper right')
    return fig


def explanation_fig(expl):
    plot = expl.iloc[::-1]
    fig, ax = new_fig(7.4, 4.6)
    ax.barh(plot['variable'], plot['contribution'], color=[CORAL if v > 0 else TEAL for v in plot['contribution']], height=.62)
    ax.axvline(0, color='#101828', lw=1)
    ax.yaxis.grid(False); ax.xaxis.grid(True, color='#EAECF0', lw=.8)
    ax.set_xlabel('Contribution au score  (orange : vers le départ · vert : vers la fidélité)', fontsize=9, color='#475467')
    chart_title(ax, 'Pourquoi ce score ?')
    return fig


def histogram_fig(p):
    fig, ax = new_fig(5.4, 3.9)
    bins = np.linspace(0, 1, 21)
    _, edges, patches = ax.hist(p, bins=bins, edgecolor='white')
    for patch, left in zip(patches, edges[:-1]):
        patch.set_facecolor(CORAL if left >= cp.TIER_HIGH else AMBER if left >= cp.TIER_MID else TEAL)
    ax.axvline(cp.TIER_MID, color=GREY, ls='--', lw=1); ax.axvline(cp.TIER_HIGH, color=GREY, ls='--', lw=1)
    ax.set_xlabel('Probabilité de départ', fontsize=9, color='#475467'); ax.set_ylabel('Clients', fontsize=9, color='#475467')
    pct_axis(ax, 'x')
    chart_title(ax, 'Distribution des probabilités')
    return fig


# ----------------------------------------------------------------------------- chargement des données (mis en cache)
@st.cache_resource
def load_scorer():
    return cp.ChurnScorer(ROOT)


@st.cache_data
def load_card():
    path = ROOT / 'models' / 'model_card.json'
    return json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}


@st.cache_data
def load_portfolio():
    scorer = load_scorer()
    test = pd.read_csv(ROOT / 'data' / 'processed' / 'test_features.csv')
    res = scorer.score(test).drop(columns=['expected_net_gain', 'action_recommended'])
    res['recommended_action'] = res['risk_tier'].map(cp.TIER_ACTIONS)
    res['churn_reel'] = (test['Churn'] == 'Yes').astype(int).values
    res['Contract'] = test['Contract'].values
    res['tenure_group'] = test['tenure_group'].astype(str).values
    X, _ = scorer.prepare(test)
    contrib = np.abs(scorer.grouped_contributions(X)).mean(axis=0)
    drivers = pd.Series(contrib, index=[cp.FR_VAR.get(g, g) for g in scorer.groups]).sort_values(ascending=False)
    return res, drivers


@st.cache_data
def example_file():
    test = pd.read_csv(ROOT / 'data' / 'processed' / 'test_features.csv')
    cols = ['customerID'] + cp.REQUIRED_COLUMNS + ['TotalCharges', 'Churn']
    return test[cols].sample(40, random_state=7).to_csv(index=False).encode('utf-8')


@st.cache_data(show_spinner=False)
def run_batch(content):
    scorer = load_scorer()
    try:
        raw = cp.read_uploaded_csv(content)
    except Exception as err:
        return {'error': f'Fichier illisible : {err}'}
    valid, problems, n_bad = cp.validate_batch(raw)
    if valid is None or len(valid) == 0:
        return {'error': problems[0] if problems else 'Aucune ligne valide dans le fichier.', 'problems': problems, 'n_rows': len(raw)}
    res = scorer.score(valid).drop(columns=['expected_net_gain', 'action_recommended'])
    res['recommended_action'] = res['risk_tier'].map(cp.TIER_ACTIONS)
    out = {'res': res, 'valid': valid, 'problems': problems, 'n_bad': n_bad, 'n_rows': len(raw)}
    if 'Churn' in valid.columns and set(valid['Churn'].dropna().unique()) <= {'Yes', 'No'}:
        y = (valid['Churn'] == 'Yes').astype(int).values
        if 0 < y.sum() < len(y):
            high = (res['risk_tier'] == 'Élevé').values
            out['perf'] = {'roc': roc_auc_score(y, res['churn_probability']), 'pr': average_precision_score(y, res['churn_probability']),
                           'base': float(y.mean()), 'real_high': float(y[high].mean()) if high.any() else float('nan')}
    return out


try:
    scorer = load_scorer()
    board, drivers = load_portfolio()
except FileNotFoundError as err:
    st.error(f'Fichier introuvable : {err}. Exécute d\'abord les notebooks 1 à 3 (dossiers models/ et data/processed/).')
    st.stop()
card = load_card()


# ----------------------------------------------------------------------------- en-tête
chip_list = [f'Modèle : {card.get("modèle", scorer.family)}']
if card.get('test'):
    chip_list.append(f'ROC-AUC {card["test"]["ROC-AUC"]:.2f}')
chip_list += ['Explication de chaque score', 'Décision finale humaine']
render('<div class="hero"><span class="hero-badge">Rétention client · IA responsable</span><h1>LoyaltyLens</h1>'
       '<p>Anticipez les départs, comprenez pourquoi, agissez au bon moment.</p>'
       f'<div class="hero-chips">{"".join(f"<span>{c}</span>" for c in chip_list)}</div></div>')

tab_portfolio, tab_client, tab_batch = st.tabs(['📊 Portefeuille', '🧑 Simuler un client', '📁 Simuler plusieurs clients'])


# ============================================================================= onglet 1 : portefeuille
with tab_portfolio:
    n = len(board)
    y = board['churn_reel'].values
    p = board['churn_probability'].values
    is_high = (board['risk_tier'] == 'Élevé').values
    n_high = int(is_high.sum())
    real_rate = float(y.mean())
    real_high = float(y[is_high].mean()) if is_high.any() else float('nan')
    pred_high = float(p[is_high].mean()) if is_high.any() else float('nan')
    share_churners_high = float(y[is_high].sum() / max(y.sum(), 1))

    section('Vue d\'ensemble du portefeuille', 'Clients du jeu de test, jamais vus par le modèle pendant son entraînement.')
    fig_gain, cover20 = gains_fig(p, y)
    kpi_row([
        kpi('Clients analysés', f'{n:,}', 'jeu de test', NAVY),
        kpi('Départs réels', f'{real_rate:.1%}', f'{int(y.sum()):,} clients partis', CORAL),
        kpi('Risque élevé', f'{n_high:,}', f'{n_high / n:.0%} du portefeuille', AMBER),
        kpi('Fiabilité du niveau élevé', f'{real_high:.0%}', f'partent réellement (prédit : {pred_high:.0%})', TEAL),
        kpi('Couverture à 20 %', f'{cover20:.0%}', 'des départs captés en contactant 20 % des clients', NAVY),
    ])
    render(f'<div class="insight">💡 Les {n_high:,} clients à risque élevé ne sont que {n_high / n:.0%} du portefeuille, '
           f'mais ils concentrent {share_churners_high:.0%} des départs réels : c\'est là qu\'il faut agir en premier.</div>')

    c1, c2, c3 = st.columns([1, 1.15, 1])
    with c1:
        show(donut_fig(board['risk_tier'].value_counts().reindex(TIER_ORDER).fillna(0)))
    with c2:
        show(fig_gain)
    with c3:
        tiers = board.groupby('risk_tier').agg(reel=('churn_reel', 'mean'), predit=('churn_probability', 'mean')).reindex(TIER_ORDER)
        show(reliability_fig(tiers))

    d1, d2 = st.columns([1, 1.6])
    with d1:
        show(drivers_fig(drivers))
    with d2:
        show(segments_fig(board))

    section('Clients à suivre', 'Triés du plus au moins à risque, avec les raisons principales de chaque score.')
    f1, f2 = st.columns([2, 1])
    chosen = f1.multiselect('Niveaux de risque', TIER_ORDER, default=['Élevé'])
    search = f2.text_input('Rechercher un client (identifiant)', value='')
    view = board[board['risk_tier'].isin(chosen)]
    if search.strip():
        view = view[view['customerID'].str.contains(search.strip(), case=False, regex=False)]
    view = view.sort_values('churn_probability', ascending=False)
    table = pd.DataFrame({'Client': view['customerID'], 'Risque (%)': (view['churn_probability'] * 100).round(1),
                          'Niveau': view['risk_tier'], 'Action recommandée': view['recommended_action'],
                          'Raisons principales': view['top_risk_factors']})
    try:
        cfg = {'Risque (%)': st.column_config.ProgressColumn('Risque (%)', min_value=0, max_value=100, format='%.0f')}
    except AttributeError:
        cfg = None
    st.caption(f'{len(table):,} clients affichés (les 200 premiers dans le tableau).')
    st.dataframe(table.head(200), column_config=cfg, hide_index=True)
    st.download_button('Télécharger cette liste (CSV)', table.to_csv(index=False, sep=';', decimal=',').encode('utf-8-sig'),
                       file_name='portefeuille_loyaltylens.csv', mime='text/csv')


# ============================================================================= onglet 2 : simuler un client
PROFILES = {
    'Client à risque (récent, contrat mensuel, fibre)': dict(
        tenure=3, MonthlyCharges=89.0, Contract='Month-to-month', PaymentMethod='Electronic check', PaperlessBilling='Yes',
        InternetService='Fiber optic', OnlineSecurity='No', OnlineBackup='No', DeviceProtection='No', TechSupport='No',
        StreamingTV='Yes', StreamingMovies='Yes', PhoneService='Yes', MultipleLines='Yes', Partner='No', Dependents='No'),
    'Client fidèle (2 ans, DSL, services de protection)': dict(
        tenure=60, MonthlyCharges=55.0, Contract='Two year', PaymentMethod='Bank transfer (automatic)', PaperlessBilling='No',
        InternetService='DSL', OnlineSecurity='Yes', OnlineBackup='Yes', DeviceProtection='Yes', TechSupport='Yes',
        StreamingTV='No', StreamingMovies='No', PhoneService='Yes', MultipleLines='No', Partner='Yes', Dependents='Yes'),
}
CONTRACTS = ['Month-to-month', 'One year', 'Two year']
INTERNET = ['DSL', 'Fiber optic', 'No']
PAYMENTS = ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)']
YN = ['Yes', 'No']

if 'log' not in st.session_state:
    st.session_state['log'] = []
if 'client_raw' not in st.session_state:
    st.session_state['client_raw'] = None


def client_result(raw, key_prefix):
    # Résultat complet pour un client : score, jauge, raisons, graphique d'explication
    res = scorer.score(raw).iloc[0]
    X, fe = scorer.prepare(raw)
    expl = scorer.explain(X, fe, 0, top=10)
    tier = res['risk_tier']
    action = cp.TIER_ACTIONS[tier]

    left, right = st.columns([1, 1.5])
    with left:
        render(f'<div class="card"><div class="bigp-label">Probabilité de départ</div><div class="bigp">{res["churn_probability"]:.0%}</div>'
               f'{risk_meter(res["churn_probability"])}<div style="margin-top:14px">{pill(tier)}</div></div>')
        render(f'<div class="action" style="--c:{TIER_COLORS[tier]}">Action recommandée : {action}</div>')
    with right:
        render('<div class="card"><h4>Ce qui augmente le risque</h4>' + chips(res['top_risk_factors'], 'up') +
               '<h4 style="margin-top:14px">Ce qui protège</h4>' + chips(res['top_protective_factors'], 'down') + '</div>')

    show(explanation_fig(expl))
    st.caption("Les contributions sont exprimées sur l'échelle du modèle (log-odds), pas en points de pourcentage.")
    return res


with tab_client:
    section('Simuler un client', 'Choisissez un profil, ajustez les caractéristiques et découvrez son score avec les raisons.')
    profile_name = st.selectbox('Point de départ', list(PROFILES.keys()))
    prof = PROFILES[profile_name]
    pk = list(PROFILES.keys()).index(profile_name)      # change les clés des champs quand on change de profil

    def pick(label, options, field, fmt=None):
        kwargs = {'format_func': fmt} if fmt else {}
        return st.selectbox(label, options, index=options.index(prof[field]), key=f'{field}_{pk}', **kwargs)

    with st.form('client_form'):
        c1, c2, c3 = st.columns(3)
        with c1:
            tenure = st.number_input('Ancienneté (mois)', min_value=0, max_value=72, value=int(prof['tenure']), step=1, key=f'tenure_{pk}')
            monthly = st.number_input('Facture mensuelle', min_value=18.0, max_value=120.0, value=float(prof['MonthlyCharges']), step=0.5, key=f'monthly_{pk}')
            total = st.number_input('Facturation cumulée (0 = ancienneté × facture)', min_value=0.0, max_value=10000.0, value=0.0, step=10.0, key=f'total_{pk}')
            contract = pick('Contrat', CONTRACTS, 'Contract', FR_OPTION)
            payment = pick('Mode de paiement', PAYMENTS, 'PaymentMethod', FR_OPTION)
            paperless = pick('Facture électronique', YN, 'PaperlessBilling', YES_NO)
        with c2:
            internet = pick('Internet', INTERNET, 'InternetService', FR_OPTION)
            sec = pick('Sécurité en ligne', YN, 'OnlineSecurity', YES_NO)
            backup = pick('Sauvegarde en ligne', YN, 'OnlineBackup', YES_NO)
            device = pick('Protection appareil', YN, 'DeviceProtection', YES_NO)
            tech = pick('Support technique', YN, 'TechSupport', YES_NO)
            tv = pick('Streaming TV', YN, 'StreamingTV', YES_NO)
            movies = pick('Streaming films', YN, 'StreamingMovies', YES_NO)
        with c3:
            phone = pick('Service téléphone', YN, 'PhoneService', YES_NO)
            lines = pick('Lignes multiples', YN, 'MultipleLines', YES_NO)
            partner = pick('Vit en couple', YN, 'Partner', YES_NO)
            kids = pick('Personnes à charge', YN, 'Dependents', YES_NO)
        submitted = st.form_submit_button('Calculer le risque', type='primary')

    if submitted:
        services = dict(OnlineSecurity=sec, OnlineBackup=backup, DeviceProtection=device, TechSupport=tech, StreamingTV=tv, StreamingMovies=movies)
        if internet == 'No':                              # cohérence : sans Internet, pas de services en ligne
            services = {k: 'No' for k in services}
        st.session_state['client_raw'] = pd.DataFrame([dict(
            customerID='CLIENT-SIMULÉ', Partner=partner, Dependents=kids, tenure=int(tenure), PhoneService=phone,
            MultipleLines='No' if phone == 'No' else lines, InternetService=internet, **services,
            Contract=contract, PaperlessBilling=paperless, PaymentMethod=payment, MonthlyCharges=float(monthly),
            TotalCharges=float(total) if total > 0 else float(tenure) * float(monthly))])

    raw_client = st.session_state['client_raw']
    if raw_client is None:
        st.info('Choisissez un point de départ, ajustez les champs, puis cliquez sur « Calculer le risque ».')
    else:
        res_client = client_result(raw_client, 'single')

        section('Validation humaine', 'Le modèle recommande, le conseiller décide : aucune action n\'est lancée sans validation (maquette du suivi dans Salesforce).')
        d1, d2 = st.columns([1, 2])
        decision = d1.radio('Décision du conseiller', ['À décider', 'Valider l\'action', 'Rejeter l\'action'], index=0)
        comment = d2.text_input('Commentaire (facultatif)', value='')
        if st.button('Enregistrer la décision'):
            if decision == 'À décider':
                st.warning('Choisissez « Valider » ou « Rejeter » avant d\'enregistrer.')
            else:
                st.session_state['log'].append({'client': 'CLIENT-SIMULÉ', 'probabilité': f'{res_client["churn_probability"]:.0%}',
                                                'niveau': res_client['risk_tier'], 'recommandation': cp.TIER_ACTIONS[res_client['risk_tier']],
                                                'décision humaine': decision, 'commentaire': comment})
        if st.session_state['log']:
            st.dataframe(pd.DataFrame(st.session_state['log']), hide_index=True)


# ============================================================================= onglet 3 : simuler plusieurs clients
SCHEMA_ROWS = [
    ('tenure', 'obligatoire', 'Ancienneté en mois (nombre)'), ('MonthlyCharges', 'obligatoire', 'Facture mensuelle (nombre)'),
    ('Contract', 'obligatoire', 'Month-to-month / One year / Two year'), ('InternetService', 'obligatoire', 'DSL / Fiber optic / No'),
    ('PaymentMethod', 'obligatoire', 'Electronic check / Mailed check / Bank transfer (automatic) / Credit card (automatic)'),
    ('PaperlessBilling, Partner, Dependents, PhoneService', 'obligatoire', 'Yes / No'),
    ('MultipleLines', 'obligatoire', 'Yes / No / No phone service'),
    ('OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies', 'obligatoire', 'Yes / No / No internet service'),
    ('customerID', 'facultatif', 'Identifiant (généré si absent)'), ('TotalCharges', 'facultatif', 'Facturation cumulée (calculée si absente)'),
    ('Churn', 'facultatif', 'Yes / No : si présent, la performance du modèle est mesurée sur votre fichier'),
]

with tab_batch:
    section('Simuler plusieurs clients', 'Déposez un fichier CSV : chaque client reçoit sa probabilité, son niveau de risque et ses raisons.')
    a, b = st.columns([1.6, 1])
    with a:
        render('<div class="card"><h4>Comment ça marche</h4>'
               '<div class="step"><b>1</b><span>Préparez un fichier CSV avec une ligne par client (colonnes ci-dessous).</span></div>'
               '<div class="step"><b>2</b><span>Déposez-le ci-dessous : virgule ou point-virgule, UTF-8 ou Excel, tout est reconnu.</span></div>'
               '<div class="step"><b>3</b><span>Consultez les niveaux de risque, ouvrez l\'explication de chaque client et téléchargez les résultats.</span></div></div>')
    
    with st.expander('Colonnes attendues'):
        st.dataframe(pd.DataFrame(SCHEMA_ROWS, columns=['Colonne(s)', 'Statut', 'Valeurs acceptées']), hide_index=True)

    uploaded = st.file_uploader('Déposez votre fichier CSV', type=['csv'])
    if uploaded is None:
        st.info('En attente d\'un fichier.')
    else:
        with st.spinner('Calcul des scores et des explications...'):
            out = run_batch(uploaded.getvalue())
        if 'error' in out:
            st.error(out['error'])
            for prob in out.get('problems', [])[1:]:
                st.warning(prob)
        else:
            res, valid = out['res'], out['valid']
            st.success(f'{len(res):,} clients scorés sur {out["n_rows"]:,} lignes lues.')
            if out['problems']:
                with st.expander(f'⚠️ {out["n_bad"]} ligne(s) écartée(s) : voir le détail'):
                    for prob in out['problems']:
                        st.write('- ' + prob)

            pb = res['churn_probability'].values
            tier_counts = res['risk_tier'].value_counts().reindex(TIER_ORDER).fillna(0)
            kpi_row([
                kpi('Clients scorés', f'{len(res):,}', 'dans votre fichier', NAVY),
                kpi('Risque élevé', f'{int(tier_counts["Élevé"]):,}', f'{tier_counts["Élevé"] / len(res):.0%} des clients', CORAL),
                kpi('Risque moyen', f'{int(tier_counts["Moyen"]):,}', f'{tier_counts["Moyen"] / len(res):.0%} des clients', AMBER),
                kpi('Probabilité moyenne', f'{pb.mean():.0%}', 'de départ dans ce fichier', TEAL),
            ])
            if 'perf' in out:
                pf = out['perf']
                render(f'<div class="insight">✅ Votre fichier contient la colonne Churn : ROC-AUC {pf["roc"]:.2f} · PR-AUC {pf["pr"]:.2f} '
                       f'(hasard : {pf["base"]:.2f}) · {pf["real_high"]:.0%} des clients classés « Élevé » sont réellement partis.</div>')

            g1, g2 = st.columns(2)
            with g1:
                show(donut_fig(tier_counts, 'clients scorés'))
            with g2:
                show(histogram_fig(pb))

            section('Résultats par client')
            f1, f2 = st.columns([2, 1])
            chosen_b = f1.multiselect('Niveaux de risque', TIER_ORDER, default=TIER_ORDER, key='batch_tiers')
            search_b = f2.text_input('Rechercher un client (identifiant)', value='', key='batch_search')
            view_b = res[res['risk_tier'].isin(chosen_b)]
            if search_b.strip():
                view_b = view_b[view_b['customerID'].str.contains(search_b.strip(), case=False, regex=False)]
            view_b = view_b.sort_values('churn_probability', ascending=False)
            table_b = pd.DataFrame({'Client': view_b['customerID'], 'Risque (%)': (view_b['churn_probability'] * 100).round(1),
                                    'Niveau': view_b['risk_tier'], 'Action recommandée': view_b['recommended_action'],
                                    'Facteurs de risque': view_b['top_risk_factors'], 'Facteurs protecteurs': view_b['top_protective_factors']})
            try:
                cfg_b = {'Risque (%)': st.column_config.ProgressColumn('Risque (%)', min_value=0, max_value=100, format='%.0f')}
            except AttributeError:
                cfg_b = None
            st.caption(f'{len(table_b):,} clients affichés (les 300 premiers dans le tableau ; le téléchargement contient tout).')
            st.dataframe(table_b.head(300), column_config=cfg_b, hide_index=True)

            full = pd.DataFrame({'Client': res['customerID'], 'Probabilité de départ (%)': (res['churn_probability'] * 100).round(1),
                                 'Niveau de risque': res['risk_tier'], 'Action recommandée': res['recommended_action'],
                                 'Facteurs de risque': res['top_risk_factors'], 'Facteurs protecteurs': res['top_protective_factors']}
                                ).sort_values('Probabilité de départ (%)', ascending=False)
            e1, e2 = st.columns(2)
            e1.download_button('Télécharger les résultats (CSV compatible Excel)', full.to_csv(index=False, sep=';', decimal=',').encode('utf-8-sig'),
                               file_name='resultats_loyaltylens.csv', mime='text/csv')
            e2.download_button('Télécharger les résultats (CSV standard)', full.to_csv(index=False).encode('utf-8'),
                               file_name='resultats_loyaltylens_standard.csv', mime='text/csv')

            section('Comprendre un client', 'Choisissez un client du fichier (les plus à risque en premier) pour voir pourquoi il est classé ainsi.')
            ranked = res.sort_values('churn_probability', ascending=False).head(300)
            labels = {i: f'{r.customerID} · {r.churn_probability:.0%} · {r.risk_tier}' for i, r in ranked.iterrows()}
            picked = st.selectbox('Client', list(labels.keys()), format_func=lambda i: labels[i])
            client_result(valid.iloc[[picked]].reset_index(drop=True), 'batch')


render('<div class="footer">LoyaltyLens · Le modèle n\'utilise ni le genre ni l\'âge · Les scores sont des aides à la décision : '
       'la décision finale revient toujours à un conseiller.</div>')
