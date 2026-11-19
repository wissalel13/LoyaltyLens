"""
LoyaltyLens : processus complet de scoring, réutilisable par l'application Streamlit,
par un script ou par l'intégration Salesforce.

Un client brut (colonnes du jeu Telco) passe par :
    nettoyage  ->  nouvelles variables  ->  encodage  ->  modèle  ->  calibration
    ->  décision business  ->  raisons principales (contributions au score)

Les fonctions clean_raw et engineer_features sont identiques à celles du notebook 2.
Dossiers attendus (relatifs à la racine du projet) :
    models/churn_model.joblib, models/model_card.json, models/preprocessor_*.joblib
    data/processed/X_train_scaled.csv (ou X_train_tree.csv), data/processed/test_features.csv
"""
import io
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

# Racine du projet : le dossier qui contient models/ et data/ (ce fichier peut être à la racine ou dans src/)
_HERE = Path(__file__).resolve().parent
ROOT = next((p for p in (_HERE, _HERE.parent) if (p / 'models').exists() and (p / 'data').exists()), _HERE.parent)

# ----------------------------------------------------------------------------- constantes (notebook 2)
SERVICE_COLS    = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
PROTECTION_COLS = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport']
AUTO_PAY        = ['Bank transfer (automatic)', 'Credit card (automatic)']
TENURE_BINS     = [-1, 6, 12, 24, 48, np.inf]      # dernière tranche ouverte : identique aux données (0-72), sans erreur au-delà
TENURE_LABELS   = ['0-6', '7-12', '13-24', '25-48', '49-72']

NUM_FEATURES     = ['tenure', 'MonthlyCharges', 'charge_trend', 'charge_per_service', 'n_addon_services', 'n_protection_services']
FLAG_FEATURES    = ['is_new_customer', 'auto_payment', 'mtm_fiber', 'mtm_echeck']
YN_FEATURES      = ['Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'OnlineSecurity', 'OnlineBackup',
                    'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'PaperlessBilling']
MULTI_FEATURES   = ['Contract', 'InternetService', 'PaymentMethod']
ORDINAL_FEATURES = ['tenure_group']
ALL_FEATURES     = NUM_FEATURES + FLAG_FEATURES + YN_FEATURES + MULTI_FEATURES + ORDINAL_FEATURES

# Valeurs par défaut des hypothèses business (remplacées par celles de model_card.json si présent)
DEFAULT_BUSINESS = {'COST_CONTACT': 50.0, 'SUCCESS_RATE': 0.30, 'HORIZON_MONTHS': 12}
TIER_HIGH, TIER_MID = 0.50, 0.25


# ----------------------------------------------------------------------------- nettoyage et variables (notebook 2)
def clean_raw(raw):
    d = raw.copy()
    for c in d.select_dtypes(include=['object', 'string']).columns:
        d[c] = d[c].astype(str).str.strip()
    d['TotalCharges'] = pd.to_numeric(d['TotalCharges'].replace('', np.nan), errors='coerce')
    d.loc[d['TotalCharges'].isna() & (d['tenure'] == 0), 'TotalCharges'] = 0.0
    d['SeniorCitizen'] = d['SeniorCitizen'].astype(str).map({'0': 'No', '1': 'Yes', 'No': 'No', 'Yes': 'Yes'})
    for c in SERVICE_COLS:
        d[c] = d[c].replace({'No internet service': 'No'})
    d['MultipleLines'] = d['MultipleLines'].replace({'No phone service': 'No'})
    return d


def engineer_features(d):
    f = d.copy()
    f['n_addon_services']      = (f[SERVICE_COLS] == 'Yes').sum(axis=1).astype(int)
    f['n_protection_services'] = (f[PROTECTION_COLS] == 'Yes').sum(axis=1).astype(int)
    f['tenure_group']          = pd.cut(f['tenure'], bins=TENURE_BINS, labels=TENURE_LABELS).astype(str)
    f['is_new_customer']       = (f['tenure'] <= 12).astype(int)
    f['auto_payment']          = f['PaymentMethod'].isin(AUTO_PAY).astype(int)
    avg = f['TotalCharges'] / f['tenure'].where(f['tenure'] > 0)
    f['avg_historical_charge'] = avg.fillna(f['MonthlyCharges'])
    f['charge_trend']          = (f['MonthlyCharges'] / f['avg_historical_charge'] - 1).fillna(0.0)
    f['charge_per_service']    = f['MonthlyCharges'] / (1 + f['n_addon_services'])
    f['mtm_fiber']             = ((f['Contract'] == 'Month-to-month') & (f['InternetService'] == 'Fiber optic')).astype(int)
    f['mtm_echeck']            = ((f['Contract'] == 'Month-to-month') & (f['PaymentMethod'] == 'Electronic check')).astype(int)
    return f


# ----------------------------------------------------------------------------- libellés lisibles pour les raisons
FR_VAR = {
    'tenure': 'Ancienneté', 'MonthlyCharges': 'Facture mensuelle', 'charge_trend': 'Évolution de la facture',
    'charge_per_service': 'Facture par service', 'n_addon_services': 'Nombre de services optionnels',
    'n_protection_services': 'Nombre de services de protection', 'is_new_customer': 'Client récent (12 mois ou moins)',
    'auto_payment': 'Paiement automatique', 'mtm_fiber': 'Contrat mensuel avec fibre',
    'mtm_echeck': 'Contrat mensuel payé par chèque électronique', 'Partner': 'Vit en couple',
    'Dependents': 'Personnes à charge', 'PhoneService': 'Service téléphone', 'MultipleLines': 'Lignes multiples',
    'OnlineSecurity': 'Sécurité en ligne', 'OnlineBackup': 'Sauvegarde en ligne', 'DeviceProtection': 'Protection appareil',
    'TechSupport': 'Support technique', 'StreamingTV': 'Streaming TV', 'StreamingMovies': 'Streaming films',
    'PaperlessBilling': 'Facture électronique', 'Contract': 'Contrat', 'InternetService': 'Internet',
    'PaymentMethod': 'Mode de paiement', 'tenure_group': "Tranche d'ancienneté",
}
FR_MODALITY = {
    'Month-to-month': 'mensuel', 'One year': '1 an', 'Two year': '2 ans', 'Fiber optic': 'fibre optique', 'DSL': 'DSL',
    'No': 'aucun', 'Electronic check': 'chèque électronique', 'Mailed check': 'chèque postal',
    'Bank transfer (automatic)': 'virement automatique', 'Credit card (automatic)': 'carte bancaire automatique',
}


def group_of(col):
    """Variable d'origine d'une colonne encodée (ex. Contract_Two year -> Contract, Partner_Yes -> Partner).
    Les trois variables d'ancienneté (tenure, tenure_group, is_new_customer) sont fusionnées : elles se compensent
    dans le modèle et se lisent mieux comme une seule raison."""
    if col in ('tenure_group', 'is_new_customer'):
        return 'tenure'
    for var in YN_FEATURES:
        if col == f'{var}_Yes':
            return var
    for var in MULTI_FEATURES:
        if col.startswith(var + '_'):
            return var
    return col


def describe_group(group, fe_row):
    """Phrase lisible pour une variable d'origine, avec la valeur du client (ex. « Ancienneté : 3 mois »)."""
    label = FR_VAR.get(group, group)
    if group in NUM_FEATURES:
        v = float(fe_row[group])
        text = {'tenure': f'{v:.0f} mois', 'MonthlyCharges': f'{v:.2f}', 'charge_trend': f'{v:+.1%}',
                'charge_per_service': f'{v:.2f}'}.get(group, f'{v:.0f}')
        return f'{label} : {text}'
    if group in FLAG_FEATURES:
        return f'{label} : {"oui" if float(fe_row[group]) >= 0.5 else "non"}'
    if group == 'tenure_group':
        return f'{label} : {fe_row[group]} mois'
    if group in YN_FEATURES:
        return f'{label} : {"oui" if str(fe_row[group]) == "Yes" else "non"}'
    if group in MULTI_FEATURES:
        return f'{label} : {FR_MODALITY.get(str(fe_row[group]), str(fe_row[group]))}'
    return group


def _logit(p, eps=1e-6):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p)).reshape(-1, 1)


# ----------------------------------------------------------------------------- action recommandée selon le niveau de risque
TIER_ACTIONS = {
    'Élevé': 'Contacter en priorité',
    'Moyen': 'Surveiller et proposer une offre si possible',
    'Faible': 'Aucune action nécessaire',
}


# ----------------------------------------------------------------------------- fichier de plusieurs clients (CSV)
REQUIRED_COLUMNS = ['tenure', 'MonthlyCharges', 'Contract', 'InternetService', 'PaymentMethod', 'PaperlessBilling',
                    'Partner', 'Dependents', 'PhoneService', 'MultipleLines'] + SERVICE_COLS
OPTIONAL_COLUMNS = ['customerID', 'TotalCharges', 'Churn', 'gender', 'SeniorCitizen']
ALLOWED_VALUES = {
    'Contract': {'Month-to-month', 'One year', 'Two year'},
    'InternetService': {'DSL', 'Fiber optic', 'No'},
    'PaymentMethod': {'Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'},
    'PaperlessBilling': {'Yes', 'No'}, 'Partner': {'Yes', 'No'}, 'Dependents': {'Yes', 'No'}, 'PhoneService': {'Yes', 'No'},
    'MultipleLines': {'Yes', 'No', 'No phone service'},
    **{c: {'Yes', 'No', 'No internet service'} for c in SERVICE_COLS},
}


def read_uploaded_csv(content):
    """Lit un CSV envoyé (octets) : détecte le séparateur (virgule ou point-virgule) et l'encodage (UTF-8 ou Latin-1)."""
    for encoding in ('utf-8-sig', 'latin-1'):
        try:
            return pd.read_csv(io.BytesIO(content), sep=None, engine='python', encoding=encoding)
        except UnicodeDecodeError:
            continue
    raise ValueError('Fichier illisible : encodage non reconnu.')


def validate_batch(df):
    """Vérifie un fichier de clients. Renvoie (clients valides, liste de problèmes, nombre de lignes écartées).
    Si des colonnes obligatoires manquent, renvoie (None, [message], 0)."""
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        return None, [f'Colonnes obligatoires manquantes : {", ".join(missing)}'], 0

    for c in df.columns:                                       # espaces parasites dans les textes
        if df[c].dtype == object or str(df[c].dtype).startswith('str'):
            df[c] = df[c].astype(str).str.strip().replace({'nan': np.nan, '': np.nan})

    problems, bad = [], pd.Series(False, index=df.index)
    for c in ['tenure', 'MonthlyCharges'] + (['TotalCharges'] if 'TotalCharges' in df.columns else []):
        df[c] = pd.to_numeric(df[c].astype(str).str.replace(',', '.', regex=False), errors='coerce')   # virgule décimale française
    for c in ['tenure', 'MonthlyCharges']:
        m = df[c].isna() | (df[c] < 0)
        if m.any():
            problems.append(f'{int(m.sum())} ligne(s) écartée(s) : « {c} » vide, négatif ou non numérique')
            bad |= m
    for c, allowed in ALLOWED_VALUES.items():
        m = ~df[c].isin(allowed)
        if m.any():
            examples = ', '.join(f'« {v} »' for v in df.loc[m, c].fillna('vide').astype(str).unique()[:3])
            problems.append(f'{int(m.sum())} ligne(s) écartée(s) : « {c} » doit valoir {" / ".join(sorted(allowed))} (trouvé : {examples})')
            bad |= m

    valid = df.loc[~bad].reset_index(drop=True)
    if 'TotalCharges' not in valid.columns:
        valid['TotalCharges'] = valid['tenure'] * valid['MonthlyCharges']
    valid['TotalCharges'] = valid['TotalCharges'].fillna(valid['tenure'] * valid['MonthlyCharges'])
    generated = pd.Series([f'CLIENT-{i + 1:04d}' for i in range(len(valid))])
    if 'customerID' not in valid.columns:
        valid['customerID'] = generated
    else:
        valid['customerID'] = valid['customerID'].astype(object).where(valid['customerID'].notna(), generated).astype(str)
    return valid, problems, int(bad.sum())


# ----------------------------------------------------------------------------- fonctions business réutilisables
def decide(p, monthly, cost, success, horizon):
    """True = contacter. Règle : probabilité x succès x (facture x horizon) > coût."""
    return np.asarray(p) * success * np.asarray(monthly) * horizon > cost


def net_value(treat, churned, monthly, cost, success, horizon):
    """Gain net attendu d'une stratégie : on gagne si le client partait vraiment, on paie le contact dans tous les cas."""
    treat, churned, monthly = np.asarray(treat), np.asarray(churned), np.asarray(monthly)
    return float((treat * (churned * success * monthly * horizon - cost)).sum())


def strategy_table(p, monthly, churned, cost, success, horizon):
    """Compare plusieurs stratégies de rétention sur des clients dont on connaît le résultat réel."""
    p, monthly, churned = np.asarray(p), np.asarray(monthly), np.asarray(churned).astype(int)
    n = len(p)
    strategies = {
        'Ne rien faire': np.zeros(n, dtype=bool),
        'Contacter tout le monde': np.ones(n, dtype=bool),
        'Contacter si probabilité ≥ 50 %': p >= 0.5,
        'Contacter les 20 % les plus à risque': p >= np.quantile(p, .8),
        'Règle de valeur attendue (ChurnGuard)': decide(p, monthly, cost, success, horizon),
        'Idéal : contacter les futurs partants': churned == 1,
    }
    rows = []
    for name, t in strategies.items():
        rows.append({'stratégie': name, 'clients contactés': int(t.sum()), 'part contactée': t.mean(),
                     'départs couverts': (t & (churned == 1)).sum() / max(churned.sum(), 1),
                     'gain net': net_value(t, churned, monthly, cost, success, horizon)})
    return pd.DataFrame(rows).set_index('stratégie')


def group_report(attr_values, churned, treat):
    """Rappel, taux d'alerte et faux positifs par groupe (audit d'équité)."""
    attr_values, churned, treat = np.asarray(attr_values), np.asarray(churned).astype(bool), np.asarray(treat).astype(bool)
    rows = []
    for g in sorted(set(attr_values)):
        m = attr_values == g
        yt, tr = churned[m], treat[m]
        tp, fn = int((tr & yt).sum()), int((~tr & yt).sum())
        fp, tn = int((tr & ~yt).sum()), int((~tr & ~yt).sum())
        rows.append({'groupe': g, 'clients': int(m.sum()), 'churn réel': yt.mean(), "taux d'alerte": tr.mean(),
                     'rappel': tp / max(tp + fn, 1), 'faux positifs': fp / max(fp + tn, 1), 'précision': tp / max(tp + fp, 1)})
    return pd.DataFrame(rows).set_index('groupe')


# ----------------------------------------------------------------------------- le processus complet
class ChurnScorer:
    """Charge le modèle sauvegardé et score des clients bruts, avec les raisons de chaque score."""

    def __init__(self, root=ROOT):
        root = Path(root)
        self.models_dir, self.data_dir = root / 'models', root / 'data' / 'processed'
        art = joblib.load(self.models_dir / 'churn_model.joblib')
        self.model, self.calibrator = art['model'], art['calibrator']
        self.family, self.matrix, self.features = art['family'], art['matrix'], list(art['features'])

        prep_name = 'preprocessor_scaled.joblib' if self.matrix == 'scaled' else 'preprocessor_tree.joblib'
        prep_path = next((p for p in (self.models_dir / prep_name, self.data_dir / prep_name) if p.exists()), None)
        if prep_path is None:
            raise FileNotFoundError(f'{prep_name} introuvable dans models/ ni dans data/processed/')
        self.preprocessor = joblib.load(prep_path)

        bg_name = 'X_train_scaled.csv' if self.matrix == 'scaled' else 'X_train_tree.csv'
        self.background = pd.read_csv(self.data_dir / bg_name)[self.features]

        card_path = self.models_dir / 'model_card.json'
        self.business = dict(DEFAULT_BUSINESS)
        if card_path.exists():
            card = json.loads(card_path.read_text(encoding='utf-8'))
            for k in DEFAULT_BUSINESS:
                self.business[k] = card.get('hypothèses_business', {}).get(k, DEFAULT_BUSINESS[k])
        self._explainer = None

        # Regroupement des colonnes encodées par variable d'origine (une raison = une variable)
        self.groups = list(dict.fromkeys(group_of(c) for c in self.features))
        self._membership = np.zeros((len(self.features), len(self.groups)))
        for j, c in enumerate(self.features):
            self._membership[j, self.groups.index(group_of(c))] = 1.0

    # --- préparation -------------------------------------------------------------
    def prepare(self, raw):
        """Client(s) bruts -> (matrice du modèle, variables lisibles). Colonnes optionnelles complétées."""
        raw = raw.copy()
        if 'TotalCharges' not in raw.columns:
            raw['TotalCharges'] = raw['tenure'] * raw['MonthlyCharges']
        for col, default in [('SeniorCitizen', 0), ('gender', 'NA'), ('customerID', 'NEW')]:
            if col not in raw.columns:            # colonnes que le modèle n'utilise pas : valeurs neutres
                raw[col] = default
        fe = engineer_features(clean_raw(raw))
        X = self.preprocessor.transform(fe[ALL_FEATURES])
        return X.reindex(columns=self.features), fe

    def probability(self, X):
        p_raw = self.model.predict_proba(X)[:, 1]
        return self.calibrator.predict_proba(_logit(p_raw))[:, 1]

    # --- explications ------------------------------------------------------------
    def contributions(self, X):
        """Contribution de chaque variable au score brut (échelle log-odds). Positif = pousse vers le départ."""
        Xv = X[self.features].values.astype(float)
        if hasattr(self.model, 'coef_'):          # modèle linéaire : formule exacte, identique à SHAP LinearExplainer
            return (Xv - self.background.values.mean(axis=0)) * self.model.coef_[0]
        try:
            import shap
        except ImportError as err:
            raise ImportError('Pour un modèle à base d\'arbres, installe shap : pip install shap') from err
        if self._explainer is None:
            self._explainer = shap.TreeExplainer(self.model)
        values = self._explainer.shap_values(Xv)
        if isinstance(values, list):
            values = values[1]
        values = np.asarray(values)
        return values[:, :, 1] if values.ndim == 3 else values

    def grouped_contributions(self, X):
        """Contributions regroupées par variable d'origine : tableau (clients, variables)."""
        return self.contributions(X) @ self._membership

    def explain(self, X, fe, i, top=10):
        """Tableau des variables qui comptent le plus pour le client i (libellé, contribution)."""
        contrib = self.grouped_contributions(X.iloc[[i]])[0]
        rows = [{'variable': describe_group(g, fe.iloc[i]), 'contribution': float(v)} for g, v in zip(self.groups, contrib)]
        out = pd.DataFrame(rows)
        return out.reindex(out['contribution'].abs().sort_values(ascending=False).index).head(top).reset_index(drop=True)

    # --- scoring complet ----------------------------------------------------------
    def score(self, raw, cost=None, success=None, horizon=None, k=3):
        """DataFrame : probabilité calibrée, niveau de risque, gain attendu, action, facteurs de risque et de protection."""
        cost = self.business['COST_CONTACT'] if cost is None else cost
        success = self.business['SUCCESS_RATE'] if success is None else success
        horizon = self.business['HORIZON_MONTHS'] if horizon is None else horizon

        X, fe = self.prepare(raw)
        p = self.probability(X)
        monthly = fe['MonthlyCharges'].values
        gain = p * success * monthly * horizon - cost
        contrib = self.grouped_contributions(X)

        risk_txt, protect_txt = [], []
        for i in range(len(X)):
            order = np.argsort(-contrib[i])
            up = [j for j in order[:k] if contrib[i, j] > 0]
            down = [j for j in order[::-1][:k] if contrib[i, j] < 0]
            risk_txt.append(' | '.join(describe_group(self.groups[j], fe.iloc[i]) for j in up))
            protect_txt.append(' | '.join(describe_group(self.groups[j], fe.iloc[i]) for j in down))

        return pd.DataFrame({
            'customerID': fe['customerID'].values,
            'churn_probability': p,
            'risk_tier': np.where(p >= TIER_HIGH, 'Élevé', np.where(p >= TIER_MID, 'Moyen', 'Faible')),
            'expected_net_gain': gain,
            'action_recommended': np.where(gain > 0, 'Contacter', 'Ne pas contacter'),
            'top_risk_factors': risk_txt,
            'top_protective_factors': protect_txt,
        })
