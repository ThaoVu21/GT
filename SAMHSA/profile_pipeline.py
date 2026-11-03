"""
Client-Profile pipeline (RQ1 + RQ4) for the 2024 SAMHSA MH-CLD data.

Steps
  1. Load mhcld_puf_2024.csv and define inpatient_use = SPHSERVICE==1 OR OPISERVICE==1.
  2. RQ1  - cluster clients on age + clinical characteristics (one-hot + K-Means), choosing k in
           4..10 by silhouette + stability across independent samples scored on a common holdout.
  3. RQ2* - fit a logistic-regression model WITHOUT state on the well-reported fields and produce
           leave-state-out (GroupKFold by state) inpatient probabilities = expected counts. To use the
           final RQ2 model's predictions instead, pass them with --probs (columns CASEID,p_inpatient).
  4. RQ4  - aggregate observed/expected by state x profile -> O/E, overdispersion-adjusted limits,
           O-E excess, reporting-quality flags,
           and a Kitagawa decomposition of each state's raw-rate gap into
           "profile mix" vs "within-profile utilization" contributions.

Outputs go to ./outputs/profile_explorer/ and feed profile_explorer_dash.py (Dash).

Usage
  python profile_pipeline.py                 # auto-select k
  python profile_pipeline.py --k 6           # force k
  python profile_pipeline.py --probs my_oof_probs.csv
"""
import argparse
import itertools
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (adjusted_rand_score, average_precision_score, normalized_mutual_info_score,
                             roc_auc_score, silhouette_score)
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder

HERE = Path(__file__).resolve().parent
DATA = HERE / "mhcld_puf_2024.csv"
OUT = HERE / "outputs" / "profile_explorer"
SEED = 42

STATE_NAMES = {
    1: "Alabama", 2: "Alaska", 4: "Arizona", 5: "Arkansas", 6: "California", 8: "Colorado",
    9: "Connecticut", 10: "Delaware", 11: "District of Columbia", 12: "Florida", 13: "Georgia",
    15: "Hawaii", 16: "Idaho", 17: "Illinois", 18: "Indiana", 19: "Iowa", 20: "Kansas",
    21: "Kentucky", 22: "Louisiana", 23: "Maine", 24: "Maryland", 25: "Massachusetts",
    26: "Michigan", 27: "Minnesota", 28: "Mississippi", 29: "Missouri", 30: "Montana",
    31: "Nebraska", 32: "Nevada", 33: "New Hampshire", 34: "New Jersey", 35: "New Mexico",
    36: "New York", 37: "North Carolina", 38: "North Dakota", 39: "Ohio", 40: "Oklahoma",
    41: "Oregon", 42: "Pennsylvania", 44: "Rhode Island", 45: "South Carolina", 46: "South Dakota",
    47: "Tennessee", 48: "Texas", 49: "Utah", 50: "Vermont", 51: "Virginia", 53: "Washington",
    54: "West Virginia", 55: "Wisconsin", 56: "Wyoming", 72: "Puerto Rico", 99: "Other jurisdictions",
}
STATE_ABBR = {
    1: "AL", 2: "AK", 4: "AZ", 5: "AR", 6: "CA", 8: "CO", 9: "CT", 10: "DE", 11: "DC", 12: "FL",
    13: "GA", 15: "HI", 16: "ID", 17: "IL", 18: "IN", 19: "IA", 20: "KS", 21: "KY", 22: "LA",
    23: "ME", 24: "MD", 25: "MA", 26: "MI", 27: "MN", 28: "MS", 29: "MO", 30: "MT", 31: "NE",
    32: "NV", 33: "NH", 34: "NJ", 35: "NM", 36: "NY", 37: "NC", 38: "ND", 39: "OH", 40: "OK",
    41: "OR", 42: "PA", 44: "RI", 45: "SC", 46: "SD", 47: "TN", 48: "TX", 49: "UT", 50: "VT",
    51: "VA", 53: "WA", 54: "WV", 55: "WI", 56: "WY", 72: "PR", 99: "OJ",
}

DX_FLAGS = {
    "TRAUSTREFLG": "Trauma/stressor", "ANXIETYFLG": "Anxiety", "ADHDFLG": "ADHD",
    "CONDUCTFLG": "Conduct", "DELIRDEMFLG": "Delirium/dementia", "BIPOLARFLG": "Bipolar",
    "DEPRESSFLG": "Depressive", "ODDFLG": "ODD", "PDDFLG": "Pervasive developmental",
    "PERSONFLG": "Personality", "SCHIZOFLG": "Schizophrenia/psychotic", "ALCSUBFLG": "Alcohol/substance",
    "OTHERDISFLG": "Other disorder",
}

# Recode tables used for both clustering features and readable summaries (-9 = missing/unknown).
AGE_BAND = {1: "0-17", 2: "0-17", 3: "0-17", 4: "18-24", 5: "18-24", 6: "25-44", 7: "25-44",
            8: "25-44", 9: "25-44", 10: "45-64", 11: "45-64", 12: "45-64", 13: "45-64", 14: "65+"}
LABELS = {
    "AGE_BAND": {k: v for k, v in AGE_BAND.items()},
    "SEX": {1: "Male", 2: "Female"},
    "SMISED": {1: "SMI", 2: "SED/at risk", 3: "Not SMI/SED"},
    "NUMMHS": {0: "No diagnosis", 1: "1 diagnosis", 2: "2 diagnoses", 3: "3+ diagnoses"},
    "SAP": {1: "SUD", 2: "No SUD"},
    # Descriptors only (recoded PUF values, codebook appendix B).
    "RACE": {1: "American Indian/Alaska Native", 2: "Asian", 3: "Black", 4: "Native Hawaiian/Pacific Isl.",
             5: "White", 6: "Other/two or more races"},
    "ETHNIC": {1: "Mexican", 2: "Puerto Rican", 3: "Other Hispanic", 4: "Not Hispanic"},
    "MARSTAT": {1: "Never married", 2: "Now married", 3: "Separated", 4: "Divorced/widowed"},
    "EMPLOY": {1: "Full-time", 2: "Part-time", 3: "Employed (not differentiated)", 4: "Unemployed",
               5: "Not in labor force"},
    "LIVARAG": {1: "Homeless", 2: "Private residence", 3: "Other residence"},
    "EDUC": {1: "Special education", 2: "8 years or less", 3: "9-11 years", 4: "12 years/GED", 5: "13+ years"},
    "VETERAN": {1: "Veteran", 2: "Not a veteran"},
}

# Clustering: well-reported clinical fields + age (progress report, Method 1). Sex is left out because it
# split every profile into male/female twins; it is described per profile instead.
CLUSTER_CATS = ["AGE_BAND", "SMISED", "NUMMHS", "SAP"]
# Described per profile but not clustered on. Most are heavily missing (35-65% unknown), so the
# percentages are among clients with a known value.
DESCRIBE_CATS = ["SEX", "RACE", "ETHNIC", "MARSTAT", "EMPLOY", "LIVARAG", "EDUC", "VETERAN"]
# Expected-count model: the well-reported tier only (progress report, Table 1), missing = own level.
# State is never a feature; the partially reported fields are left to the sensitivity analysis.
MODEL_CATS = ["AGE", "SEX", "SMISED", "NUMMHS", "SAP", "RACE", "ETHNIC", "MH1"] + list(DX_FLAGS)

MIN_N, MIN_E = 100, 5  # small/unstable cell thresholds for flagging
K_RANGE = range(4, 11)  # candidate numbers of profiles (progress report)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load():
    cols = ["CASEID", "STATEFIP", "SPHSERVICE", "OPISERVICE", "AGE", "SEX", "SMISED", "NUMMHS", "SAP",
            "RACE", "ETHNIC", "MARSTAT", "EMPLOY", "LIVARAG", "EDUC", "VETERAN", "MH1"] + list(DX_FLAGS)
    if not DATA.exists():
        raise SystemExit(f"{DATA.name} not found next to this script (download it from SAMHSA)")
    missing = set(cols) - set(pd.read_csv(DATA, nrows=0).columns)
    if missing:
        raise SystemExit(f"{DATA.name} is missing columns: {sorted(missing)}")
    dtypes = {c: "int16" for c in cols}
    dtypes["CASEID"] = "int64"
    df = pd.read_csv(DATA, usecols=cols, dtype=dtypes)
    # The diagnosis flags are 0/1 in the 2024 PUF; clustering and prevalence assume that.
    bad = [f for f in DX_FLAGS if not df[f].isin([0, 1]).all()]
    if bad:
        raise SystemExit(f"diagnosis flags with values other than 0/1: {bad}")
    if not df.CASEID.is_unique:
        raise SystemExit("CASEID is not unique")
    df["inpatient"] = ((df.SPHSERVICE == 1) | (df.OPISERVICE == 1)).astype("int8")
    df["AGE_BAND"] = df.AGE.map(AGE_BAND).fillna("Unknown")
    log(f"loaded {len(df):,} rows; national inpatient rate {df.inpatient.mean():.4f}")
    return df


def cluster_matrix(df, enc=None):
    cats = df[CLUSTER_CATS].astype(str)
    if enc is None:
        enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float32).fit(cats)
    X = np.hstack([enc.transform(cats), df[list(DX_FLAGS)].to_numpy(np.float32)])
    return X, enc


def run_clustering(df, k_forced=None, sample_n=500_000):
    # Stability = agreement between fits on 4 independent 200k samples, all scored on one common
    # 100k holdout. (Seeds on a single sample only measure initialisation sensitivity.)
    idx = np.random.RandomState(SEED).permutation(len(df))
    if len(df) < 900_000:
        raise SystemExit(f"need at least 900,000 rows for the stability check, got {len(df):,}")
    holdout = df.iloc[idx[:100_000]]
    parts = [df.iloc[idx[100_000 + i * 200_000:100_000 + (i + 1) * 200_000]] for i in range(4)]
    sample = df.iloc[idx[:sample_n]]
    Xs, enc = cluster_matrix(sample)
    Xh, _ = cluster_matrix(holdout, enc)
    Xp = [cluster_matrix(p, enc)[0] for p in parts]

    rows = []
    for k in K_RANGE:
        fits = [KMeans(k, random_state=s, n_init=10).fit(X) for s, X in enumerate(Xp)]
        labs = [f.predict(Xh) for f in fits]
        ari = np.mean([adjusted_rand_score(a, b) for a, b in itertools.combinations(labs, 2)])
        sil = silhouette_score(Xh[:20_000], labs[0][:20_000])
        sizes = np.bincount(labs[0], minlength=k) / len(Xh)
        rows.append({"k": k, "silhouette": sil, "stability_ari": ari, "min_cluster_share": sizes.min(),
                     "inertia": fits[0].inertia_})
        log(f"k={k}: silhouette={sil:.3f} ARI={ari:.3f} min share={sizes.min():.3f}")
    sel = pd.DataFrame(rows)
    if k_forced:
        k = k_forced
    else:
        # Stable (ARI >= .8), no tiny clusters (>= 2%), then best silhouette.
        ok = sel[(sel.stability_ari >= 0.8) & (sel.min_cluster_share >= 0.02)]
        k = int((ok if len(ok) else sel).sort_values("silhouette", ascending=False).k.iloc[0])
    log(f"using k={k}")

    km = KMeans(k, random_state=SEED, n_init=10).fit(Xs)
    labels = np.empty(len(df), dtype=np.int16)
    for start in range(0, len(df), 1_000_000):
        X, _ = cluster_matrix(df.iloc[start:start + 1_000_000], enc)
        labels[start:start + 1_000_000] = km.predict(X)
    # Order profiles by size so P1 is the largest.
    order = pd.Series(labels).value_counts().index
    remap = {old: new for new, old in enumerate(order)}
    df["profile_id"] = pd.Series(labels).map(remap).to_numpy() + 1
    sel["selected"] = sel.k == k
    return sel


def describe_profiles(df):
    """Profile characteristic table (long format) + auto-generated descriptive labels."""
    nat = {}
    rows = []
    for var in CLUSTER_CATS:
        lab = LABELS.get(var, {})
        col = df[var].map(lambda v: lab.get(v, v) if v != -9 else "Unknown") if var != "AGE_BAND" else df[var]
        ct = pd.crosstab(df.profile_id, col, normalize="index")
        nat[var] = col.value_counts(normalize=True)
        for pid in ct.index:
            for level, pct in ct.loc[pid].items():
                rows.append({"profile_id": pid, "group": var, "feature": f"{var}: {level}", "pct": pct,
                             "national_pct": nat[var][level], "in_clustering": True})
    for var in DESCRIBE_CATS:
        known = df[df[var] != -9]
        col = known[var].map(LABELS[var])
        ct = pd.crosstab(known.profile_id, col, normalize="index")
        nat_pct = col.value_counts(normalize=True)
        coverage = (df[var] != -9).groupby(df.profile_id).mean()  # share of the profile with a known value
        for pid in ct.index:
            for level, pct in ct.loc[pid].items():
                rows.append({"profile_id": pid, "group": var, "feature": f"{var}: {level}", "pct": pct,
                             "national_pct": nat_pct[level], "in_clustering": False,
                             "known_pct": coverage[pid], "national_known_pct": len(known) / len(df)})
    flag_pct = df.groupby("profile_id")[list(DX_FLAGS)].mean()
    flag_nat = df[list(DX_FLAGS)].mean()
    for pid in flag_pct.index:
        for f, name in DX_FLAGS.items():
            rows.append({"profile_id": pid, "group": "Diagnosis", "feature": f"Diagnosis: {name}",
                         "pct": flag_pct.loc[pid, f], "national_pct": flag_nat[f], "in_clustering": True})
    feats = pd.DataFrame(rows)
    feats["lift"] = feats.pct / feats.national_pct.replace(0, np.nan)

    labels = {}
    for pid, g in feats.groupby("profile_id"):
        pct = g.set_index("feature").pct
        youth = pct.get("AGE_BAND: 0-17", 0)
        older = pct.get("AGE_BAND: 45-64", 0) + pct.get("AGE_BAND: 65+", 0)
        if youth >= 0.5:
            age_txt = "Youth 0-17"
        elif youth <= 0.1 and older >= 0.5:
            age_txt = "Adults 45+"
        elif youth <= 0.1:
            age_txt = "Adults"
        else:
            age_txt = "Mixed ages"
        smi = g[g.group == "SMISED"].sort_values("pct").iloc[-1]
        smi_txt = smi.feature.split(": ")[1] if smi.pct >= 0.5 else "Mixed SMI/SED"
        # Most distinctive remaining traits: common in the profile (>=40%) and over-represented (lift>=1.3).
        traits = g[g.group.isin(["Diagnosis", "NUMMHS", "SAP"]) & (g.pct >= 0.4) & (g.lift >= 1.3)
                   & ~g.feature.str.endswith("Unknown")].sort_values("lift", ascending=False)
        trait_txt = " + ".join(traits.feature.str.split(": ").str[1].head(2)) or "mixed diagnoses"
        labels[pid] = f"P{pid} · {age_txt} · {smi_txt} · {trait_txt}"
    feats["profile"] = feats.profile_id.map(labels)
    return feats, labels


def oof_probabilities(df, n_splits=5, train_n=1_200_000):
    """Leave-state-out predictions (progress report, Method 2): 5-fold GroupKFold with state as the group,
    so every client is scored by a model that never saw their state. The encoder is fit inside each
    fold, and each fold trains on a random 1.2M-client sample of its training states."""
    X = df[MODEL_CATS].astype("int16")
    y = df.inpatient.to_numpy()
    p = np.zeros(len(df), dtype=np.float64)
    rng = np.random.RandomState(SEED)
    for i, (tr, te) in enumerate(GroupKFold(n_splits).split(X, y, groups=df.STATEFIP), 1):
        tr = rng.choice(tr, min(train_n, len(tr)), replace=False)
        m = make_pipeline(OneHotEncoder(handle_unknown="ignore", dtype=np.float32, min_frequency=50),
                          LogisticRegression(max_iter=300, C=1.0, solver="lbfgs"))
        m.fit(X.iloc[tr], y[tr])
        p[te] = m.predict_proba(X.iloc[te])[:, 1]
        log(f"fold {i}/{n_splits} done ({df.STATEFIP.iloc[te].nunique()} held-out states)")
    log(f"OOF ROC-AUC={roc_auc_score(y, p):.3f}  PR-AUC={average_precision_score(y, p):.3f}  "
        f"sum(E)/sum(O)={p.sum() / y.sum():.4f}")
    return p


def poisson_ci(o, e, alpha=0.05):
    """Exact Poisson interval for O/E treating E as fixed. Ignores prediction uncertainty and
    between-state overdispersion, so it is far too narrow for state comparisons; kept for reference."""
    with np.errstate(divide="ignore", invalid="ignore"):
        lo = np.where(o > 0, chi2.ppf(alpha / 2, 2 * o) / 2, 0.0) / e
        hi = chi2.ppf(1 - alpha / 2, 2 * (o + 1)) / 2 / e
    return np.where(e > 0, lo, np.nan), np.where(e > 0, hi, np.nan)


def overdispersed_limits(o, e, usable):
    """Overdispersion-adjusted intervals (progress report, Method 3; Spiegelhalter 2005).
    On log(O/E) with SE 1/sqrt(E): estimate phi from z-scores winsorised at 10%, convert it to an additive
    random-effects variance tau^2, then use SE = sqrt(1/E + tau^2). `usable` marks the units used to
    estimate phi (no small cells, O > 0). Returns 95% interval around O/E, 99.8% outlier flag, phi, tau^2."""
    o, e = np.asarray(o, float), np.asarray(e, float)
    ok = usable & (o > 0) & (e > 0)
    z = np.log(o[ok] / e[ok]) * np.sqrt(e[ok])
    z = np.clip(z, *np.percentile(z, [10, 90]))
    n, w = len(z), e[ok]
    phi = np.mean(z ** 2)
    tau2 = max(0.0, (n * phi - (n - 1)) / (w.sum() - (w ** 2).sum() / w.sum()))
    with np.errstate(divide="ignore", invalid="ignore"):
        se = np.sqrt(1 / np.where(e > 0, e, np.nan) + tau2)
        oe = np.where(e > 0, o / e, np.nan)
        log_oe = np.log(np.where(o > 0, oe, np.nan))
    lo = np.where(o > 0, oe * np.exp(-1.96 * se), 0.0)
    hi = np.where(o > 0, oe * np.exp(1.96 * se), np.exp(1.96 * se) * 0.5 / e)  # O = 0: use 0.5 as O
    outlier = np.abs(np.nan_to_num(log_oe)) > 3.09 * se  # outside 99.8% funnel limits
    return np.where(e > 0, lo, np.nan), np.where(e > 0, hi, np.nan), outlier, phi, tau2


def aggregate(df, labels):
    nat_rate = df.inpatient.mean()
    sp = (df.groupby(["STATEFIP", "profile_id"])
            .agg(n=("inpatient", "size"), observed=("inpatient", "sum"), expected=("p_inpatient", "sum"))
            .reset_index())
    st = (df.groupby("STATEFIP")
            .agg(n=("inpatient", "size"), observed=("inpatient", "sum"), expected=("p_inpatient", "sum"))
            .reset_index())
    prof = (df.groupby("profile_id")
              .agg(n=("inpatient", "size"), observed=("inpatient", "sum"), expected=("p_inpatient", "sum"))
              .reset_index())
    prof["share"] = prof.n / prof.n.sum()
    prof["rate"] = prof.observed / prof.n

    for t in (sp, st):
        t["rate"] = t.observed / t.n
        t["oe"] = (t.observed / t.expected).where(t.expected > 0)
        t["oe_lo_pois"], t["oe_hi_pois"] = poisson_ci(t.observed.to_numpy(), t.expected.to_numpy())
        t["excess"] = t.observed - t.expected
        t["adj_rate"] = t.oe * nat_rate
        t["state"] = t.STATEFIP.map(STATE_NAMES)
        t["abbr"] = t.STATEFIP.map(STATE_ABBR)
        t["small_cell"] = (t.n < MIN_N) | (t.expected < MIN_E)
    sp["profile"] = sp.profile_id.map(labels)

    # Overdispersion-adjusted limits: across states overall, and across states within each profile.
    lo, hi, out, phi, tau2 = overdispersed_limits(st.observed, st.expected, ~st.small_cell & (st.n >= 1000))
    st["oe_lo"], st["oe_hi"], st["outlier_998"] = lo, hi, out
    log(f"state O/E: phi={phi:,.0f} tau^2={tau2:.3f}; {out.sum()} states outside 99.8% limits")
    for pid, idx in sp.groupby("profile_id").groups.items():
        c = sp.loc[idx]
        lo, hi, out, phi, tau2 = overdispersed_limits(c.observed, c.expected, ~c.small_cell)
        sp.loc[idx, "oe_lo"], sp.loc[idx, "oe_hi"], sp.loc[idx, "outlier_998"] = lo, hi, out
        log(f"P{pid} O/E: phi={phi:,.0f} tau^2={tau2:.3f}; {out.sum()} cells outside 99.8% limits")
    sp["outlier_998"] = sp.outlier_998.astype(bool)
    sp["share_in_state"] = sp.n / sp.groupby("STATEFIP").n.transform("sum")
    sp = sp.merge(prof[["profile_id", "share", "rate"]].rename(columns={"share": "national_share",
                                                                        "rate": "national_rate"}),
                  on="profile_id")
    # Signed contribution of each profile to the state's excess: profile O-E divided by the state's total
    # |O-E|. Absolute values sum to 1 within a state; this is not a conventional percentage share.
    sp["excess_contribution"] = sp.excess / sp.groupby("STATEFIP").excess.transform(lambda s: s.abs().sum())

    # Kitagawa decomposition of (state raw rate - national raw rate) by profile:
    #   mix    = (share_state - share_nat) * (rate_state + rate_nat) / 2
    #   within = (rate_state - rate_nat)  * (share_state + share_nat) / 2
    sp["mix_effect"] = (sp.share_in_state - sp.national_share) * (sp.rate + sp.national_rate) / 2
    sp["within_effect"] = (sp.rate - sp.national_rate) * (sp.share_in_state + sp.national_share) / 2
    # Convention for profiles absent from a state: the state rate is undefined, so it is set to the national
    # rate. The whole contribution (-share_nat * rate_nat) is then a mix effect and the within effect is 0,
    # which keeps the decomposition exact (mix + within summed over profiles = state rate - national rate).
    full = pd.MultiIndex.from_product([st.STATEFIP, prof.profile_id], names=["STATEFIP", "profile_id"])
    missing = full.difference(sp.set_index(["STATEFIP", "profile_id"]).index).to_frame(index=False)
    if len(missing):
        missing = missing.merge(prof[["profile_id", "share", "rate"]], on="profile_id")
        missing["mix_effect"] = -missing.share * missing.rate
        missing["within_effect"] = 0.0
        missing = missing.rename(columns={"share": "national_share", "rate": "national_rate"})
        missing = missing.assign(n=0, observed=0, expected=0.0, share_in_state=0.0, small_cell=True,
                                 outlier_998=False,
                                 state=missing.STATEFIP.map(STATE_NAMES), abbr=missing.STATEFIP.map(STATE_ABBR),
                                 profile=missing.profile_id.map(labels))
        sp = pd.concat([sp, missing], ignore_index=True)
        log(f"{len(missing)} empty state x profile cells (kept as n=0 rows)")

    gap = sp.groupby("STATEFIP")[["mix_effect", "within_effect"]].sum().sum(axis=1)
    assert np.allclose(gap.reindex(st.STATEFIP).to_numpy(), st.rate - nat_rate), "decomposition does not add up"
    assert sp.n.sum() == st.n.sum() == prof.n.sum() == len(df), "state x profile counts do not add up"

    # Reporting quality (progress report, Method 3): inpatient settings reported and key-field missingness.
    q = df.groupby("STATEFIP").agg(reports_sph=("SPHSERVICE", lambda s: (s == 1).any()),
                                   reports_opi=("OPISERVICE", lambda s: (s == 1).any()),
                                   smised_missing=("SMISED", lambda s: (s == -9).mean()),
                                   mh1_missing=("MH1", lambda s: (s == -9).mean()))
    q["scope"] = np.select([q.reports_sph & q.reports_opi, q.reports_sph, q.reports_opi],
                           ["both inpatient settings", "state hospitals only", "other inpatient only"], "none")
    st = st.merge(q.reset_index(), on="STATEFIP")
    st["near_zero_rate"] = st.rate < 0.005  # e.g. Arizona, Kansas: settings "reported" but almost no inpatients
    st["reporting_flag"] = ((st.scope != "both inpatient settings") | (st.smised_missing > 0.3)
                            | (st.mh1_missing > 0.3) | st.near_zero_rate)
    q = st.set_index("STATEFIP")
    log(f"{q.reporting_flag.sum()} states flagged for reporting scope or >30% missing SMISED/MH1")

    prof["profile"] = prof.profile_id.map(labels)
    prof["oe"] = prof.observed / prof.expected
    # Reporting-artifact audit (progress report, Method 1): is a profile concentrated in one state?
    state_share = df.STATEFIP.value_counts(normalize=True)
    in_prof = df.groupby("profile_id").STATEFIP.value_counts(normalize=True).rename("p").reset_index()
    in_prof["lift"] = in_prof.p / in_prof.STATEFIP.map(state_share)
    top = in_prof.sort_values("p", ascending=False).drop_duplicates("profile_id").set_index("profile_id")
    prof["top_state"] = prof.profile_id.map(top.STATEFIP.map(STATE_NAMES))
    prof["top_state_share"] = prof.profile_id.map(top.p)
    prof["top_state_lift"] = prof.profile_id.map(top.lift)
    st["small_state"] = st.n < 1000
    return sp, st, prof


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, help="force number of profiles")
    ap.add_argument("--probs", type=Path, help="CSV with CASEID,p_inpatient (out-of-fold) to use instead")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    df = load()
    sel = run_clustering(df, args.k)
    nmi = normalized_mutual_info_score(df.STATEFIP, df.profile_id)
    sel.loc[sel.selected, "nmi_with_state"] = nmi
    log(f"NMI(profile, state) = {nmi:.3f} (near 0 = profiles are not state proxies)")
    feats, labels = describe_profiles(df)

    if args.probs:
        pr = pd.read_csv(args.probs, usecols=["CASEID", "p_inpatient"])
        problems = [msg for bad, msg in [
            (pr.CASEID.duplicated().any(), "duplicate CASEIDs"),
            (pr.p_inpatient.isna().any(), "missing probabilities"),
            (~pr.p_inpatient.between(0, 1).all(), "probabilities outside [0, 1]"),
            (not set(df.CASEID) <= set(pr.CASEID), "does not cover every CASEID in the data"),
        ] if bad]
        if problems:
            raise SystemExit(f"{args.probs}: " + "; ".join(problems))
        n_before = len(df)
        df = df.merge(pr, on="CASEID", how="left", validate="one_to_one")
        assert len(df) == n_before
        log(f"using external probabilities from {args.probs}; sum(E)/sum(O)="
            f"{df.p_inpatient.sum() / df.inpatient.sum():.4f}")
    else:
        df["p_inpatient"] = oof_probabilities(df)

    sp, st, prof = aggregate(df, labels)

    sel.to_csv(OUT / "cluster_selection.csv", index=False)
    feats.to_csv(OUT / "profile_features.csv", index=False)
    prof.to_csv(OUT / "profile_summary.csv", index=False)
    sp.to_csv(OUT / "state_profile.csv", index=False)
    st.to_csv(OUT / "state_summary.csv", index=False)
    df[["CASEID", "STATEFIP", "profile_id", "inpatient", "p_inpatient"]].to_parquet(
        OUT / "client_profiles.parquet", index=False)
    log(f"wrote outputs to {OUT}")
    print(prof[["profile", "n", "share", "rate", "oe"]].to_string(index=False))


if __name__ == "__main__":
    main()
