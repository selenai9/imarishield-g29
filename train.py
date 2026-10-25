"""ImariShield training pipeline.

Reads the FinScope Rwanda 2024 microdata (NISR) and writes everything the dashboard needs to ./artifacts and ./data:
  * the deployed model (XGBoost, one per target) and its input specification
  * survey-weighted district and segment estimates with confidence intervals
  * SHAP-style drivers, a fairness audit and association-based scenario results
  * a multi-task PyTorch network trained for comparison, logged to TensorBoard (./runs)

Usage:  python train.py --data "FinScope_2024_Rwanda.dta"
The raw microdata is NOT redistributed in this repository; download it from the NISR microdata portal
(see data/README_DATA.md).
"""
import argparse, json, os, re, random, warnings
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
import xgboost as xgb

warnings.filterwarnings("ignore")
SEED = 42
random.seed(SEED); np.random.seed(SEED)

# ----------------------------------------------------------------------------- configuration
TARGETS = {  # name -> (FinScope column, resilient answers, vulnerable answers matcher)
    "savings": ("e7d", ["You could do it", "You probably could do it"]),
    "expense": ("e9c", ["Not difficult at all", "Somewhat difficult"]),
}
WEIGHT_COL = "pop_wt"
CAT_COLS = ["a2", "a6", "b2", "b3", "c4a", "n1b", "n4", "c13a", "d1b5", "d1a7"]
CAT_NAMES = {"a2": "District", "a6": "Urban or rural", "b2": "Sex", "b3": "Marital status", "c4a": "Education",
             "n1b": "Main source of income", "n4": "Working status", "c13a": "Ubudehe category",
             "d1b5": "Time to health facility", "d1a7": "Distance to SACCO"}
FLAG_PREFIXES = ["n1a_", "qf1_", "qf4_", "g3a_", "g10_", "h3_", "i2_", "i5a_", "i6_", "c12_", "cc1_"]
FLAG_SINGLES = ["h2_1", "c13d", "f3", "c14_1", "c14_3", "c14_5"]
YES_VALUES = {"Yes", "Have", "Yes, have access", "yes"}
RARE = {"Don?t know", "Appellxed", "Do not know"}

GROUP_PREFIX = {"n1a_": "Income from", "qf1_": "Ever used", "qf4_": "Currently use", "g3a_": "Saves via",
                "g10_": "Other savings", "h3_": "Borrowed from", "i2_": "Experienced", "i5a_": "Has",
                "i6_": "Insurance", "c12_": "Owns", "cc1_": "Climate problem"}


def friendly_label(col, raw_label):
    """Turn a FinScope variable label into a short readable name."""
    s = re.sub(r"^\s*[A-Za-z0-9_]+(\.\d+)?\s*[\.:]\s*", "", str(raw_label)).strip()
    s = re.sub(r"^(Does you or your household own-|Do you have:|Saving (with|through)\s*-?\s*|Insurance product\s*-\s*|climate change:\s*|"
               r"Have you invested money or bought-|Other types of savings -\s*|Borrowed money (from )?|Ever used\s*-\s*|Do you currently use\s*-\s*)", "", s, flags=re.I).strip()
    s = s.replace("  ", " ")
    for p, pre in GROUP_PREFIX.items():
        if col.startswith(p):
            return f"{pre}: {s}"
    names = {"h2_1": "Borrowed money in past 12 months", "c13d": "Received a VUP cash transfer", "f3": "Owns a mobile phone",
             "c14_1": "Has access to a cell phone", "c14_3": "Has access to a computer", "c14_5": "Has access to the internet",
             "b1": "Age", "hh_size": "Household size", "n11": "Personal monthly income band", "n11_missing": "Income not reported",
             "female_head": "Female-headed household"}
    return names.get(col, s)


# ----------------------------------------------------------------------------- data
def load_raw(path):
    labels = pd.io.stata.StataReader(path).variable_labels()
    return pd.read_stata(path, convert_categoricals=True), labels


def is_e7d_negative(s):
    s = s.astype(str)
    return s.str.startswith("It is very unlikely") | s.str.contains("couldn", regex=False)


def build_targets(df):
    out = df.copy()
    out["y_savings"] = np.where(out["e7d"].isin(TARGETS["savings"][1]), 1, np.where(is_e7d_negative(out["e7d"]), 0, np.nan))
    out["y_expense"] = np.where(out["e9c"].isin(TARGETS["expense"][1]), 1, np.where(out["e9c"] == "Very difficult", 0, np.nan))
    return out


def flag_columns(frame):
    cols = [c for c in frame.columns if any(c.startswith(p) for p in FLAG_PREFIXES)] + FLAG_SINGLES
    return sorted({c for c in cols if "other_specify" not in c and frame[c].nunique(dropna=True) <= 3})


def build_features(frame, flag_cols, n11_levels=None):
    X = pd.DataFrame(index=frame.index)
    for c in CAT_COLS:
        X[c] = frame[c].astype(str).fillna("missing")
    for c in flag_cols:
        X[c] = frame[c].astype(str).isin(YES_VALUES).astype(int)
    X["female_head"] = (frame["hh_head"].astype(str) == "female").astype(int)
    X["b1"] = pd.to_numeric(frame["b1"], errors="coerce")
    X["hh_size"] = pd.to_numeric(frame["hh_size"], errors="coerce")
    codes = frame["n11"].cat.codes
    X["n11"] = codes + 1
    X["n11_missing"] = (codes < 0).astype(int)
    return X


def leakage_guard(columns):
    bad = [c for c in columns if re.match(r"^e\d", c) or c.startswith("c5_") or c in ("e7d", "e9c", "y_savings", "y_expense")]
    assert not bad, f"leakage columns in features: {bad}"


def to_xgb_frame(X, cat_levels):
    out = X.copy()
    for c in CAT_COLS:
        out[c] = pd.Categorical(out[c], categories=cat_levels[c])
    return out


# ----------------------------------------------------------------------------- stats helpers
def wmean(x, w):
    return float(np.average(x, weights=w))


def boot_ci(y, w, reps=400, seed=0):
    rng = np.random.RandomState(seed)
    n = len(y)
    est = [np.average(y[i], weights=w[i]) for i in (rng.randint(0, n, n) for _ in range(reps))]
    return float(np.percentile(est, 2.5)), float(np.percentile(est, 97.5))


def wquantiles(p, w, n=101):
    order = np.argsort(p); cw = np.cumsum(w[order]) / w.sum()
    return [float(p[order][min(np.searchsorted(cw, q), len(p) - 1)]) for q in np.linspace(0, 1, n)]


def metrics(y, p, w=None):
    return {"auc": float(roc_auc_score(y, p, sample_weight=w)), "ap": float(average_precision_score(y, p, sample_weight=w)),
            "brier": float(brier_score_loss(y, p, sample_weight=w))}


# ----------------------------------------------------------------------------- main
def main(args):
    os.makedirs("artifacts", exist_ok=True); os.makedirs("data", exist_ok=True)
    raw, labels = load_raw(args.data)
    full = build_targets(raw)
    df = full.dropna(subset=["y_savings", "y_expense"]).reset_index(drop=True)
    df[["y_savings", "y_expense"]] = df[["y_savings", "y_expense"]].astype(int)
    print("respondents", len(raw), "| modelling sample", len(df))

    flag_cols = flag_columns(raw)
    X_all = build_features(df, flag_cols)
    leakage_guard(X_all.columns)
    cat_levels = {c: [v for v in X_all[c].value_counts().index] for c in CAT_COLS}  # most frequent first
    num_cols = [c for c in X_all.columns if c not in CAT_COLS]
    feature_names = CAT_COLS + num_cols
    friendly = {c: CAT_NAMES.get(c) or friendly_label(c, labels.get(c, c)) for c in feature_names}
    w = df[WEIGHT_COL].values.astype(float)
    wn = w / w.mean()
    y = {"savings": df["y_savings"].values, "expense": df["y_expense"].values}

    idx = np.arange(len(df))
    tr, tmp = train_test_split(idx, test_size=0.30, stratify=y["savings"], random_state=SEED)
    va, te = train_test_split(tmp, test_size=0.50, stratify=y["savings"][tmp], random_state=SEED)
    Xg = to_xgb_frame(X_all[feature_names], cat_levels)

    # ---------------- XGBoost: small validation search, native categorical handling, survey weights
    models, report, test_prob = {}, {}, {}
    for t in ("savings", "expense"):
        best = None
        for depth in (3, 4, 5):
            for mcw in (5, 20):
                m = xgb.XGBClassifier(n_estimators=1500, learning_rate=0.03, max_depth=depth, min_child_weight=mcw,
                                      subsample=0.8, colsample_bytree=0.7, reg_lambda=5.0, tree_method="hist",
                                      enable_categorical=True, max_cat_to_onehot=8, eval_metric="auc",
                                      early_stopping_rounds=60, random_state=SEED, n_jobs=-1)
                m.fit(Xg.iloc[tr], y[t][tr], sample_weight=wn[tr], eval_set=[(Xg.iloc[va], y[t][va])],
                      sample_weight_eval_set=[wn[va]], verbose=False)
                score = m.best_score
                if best is None or score > best[0]:
                    best = (score, m, depth, mcw)
        score, m, depth, mcw = best
        models[t] = m
        p = m.predict_proba(Xg.iloc[te])[:, 1]
        test_prob[t] = p
        report[t] = {"xgb_val_auc": float(score), "best_depth": depth, "min_child_weight": mcw,
                     "best_iteration": int(m.best_iteration), "xgb_test": metrics(y[t][te], p),
                     "xgb_test_weighted": metrics(y[t][te], p, wn[te]),
                     "base_rate_weighted": wmean(y[t], w)}
        m.save_model(f"artifacts/xgb_{t}.json")
        print(t, report[t]["xgb_test"], "depth", depth)

    # ---------------- multi-task PyTorch network (comparison model, logged to TensorBoard)
    if not args.skip_nn:
        report["mlp"] = train_mlp(X_all, feature_names, num_cols, y, wn, tr, va, te, report)

    # ---------------- decide the deployed model and risk tiers from the validation set
    pv = {t: models[t].predict_proba(Xg.iloc[va])[:, 1] for t in models}
    pall = {t: models[t].predict_proba(Xg)[:, 1] for t in models}
    tiers = {}
    for t in models:
        # population-weighted terciles of predicted resilience -> high / medium / lower risk
        order = np.argsort(pall[t]); cw = np.cumsum(w[order]) / w.sum()
        q1 = float(pall[t][order][np.searchsorted(cw, 1 / 3)]); q2 = float(pall[t][order][np.searchsorted(cw, 2 / 3)])
        tiers[t] = {"high_risk_below": q1, "lower_risk_above": q2}

    # ---------------- survey-weighted district and segment estimates (descriptive, not model-based)
    pop = full.groupby("a2", observed=True)[WEIGHT_COL].sum()
    rows = []
    for d, g in df.groupby("a2", observed=True):
        row = {"district": d, "n": len(g), "adults_represented": float(pop[d])}
        for t in ("savings", "expense"):
            yy, ww = g[f"y_{t}"].values, g[WEIGHT_COL].values.astype(float)
            p = wmean(yy, ww); lo, hi = boot_ci(yy, ww)
            row.update({f"resilient_{t}": p, f"resilient_{t}_lo": lo, f"resilient_{t}_hi": hi,
                        f"not_resilient_adults_{t}": (1 - p) * float(pop[d])})
        rows.append(row)
    district = pd.DataFrame(rows).sort_values("district")
    district.to_csv("data/district_estimates.csv", index=False)

    seg_defs = {"Sex": "b2", "Urban or rural": "a6", "Education": "c4a", "Main source of income": "n1b",
                "Working status": "n4", "Ubudehe category": "c13a", "Marital status": "b3"}
    seg_rows = []
    dseg = df.copy()
    dseg["Age group"] = pd.cut(pd.to_numeric(dseg["b1"]), [15, 24, 34, 44, 54, 64, 120],
                               labels=["16-24", "25-34", "35-44", "45-54", "55-64", "65+"]).astype(str)
    seg_defs["Age group"] = "Age group"
    for name, col in seg_defs.items():
        for lvl, g in dseg.groupby(col, observed=True):
            if len(g) < 40 or str(lvl) in RARE:
                continue
            lvl_txt = {"Government": "Government (grant or pension)"}.get(str(lvl), str(lvl))
            r = {"segment": name, "group": lvl_txt, "n": len(g)}
            for t in ("savings", "expense"):
                yy, ww = g[f"y_{t}"].values, g[WEIGHT_COL].values.astype(float)
                lo, hi = boot_ci(yy, ww, reps=300)
                r.update({f"resilient_{t}": wmean(yy, ww), f"resilient_{t}_lo": lo, f"resilient_{t}_hi": hi})
            seg_rows.append(r)
    pd.DataFrame(seg_rows).to_csv("data/segment_estimates.csv", index=False)

    # district profile: weighted prevalence of conditions that drive fragility
    prof_defs = {"Does not save at all": ("g3a_17", 1), "Saves via a bank": ("g3a_01", 1), "Saves via a SACCO": ("g3a_08", 1),
                 "Saves via a savings group": ("g3a_11", 1), "Owns a mobile phone": ("f3", 1), "Has internet access": ("c14_5", 1),
                 "Received VUP transfer": ("c13d", 1), "Income from farming": ("n1a_6", 1), "Income from piece work": ("n1a_17", 1),
                 "Household member lost job or income": ("i2_1", 1), "Serious illness in household": ("i2_2", 1),
                 "Experienced drought or water scarcity": ("cc1_3", 1), "Experienced heavy rain or floods": ("cc1_1", 1)}
    prof_rows = []
    for d, idxs in df.groupby("a2", observed=True).groups.items():
        for name, (col, _) in prof_defs.items():
            if col in X_all.columns:
                prof_rows.append({"district": d, "indicator": name, "share": wmean(X_all.loc[idxs, col].values, w[idxs])})
    for name, (col, _) in prof_defs.items():
        if col in X_all.columns:
            prof_rows.append({"district": "Rwanda", "indicator": name, "share": wmean(X_all[col].values, w)})
    pd.DataFrame(prof_rows).to_csv("data/district_profile.csv", index=False)

    # ---------------- drivers (SHAP-style contributions from the deployed XGBoost)
    drivers = []
    for t, m in models.items():
        contrib = m.get_booster().predict(xgb.DMatrix(Xg.iloc[te], enable_categorical=True), pred_contribs=True)[:, :-1]
        for j, f in enumerate(feature_names):
            r = {"target": t, "feature": f, "label": friendly[f], "mean_abs": float(np.abs(contrib[:, j]).mean())}
            if f in flag_cols or f in ("female_head", "n11_missing"):
                on = Xg.iloc[te][f].values == 1
                r["effect_when_present"] = float(contrib[on, j].mean()) if on.sum() >= 20 else np.nan
                r["share_present"] = float(on.mean())
            drivers.append(r)
    pd.DataFrame(drivers).sort_values(["target", "mean_abs"], ascending=[True, False]).to_csv("data/drivers.csv", index=False)

    # ---------------- fairness audit on the held-out test set (one fixed rule for every group)
    # A person is "flagged" when the model puts their chance of being resilient below FLAG_BELOW.
    FLAG_BELOW = 0.20
    fair = []
    dte = df.iloc[te].copy(); dte["Age group"] = dseg.iloc[te]["Age group"].values
    for t in models:
        dte[f"p_{t}"] = test_prob[t]
    for label, col in [("Sex", "b2"), ("Urban or rural", "a6"), ("Education", "c4a"), ("Ubudehe category", "c13a"), ("District", "a2")]:
        for lvl, g in dte.groupby(col, observed=True):
            if len(g) < 25 or str(lvl) in RARE:
                continue
            r = {"dimension": label, "group": str(lvl), "n_test": len(g)}
            for t in models:
                yy, pp = g[f"y_{t}"].values, g[f"p_{t}"].values
                flagged = pp < FLAG_BELOW
                r.update({f"observed_resilient_{t}": float(yy.mean()), f"mean_predicted_{t}": float(pp.mean()),
                          f"auc_{t}": float(roc_auc_score(yy, pp)) if len(set(yy)) == 2 else np.nan,
                          f"flagged_{t}": float(flagged.mean()),
                          f"recall_nonresilient_{t}": float(flagged[yy == 0].mean()) if (yy == 0).sum() else np.nan,
                          f"wrongly_flagged_resilient_{t}": float(flagged[yy == 1].mean()) if (yy == 1).sum() else np.nan})
            fair.append(r)
    pd.DataFrame(fair).to_csv("data/fairness.csv", index=False)

    # ---------------- calibration on the test set (predicted vs observed, by decile of predicted chance)
    cal = []
    for t in models:
        d = pd.DataFrame({"p": test_prob[t], "y": y[t][te]})
        d["bin"] = pd.qcut(d["p"], 10, duplicates="drop")
        for b_, g in d.groupby("bin", observed=True):
            cal.append({"target": t, "mean_predicted": float(g["p"].mean()), "observed": float(g["y"].mean()), "n": len(g)})
    pd.DataFrame(cal).to_csv("data/calibration.csv", index=False)

    # ---------------- association-based levers: change in predicted chance for people who do NOT yet have the lever
    levers = {
        "Saves through a bank account": {"g3a_01": 1, "g3a_17": 0},
        "Saves through a SACCO": {"g3a_08": 1, "g3a_17": 0},
        "Saves in a savings group (VSLA, ikimina)": {"g3a_11": 1, "g3a_17": 0},
        "Saves through mobile money": {"g3a_03": 1, "g3a_17": 0},
        "Uses mobile money": {"qf1_05": 1, "qf4_05": 1},
        "Owns a mobile phone": {"f3": 1},
        "Has internet access": {"c14_5": 1},
        "Receives a VUP cash transfer": {"c13d": 1},
        "Holds medical insurance": {"i5a_1": 1},
    }
    lever_key = {"Saves through a bank account": "g3a_01", "Saves through a SACCO": "g3a_08",
                 "Saves in a savings group (VSLA, ikimina)": "g3a_11", "Saves through mobile money": "g3a_03",
                 "Uses mobile money": "qf1_05", "Owns a mobile phone": "f3", "Has internet access": "c14_5",
                 "Receives a VUP cash transfer": "c13d", "Holds medical insurance": "i5a_1"}
    srows = []
    for name, edits in levers.items():
        edits = {k: v for k, v in edits.items() if k in X_all.columns}
        lacking = (X_all[lever_key[name]] == 0).values
        Xe = X_all[feature_names].copy()
        for k, v in edits.items():
            Xe[k] = v
        Xe = to_xgb_frame(Xe, cat_levels)
        r = {"lever": name, "share_without": float(wmean(lacking.astype(float), w))}
        for t, m in models.items():
            diff = m.predict_proba(Xe)[:, 1] - pall[t]
            r[f"change_{t}"] = wmean(diff[lacking], w[lacking])
            for grp in ("Rural", "Urban"):
                msk = lacking & (df["a6"] == grp).values
                r[f"change_{t}_{grp.lower()}"] = wmean(diff[msk], w[msk]) if msk.sum() > 30 else np.nan
        srows.append(r)
    pd.DataFrame(srows).to_csv("data/levers.csv", index=False)

    # ---------------- headline numbers
    emerg = (raw["e9b"] == "Yes").astype(float).values
    both_fail = ((y["savings"] == 0) & (y["expense"] == 0)).astype(float)
    both_ok = ((y["savings"] == 1) & (y["expense"] == 1)).astype(float)
    headline = {"adults_represented": float(raw[WEIGHT_COL].sum()), "respondents": int(len(raw)), "modelling_sample": int(len(df)),
                "not_resilient_savings": 1 - wmean(y["savings"], w), "not_resilient_savings_ci": [1 - c for c in boot_ci(y["savings"], w)[::-1]],
                "not_resilient_expense": 1 - wmean(y["expense"], w), "not_resilient_expense_ci": [1 - c for c in boot_ci(y["expense"], w)[::-1]],
                "fail_both": wmean(both_fail, w), "pass_both": wmean(both_ok, w),
                "emergency_fund": wmean(emerg, raw[WEIGHT_COL].values.astype(float))}
    json.dump(headline, open("data/headline.json", "w"), indent=1)

    # ---------------- model specification for the app
    defaults = {}
    for c in CAT_COLS:
        defaults[c] = cat_levels[c][0]
    meta = {
        "cat_cols": CAT_COLS, "num_cols": num_cols, "flag_cols": flag_cols, "feature_names": feature_names,
        "cat_levels": cat_levels, "labels": friendly, "defaults": defaults,
        "n11_levels": list(raw["n11"].cat.categories),
        "age_median": float(pd.to_numeric(df["b1"]).median()), "hh_size_median": float(pd.to_numeric(df["hh_size"]).median()),
        "tiers": tiers,
        "national": {t: wmean(y[t], w) for t in y},
        "n_modelling": int(len(df)), "n_total": int(len(raw)),
        "quantiles": {t: wquantiles(pall[t], w) for t in pall},
        "flag_below": FLAG_BELOW,
    }
    json.dump(meta, open("artifacts/model_spec.json", "w"), indent=1)
    report["sample"] = {"total": int(len(raw)), "modelling": int(len(df)), "train": int(len(tr)), "val": int(len(va)), "test": int(len(te)),
                        "dont_know_removed": int(len(raw) - len(df)), "n_inputs": len(feature_names)}
    json.dump(report, open("artifacts/metrics.json", "w"), indent=1)
    print(json.dumps(report, indent=1)[:2500])


# ----------------------------------------------------------------------------- neural network
def train_mlp(X_all, feature_names, num_cols, y, wn, tr, va, te, report):
    import torch, torch.nn as nn, torch.nn.functional as F
    from torch.utils.data import TensorDataset, DataLoader
    from torch.utils.tensorboard import SummaryWriter
    torch.manual_seed(SEED)
    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1).fit(X_all.iloc[tr][CAT_COLS])
    Xc = enc.transform(X_all[CAT_COLS]).astype(int) + 1
    med = X_all.iloc[tr][num_cols].median()
    sc = StandardScaler().fit(X_all.iloc[tr][num_cols].fillna(med))
    Xn = sc.transform(X_all[num_cols].fillna(med)).astype(np.float32)
    cards = [len(c) + 1 for c in enc.categories_]

    class Net(nn.Module):
        def __init__(self, emb=8, h1=64, h2=32, drop=0.3):
            super().__init__()
            self.emb = nn.ModuleList([nn.Embedding(c, emb) for c in cards])
            self.trunk = nn.Sequential(nn.Linear(emb * len(cards) + Xn.shape[1], h1), nn.ReLU(), nn.Dropout(drop),
                                       nn.Linear(h1, h2), nn.ReLU())
            self.h_s, self.h_e = nn.Linear(h2, 1), nn.Linear(h2, 1)

        def forward(self, xc, xn):
            z = torch.cat([e(xc[:, i]) for i, e in enumerate(self.emb)] + [xn], 1)
            h = self.trunk(z)
            return self.h_s(h).squeeze(1), self.h_e(h).squeeze(1)

    def loader(ids, shuffle):
        ds = TensorDataset(torch.tensor(Xc[ids]), torch.tensor(Xn[ids]), torch.tensor(y["savings"][ids], dtype=torch.float32),
                           torch.tensor(y["expense"][ids], dtype=torch.float32), torch.tensor(wn[ids], dtype=torch.float32))
        return DataLoader(ds, batch_size=128, shuffle=shuffle)

    net = Net(); opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
    trl = loader(tr, True)
    writer = SummaryWriter("runs/imarishield_full")
    xb = next(iter(trl)); writer.add_graph(net, (xb[0], xb[1]))

    @torch.no_grad()
    def predict(ids):
        net.eval()
        a, b = net(torch.tensor(Xc[ids]), torch.tensor(Xn[ids]))
        return torch.sigmoid(a).numpy(), torch.sigmoid(b).numpy()

    best, best_state, bad = -1, None, 0
    for ep in range(80):
        net.train()
        for xc, xn, y1, y2, ww in trl:
            opt.zero_grad(); o1, o2 = net(xc, xn)
            l1 = (F.binary_cross_entropy_with_logits(o1, y1, reduction="none") * ww).mean()
            l2 = (F.binary_cross_entropy_with_logits(o2, y2, reduction="none") * ww).mean()
            (l1 + l2).backward(); opt.step()
        p1, p2 = predict(va)
        a1, a2 = roc_auc_score(y["savings"][va], p1), roc_auc_score(y["expense"][va], p2)
        writer.add_scalar("loss/train_savings", l1.item(), ep); writer.add_scalar("loss/train_expense", l2.item(), ep)
        writer.add_scalar("auc/val_savings", a1, ep); writer.add_scalar("auc/val_expense", a2, ep)
        if (a1 + a2) / 2 > best + 1e-4:
            best, bad = (a1 + a2) / 2, 0
            best_state = {k: v.clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= 8:
                break
    net.load_state_dict(best_state)
    p1, p2 = predict(te)
    writer.add_hparams({"emb": 8, "h1": 64, "h2": 32, "dropout": 0.3, "lr": 1e-3, "wd": 1e-4, "epochs_run": ep + 1},
                       {"hparam/test_auc_savings": roc_auc_score(y["savings"][te], p1), "hparam/test_auc_expense": roc_auc_score(y["expense"][te], p2)})
    writer.close()
    out = {"epochs_run": ep + 1, "best_val_mean_auc": float(best),
           "mlp_test_savings": metrics(y["savings"][te], p1), "mlp_test_expense": metrics(y["expense"][te], p2),
           "parameters": int(sum(p.numel() for p in net.parameters()))}
    print("MLP", out)
    torch.save(net.state_dict(), "artifacts/mlp_multitask.pt")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="path to FinScope_2024_Rwanda.dta")
    ap.add_argument("--skip-nn", action="store_true", help="skip the PyTorch comparison model")
    main(ap.parse_args())
