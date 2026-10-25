"""ImariShield - explainable financial-resilience dashboard for Rwanda (NISR 2026 Big Data Hackathon, Track 2).

Run locally:  streamlit run app.py
Everything the app needs is in artifacts/ (trained models) and data/ (aggregate results). No raw microdata is used at run time.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import xgboost as xgb

ROOT = Path(__file__).parent
DATA, ART = ROOT / "data", ROOT / "artifacts"

st.set_page_config(page_title="ImariShield | Financial resilience in Rwanda", page_icon="🛡️", layout="wide")

# ----------------------------------------------------------------------------- style
INK, MUTED, TEAL, AMBER, RED, GREY = "#1b2a3a", "#5d6b7a", "#0f766e", "#d97706", "#b42318", "#e6eaee"
SEQ = ["#e8f3f1", "#a6d5cf", "#5db3a8", "#1f8a7d", "#0b5a52"]       # resilience share (higher = better)
RISK = ["#fdf2e3", "#f7cf94", "#ec9a4b", "#c85a1c", "#8f2d0b"]      # share unable to cope (higher = worse)
st.markdown(
    f"""
<style>
.block-container {{padding-top: 1.6rem; max-width: 1250px;}}
h1, h2, h3 {{color:{INK}; letter-spacing:-0.01em;}}
.kpi {{border:1px solid {GREY}; border-radius:12px; padding:16px 18px; background:#fff;}}
.kpi .v {{font-size:2.1rem; font-weight:700; color:{INK}; line-height:1.1;}}
.kpi .l {{color:{MUTED}; font-size:0.9rem; margin-top:4px;}}
.kpi .c {{color:{MUTED}; font-size:0.78rem; margin-top:2px;}}
.note {{border-left:4px solid {AMBER}; background:#fff8ec; padding:10px 14px; border-radius:6px; color:{INK}; font-size:0.9rem;}}
.good {{color:{TEAL}; font-weight:600;}} .bad {{color:{RED}; font-weight:600;}}
</style>
""",
    unsafe_allow_html=True,
)


def kpi(value, label, caption=""):
    st.markdown(f'<div class="kpi"><div class="v">{value}</div><div class="l">{label}</div><div class="c">{caption}</div></div>',
                unsafe_allow_html=True)


def pct(x, d=0):
    return f"{100 * x:.{d}f}%"


# ----------------------------------------------------------------------------- load
@st.cache_data
def load_tables():
    t = {n: pd.read_csv(DATA / f"{n}.csv") for n in
         ["district_estimates", "segment_estimates", "district_profile", "drivers", "fairness", "calibration", "levers"]}
    t["headline"] = json.loads((DATA / "headline.json").read_text())
    t["geo"] = json.loads((DATA / "rwanda_districts.geojson").read_text())
    t["metrics"] = json.loads((ART / "metrics.json").read_text())
    t["spec"] = json.loads((ART / "model_spec.json").read_text())
    return t


@st.cache_resource
def load_models():
    out = {}
    for name in ("savings", "expense"):
        m = xgb.XGBClassifier()
        m.load_model(str(ART / f"xgb_{name}.json"))
        out[name] = m.get_booster()
    return out


T = load_tables()
SPEC, HEAD = T["spec"], T["headline"]
TARGET_TXT = {
    "savings": "Could cover 4 weeks of essentials from savings",
    "expense": "Could find RWF 56,000 within a month",
}

# ----------------------------------------------------------------------------- sidebar
st.sidebar.markdown("## 🛡️ ImariShield")
st.sidebar.caption("Explainable financial resilience for Rwanda · FinScope Rwanda 2024")
PAGE = st.sidebar.radio(
    "Explore",
    ["Overview", "District map", "Who is most exposed", "Risk calculator", "Model, fairness & limits", "About & data"],
    label_visibility="collapsed",
)
st.sidebar.markdown("---")
st.sidebar.caption("Team G29 · African Leadership University · NISR 2026 Big Data Hackathon, Track 2: Financial Inclusion & Poverty Reduction")


# ============================================================================= pages
def page_overview():
    st.title("Who could weather a financial shock in Rwanda?")
    st.markdown(
        "FinScope Rwanda 2024 asks adults two direct stress questions. ImariShield turns the answers into a "
        "**resilience score for every district and household profile**, and explains what drives it, so that programmes "
        "can be aimed at the people who would struggle most."
    )
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        kpi(pct(HEAD["not_resilient_savings"]), "could NOT cover 4 weeks of essentials from savings",
            f"95% CI {pct(HEAD['not_resilient_savings_ci'][0], 1)} – {pct(HEAD['not_resilient_savings_ci'][1], 1)}")
    with c2:
        kpi(pct(HEAD["not_resilient_expense"]), "would find RWF 56,000 within a month very difficult",
            f"95% CI {pct(HEAD['not_resilient_expense_ci'][0], 1)} – {pct(HEAD['not_resilient_expense_ci'][1], 1)}")
    with c3:
        kpi(pct(HEAD["fail_both"]), "fail BOTH stress tests", "the most exposed group")
    with c4:
        kpi(pct(HEAD["emergency_fund"]), "have an emergency fund", "only about one adult in five")
    st.markdown("")

    d = T["district_estimates"].copy()
    d["not_savings"], d["not_expense"] = 1 - d["resilient_savings"], 1 - d["resilient_expense"]
    d = d.sort_values("not_savings")
    left, right = st.columns([3, 2])
    with left:
        st.subheader("Share unable to cope, by district")
        fig = go.Figure()
        fig.add_bar(y=d["district"], x=d["not_savings"], orientation="h", name="Savings test", marker_color=AMBER)
        fig.add_bar(y=d["district"], x=d["not_expense"], orientation="h", name="RWF 56,000 test", marker_color=RED)
        fig.update_layout(barmode="group", height=760, margin=dict(l=0, r=10, t=10, b=0), xaxis_tickformat=".0%",
                          xaxis_range=[0.4, 1], legend=dict(orientation="h", y=1.04), plot_bgcolor="white")
        st.plotly_chart(fig, use_container_width=True)
    with right:
        st.subheader("What the data says")
        worst = d.sort_values("not_savings", ascending=False).iloc[0]
        best = d.iloc[0]
        seg = T["segment_estimates"]
        sx = seg[seg.segment == "Sex"].set_index("group")["resilient_savings"]
        ur = seg[seg.segment == "Urban or rural"].set_index("group")["resilient_savings"]
        st.markdown(
            f"- **Gap between districts:** from {pct(best['not_savings'])} in {best['district']} to "
            f"{pct(worst['not_savings'])} in {worst['district']} cannot cover four weeks from savings.\n"
            f"- **Women are more exposed:** {pct(sx.get('Female', np.nan))} of women could cover four weeks, versus "
            f"{pct(sx.get('Male', np.nan))} of men.\n"
            f"- **Rural vs urban:** {pct(ur.get('Rural', np.nan))} of rural adults and {pct(ur.get('Urban', np.nan))} of urban adults pass the savings test.\n"
            "- **Two tests, one story:** the two stress questions agree for about 80% of adults, so the picture is not an artefact of one question."
        )
        st.markdown('<div class="note">Estimates are weighted to the 8.16 million adults FinScope represents. '
                    'Intervals ignore the survey clustering, so treat them as <b>optimistic</b>.</div>', unsafe_allow_html=True)
        st.markdown("")
        st.info("Open **District map** to see where, **Who is most exposed** to see who, and **Risk calculator** to try a profile.")


def district_map(d, col, is_share):
    """Choropleth drawn as filled SVG polygons (no WebGL, no external tiles or topojson, so it renders everywhere)."""
    vals = d.set_index("district")[col]
    lo, hi = float(vals.min()), float(vals.max())
    scale = RISK

    def colour(v):
        i = 0 if hi == lo else (v - lo) / (hi - lo) * (len(scale) - 1)
        j = min(int(i), len(scale) - 2)
        f = i - j
        c1, c2 = [tuple(int(scale[k][x:x + 2], 16) for x in (1, 3, 5)) for k in (j, j + 1)]
        return "#" + "".join(f"{round(a + (b - a) * f):02x}" for a, b in zip(c1, c2))

    fig = go.Figure()
    nmap = d.set_index("district")["n"]
    for ft in T["geo"]["features"]:
        name = ft["properties"]["shapeName"]
        if name not in vals.index:
            continue
        g = ft["geometry"]
        polys = [g["coordinates"]] if g["type"] == "Polygon" else g["coordinates"]
        xs, ys = [], []
        for poly in polys:
            ring = poly[0]
            xs += [p[0] for p in ring] + [None]
            ys += [p[1] for p in ring] + [None]
        v = vals[name]
        txt = f"<b>{name}</b><br>" + (f"{100 * v:.0f}% cannot cope" if is_share else f"{v:,.0f} adults") + f"<br>{int(nmap[name])} respondents"
        fig.add_scatter(x=xs, y=ys, mode="lines", fill="toself", fillcolor=colour(v), line=dict(color="white", width=1),
                        text=txt, hoverinfo="text", hoveron="fills", showlegend=False)
    fig.add_scatter(x=[None], y=[None], mode="markers", showlegend=False, hoverinfo="skip",
                    marker=dict(colorscale=[[i / (len(scale) - 1), c] for i, c in enumerate(scale)], cmin=lo, cmax=hi, color=[lo], showscale=True,
                                colorbar=dict(title="", tickformat=".0%" if is_share else ",.0f", len=0.7)))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False, scaleanchor="x", scaleratio=1)
    fig.update_layout(plot_bgcolor="white")
    return fig


def page_map():
    st.title("District map")
    d = T["district_estimates"].copy()
    d["not_savings"], d["not_expense"] = 1 - d["resilient_savings"], 1 - d["resilient_expense"]
    d["not_savings_lo"], d["not_savings_hi"] = 1 - d["resilient_savings_hi"], 1 - d["resilient_savings_lo"]
    d["not_expense_lo"], d["not_expense_hi"] = 1 - d["resilient_expense_hi"], 1 - d["resilient_expense_lo"]
    choice = st.radio("Show", ["Share who could not cover 4 weeks from savings", "Share who would find RWF 56,000 very difficult",
                               "Number of adults who could not cover 4 weeks from savings"], horizontal=True)
    col = {"S": "not_savings", "h": "not_expense", "N": "not_resilient_adults_savings"}[
        "S" if choice.startswith("Share who could") else ("h" if "56,000" in choice else "N")]
    is_share = col != "not_resilient_adults_savings"
    fig = district_map(d, col, is_share)
    fig.update_layout(height=560, margin=dict(l=0, r=0, t=0, b=0))
    left, right = st.columns([3, 2])
    with left:
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Boundaries: geoBoundaries (CC BY 4.0). Estimates are survey-weighted; about 300–470 adults answered in a typical district.")
    with right:
        st.subheader("Priority ranking")
        tbl = d.sort_values(col, ascending=False)[["district", "not_savings", "not_savings_lo", "not_savings_hi",
                                                    "not_expense", "not_resilient_adults_savings"]].copy()
        tbl["95% interval"] = tbl.apply(lambda r: f"{pct(r.not_savings_lo)}–{pct(r.not_savings_hi)}", axis=1)
        tbl["not_savings"], tbl["not_expense"] = 100 * tbl["not_savings"], 100 * tbl["not_expense"]
        tbl = tbl.rename(columns={"district": "District", "not_savings": "Cannot cover 4 wks", "not_expense": "Cannot find RWF 56k",
                                  "not_resilient_adults_savings": "Adults affected"})
        st.dataframe(tbl[["District", "Cannot cover 4 wks", "95% interval", "Cannot find RWF 56k", "Adults affected"]],
                     hide_index=True, height=420, use_container_width=True,
                     column_config={"Cannot cover 4 wks": st.column_config.ProgressColumn(format="%.0f%%", min_value=0, max_value=100),
                                    "Cannot find RWF 56k": st.column_config.ProgressColumn(format="%.0f%%", min_value=0, max_value=100),
                                    "Adults affected": st.column_config.NumberColumn(format="%d")})
        st.caption("Large, populous districts can matter more for programmes than the highest share: compare 'Adults affected'.")
    st.subheader("District profile")
    sel = st.selectbox("Choose a district", sorted(d["district"]), index=sorted(d["district"]).index("Gasabo") if "Gasabo" in set(d["district"]) else 0)
    prof = T["district_profile"]
    p = prof[prof.district == sel].copy()
    nat = prof.groupby("indicator")["share"].mean().rename("avg")  # simple mean of district shares, for reference
    p = p.merge(nat, left_on="indicator", right_index=True)
    fig2 = go.Figure()
    fig2.add_bar(y=p["indicator"], x=p["share"], orientation="h", name=sel, marker_color=TEAL)
    fig2.add_scatter(y=p["indicator"], x=p["avg"], mode="markers", name="Average district", marker=dict(color=INK, symbol="diamond", size=9))
    fig2.update_layout(height=430, margin=dict(l=0, r=10, t=10, b=0), xaxis_tickformat=".0%", plot_bgcolor="white",
                       legend=dict(orientation="h", y=1.06), yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig2, use_container_width=True)
    row = d[d.district == sel].iloc[0]
    st.markdown(f"**{sel}:** {pct(row.not_savings)} cannot cover four weeks from savings "
                f"(95% CI {pct(row.not_savings_lo)}–{pct(row.not_savings_hi)}), based on {int(row.n)} respondents.")


def page_segments():
    st.title("Who is most exposed")
    seg = T["segment_estimates"].copy()
    seg["not_s"], seg["lo"], seg["hi"] = 1 - seg.resilient_savings, 1 - seg.resilient_savings_hi, 1 - seg.resilient_savings_lo
    seg["not_e"], seg["lo_e"], seg["hi_e"] = 1 - seg.resilient_expense, 1 - seg.resilient_expense_hi, 1 - seg.resilient_expense_lo
    test = st.radio("Stress test", ["Cover 4 weeks of essentials from savings", "Find RWF 56,000 within a month"], horizontal=True)
    use_s = test.startswith("Cover")
    v, lo, hi = ("not_s", "lo", "hi") if use_s else ("not_e", "lo_e", "hi_e")
    segs = list(seg.segment.unique())
    sel = st.selectbox("Compare adults by", segs)
    s = seg[seg.segment == sel].sort_values(v)
    fig = go.Figure(go.Bar(y=s["group"], x=s[v], orientation="h", marker_color=AMBER if use_s else RED,
                           error_x=dict(type="data", symmetric=False, array=(s[hi] - s[v]), arrayminus=(s[v] - s[lo]), color=INK),
                           text=[pct(x) for x in s[v]], textposition="inside"))
    fig.add_vline(x=HEAD["not_resilient_savings" if use_s else "not_resilient_expense"], line_dash="dash", line_color=MUTED,
                  annotation_text="national", annotation_position="top")
    fig.update_layout(height=max(280, 60 * len(s) + 100), margin=dict(l=0, r=10, t=30, b=0), xaxis_tickformat=".0%",
                      xaxis_range=[0, 1.02], plot_bgcolor="white", xaxis_title="Share who could not cope")
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Bars show the share who could NOT cope; whiskers are bootstrap 95% intervals (optimistic: survey clustering is ignored). "
               "Groups with few respondents are noisy.")
    st.subheader("Biggest differences at a glance")
    rows = []
    for sg in segs:
        x = seg[seg.segment == sg]
        x = x[x.n >= 150]
        if len(x) > 1:
            a, b = x.loc[x.not_s.idxmax()], x.loc[x.not_s.idxmin()]
            rows.append({"Dimension": sg, "Most exposed group": a.group, "Share": a.not_s, "Least exposed group": b.group,
                         "Share ": b.not_s, "Gap (pts)": 100 * (a.not_s - b.not_s)})
    gap = pd.DataFrame(rows).sort_values("Gap (pts)", ascending=False)
    st.dataframe(gap, hide_index=True, use_container_width=True,
                 column_config={"Share": st.column_config.NumberColumn(format="percent"), "Share ": st.column_config.NumberColumn(format="percent"),
                                "Gap (pts)": st.column_config.NumberColumn(format="%.0f")})
    st.caption("Savings test only; groups with at least 150 respondents.")


# ----------------------------------------------------------------------------- calculator
LEVERS = {
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
LEVER_KEY = {"Saves through a bank account": "g3a_01", "Saves through a SACCO": "g3a_08", "Saves in a savings group (VSLA, ikimina)": "g3a_11",
             "Saves through mobile money": "g3a_03", "Uses mobile money": "qf1_05", "Owns a mobile phone": "f3", "Has internet access": "c14_5",
             "Receives a VUP cash transfer": "c13d", "Holds medical insurance": "i5a_1"}


def make_row(inp):
    """One-row DataFrame in the exact feature order and categorical coding the models were trained on."""
    row = {c: SPEC["defaults"].get(c, "missing") for c in SPEC["cat_cols"]}
    row.update({k: v for k, v in inp.items() if k in SPEC["cat_cols"]})
    for c in SPEC["flag_cols"]:
        row[c] = int(inp.get(c, 0))
    for c in ("female_head", "b1", "hh_size", "n11", "n11_missing"):
        row[c] = inp[c]
    df = pd.DataFrame([row])[SPEC["feature_names"]]
    for c in SPEC["cat_cols"]:
        df[c] = pd.Categorical(df[c], categories=SPEC["cat_levels"][c])
    return df


def predict(df, models, contribs=False):
    d = xgb.DMatrix(df, enable_categorical=True)
    if contribs:
        return {k: m.predict(d, pred_contribs=True)[0] for k, m in models.items()}
    return {k: float(m.predict(d)[0]) for k, m in models.items()}


def percentile_of(p, target):
    q = np.array(SPEC["quantiles"][target])
    return float((q <= p).mean())  # share of adults with a lower-or-equal predicted chance


def describe(feature, value):
    lab = SPEC["labels"].get(feature, feature)
    if feature in SPEC["cat_cols"]:
        return f"{lab}: {value}"
    if feature == "b1":
        return f"Age: {int(value)}"
    if feature == "hh_size":
        return f"Household size: {int(value)}"
    if feature == "n11":
        return "Income band: " + (SPEC["n11_levels"][int(value) - 1] if value >= 1 else "not reported")
    return f"{lab}: {'yes' if value else 'no'}"


def page_calculator():
    st.title("Risk calculator & stress test")
    st.markdown('<div class="note"><b>For planning and exploration only.</b> It estimates how adults with a similar profile answered the FinScope '
                'stress questions. It must not be used to decide whether a real person receives a loan, a transfer or any service.</div>',
                unsafe_allow_html=True)
    st.markdown("")
    L = SPEC["cat_levels"]
    models = load_models()
    with st.form("profile"):
        a, b, c = st.columns(3)
        with a:
            st.markdown("**Who and where**")
            district = st.selectbox("District", sorted(L["a2"]), index=sorted(L["a2"]).index("Gasabo"))
            area = st.radio("Area", L["a6"], horizontal=True)
            sex = st.radio("Sex", L["b2"], horizontal=True)
            age = st.slider("Age", 16, 90, int(SPEC["age_median"]))
            hh = st.slider("Household size", 1, 15, int(SPEC["hh_size_median"]))
            marital = st.selectbox("Marital status", L["b3"])
            fhead = st.checkbox("Household headed by a woman")
        with b:
            st.markdown("**Livelihood**")
            edu_order = ["No formal education", "Primary 1-3", "Primary 4-6", "Secondary 1-3", "Secondary 4-6", "Vocational training",
                         "University or other higher education"]
            edu = st.selectbox("Education", [e for e in edu_order if e in L["c4a"]], index=2)
            inc_src = st.selectbox("Main source of income", sorted(L["n1b"]), index=sorted(L["n1b"]).index("Money from farming"))
            work_opts = L["n4"]
            work = st.selectbox("Working status", work_opts, index=0)
            ubu = st.selectbox("Ubudehe category", [x for x in ["1st category", "2nd category", "3rd category", "4th category"] if x in L["c13a"]], index=1)
            inc_band = st.selectbox("Personal monthly income", ["Not reported"] + SPEC["n11_levels"])
        with c:
            st.markdown("**Financial access & assets**")
            f_bank = st.checkbox("Saves through a bank account")
            f_sacco = st.checkbox("Saves through a SACCO")
            f_group = st.checkbox("Saves in a savings group (VSLA, ikimina)")
            f_momo_s = st.checkbox("Saves through mobile money")
            f_nosave = st.checkbox("Does not save at all")
            f_momo = st.checkbox("Uses mobile money")
            f_phone = st.checkbox("Owns a mobile phone", value=True)
            f_net = st.checkbox("Has internet access")
            f_vup = st.checkbox("Receives a VUP cash transfer")
            f_ins = st.checkbox("Holds medical insurance")
            f_borrow = st.checkbox("Borrowed money in the past 12 months")
            f_land = st.checkbox("Holds savings in land")
            f_live = st.checkbox("Holds savings in farming / livestock")
            f_tv = st.checkbox("Owns a television")
        st.caption("Anything not listed here is treated as absent (for example no radio, no bed), so profiles with few assets score lower.")
        go_btn = st.form_submit_button("Estimate resilience", type="primary")

    flags = {"g3a_01": f_bank, "g3a_08": f_sacco, "g3a_11": f_group, "g3a_03": f_momo_s, "g3a_17": f_nosave, "qf1_05": f_momo, "qf4_05": f_momo,
             "f3": f_phone, "c14_5": f_net, "c14_1": f_phone, "c13d": f_vup, "i5a_1": f_ins, "h2_1": f_borrow, "g10_2": f_land, "g10_3": f_live,
             "c12_08": f_tv}
    n11 = 0 if inc_band == "Not reported" else SPEC["n11_levels"].index(inc_band) + 1
    inp = {"a2": district, "a6": area, "b2": sex, "b3": marital, "c4a": edu, "n1b": inc_src, "n4": work, "c13a": ubu,
           "female_head": int(fhead), "b1": age, "hh_size": hh, "n11": n11, "n11_missing": int(n11 == 0), **{k: int(v) for k, v in flags.items()}}
    df = make_row(inp)
    probs = predict(df, models)

    st.subheader("Result")
    g1, g2 = st.columns(2)
    for col, key in ((g1, "savings"), (g2, "expense")):
        p = probs[key]
        pr = percentile_of(p, key)
        flagged = p < SPEC["flag_below"]
        with col:
            fig = go.Figure(go.Indicator(mode="gauge+number", value=100 * p, number={"suffix": "%", "valueformat": ".0f"},
                                         title={"text": TARGET_TXT[key], "font": {"size": 15}},
                                         gauge={"axis": {"range": [0, 100]}, "bar": {"color": TEAL if not flagged else RED},
                                                "steps": [{"range": [0, 100 * SPEC["flag_below"]], "color": "#fde8e6"},
                                                          {"range": [100 * SPEC["flag_below"], 100], "color": "#eef6f5"}],
                                                "threshold": {"line": {"color": INK, "width": 3}, "value": 100 * SPEC["national"][key]}}))
            fig.update_layout(height=250, margin=dict(l=20, r=20, t=60, b=0))
            st.plotly_chart(fig, use_container_width=True)
            st.markdown(f"Chance of passing: **{pct(p)}** · national average {pct(SPEC['national'][key])} (black line) · "
                        f"higher than **{pct(pr)}** of adults. " +
                        ('<span class="bad">Flagged as likely to struggle.</span>' if flagged else '<span class="good">Not flagged.</span>'),
                        unsafe_allow_html=True)

    st.subheader("Why this estimate?")
    tabs = st.tabs([TARGET_TXT["savings"], TARGET_TXT["expense"]])
    contrib = predict(df, models, contribs=True)
    for tab, key in zip(tabs, ("savings", "expense")):
        with tab:
            cv = contrib[key][:-1]
            order = np.argsort(-np.abs(cv))[:8]
            names = [describe(SPEC["feature_names"][i], df.iloc[0, i]) for i in order]
            vals = [cv[i] for i in order]
            fig = go.Figure(go.Bar(y=names[::-1], x=vals[::-1], orientation="h",
                                   marker_color=[TEAL if v > 0 else RED for v in vals[::-1]]))
            fig.update_layout(height=340, margin=dict(l=0, r=10, t=10, b=0), plot_bgcolor="white",
                              xaxis_title="← pushes toward struggling      pushes toward coping →")
            st.plotly_chart(fig, use_container_width=True)
    st.caption("Bars are SHAP-style contributions (log-odds) from the XGBoost model: how much each answer moves this profile away from the average adult.")

    st.subheader("What if? Associations, not causes")
    rows = []
    for name, edits in LEVERS.items():
        if flags_have(inp, LEVER_KEY[name]):
            continue
        alt = dict(inp)
        alt.update({k: v for k, v in edits.items() if k in SPEC["flag_cols"]})
        pa = predict(make_row(alt), models)
        rows.append({"If this profile also…": name, "Savings test": 100 * (pa["savings"] - probs["savings"]),
                     "RWF 56,000 test": 100 * (pa["expense"] - probs["expense"])})
    if rows:
        w = pd.DataFrame(rows).sort_values("Savings test", ascending=False)
        st.dataframe(w, hide_index=True, use_container_width=True,
                     column_config={"Savings test": st.column_config.NumberColumn("Savings test (pts)", format="%+.1f"),
                                    "RWF 56,000 test": st.column_config.NumberColumn("RWF 56k test (pts)", format="%+.1f")})
    st.caption("Change in the estimated chance of passing, in percentage points, if people like this profile also had the item. "
               "These are patterns in survey data. Owning or using something may reflect existing wealth, so they do not prove that giving it would cause the change.")


def flags_have(inp, key):
    return bool(inp.get(key, 0))


# ----------------------------------------------------------------------------- model & fairness
def page_model():
    st.title("Model, fairness & limits")
    m = T["metrics"]
    st.subheader("How well does it work?")
    rows = []
    for key in ("savings", "expense"):
        x = m[key]["xgb_test"]
        rows.append({"Stress test": TARGET_TXT[key], "Model": "XGBoost (deployed)", "AUC": x["auc"], "Avg. precision": x["ap"], "Brier": x["brier"]})
    mlp = m.get("mlp")
    if mlp:
        for key in ("savings", "expense"):
            node = mlp.get(f"mlp_test_{key}")
            if isinstance(node, dict) and "auc" in node:
                rows.append({"Stress test": TARGET_TXT[key], "Model": "Multi-task neural network", "AUC": node["auc"],
                             "Avg. precision": node.get("ap", np.nan), "Brier": node.get("brier", np.nan)})
    tbl = pd.DataFrame(rows).sort_values(["Stress test", "Model"], ascending=[True, False])
    st.dataframe(tbl, hide_index=True, use_container_width=True,
                 column_config={c: st.column_config.NumberColumn(format="%.3f") for c in ("AUC", "Avg. precision", "Brier")})
    st.caption(f"Held-out 15% test set (stratified, seed 42). {m['sample']['modelling']:,} adults modelled from {m['sample']['total']:,}. "
               "AUC 0.5 = guessing, 1.0 = perfect. Both models see only pre-shock characteristics: every question that defines the targets is excluded "
               "(automated leakage check in train.py). XGBoost matched or beat the neural network on this tabular survey, so it is the model deployed here.")

    st.subheader("What drives resilience?")
    dr = T["drivers"]
    tg = st.radio("Stress test", ["savings", "expense"], format_func=lambda k: TARGET_TXT[k], horizontal=True)
    top = dr[dr.target == tg].sort_values("mean_abs", ascending=False).head(14).iloc[::-1]
    fig = go.Figure(go.Bar(y=top["label"], x=top["mean_abs"], orientation="h", marker_color=TEAL))
    fig.update_layout(height=470, margin=dict(l=0, r=10, t=10, b=0), plot_bgcolor="white", xaxis_title="Average absolute SHAP contribution (log-odds)")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Are the probabilities trustworthy?")
    cal = T["calibration"]
    fig = go.Figure()
    fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color=MUTED, dash="dash"), name="perfect")
    for k, colr in (("savings", AMBER), ("expense", RED)):
        c = cal[cal.target == k]
        fig.add_scatter(x=c["mean_predicted"], y=c["observed"], mode="lines+markers", name=TARGET_TXT[k], line=dict(color=colr))
    fig.update_layout(height=380, margin=dict(l=0, r=10, t=10, b=0), plot_bgcolor="white", xaxis_title="Predicted chance of coping",
                      yaxis_title="Observed share who coped", xaxis_tickformat=".0%", yaxis_tickformat=".0%", legend=dict(orientation="h", y=1.1))
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Points close to the dashed line mean predicted chances match what happened in held-out data.")

    st.subheader("Fairness audit")
    st.markdown(f"Rule used for the audit: flag an adult as *likely to struggle* when the predicted chance of coping is below {pct(SPEC['flag_below'])}. "
                "**Recall** = of the adults who really could not cope, the share we flag. **Missed-resilient** = of those flagged, the share who actually coped (wrongly flagged).")
    fa = T["fairness"]
    dim = st.selectbox("Group by", list(fa.dimension.unique()))
    f = fa[fa.dimension == dim][["group", "n_test", "auc_savings", "flagged_savings", "recall_nonresilient_savings", "wrongly_flagged_resilient_savings"]]
    f = f.rename(columns={"group": "Group", "n_test": "Test n", "auc_savings": "AUC", "flagged_savings": "Share flagged",
                          "recall_nonresilient_savings": "Recall", "wrongly_flagged_resilient_savings": "Wrongly flagged"})
    st.dataframe(f, hide_index=True, use_container_width=True,
                 column_config={"AUC": st.column_config.NumberColumn(format="%.2f"), "Share flagged": st.column_config.NumberColumn(format="percent"),
                                "Recall": st.column_config.NumberColumn(format="percent"), "Wrongly flagged": st.column_config.NumberColumn(format="percent")})
    st.markdown("""
**What we found:** the model is more accurate for urban adults (AUC 0.85) than rural (0.79); women are flagged more often than men (69% vs 58%), and
the share wrongly flagged is higher in rural areas and among adults with no formal education. District-level accuracy varies widely, with the lowest
recall in the Kigali districts (Kicukiro, Nyarugenge) and Bugesera. These gaps are why ImariShield supports **area-level planning, not individual decisions**.
""")

    st.subheader("Limits you should know")
    st.markdown("""
- **Self-reported and hypothetical.** Targets come from what people say they *could* do, not from observed shocks or bank data.
- **Association, not causation.** “What if” rows describe patterns; they are not impact estimates.
- **Intervals are optimistic.** Bootstrap intervals ignore the survey’s clustering and stratification; true uncertainty is wider.
- **Small district samples.** Roughly 300–470 respondents per district; fairness numbers by district are noisier still.
- **One survey, one year.** FinScope Rwanda 2024; the model has not been validated on other years or on administrative data.
- **Income is thinly reported.** About 72% of income answers are missing; an explicit “not reported” flag is used.
""")


def page_about():
    st.title("About & data")
    st.markdown(f"""
### Problem
Rwanda has expanded financial access quickly, but access is not resilience. FinScope Rwanda 2024 shows that about **{pct(HEAD['not_resilient_savings'])}** of adults could not
cover four weeks of essentials from savings. Programmes need to know *where* these adults live, *who* they are and *which* levers are associated with better outcomes.

### Data
- **FinScope Rwanda 2024** (Access to Finance Rwanda; NISR reference RWA-NISR-FINSCOPE-2024-v01): {HEAD['respondents']:,} adults, survey weights `pop_wt`. Obtained from the NISR microdata portal
  (<https://microdata.statistics.gov.rw>). The raw microdata is **not redistributed** in this repository; only aggregate outputs and trained models are included.
- **District boundaries:** geoBoundaries Rwanda ADM2, licensed CC BY 4.0 (<https://www.geoboundaries.org>).

### Method in brief
1. Two outcome measures from FinScope’s own stress questions: *could pay essentials for 4 weeks from savings* (E7d) and *could find RWF 56,000 within a month* (E9c).
2. Features: {len(SPEC['feature_names'])} pre-shock characteristics (demographics, location, livelihood, assets, financial access, shocks and climate exposure). All outcome-related questions are excluded and checked by code.
3. Gradient-boosted trees (XGBoost) with survey weights and native categorical handling; a multi-task neural network (PyTorch, logged in TensorBoard) is the comparison model.
4. Weighted district and segment estimates with bootstrap 95% intervals; SHAP-style explanations; fairness audit; association-based “levers”.

### Reproduce
`python train.py --data <FinScope_2024_Rwanda.dta>` regenerates every file in `artifacts/` and `data/`. `streamlit run app.py` launches this app. See the README.

### Team & declarations
Team G29, African Leadership University, Kigali. Built for the NISR 2026 Big Data Hackathon, Track 2. Intellectual property is assigned to NISR under the hackathon rules.
AI assistance (Claude) was used for coding and drafting; the team is responsible for all analysis and claims. Third-party libraries: pandas, XGBoost, PyTorch, Streamlit, Plotly. No text or code was copied from other participants.
""")


{"Overview": page_overview, "District map": page_map, "Who is most exposed": page_segments, "Risk calculator": page_calculator,
 "Model, fairness & limits": page_model, "About & data": page_about}[PAGE]()
