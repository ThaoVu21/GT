"""
Client-Profile Explorer (RQ1 + RQ4) - Dash version.

Run `python profile_pipeline.py` first, then either
    python profile_explorer_dash.py                  # http://localhost:8050
    gunicorn profile_explorer_dash:server            # production / Render

Mounting inside a larger Dash app:
    from profile_explorer_dash import layout as profile_layout, register_callbacks
    register_callbacks(app)            # once, after creating `app`
    ... put profile_layout() wherever the explorer should appear (a tab or page)
Every component id starts with "pe-" so it does not clash with other components.

Views (one takeaway line on top of each)
  1. Profiles        - who is in each cluster (clinical traits and other demographics),
                       its size and inpatient rate.                                  (RQ1)
  2. State x profile - heatmap of profile-specific O/E. Click a row to add/remove
                       that state in the comparison.                                 (RQ4)
  3. Compare states  - network of states grouped by their main-driver profile, then
                       for the selected states: profile mix, O − E by profile and the
                       mix vs. within-profile split of the raw-rate gap.             (RQ4)
Each tab opens with its question, audience and main limitation, and each chart has a
short Purpose / How it helps in practice / How to use it panel.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, ctx, dash_table, dcc, html

OUT = Path(__file__).resolve().parent / "outputs" / "profile_explorer"

# Reference palette (dataviz skill): fixed categorical order, blue<->red diverging, gray midpoint.
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
DIVERGING = [[0.0, "#1c5cab"], [0.25, "#86b6ef"], [0.5, "#f0efec"], [0.75, "#f0a3a2"], [1.0, "#b8302f"]]
INK, MUTED, GRID, SURFACE, PAGE = "#0b0b0b", "#52514e", "#e6e5e0", "#fcfcfb", "#f9f9f7"
NATIONAL = "National"
MIN_STATE_N = 1000  # states/jurisdictions with fewer clients are left out of the heatmap
MIN_N, MIN_E = 100, 5  # small-cell thresholds (same as profile_pipeline.py)

SP = pd.read_csv(OUT / "state_profile.csv")
STT = pd.read_csv(OUT / "state_summary.csv")
PROF = pd.read_csv(OUT / "profile_summary.csv").sort_values("profile_id")
FEATS = pd.read_csv(OUT / "profile_features.csv")
SEL = pd.read_csv(OUT / "cluster_selection.csv")

PROFILES = PROF.profile.tolist()
SHORT = {p: p.split(" · ")[0] for p in PROFILES}  # "P3"
COLOR_OF = {p: CAT[i % len(CAT)] for i, p in enumerate(PROFILES)}
NAT_RATE = PROF.observed.sum() / PROF.n.sum()
STATE_LIST = STT.sort_values("state").state.tolist()
DEFAULT_STATES = STT.sort_values("n", ascending=False).state.head(2).tolist()


def style(fig, height=None):
    fig.update_layout(template="plotly_white", font=dict(color=INK, size=12),
                      margin=dict(l=10, r=10, t=40, b=10), legend_title_text="",
                      plot_bgcolor=SURFACE, paper_bgcolor=SURFACE, **({"height": height} if height else {}))
    fig.update_xaxes(gridcolor=GRID, zerolinecolor=GRID)
    fig.update_yaxes(gridcolor=GRID, zerolinecolor=GRID)
    return fig


def empty(msg, height=300):
    fig = go.Figure()
    fig.add_annotation(text=msg, showarrow=False, font=dict(color=MUTED, size=14))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return style(fig, height)


def records(df, fmt):
    """DataFrame -> DataTable records with columns pre-formatted as strings."""
    out = df.copy()
    for col, f in fmt.items():
        out[col] = out[col].map(lambda v: "" if pd.isna(v) else f.format(v))
    return out.to_dict("records"), [{"name": c, "id": c} for c in out.columns]


def pts(x):
    return f"{x * 100:+.1f} pts"


# ---------------------------------------------------------------- takeaways
def profiles_takeaway():
    hi, lo = PROF.loc[PROF.rate.idxmax()], PROF.loc[PROF.rate.idxmin()]
    return (f"{SHORT[hi.profile]} ({hi.profile.split(' · ', 1)[1]}) has the highest inpatient rate, "
            f"{hi.rate:.1%}: {hi.rate / lo.rate:.1f}× {SHORT[lo.profile]} ({lo.rate:.1%}). "
            f"The national rate is {NAT_RATE:.1%}.")


FLAGGED = set(STT[STT.reporting_flag].state)


def reporting_note(state):
    r = STT.set_index("state").loc[state]
    issues = ([f"reports {r.scope}"] if r.scope != "both inpatient settings" else []) + \
             ([f"near-zero inpatient rate ({r.rate:.2%})"] if r.near_zero_rate else []) + \
             [f"{name} missing for {v:.0%} of clients" for name, v in
              [("SMI/SED", r.smised_missing), ("primary diagnosis", r.mh1_missing)] if v > 0.3]
    return "; ".join(issues)


def heatmap_takeaway():
    big = SP[~SP.small_cell & (SP.n >= MIN_STATE_N) & np.isfinite(SP.oe)]
    top = big.loc[big.oe.idxmax()]
    note = reporting_note(top.state)
    return (f"Biggest outlier: {top.state} · {SHORT[top.profile]}, with {top.oe:.1f}× the expected inpatient "
            f"use ({top.n:,.0f} clients, {top.rate:.0%} inpatient)"
            + (f". Reporting caveat: {top.state} {note}." if note else ".")
            + " A state's overall O/E can hide profiles that run far above or below it.")


def compare_takeaways(states):
    g = SP[SP.state.isin(states)].groupby("state")[["mix_effect", "within_effect"]].sum()
    out = []
    for s in states:
        row, st = g.loc[s], STT.set_index("state").loc[s]
        bigger = "profile mix" if abs(row.mix_effect) > abs(row.within_effect) else "within-profile rates"
        note = reporting_note(s)
        out.append(html.Li([
            html.B(s), f": raw inpatient rate {st.rate:.1%} ({pts(st.rate - NAT_RATE)} vs. national) = "
            f"profile mix {pts(row.mix_effect)} + within-profile rates {pts(row.within_effect)}. "
            f"Most of the gap comes from {bigger}."
            + (f" Reporting caveat: {note}." if note else "")]))
    return out


# ---------------------------------------------------------------- figure builders
def fig_profile_bar(col):
    is_share = col == "share"
    fig = px.bar(PROF, x=col, y="profile", orientation="h", color="profile", color_discrete_map=COLOR_OF,
                 text=PROF[col].map("{:.1%}".format), custom_data=["profile", "n", "share", "observed", "rate"])
    fig.update_traces(hovertemplate="<b>%{customdata[0]}</b><br>Clients: %{customdata[1]:,}"
                                    "<br>Share of all clients: %{customdata[2]:.1%}"
                                    "<br>Inpatient clients: %{customdata[3]:,}"
                                    "<br>Inpatient rate: %{customdata[4]:.1%}<extra></extra>")
    if not is_share:
        fig.add_vline(NAT_RATE, line_dash="dot", line_color=MUTED,
                      annotation_text=f"National {NAT_RATE:.1%}", annotation_font_color=MUTED)
    fig.update_traces(textfont_color=INK, textposition="outside", marker_line_width=0, cliponaxis=False)
    fig.update_layout(showlegend=False, xaxis_tickformat=".0%",
                      title="Share of all clients" if is_share else "Inpatient rate",
                      yaxis=dict(categoryorder="array", categoryarray=PROF.profile[::-1], title="",
                                 showticklabels=is_share),
                      xaxis=dict(range=[0, PROF[col].max() * 1.25], title=""))
    return style(fig, 300)


PRETTY = {"AGE_BAND": "Age", "SEX": "Sex", "SMISED": "SMI/SED", "NUMMHS": "Diagnosis count", "SAP": "Substance use",
          "RACE": "Race", "ETHNIC": "Ethnicity", "MARSTAT": "Marital status", "EMPLOY": "Employment",
          "LIVARAG": "Residence", "EDUC": "Education", "VETERAN": "Veteran", "Diagnosis": "Diagnosis"}


def fig_characteristics(in_clustering):
    """Cell text = % of the profile with that trait; colour = over/under-representation vs. national."""
    ff = FEATS[(FEATS.in_clustering == in_clustering) & ~FEATS.feature.str.endswith("Unknown")].copy()
    var, level = ff.feature.str.split(": ", n=1).str[0], ff.feature.str.split(": ", n=1).str[1]
    ff["feature"] = var.map(PRETTY).fillna(var) + ": " + level
    if "known_pct" not in ff:
        ff["known_pct"] = 1.0
    ff["known_pct"] = ff.known_pct.fillna(1.0)
    order = ff.drop_duplicates("feature").feature.tolist()
    pct = ff.pivot(index="feature", columns="profile", values="pct").reindex(order)[PROFILES]
    known = ff.pivot(index="feature", columns="profile", values="known_pct").reindex(order)[PROFILES]
    nat = ff.drop_duplicates("feature").set_index("feature").national_pct.reindex(order)
    ratio = pct / nat.values[:, None]
    fig = go.Figure(go.Heatmap(
        x=[SHORT[p] for p in PROFILES], y=order, xgap=2, ygap=2,
        z=np.log2(ratio.clip(lower=1 / 8, upper=8)).values, colorscale=DIVERGING, zmin=-3, zmax=3,
        colorbar=dict(title="Vs. national", tickvals=[-3, -1, 0, 1, 3], ticktext=["⅛×", "½×", "1×", "2×", "8×"]),
        text=pct.map(lambda v: f"{v:.0%}").values, texttemplate="%{text}", textfont=dict(size=10),
        customdata=np.dstack([pct.values, np.repeat(nat.values[:, None], len(PROFILES), 1), ratio.values,
                              known.values]),
        hovertemplate="<b>%{x}</b> · %{y}<br>In this profile: %{customdata[0]:.1%}"
                      "<br>All clients: %{customdata[1]:.1%}<br>Ratio to all clients: %{customdata[2]:.2f}×"
                      "<br>Known for %{customdata[3]:.0%} of this profile<extra></extra>"))
    fig.update_yaxes(autorange="reversed", dtick=1, tickfont=dict(size=10))
    fig.update_xaxes(dtick=1, side="top")
    return style(fig, 22 * len(order) + 80)


def fig_heatmap(states):
    stt_f = STT[STT.n >= MIN_STATE_N].sort_values("oe", ascending=False)
    order = stt_f.state.tolist()
    h = SP[SP.STATEFIP.isin(stt_f.STATEFIP)].copy()
    h.loc[h.small_cell, "oe"] = np.nan

    grid = h.pivot(index="state", columns="profile", values="oe").reindex(index=order, columns=PROFILES)
    ov = stt_f.set_index("state", drop=False).reindex(order)  # leading "All" column = state's overall O/E
    grid.insert(0, "All", ov.oe.values)
    xs = ["All"] + [SHORT[p] for p in PROFILES]

    def small(v):
        return "below 0.01" if 0 < v < 0.005 else f"{v:.2f}"

    def hover(r, state, col):
        head = f"<b>{state}</b> · {col}<br>Clients: {r.n:,.0f}"
        if r.small_cell:  # suppressed: show size only, never the unstable ratio
            return head + f"<br>Suppressed: fewer than {MIN_N} clients or {MIN_E} expected inpatients"
        flag = "<br>Outside the 99.8% limits" if r.outlier_998 else ""
        return (head + f"<br>Observed inpatients: {r.observed:,.0f}<br>Expected inpatients: {r.expected:,.1f}"
                f"<br>O/E (observed ÷ expected): {small(r.oe)}<br>95% interval: {small(r.oe_lo)}–{small(r.oe_hi)} "
                f"(adjusted for overdispersion)<br>Difference (O − E): {r.excess:+,.1f}" + flag)

    cells = SP.set_index(["state", "profile"])
    hovertext = [[hover(ov.loc[s], s, "All profiles")] +
                 [hover(cells.loc[(s, p)], s, p) if (s, p) in cells.index else "" for p in PROFILES]
                 for s in order]
    ylabels = [f"{s} †" if s in FLAGGED else s for s in order]

    fig = go.Figure(go.Heatmap(
        x=xs, y=ylabels, xgap=2, ygap=1, hovertext=hovertext, texttemplate="%{text}", textfont=dict(size=9),
        z=np.log2(grid.clip(lower=0.125, upper=8)).values, zmin=-3, zmax=3, colorscale=DIVERGING,
        colorbar=dict(title="O/E", tickvals=[-3, -2, -1, 0, 1, 2, 3], ticktext=["⅛", "¼", "½", "1", "2", "4", "8"]),
        text=grid.map(lambda v: "" if pd.isna(v) else f"{v:.2f}").values,
        hovertemplate="%{hovertext}<extra>Click to compare</extra>"))
    for s in states:  # outline the selected states
        if s in order:
            i = order.index(s)
            fig.add_shape(type="rect", x0=-0.5, x1=len(xs) - 0.5, y0=i - 0.5, y1=i + 0.5,
                          line=dict(color=INK, width=2))
    fig.update_yaxes(autorange="reversed", tickfont=dict(size=10), dtick=1)
    fig.update_xaxes(side="top", dtick=1)
    return style(fig, 18 * len(order) + 120)


def national_gap_share():
    """Each profile's share of all state-level |O − E| (states with enough clients)."""
    d = SP[SP.STATEFIP.isin(STT[STT.n >= MIN_STATE_N].STATEFIP)]
    gaps = d.excess.abs().groupby(d.profile).sum().reindex(PROFILES, fill_value=0)
    return gaps / gaps.sum()


def network_links():
    """Each state's two largest |O − E| profiles (suppressed cells excluded); rank 1 = the state's main driver."""
    d = SP[SP.STATEFIP.isin(STT[STT.n >= MIN_STATE_N].STATEFIP) & ~SP.small_cell].copy()
    d["gap_share"] = d.excess.abs() / d.groupby("state").excess.transform(lambda s: s.abs().sum())
    d = d.sort_values("gap_share", ascending=False)
    d["rank"] = d.groupby("state").cumcount() + 1
    return d[d["rank"] <= 2]


def network_takeaway():
    main = network_links().query("rank == 1")
    counts, share = main.profile.value_counts(), national_gap_share()
    top = counts.index[0]
    return (f"{SHORT[top]} ({top.split(' · ', 1)[1]}) is the main driver in {counts.iloc[0]} of {len(main)} states "
            f"and accounts for {share[top]:.0%} of all observed − expected gaps nationally. Next: "
            + ", ".join(f"{SHORT[p]} ({c} states)" for p, c in counts.iloc[1:].items()) + ".")


def fig_network(states):
    """States grouped around the profile that drives most of their observed − expected gap."""
    links = network_links()
    main = links[links["rank"] == 1].sort_values("n", ascending=False)
    hub_r, step = 1.8, 2 * np.pi / len(PROFILES)
    hub = {p: (hub_r * np.cos(np.pi / 2 - i * step), hub_r * np.sin(np.pi / 2 - i * step))
           for i, p in enumerate(PROFILES)}
    pos = {}
    for i, p in enumerate(PROFILES):  # fan each group's states out on the outer side of its hub
        group = main[main.profile == p].state.tolist()
        out_angle = np.pi / 2 - i * step
        for j, s in enumerate(group):
            ring, k = (0, j) if j < 8 else (1, j - 8)
            n_ring = min(len(group), 8) if ring == 0 else len(group) - 8
            spread = np.radians(min(240, 34 * n_ring))
            a = out_angle + (0 if n_ring == 1 else -spread / 2 + spread * k / (n_ring - 1))
            r = 0.55 + 0.35 * ring
            pos[s] = (hub[p][0] + r * np.cos(a), hub[p][1] + r * np.sin(a))

    traces = []
    for rank, width, opacity in [(2, 1, 0.35), (1, 2.5, 0.8)]:  # weaker links first, drawn underneath
        for p in PROFILES:
            for above, dash in [(True, "solid"), (False, "dot")]:
                e = links[(links["rank"] == rank) & (links.profile == p) & ((links.excess > 0) == above)]
                xs, ys = [], []
                for s in e.state:
                    xs += [pos[s][0], hub[p][0], None]
                    ys += [pos[s][1], hub[p][1], None]
                if xs:
                    traces.append(go.Scatter(x=xs, y=ys, mode="lines", hoverinfo="skip", showlegend=False,
                                             opacity=opacity, line=dict(color=COLOR_OF[p], width=width, dash=dash)))

    stt, gap = STT.set_index("state"), national_gap_share()
    second = links[links["rank"] == 2].set_index("state")
    size_of = lambda n: 12 + 22 * np.sqrt(n / STT.n.max())
    for p in PROFILES:
        g = main[main.profile == p]
        hover = [f"<b>{r.state}</b> · overall O/E {stt.loc[r.state].oe:.2f}"
                 f"<br>Main driver: {SHORT[p]}, {'more' if r.excess > 0 else 'fewer'} inpatients than expected "
                 f"({r.excess:+,.0f}, O/E {r.oe:.2f})<br>Share of the state's gaps: {r.gap_share:.0%}"
                 + (f"<br>Second: {SHORT[second.loc[r.state].profile]} ({second.loc[r.state].excess:+,.0f}, "
                    f"{second.loc[r.state].gap_share:.0%} of gaps)" if r.state in second.index else "")
                 for r in g.itertuples()]
        traces.append(go.Scatter(
            x=[pos[s][0] for s in g.state], y=[pos[s][1] for s in g.state], mode="markers+text",
            name=p, legendgroup=p, text=g.abbr, textposition="top center", textfont=dict(size=10, color=INK),
            customdata=g.state, hovertext=hover, hovertemplate="%{hovertext}<extra>Click to compare</extra>",
            marker=dict(color=COLOR_OF[p], size=size_of(g.n), opacity=0.9,
                        line=dict(color=[INK if s in states else SURFACE for s in g.state],
                                  width=[3 if s in states else 1.5 for s in g.state]))))
    traces.append(go.Scatter(
        x=[hub[p][0] for p in PROFILES], y=[hub[p][1] for p in PROFILES], mode="markers+text",
        showlegend=False, text=[f"<b>{SHORT[p]}</b>" for p in PROFILES], textposition="middle center",
        textfont=dict(size=13, color="#ffffff"), hoverinfo="text",
        hovertext=[f"<b>{p}</b><br>Clients: {r.n:,}<br>Inpatient rate: {r.rate:.1%}"
                   f"<br>Main driver in {(main.profile == p).sum()} states<br>Share of all gaps nationally: {gap[p]:.0%}"
                   for p, r in zip(PROFILES, PROF.itertuples())],
        marker=dict(color=[COLOR_OF[p] for p in PROFILES], size=46, line=dict(color=SURFACE, width=3))))
    fig = go.Figure(traces)
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False, scaleanchor="x")
    style(fig, 760)
    fig.update_layout(legend=dict(orientation="h", y=1.0, yanchor="bottom", x=0, font=dict(size=11)),
                      margin=dict(t=70, b=10), hoverlabel=dict(align="left"))
    return fig


def fig_mix(states):
    comp = SP[SP.state.isin(states)]
    nat = PROF.assign(state=NATIONAL, share_in_state=PROF.share)
    comp = pd.concat([comp, nat], ignore_index=True)
    fig = px.bar(comp, y="state", x="share_in_state", color="profile", orientation="h",
                 category_orders={"state": states + [NATIONAL], "profile": PROFILES}, color_discrete_map=COLOR_OF,
                 custom_data=["state", "profile", "n", "share_in_state"])
    fig.update_traces(marker_line_color=SURFACE, marker_line_width=2,
                      hovertemplate="<b>%{customdata[0]}</b> · %{customdata[1]}<br>Clients: %{customdata[2]:,}"
                                    "<br>Share of clients: %{customdata[3]:.1%}<extra></extra>")
    fig.update_layout(title="Profile mix (share of clients)", xaxis_tickformat=".0%", xaxis_title="",
                      yaxis_title="", legend=dict(orientation="h", y=-0.2, font=dict(size=10)))
    return style(fig, 400)


def fig_drivers(states):
    rows = SP[SP.state.isin(states)].assign(p=lambda d: d.profile.map(SHORT))
    fig = px.bar(rows, x="p", y="excess", color="state", barmode="group", color_discrete_sequence=CAT,
                 category_orders={"p": [SHORT[p] for p in PROFILES], "state": states},
                 custom_data=["state", "profile", "observed", "expected", "oe", "excess"])
    fig.update_traces(hovertemplate="<b>%{customdata[0]}</b> · %{customdata[1]}"
                                    "<br>Observed inpatients: %{customdata[2]:,}"
                                    "<br>Expected inpatients: %{customdata[3]:,.1f}"
                                    "<br>O/E (observed ÷ expected): %{customdata[4]:.2f}"
                                    "<br>Difference (O − E): %{customdata[5]:+,.0f}<extra></extra>")
    fig.add_hline(0, line_color=MUTED)
    fig.update_layout(title="Which profiles drive it? (observed − expected)",
                      yaxis_title="", xaxis_title="", legend=dict(orientation="h", y=-0.15))
    return style(fig, 400)


# ---------------------------------------------------------------- layout
CARD = {"background": SURFACE, "border": f"1px solid {GRID}", "borderRadius": "8px", "padding": "12px"}
LABEL = {"fontSize": "12px", "color": MUTED, "marginBottom": "4px", "fontWeight": 600}
CAPTION = {"fontSize": "12px", "color": MUTED, "margin": "6px 0 0"}
TAKEAWAY = {"background": "#eaf2fc", "borderLeft": "4px solid #2a78d6", "padding": "10px 14px",
            "margin": "12px 0", "fontSize": "15px", "color": INK, "borderRadius": "4px"}
TABLE_STYLE = dict(style_table={"overflowX": "auto"}, page_size=12,
                   style_cell={"fontFamily": "inherit", "fontSize": "12px", "padding": "4px 8px",
                               "textAlign": "left", "backgroundColor": SURFACE, "color": INK},
                   style_header={"fontWeight": 600, "backgroundColor": PAGE, "color": INK})


def kpi(label, value):
    return html.Div([html.Div(label, style=LABEL), html.Div(value, style={"fontSize": "26px", "color": INK})],
                    style={**CARD, "flex": "1", "minWidth": "150px"})


def graph(id_, figure=None):
    return dcc.Graph(id=id_, figure=figure, config={"displaylogo": False})


def tab_intro(question, audience):
    """Top-of-tab banner: the research question and who it serves."""
    return html.Div([html.Div([html.B("Question: "), question]),
                     html.Div([html.B("Who it helps: "), audience], style={"marginTop": "4px"})],
                    style={**CARD, "margin": "12px 0 0", "fontSize": "14px"})


def about(purpose, use, how):
    """Three short notes under a chart heading: what it is for, a real-world use, how to read/interact."""
    def cell(label, body):
        return html.Div([html.Div(label, style=LABEL), html.Div(body)],
                        style={"flex": "1", "minWidth": "240px"})
    return html.Div([cell("Purpose", purpose), cell("How it helps in practice", use), cell("How to use it", how)],
                    style={"display": "flex", "gap": "16px", "flexWrap": "wrap", "fontSize": "13px",
                           "color": INK, "background": PAGE, "border": f"1px solid {GRID}", "borderRadius": "6px",
                           "padding": "10px 12px", "margin": "6px 0 8px"})


def section(title):
    return html.H3(title, style={"margin": "20px 0 4px"})


def layout():
    kpis = html.Div([
        kpi("Clients", f"{PROF.n.sum():,}"), kpi("Client profiles", str(len(PROFILES))),
        kpi("National inpatient rate", f"{NAT_RATE:.2%}"), kpi("States in data", str(len(STT))),
    ], style={"display": "flex", "gap": "12px", "flexWrap": "wrap", "margin": "12px 0"})

    sel_view = SEL.assign(chosen=SEL.selected.map({True: "✓", False: ""}))[
        ["k", "silhouette", "stability_ari", "min_cluster_share", "chosen"]].rename(columns={
        "k": "Profiles (k)", "silhouette": "Silhouette", "stability_ari": "Stability (ARI)",
        "min_cluster_share": "Smallest profile", "chosen": "Chosen"})
    sel_rec, sel_cols = records(sel_view, {"Silhouette": "{:.3f}", "Stability (ARI)": "{:.2f}",
                                           "Smallest profile": "{:.1%}"})
    row = {"display": "flex", "gap": "12px", "flexWrap": "wrap"}
    col = {"flex": "1", "minWidth": "320px"}
    tab_profiles = html.Div([
        tab_intro("Who is being served? Seven client profiles replace one national average with groups that can be "
                  "planned for.",
                  "State mental-health agencies, community providers, program administrators and researchers."),
        html.Div(profiles_takeaway(), style=TAKEAWAY),

        section("How big is each profile, and how often does it use inpatient care?"),
        about("Shows the size of each profile (left) and its inpatient rate compared with the national rate (right).",
              "Shows whether a small group accounts for a large share of inpatient use, so agencies can prioritise "
              "reviewing crisis, discharge follow-up and case-management capacity for that group.",
              "Bars to the right of the dotted line are above the national rate. Hover a bar for client and "
              "inpatient counts."),
        html.Div([html.Div(graph("pe-share", fig_profile_bar("share")), style={**col, "flex": "3"}),
                  html.Div(graph("pe-rate", fig_profile_bar("rate")), style={**col, "flex": "2"})], style=row),

        section("What does each profile look like?"),
        about("Describes who is in each profile: the clinical traits and age used to build it (left) and other "
              "demographics that only describe it (right).",
              "Program planners can match services to a group, such as integrated mental-health and substance-use "
              "care for adults with SMI and SUD. Equity researchers can check whether some groups are "
              "over-represented in high-use profiles.",
              "Read one column top to bottom. The number is the % of the profile with that trait; red = more "
              "common than nationally, blue = less common. Hover for the national % and data coverage."),
        html.Div([
            html.Div([html.B("Clinical traits and age (used to build the profiles)"),
                      graph("pe-chars", fig_characteristics(True))], style=col),
            html.Div([html.B("Other demographics (describe the profiles, not used to build them)"),
                      graph("pe-demo", fig_characteristics(False)),
                      html.P("Percentages are among clients with a known value. Share of all clients with a known "
                             "value: " + ", ".join(
                                 f"{PRETTY[g]} {v:.0%}" for g, v in
                                 FEATS[~FEATS.in_clustering].groupby("group", sort=False)
                                 .national_known_pct.first().items()) + ".", style=CAPTION)], style=col),
        ], style=row),
        html.Details([
            html.Summary("How the profiles were built (for analysts and reviewers)"),
            html.P("Lets reviewers judge how stable the profiles are and whether any of them is a reporting artifact "
                   "rather than a real client group.", style=CAPTION),
            html.P("Features: age band, SMI/SED status, number of diagnoses, SUD and 13 diagnosis flags (one-hot, "
                   "missing kept as its own level). Sex and other demographics are shown as descriptors rather than "
                   "clustering features, so the profiles reflect clinical differences. K-Means (10 restarts) is fit "
                   "on a 500k sample, then every client is assigned. To pick k (4–10), it is fit on 4 independent "
                   "200k samples scored on a common 100k holdout; stability = mean pairwise ARI. Rule: ARI ≥ 0.8 and "
                   "smallest profile ≥ 2%, then best silhouette. Silhouette scores are low (~0.15), so the profiles "
                   "overlap: they are useful summaries, not natural categories.", style=CAPTION),
            dash_table.DataTable(data=sel_rec, columns=sel_cols, **TABLE_STYLE),
            html.P(f"Reporting-artifact audit: NMI between profile and state = {SEL.nmi_with_state.max():.2f} "
                   "(0 = profiles are not state proxies). Largest single-state concentration: "
                   + "; ".join(f"{SHORT[r.profile]} {r.top_state_share:.0%} {r.top_state} "
                               f"(lift {r.top_state_lift:.1f})"
                               for r in PROF.sort_values("top_state_lift", ascending=False).head(3).itertuples())
                   + ". A profile is treated as a possible reporting artifact only if it is concentrated in one "
                     "state (lift > 3) and defined by a missing value; no profile meets both conditions.",
                   style=CAPTION),
        ], style={"marginTop": "12px"}),
    ])

    tab_heatmap = html.Div([
        tab_intro("Where is inpatient use higher or lower than expected, and for which client group?",
                  "State administrators, federal researchers and policymakers."),
        html.Div(heatmap_takeaway(), style=TAKEAWAY),
        section("Observed ÷ expected inpatient use, by state and profile"),
        about("The relative view: for each state and profile, how many inpatient clients there were compared with "
              "what the clients' characteristics predict.",
              "A state with an average overall O/E can still hospitalise one group far more than expected. "
              "Administrators can focus a review on that group's services instead of every program, and "
              "find states where the same group sits near 1 to learn from.",
              ["Rows are sorted by overall O/E (the All column). Red = more than expected, blue = fewer. Hover a "
               "cell for counts, a 95% interval and O − E. Blank = too few clients to report. † = check reporting "
               "first. ", html.B("Click a row"), " to compare that state in tab ③."]),
        graph("pe-heatmap"),
        html.P([f"Left out: states with < {MIN_STATE_N:,} clients. Blank cells have fewer than {MIN_N} clients or "
                f"{MIN_E} expected inpatients. Intervals are adjusted for overdispersion between states, so they are "
                "much wider than a plain Poisson interval. † = the state reports only one inpatient setting, has a "
                "near-zero inpatient rate, or is missing SMI/SED or primary diagnosis for over 30% of clients. "] +
               [f"{SHORT[p]} = {p.split(' · ', 1)[1]}.  " for p in PROFILES], style=CAPTION),
    ])

    tab_compare = html.Div([
        tab_intro("Which client groups account for the differences between states?",
                  "State administrators, clinical program planners and policymakers."),
        section("Which profile drives each state?"),
        html.Div(network_takeaway(), style=TAKEAWAY),
        about("Groups every state with the profile behind the largest part of its observed − expected gap, so "
              "states that differ for the same reason sit together.",
              "National program staff can bring together states with the same driver for joint technical "
              "assistance or peer learning, and each state sees which client group to look at first.",
              ["Big circles = profiles; dots = states (size = clients). Thick line = main driver, thin = second; "
               "solid = more than expected, dotted = fewer. Click a legend item to hide a group. ",
               html.B("Click a state"), " to add it to the comparison below."]),
        graph("pe-network"),
        html.P(f"States with < {MIN_STATE_N:,} clients and suppressed cells are left out.", style=CAPTION),

        section("Compare selected states"),
        html.P("Pick states in the box at the top of the page, or click them in the heatmap or the network.",
               style=CAPTION),
        html.Ul(id="pe-takeaways", style={**TAKEAWAY, "paddingLeft": "28px"}),
        html.Div([
            html.Div([about("Shows each state's mix of client profiles next to the national mix.",
                            "Supports fair benchmarking: a state serving more high-use clients should be expected "
                            "to have a higher raw rate. The summary above splits the raw-rate gap into profile mix "
                            "and within-profile rates (descriptive, not causal).",
                            "Compare each state's segments with the National bar. Hover a segment for its share."),
                      graph("pe-mix")], style=col),
            html.Div([about("The absolute view: inpatient clients above (+) or below (−) expected, per profile.",
                            "Shows where the volume is, so administrators can prioritise the profiles with the "
                            "largest bars rather than those with the highest ratio.",
                            "Bars above 0 = more than expected. Hover for observed, expected and O/E."),
                      graph("pe-drivers")], style=col),
        ], style=row),
    ])

    return html.Div([
        html.H1("Client-Profile Explorer", style={"margin": "0 0 4px"}),
        html.P("2024 SAMHSA Mental Health Client-Level Data (MH-CLD) · RQ1 client profiles (clustering) · RQ4 "
               "profile-level state comparisons.", style=CAPTION),
        html.Div("Abbreviations", style={**LABEL, "marginTop": "8px"}),
        html.Div([
            html.Div([html.B(k), f" = {v}"]) for k, v in [
                ("RQ", "research question"), ("P1–P7", "client profiles 1–7"),
                ("O", "observed inpatient clients"),
                ("E", "expected inpatient clients, from a model of client characteristics that excludes state"),
                ("O/E", "observed ÷ expected (above 1 = more than expected; not a measure of care quality)"),
                ("O − E", "observed minus expected, in clients"),
                ("SMI", "serious mental illness (adults)"), ("SED", "serious emotional disturbance (children)"),
                ("SUD", "substance use disorder"), ("†", "reporting caveat for that state")]],
            style={"display": "grid", "gridTemplateColumns": "repeat(auto-fill, minmax(280px, 1fr))",
                   "gap": "2px 16px", "fontSize": "12px", "color": MUTED}),
        kpis,
        html.Div([html.Div("States to compare (or click a state in the heatmap or network)", style=LABEL),
                  dcc.Dropdown(id="pe-states", options=STATE_LIST, value=DEFAULT_STATES, multi=True)], style=CARD),
        dcc.Tabs(id="pe-view", value="profiles", style={"marginTop": "12px"}, children=[
            dcc.Tab(label="① Who are the clients?", value="profiles", children=tab_profiles),
            dcc.Tab(label="② Where is use higher than expected?", value="heatmap", children=tab_heatmap),
            dcc.Tab(label="③ Why do states differ?", value="compare", children=tab_compare),
        ]),
    ], style={"maxWidth": "1300px", "margin": "0 auto", "padding": "16px",
              "fontFamily": "system-ui, -apple-system, sans-serif", "color": INK, "background": PAGE})


# ---------------------------------------------------------------- callbacks
def register_callbacks(app):
    @app.callback(Output("pe-heatmap", "figure"), Input("pe-states", "value"))
    def update_heatmap(states):
        return fig_heatmap(states or [])

    @app.callback(Output("pe-network", "figure"), Input("pe-states", "value"))
    def update_network(states):
        return fig_network(states or [])

    @app.callback(Output("pe-mix", "figure"), Output("pe-drivers", "figure"), Output("pe-takeaways", "children"),
                  Input("pe-states", "value"))
    def update_compare(states):
        if not states:
            return empty("Pick one or more states"), empty("Pick one or more states"), [html.Li("No states selected.")]
        return fig_mix(states), fig_drivers(states), compare_takeaways(states)

    # Linked selection: clicking a heatmap row or a network state toggles that state in the comparison.
    @app.callback(Output("pe-states", "value"), Input("pe-heatmap", "clickData"), Input("pe-network", "clickData"),
                  State("pe-states", "value"), prevent_initial_call=True)
    def click_select(heat_click, net_click, states):
        click = heat_click if ctx.triggered_id == "pe-heatmap" else net_click
        if not click:
            return states
        point = click["points"][0]
        if ctx.triggered_id == "pe-network":
            if not isinstance(point.get("customdata"), str):  # profile hub, not a state
                return states
            state = point["customdata"]
        else:
            state = point["y"].removesuffix(" †")
        states = list(states or [])
        return [s for s in states if s != state] if state in states else states + [state]


app = Dash(__name__, title="Client-Profile Explorer")
app.layout = layout
register_callbacks(app)
server = app.server  # for gunicorn: `gunicorn profile_explorer_dash:server`

if __name__ == "__main__":
    app.run(debug=False, port=8050)
