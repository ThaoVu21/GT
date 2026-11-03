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
  3. Compare states  - which profiles drive state differences overall, a network of
                       states grouped by their main-driver profile, then for the
                       selected states: profile mix, excess by profile and the
                       mix vs. within-profile split of the raw-rate gap.             (RQ4)
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
        driver = SP[SP.state == s].sort_values("excess", key=abs, ascending=False).iloc[0]
        note = reporting_note(s)
        out.append(html.Li([
            html.B(s), f": raw rate {st.rate:.1%} ({pts(st.rate - NAT_RATE)} vs. national). "
            f"Profile mix {pts(row.mix_effect)}, within-profile rates {pts(row.within_effect)}. "
            f"Biggest driver: {SHORT[driver.profile]} ({driver.excess:+,.0f} inpatients vs. expected)."
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


def overall_drivers():
    """Each profile's share of all state-level |O − E|, split into more- and fewer-than-expected."""
    d = SP[SP.STATEFIP.isin(STT[STT.n >= MIN_STATE_N].STATEFIP)]
    more = d[d.excess > 0].groupby("profile").excess.sum().reindex(PROFILES, fill_value=0)
    fewer = -d[d.excess < 0].groupby("profile").excess.sum().reindex(PROFILES, fill_value=0)
    return more, fewer, (more + fewer) / (more + fewer).sum()


def drivers_takeaway():
    _, _, share = overall_drivers()
    top = share.sort_values(ascending=False)
    return (f"Across all states, {SHORT[top.index[0]]} ({top.index[0].split(' · ', 1)[1]}) accounts for "
            f"{top.iloc[0]:.0%} of all gaps between observed and expected inpatients (above or below), followed by "
            + ", ".join(f"{SHORT[p]} {v:.0%}" for p, v in top.iloc[1:].items()) + ".")


def fig_overall_drivers():
    more, fewer, share = overall_drivers()
    ys = PROFILES[::-1]
    fig = go.Figure([
        go.Bar(y=ys, x=more[ys], name="More than expected", orientation="h", marker_color="#b8302f",
               hovertemplate="<b>%{y}</b><br>Above expected: +%{x:,.0f} inpatients, summed over states"
                             "<extra></extra>"),
        go.Bar(y=ys, x=fewer[ys], name="Fewer than expected", orientation="h", marker_color="#1c5cab",
               text=[f"{share[p]:.0%} of all gaps" for p in ys], textposition="outside", textfont_color=INK,
               customdata=[share[p] for p in ys],
               hovertemplate="<b>%{y}</b><br>Below expected: −%{x:,.0f} inpatients, summed over states"
                             "<br>Share of all gaps: %{customdata:.0%}<extra></extra>"),
    ])
    fig.update_traces(marker_line_color=SURFACE, marker_line_width=2)
    fig.update_layout(barmode="stack", title="Inpatients above/below expected, summed over all states",
                      xaxis=dict(title="", tickformat=",", range=[0, (more + fewer).max() * 1.25]),
                      legend=dict(orientation="h", y=-0.2))
    return style(fig, 300)


def network_links():
    """Each state's two largest |O − E| profiles (suppressed cells excluded); rank 1 = the state's main driver."""
    d = SP[SP.STATEFIP.isin(STT[STT.n >= MIN_STATE_N].STATEFIP) & ~SP.small_cell].copy()
    d["gap_share"] = d.excess.abs() / d.groupby("state").excess.transform(lambda s: s.abs().sum())
    d = d.sort_values("gap_share", ascending=False)
    d["rank"] = d.groupby("state").cumcount() + 1
    return d[d["rank"] <= 2]


def network_takeaway():
    main = network_links().query("rank == 1")
    counts = main.profile.value_counts()
    return (f"{SHORT[counts.index[0]]} is the main driver in {counts.iloc[0]} of {len(main)} states, "
            f"followed by " + ", ".join(f"{SHORT[p]} ({c})" for p, c in counts.iloc[1:].items())
            + ". States in the same colour group differ from expectation mainly through the same kind of client.")


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

    stt = STT.set_index("state")
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
                   f"<br>Main driver in {(main.profile == p).sum()} states" for p, r in zip(PROFILES, PROF.itertuples())],
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
    tab_profiles = html.Div([
        html.Div(profiles_takeaway(), style=TAKEAWAY),
        html.Div([html.Div(graph("pe-share", fig_profile_bar("share")), style={"flex": "3", "minWidth": "320px"}),
                  html.Div(graph("pe-rate", fig_profile_bar("rate")), style={"flex": "2", "minWidth": "320px"})],
                 style={"display": "flex", "gap": "12px", "flexWrap": "wrap"}),
        html.H3("What each profile looks like", style={"margin": "16px 0 4px"}),
        html.P("Numbers are the % of the profile with that trait. Red = more common than nationally, "
               "blue = less common.", style=CAPTION),
        html.Div([
            html.Div([html.B("Clinical traits and age (used to build the profiles)"),
                      graph("pe-chars", fig_characteristics(True))], style={"flex": "1", "minWidth": "320px"}),
            html.Div([html.B("Other demographics (describe the profiles, not used to build them)"),
                      graph("pe-demo", fig_characteristics(False)),
                      html.P("Percentages are among clients with a known value. Share of all clients with a known "
                             "value: " + ", ".join(
                                 f"{PRETTY[g]} {v:.0%}" for g, v in
                                 FEATS[~FEATS.in_clustering].groupby("group", sort=False)
                                 .national_known_pct.first().items()) + ".", style=CAPTION)],
                     style={"flex": "1", "minWidth": "320px"}),
        ], style={"display": "flex", "gap": "12px", "flexWrap": "wrap"}),
        html.Details([
            html.Summary("How the profiles were built"),
            html.P("Features: age band, SMI/SED status, number of diagnoses, SUD and 13 diagnosis flags (one-hot, "
                   "missing kept as its own level). Sex and other demographics are shown as descriptors rather than "
                   "clustering features, so the profiles reflect clinical differences. K-Means (10 restarts) is fit on a 500k sample. To pick k (4–10), "
                   "it is fit on 4 independent 200k samples scored on a common 100k holdout; stability = mean "
                   "pairwise ARI. Rule: ARI ≥ 0.8 and smallest profile ≥ 2%, then best silhouette. Silhouette "
                   "scores are low (~0.15), so the profiles overlap: they are useful summaries, not natural "
                   "categories.", style=CAPTION),
            dash_table.DataTable(data=sel_rec, columns=sel_cols, **TABLE_STYLE),
            html.P(f"Reporting-artifact audit: NMI between profile and state = {SEL.nmi_with_state.max():.2f} "
                   "(0 = profiles are not state proxies). Largest single-state concentration: "
                   + "; ".join(f"{SHORT[r.profile]} {r.top_state_share:.0%} {r.top_state} "
                               f"(lift {r.top_state_lift:.1f})"
                               for r in PROF.sort_values("top_state_lift", ascending=False).head(3).itertuples())
                   + ". A profile is treated as a possible reporting artifact only if it is concentrated in one "
                     "state (lift > 3) and defined by a missing value; no profile meets both conditions.",
                   style=CAPTION),
        ], style={"marginTop": "8px"}),
    ])

    tab_heatmap = html.Div([
        html.Div(heatmap_takeaway(), style=TAKEAWAY),
        graph("pe-heatmap"),
        html.P(["Red = more inpatient clients than expected for that profile, blue = fewer. States sorted by "
                f"overall O/E (the All column); states with < {MIN_STATE_N:,} clients are left out and cells with "
                f"fewer than {MIN_N} clients or {MIN_E} expected inpatients are suppressed (blank, and the hover "
                "shows only the number of clients). Hover intervals are 95% and adjusted for overdispersion between states, "
                "so they are much wider than a plain Poisson interval. † = reporting caveat: the state reports only "
                "one inpatient setting, has a near-zero inpatient rate, or is missing SMI/SED or primary diagnosis for over "
                "30% of clients; read "
                "its O/E as a reporting difference first. ", html.B("Click a row"),
                " to add or remove that state in tab ③. "] +
               [f"{SHORT[p]} = {p.split(' · ', 1)[1]}.  " for p in PROFILES], style=CAPTION),
    ])

    tab_compare = html.Div([
        html.H3("Across all states", style={"margin": "16px 0 4px"}),
        html.Div(drivers_takeaway(), style=TAKEAWAY),
        graph("pe-overall", fig_overall_drivers()),
        html.H3("Which profile drives each state?", style={"margin": "16px 0 4px"}),
        html.Div(network_takeaway(), style=TAKEAWAY),
        graph("pe-network"),
        html.P(["Each state sits next to the profile behind the largest share of its gap between observed and "
                "expected inpatients (thick line) and is also linked to its second-largest (thin line). Solid line "
                "= more inpatients than expected, dotted = fewer. Colour = the state's main-driver profile; dot "
                f"size = number of clients. States with < {MIN_STATE_N:,} clients and suppressed cells are left "
                "out. ", html.B("Click a state"), " to add or remove it in the comparison below."], style=CAPTION),
        html.H3("Selected states", style={"margin": "16px 0 4px"}),
        html.Ul(id="pe-takeaways", style={**TAKEAWAY, "paddingLeft": "28px"}),
        html.Div([html.Div(graph("pe-mix"), style={"flex": "1", "minWidth": "320px"}), html.Div(graph("pe-drivers"), style={"flex": "1", "minWidth": "320px"})],
                 style={"display": "flex", "gap": "12px", "flexWrap": "wrap"}),
        html.P("Bars above 0: the profile has more inpatient clients than its client characteristics predict "
               "(O − E, risk-adjusted). The mix/within split is different: it decomposes the raw-rate gap, before "
               "adjustment. "
               "Profile mix = the state serves more/fewer clients of high-use profiles; within-profile = the same "
               "profile is hospitalised more/less often there (the two add up to the raw-rate gap). A profile with no "
               "clients in a state counts entirely as a mix effect.", style=CAPTION),
    ])

    return html.Div([
        html.H1("Client-Profile Explorer", style={"margin": "0 0 4px"}),
        html.P("2024 SAMHSA MH-CLD · RQ1 client profiles (clustering) · RQ4 profile-level state comparisons (O/E). "
               "O/E = observed inpatient clients ÷ expected under a model of client characteristics that excludes "
               "state; O/E > 1 means more reported inpatient use than expected, not worse care. SMI = serious mental "
               "illness (adults); SED = serious emotional disturbance (children); SUD = substance use disorder.",
               style=CAPTION),
        kpis,
        html.Div([html.Div("States to compare (or click the heatmap)", style=LABEL),
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
