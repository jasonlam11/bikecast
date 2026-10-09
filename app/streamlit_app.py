"""BikeCast dashboard. Reads the committed snapshot in reports/app_data/; never trains a model.

Run with `make app`. Rebuild the snapshot with `make backtest && make snapshot`.
"""

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from bikecast import viz
from bikecast.dashboard import (
    MODEL_LABELS,
    WINDOWS,
    backtest_days,
    comparison_table,
    load_snapshot,
    rebalancing_list,
    week_of,
    window_net_flow,
)

st.set_page_config(page_title="BikeCast", page_icon=":bike:", layout="wide")
NET_FLOW_SCALE = [[0.0, "#b3261e"], [0.25, "#e34948"], [0.5, "#f0efec"],
                  [0.75, "#3987e5"], [1.0, "#184f95"]]  # fmt: skip
LINE_COLORS = {
    "historical_average": viz.SERIES[3],
    "prophet": viz.SERIES[1],
    "torch_mlp": viz.SERIES[0],
    "seasonal_naive": viz.MUTED,
}


@st.cache_data
def data():
    return load_snapshot()


snap = data()
preds, info = snap["stations"], snap["stations_info"]
days = backtest_days(preds)

st.title("BikeCast: day-ahead Bluebikes demand")
st.caption(
    "Hourly pickups and dropoffs at Boston's 50 busiest Bluebikes stations, forecast at midnight "
    "for the next day. Everything shown is from the rolling backtest (10 test weeks, 2025 to "
    "2026): forecasts made with only the data available the night before, next to what actually "
    "happened."
)

with st.sidebar:
    st.header("Controls")
    day = st.selectbox(
        "Backtest day",
        days,
        index=next((i for i, d in enumerate(days) if d.dayofweek == 1), 0),
        format_func=lambda d: d.strftime("%a %Y-%m-%d"),
    )
    model = st.selectbox(
        "Model",
        ["torch_mlp", "prophet", "historical_average", "seasonal_naive"],
        format_func=MODEL_LABELS.get,
    )
    window_label = st.radio("Time window", list(WINDOWS))
    st.caption("Backtest days are past days scored after the fact. There are no live forecasts.")

hours = WINDOWS[window_label]
flow = window_net_flow(preds, model, day, hours)
map_data = flow.merge(info, on="station_id")
day_rows = preds[preds["date"] == day]
context = []
if day_rows["holiday_name"].notna().any():
    context.append(f"holiday: {day_rows['holiday_name'].dropna().iloc[0]}")
if (day_rows["weather"] == "rainy").any():
    context.append("rainy day")

st.subheader(f"{window_label} net flow on {day.strftime('%A %Y-%m-%d')}")
if context:
    st.caption(", ".join(context).capitalize())
col_map, col_list = st.columns([3, 2])
with col_map:
    lim = max(5.0, map_data["forecast"].abs().max())
    fig = px.scatter_map(
        map_data,
        lat="lat",
        lon="lng",
        color="forecast",
        color_continuous_scale=NET_FLOW_SCALE,
        range_color=(-lim, lim),
        hover_name="name",
        hover_data={"forecast": ":.1f", "actual": ":.1f", "lat": False, "lng": False},
        center={"lat": info["lat"].mean(), "lon": info["lng"].mean()},
        zoom=11.8,
        height=520,
        map_style="carto-positron",
    )
    fig.update_traces(marker={"size": 13})
    fig.update_layout(
        margin={"l": 0, "r": 0, "t": 0, "b": 0},
        coloraxis_colorbar={"title": "bikes"},
    )
    st.plotly_chart(fig, width="stretch")
    st.caption(
        "Forecast arrivals minus departures over the window. Red stations lose bikes (restock "
        "them first), blue stations gain bikes (their docks fill up)."
    )
with col_list:
    drains, fills = rebalancing_list(flow, info)
    st.markdown("**Bring bikes to** (largest forecast deficit)")
    st.dataframe(drains.rename(columns=str.capitalize), hide_index=True, width="stretch")
    st.markdown("**Free docks at** (largest forecast surplus)")
    st.dataframe(fills.rename(columns=str.capitalize), hide_index=True, width="stretch")

st.divider()
st.subheader("Station view: forecast vs actual for the test week")
c1, c2 = st.columns([3, 1])
names = info.set_index("station_id")["name"]
station = c1.selectbox("Station", info["station_id"], format_func=lambda s: f"{names[s]} ({s})")
target = c2.radio("Target", ["departures", "arrivals"], horizontal=True)
week = week_of(preds, day)
sel = preds[(preds["station_id"] == station) & (preds["test_week"] == week)
            & (preds["target"] == target)].sort_values("ts")  # fmt: skip
fig = go.Figure()
p = sel[sel["model"] == "prophet"]
if not p.empty:
    fig.add_trace(go.Scatter(x=p["ts"], y=p["yhat_upper"], line={"width": 0}, showlegend=False,
                             hoverinfo="skip"))  # fmt: skip
    fig.add_trace(
        go.Scatter(
            x=p["ts"],
            y=p["yhat_lower"],
            fill="tonexty",
            line={"width": 0},
            fillcolor="rgba(235,104,52,0.15)",
            name="Prophet 80% interval",
        )  # fmt: skip
    )
actual = sel[sel["model"] == "torch_mlp"]
fig.add_trace(go.Scatter(x=actual["ts"], y=actual["y"], name="Actual",
                         line={"color": viz.INK, "width": 1.5}))  # fmt: skip
for m in ["historical_average", "prophet", "torch_mlp"]:
    d = sel[sel["model"] == m]
    fig.add_trace(go.Scatter(x=d["ts"], y=d["yhat"], name=MODEL_LABELS[m],
                             line={"color": LINE_COLORS[m], "width": 1.5}))  # fmt: skip
fig.update_layout(height=380, margin={"l": 0, "r": 0, "t": 10, "b": 0},
                  yaxis_title=f"{target} per hour", hovermode="x unified",
                  legend={"orientation": "h", "y": 1.12})  # fmt: skip
st.plotly_chart(fig, width="stretch")
st.caption(f"Test week of {week.strftime('%Y-%m-%d')}. Closed hours are not shown.")

st.divider()
st.subheader("Model comparison (all 10 test weeks, departures and arrivals)")
table = comparison_table(preds)
c1, c2 = st.columns([3, 2])
with c1:
    shown = table.rename(index=MODEL_LABELS).rename(
        columns={"skill": "Skill vs naive", "skill vs hist avg": "Skill vs hist avg"}
    )
    st.dataframe(
        shown.style.format(
            {
                "MAE": "{:.3f}",
                "RMSE": "{:.3f}",
                "WAPE": "{:.1%}",
                "Skill vs naive": "{:+.1%}",
                "Skill vs hist avg": "{:+.1%}",
            }  # fmt: skip
        ),
        width="stretch",
    )
    st.caption(
        "MAE is in bikes per station per hour. Skill = 1 - MAE / baseline MAE on the same rows; "
        "positive means better. Source of record: reports/results.md."
    )
with c2:
    skills = table["skill"].drop("seasonal_naive")
    fig = go.Figure(go.Bar(
        x=skills.values, y=[MODEL_LABELS[m] for m in skills.index], orientation="h",
        marker_color=[LINE_COLORS[m] for m in skills.index],
        text=[f"{v:+.1%}" for v in skills.values], textposition="outside",
    ))  # fmt: skip
    fig.update_layout(height=260, margin={"l": 0, "r": 40, "t": 30, "b": 0},
                      title="Skill vs seasonal naive", xaxis_tickformat="+.0%",
                      xaxis_range=[0, skills.max() * 1.3])  # fmt: skip
    st.plotly_chart(fig, width="stretch")

st.caption(
    "Data: Bluebikes trip history (Jan 2024 to Sep 2026) and Open-Meteo archived weather "
    "forecasts. Code and method: see the README."
)
