# app.py
# Dark, compact, interactive SE + Trust simulator with JSON policies editor,
# 2-plots-per-row layout, and detailed traces per policy.

import json
from dataclasses import dataclass
from typing import List, Dict, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st


# -----------------------------
# Dark plot style (global)
# -----------------------------
plt.style.use("dark_background")
plt.rcParams.update({
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "legend.fontsize": 7,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "figure.facecolor": "black",
    "axes.facecolor": "black",
})


# -----------------------------
# Math utilities
# -----------------------------
def normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    s = float(np.sum(v))
    if s < eps:
        return np.ones_like(v) / len(v)
    return v / s


def markov_predict(b: np.ndarray, T: np.ndarray) -> np.ndarray:
    # b^- = T^T b
    return T.T @ b


def hmm_correct(b_pred: np.ndarray, O: np.ndarray) -> np.ndarray:
    # b ∝ O ⊙ b_pred
    return normalize(O * b_pred)


def exm(m_obs: np.ndarray, w: np.ndarray) -> float:
    return float(np.dot(w, m_obs))


def se_posterior(exm_val: float, pT: float, gamma: float) -> float:
    return float(exm_val * pT + gamma * (1.0 - pT))


def abs_contrib(e: np.ndarray, w: np.ndarray) -> np.ndarray:
    return w * np.abs(e)


def rel_attrib(c: np.ndarray) -> np.ndarray:
    s = float(np.sum(c))
    if s < 1e-12:
        return np.zeros_like(c)
    return c / s


# -----------------------------
# Data structures
# -----------------------------
@dataclass
class Policy:
    name: str
    m_pred: np.ndarray
    m_obs: np.ndarray


@dataclass
class Model:
    meta_names: List[str]
    w0: np.ndarray
    taus: np.ndarray
    T: np.ndarray
    p_match_T: float
    p_match_D: float
    gamma: float
    phi: float
    adapt: bool


# -----------------------------
# Simulation
# -----------------------------
def simulate(policies: List[Policy], model: Model, steps: int) -> Dict[str, pd.DataFrame]:
    K = len(model.meta_names)
    results: Dict[str, pd.DataFrame] = {}

    for pol in policies:
        # per-meta-parameter belief b_k = [p(T), p(D)]
        b = np.tile(np.array([0.5, 0.5], dtype=float), (K, 1))
        w = model.w0.copy()

        rows = []
        for t in range(steps):
            pT_local = np.zeros(K, dtype=float)
            match_flags = np.zeros(K, dtype=bool)

            for k in range(K):
                b_pred = markov_predict(b[k], model.T)
                err = abs(float(pol.m_obs[k] - pol.m_pred[k]))
                match = err <= float(model.taus[k])

                O = np.array([
                    model.p_match_T if match else 1.0 - model.p_match_T,
                    model.p_match_D if match else 1.0 - model.p_match_D
                ], dtype=float)

                b[k] = hmm_correct(b_pred, O)
                pT_local[k] = b[k, 0]
                match_flags[k] = match

            pT_global = float(np.dot(w, pT_local))
            exm_val = exm(pol.m_obs, w)
            se_val = se_posterior(exm_val, pT_global, model.gamma)

            e = pol.m_obs - pol.m_pred
            c = abs_contrib(e, w)
            rho = rel_attrib(c)

            # Optional weight adaptation: w <- w + phi * rho * (SE-ExM) * g, with g = m_obs
            if model.adapt:
                delta = se_val - exm_val
                w = w + model.phi * rho * delta * pol.m_obs
                w = normalize(np.clip(w, 0.0, None))

            row = {
                "t": t,
                "ExM": exm_val,
                "SE": se_val,
                "SE_minus_ExM": se_val - exm_val,
                "pT_global": pT_global,
            }

            for k, name in enumerate(model.meta_names):
                row[f"m_pred_{name}"] = float(pol.m_pred[k])
                row[f"m_obs_{name}"] = float(pol.m_obs[k])
                row[f"e_{name}"] = float(e[k])
                row[f"abs_e_{name}"] = float(abs(e[k]))
                row[f"match_{name}"] = bool(match_flags[k])
                row[f"pT_{name}"] = float(pT_local[k])
                row[f"c_{name}"] = float(c[k])
                row[f"rho_{name}"] = float(rho[k])
                row[f"w_{name}"] = float(w[k])

            rows.append(row)

        results[pol.name] = pd.DataFrame(rows)

    return results


# -----------------------------
# Helpers for UI parsing
# -----------------------------
def parse_csv_floats(text: str, K: int, fallback: float) -> np.ndarray:
    parts = [p.strip() for p in text.split(",") if p.strip()]
    vals = []
    for p in parts:
        try:
            vals.append(float(p))
        except Exception:
            pass
    if len(vals) != K:
        vals = [fallback] * K
    return np.array(vals, dtype=float)


def clamp01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))


def default_policies_json(meta_names: List[str]) -> str:
    # Default stick/fork, using the given meta names
    # We map by key names; if user changes meta names they should update JSON keys too.
    example = [
        {
            "name": "Fork",
            "m_pred": {meta_names[0]: 0.6, meta_names[1]: 0.8, meta_names[2]: 1.0} if len(meta_names) >= 3 else {},
            "m_obs":  {meta_names[0]: 0.6, meta_names[1]: 0.8, meta_names[2]: 1.0} if len(meta_names) >= 3 else {},
        },
        {
            "name": "Sticks",
            "m_pred": {meta_names[0]: 0.6, meta_names[1]: 0.8, meta_names[2]: 1.0} if len(meta_names) >= 3 else {},
            "m_obs":  {meta_names[0]: 0.5, meta_names[1]: 0.2, meta_names[2]: 0.7} if len(meta_names) >= 3 else {},
        },
    ]
    return json.dumps(example, indent=2)


def parse_policies_from_json(raw: str, meta_names: List[str]) -> List[Policy]:
    data = json.loads(raw)
    if not isinstance(data, list) or len(data) == 0:
        raise ValueError("Policies JSON must be a non-empty list.")
    policies: List[Policy] = []
    for item in data:
        name = str(item.get("name", "Unnamed"))
        mp = item.get("m_pred", {})
        mo = item.get("m_obs", {})
        m_pred = np.array([clamp01(float(mp.get(m, 0.0))) for m in meta_names], dtype=float)
        m_obs  = np.array([clamp01(float(mo.get(m, 0.0))) for m in meta_names], dtype=float)
        policies.append(Policy(name=name, m_pred=m_pred, m_obs=m_obs))
    return policies


def make_policy_colors(policy_names: List[str]) -> Dict[str, Any]:
    # Use matplotlib default cycle; keep consistent mapping
    cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["#00ffcc", "#ff5555", "#aaaaaa"])
    colors = {}
    for idx, n in enumerate(policy_names):
        colors[n] = cycle[idx % len(cycle)]
    return colors


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="SE & Trust Simulator", layout="wide")
st.title("🖤 SE + Trust Simulator (Interactive)")

st.markdown(
    "Edit predicted/observed evaluative meta-parameters in the **Policies JSON** and click **Run simulation**. "
    "Plots are shown **two per row** and traces are expandable."
)

# Session state for “Run” control
if "run_clicked" not in st.session_state:
    st.session_state.run_clicked = True  # run once by default on load

with st.sidebar:
    st.header("Controls")

    # Keep sidebar clean with expanders (prevents “hidden” feeling)
    with st.expander("Simulation", expanded=True):
        steps = st.number_input("Iterations", 1, 500, 30, 1)

    with st.expander("Meta-parameters", expanded=True):
        meta_names_str = st.text_input(
            "Meta-parameter names (comma-separated)",
            value="Efficiency, Comfort, TaskCompletion"
        )
        meta_names = [x.strip() for x in meta_names_str.split(",") if x.strip()]
        K = len(meta_names)
        if K < 1:
            st.error("Please provide at least one meta-parameter name.")

        w0_str = st.text_input("Initial weights (comma-separated)", value="0.4, 0.4, 0.2")
        w0 = parse_csv_floats(w0_str, K, 1.0 / max(K, 1))
        w0 = normalize(np.clip(w0, 0.0, None))

        taus_str = st.text_input("Match thresholds τ (comma-separated)", value="0.10, 0.10, 0.05")
        taus = parse_csv_floats(taus_str, K, 0.10)
        taus = np.clip(taus, 0.0, 1.0)

    with st.expander("Trust model (Markov + observations)", expanded=False):
        c1, c2 = st.columns(2)
        with c1:
            T00 = st.slider("P(T→T)", 0.0, 1.0, 0.85, 0.01)
            T10 = st.slider("P(D→T)", 0.0, 1.0, 0.20, 0.01)
        with c2:
            T01 = st.slider("P(T→D)", 0.0, 1.0, 0.15, 0.01)
            T11 = st.slider("P(D→D)", 0.0, 1.0, 0.80, 0.01)

        # Normalize rows in case user makes them inconsistent
        row0 = normalize(np.array([T00, T01], dtype=float))
        row1 = normalize(np.array([T10, T11], dtype=float))
        T = np.vstack([row0, row1])

        p_match_T = st.slider("P(match | Trust)", 0.0, 1.0, 0.80, 0.01)
        p_match_D = st.slider("P(match | Distrust)", 0.0, 1.0, 0.30, 0.01)

    with st.expander("SE + weight adaptation", expanded=False):
        gamma = st.slider("γ (distrust penalty, <0)", -2.0, -0.01, -0.40, 0.01)
        adapt = st.checkbox("Enable weight calibration", value=True)
        phi = st.slider("φ (learning rate)", 0.0, 1.0, 0.10, 0.01)

    with st.expander("Policies (JSON editor)", expanded=True):
        if "policies_json" not in st.session_state:
            st.session_state.policies_json = default_policies_json(meta_names if len(meta_names) >= 3 else ["m1","m2","m3"])

        st.caption("Edit **m_pred** and **m_obs** here. Values must be in [0,1]. Keys must match meta-parameter names.")
        policies_json = st.text_area("Policies JSON", value=st.session_state.policies_json, height=280)
        st.session_state.policies_json = policies_json

    run = st.button("Run simulation ✅", type="primary")
    if run:
        st.session_state.run_clicked = True

# If we haven't run yet (should run once on first load), stop
if not st.session_state.run_clicked:
    st.stop()

# Build model + parse policies
model = Model(
    meta_names=meta_names,
    w0=w0,
    taus=taus,
    T=T,
    p_match_T=float(p_match_T),
    p_match_D=float(p_match_D),
    gamma=float(gamma),
    phi=float(phi),
    adapt=bool(adapt),
)

try:
    policies = parse_policies_from_json(st.session_state.policies_json, meta_names)
except Exception as e:
    st.error(f"Policies JSON error: {e}")
    st.stop()

results = simulate(policies, model, int(steps))
policy_names = list(results.keys())
colors = make_policy_colors(policy_names)

# -----------------------------
# Plot helpers
# -----------------------------
def fig_small():
    return plt.subplots(figsize=(4.8, 2.2))

def fig_small_tall():
    return plt.subplots(figsize=(4.8, 2.6))


# -----------------------------
# Summary table
# -----------------------------
st.subheader("Final summary")
summary = []
for name, df in results.items():
    last = df.iloc[-1]
    summary.append({
        "Policy": name,
        "ExM_final": round(float(last["ExM"]), 4),
        "SE_final": round(float(last["SE"]), 4),
        "pT_global_final": round(float(last["pT_global"]), 4),
        "SE_minus_ExM_final": round(float(last["SE_minus_ExM"]), 4),
    })
st.dataframe(pd.DataFrame(summary).sort_values("SE_final", ascending=False), use_container_width=True)


# -----------------------------
# Plots: always 2 per row
# -----------------------------
st.subheader("Plots (2 per row)")

colA, colB = st.columns(2)

# Plot 1: ExM vs SE
with colA:
    fig, ax = fig_small_tall()
    for name, df in results.items():
        ax.plot(df["t"], df["SE"], label=f"{name} SE", color=colors[name])
        ax.plot(df["t"], df["ExM"], linestyle="--", alpha=0.6, color=colors[name], label=f"{name} ExM")
    ax.set_title("SE vs ExM")
    ax.set_xlabel("t")
    ax.legend()
    st.pyplot(fig, use_container_width=True)

# Plot 2: Global trust
with colB:
    fig, ax = fig_small_tall()
    for name, df in results.items():
        ax.plot(df["t"], df["pT_global"], label=name, color=colors[name])
    ax.set_ylim(0, 1)
    ax.set_title("Global trust $p_T(\\pi)$")
    ax.set_xlabel("t")
    ax.legend()
    st.pyplot(fig, use_container_width=True)

colC, colD = st.columns(2)

# Plot 3: SE - ExM (distrust penalty)
with colC:
    fig, ax = fig_small()
    for name, df in results.items():
        ax.plot(df["t"], df["SE_minus_ExM"], label=name, color=colors[name])
    ax.axhline(0, color="gray", linestyle=":", linewidth=1)
    ax.set_title("Distrust penalty: SE − ExM")
    ax.set_xlabel("t")
    ax.legend()
    st.pyplot(fig, use_container_width=True)

# Plot 4: Weight evolution (per policy — pick first policy for readability)
with colD:
    fig, ax = fig_small()
    pick = policy_names[0]
    dfp = results[pick]
    for m in meta_names:
        ax.plot(dfp["t"], dfp[f"w_{m}"], label=f"w_{m}")
    ax.set_title(f"Weight evolution (policy: {pick})")
    ax.set_xlabel("t")
    ax.set_ylim(0, 1)
    ax.legend()
    st.pyplot(fig, use_container_width=True)

# Local trust plots (2 per row, per meta-parameter)
st.subheader("Local trust per meta-parameter (2 per row)")
pairs = [meta_names[i:i+2] for i in range(0, len(meta_names), 2)]
for pair in pairs:
    c1, c2 = st.columns(2)
    for idx, m in enumerate(pair):
        target_col = c1 if idx == 0 else c2
        with target_col:
            fig, ax = fig_small()
            for name, df in results.items():
                ax.plot(df["t"], df[f"pT_{m}"], label=name, color=colors[name])
            ax.set_ylim(0, 1)
            ax.set_title(f"Local trust in {m}")
            ax.set_xlabel("t")
            ax.legend()
            st.pyplot(fig, use_container_width=True)

# Attribution bar charts at final step (2 per row)
st.subheader("Attribution at final timestep (2 per row)")
policy_pairs = [policy_names[i:i+2] for i in range(0, len(policy_names), 2)]
for pair in policy_pairs:
    c1, c2 = st.columns(2)
    for idx, pol in enumerate(pair):
        target_col = c1 if idx == 0 else c2
        with target_col:
            df = results[pol]
            last = df.iloc[-1]
            c_vals = np.array([last[f"c_{m}"] for m in meta_names], dtype=float)
            rho_vals = np.array([last[f"rho_{m}"] for m in meta_names], dtype=float)

            fig, ax = fig_small()
            ax.bar(meta_names, c_vals)
            ax.set_title(f"{pol}: absolute contrib c")
            ax.set_xticklabels(meta_names, rotation=25, ha="right")
            st.pyplot(fig, use_container_width=True)

            fig, ax = fig_small()
            ax.bar(meta_names, rho_vals)
            ax.set_title(f"{pol}: relative attrib ρ")
            ax.set_ylim(0, 1)
            ax.set_xticklabels(meta_names, rotation=25, ha="right")
            st.pyplot(fig, use_container_width=True)

# -----------------------------
# Detailed traces (expanders)
# -----------------------------
st.subheader("Detailed traces (expandable)")
for name, df in results.items():
    with st.expander(f"Trace table: {name}", expanded=False):
        st.dataframe(df, use_container_width=True)

st.caption(
    "How to edit obs/pred: In the sidebar JSON, modify `m_obs` and `m_pred` values. "
    "Click **Run simulation** to refresh plots and tables."
)