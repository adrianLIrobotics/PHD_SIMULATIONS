# app.py
# Dark, compact, interactive SE + Trust simulator with:
# - JSON policies editor
# - 2 plots per row
# - detailed traces per policy
# - Documentation page with robust math rendering (st.latex everywhere for equations)

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
    # SE = ExM * pT + gamma * (1 - pT), gamma < 0
    return float(exm_val * pT + gamma * (1.0 - pT))


def abs_contrib(e: np.ndarray, w: np.ndarray) -> np.ndarray:
    # c = w ⊙ |e|
    return w * np.abs(e)


def rel_attrib(c: np.ndarray) -> np.ndarray:
    # rho = c / sum(c)
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

                # Observation likelihood vector O(z) = [P(z|T), P(z|D)]^T
                O = np.array([
                    model.p_match_T if match else 1.0 - model.p_match_T,
                    model.p_match_D if match else 1.0 - model.p_match_D
                ], dtype=float)

                b[k] = hmm_correct(b_pred, O)
                pT_local[k] = b[k, 0]
                match_flags[k] = match

            # Global trust: attention-weighted aggregation of local trusts
            pT_global = float(np.dot(w, pT_local))

            exm_val = exm(pol.m_obs, w)
            se_val = se_posterior(exm_val, pT_global, model.gamma)

            e = pol.m_obs - pol.m_pred
            c = abs_contrib(e, w)
            rho = rel_attrib(c)

            # Weight calibration: w <- w + phi * rho * (SE - ExM) * g, with g = m_obs
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
    keys = meta_names[:]
    while len(keys) < 3:
        keys.append(f"m{len(keys)+1}")
    e, c, tc = keys[0], keys[1], keys[2]
    example = [
        {"name": "Fork",
         "m_pred": {e: 0.6, c: 0.8, tc: 1.0},
         "m_obs":  {e: 0.6, c: 0.8, tc: 1.0}},
        {"name": "Sticks",
         "m_pred": {e: 0.6, c: 0.8, tc: 1.0},
         "m_obs":  {e: 0.5, c: 0.2, tc: 0.7}},
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
    cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["#00ffcc", "#ff5555", "#aaaaaa"])
    return {n: cycle[i % len(cycle)] for i, n in enumerate(policy_names)}


# -----------------------------
# Documentation page (robust math rendering)
# -----------------------------
def render_docs():

    st.title("Documentation: Terminology & Equations")

    st.header("What is the outcome of this simulator?")

    st.markdown(
        "The simulator produces a **dynamic trace of subjective experience** for one or more policies. "
        "For each policy, it shows how:\n"
        "- evaluative outcomes (meta-utility),\n"
        "- epistemic trust (local and global), and\n"
        "- attention over evaluative dimensions\n"
        "co-evolve over repeated executions.\n\n"
        "The primary outcome of the simulator is **not a best policy**, but an explanation of *why* "
        "certain policies feel better or worse over time, even when their observable outcomes are similar."
    )

    st.divider()

    st.header("Policies")

    st.markdown(
        "A **policy** $\\pi$ represents a concrete way of acting in the world (e.g., using a fork versus using sticks). "
        "Each policy is evaluated along a fixed set of **meta-parameters** that describe *how the action is experienced*, "
        "not just whether the task succeeds.\n\n"
        "For each policy, the simulator distinguishes between:\n"
        "- **Predicted evaluative meta-parameters** $\\hat m(\\pi)$: what the agent expects based on analogy or prior knowledge.\n"
        "- **Observed evaluative meta-parameters** $m^{obs}(\\pi)$: what is actually measured after executing the policy."
    )

    st.divider()

    st.header("Analogy")

    st.markdown(
        "**Analogy** is used as an *evaluative prior*, not as a hard model of the environment.\n\n"
        "An analogy proposes that a new policy $\\pi$ will *feel similar* to a known reference, "
        "by predicting its evaluative meta-parameters $\\hat m(\\pi)$. "
        "This prediction does **not** need to be correct in all dimensions; "
        "it is a hypothesis that must be evaluated through experience.\n\n"
        "Crucially, an analogy can be:\n"
        "- **Good at the ground level** (task success, reward),\n"
        "- but **poor at the meta level** if the resulting experience violates comfort, safety, or trust constraints.\n\n"
        "The simulator demonstrates how such analogies are accepted, weakened, or rejected "
        "based on their *subjective experiential consequences*."
    )

    st.divider()

    st.header("What the simulator is not")

    st.markdown(
        "- It is **not** a planner or a reinforcement learning algorithm.\n"
        "- It does **not** search for optimal actions.\n"
        "- It does **not** assume that higher reward always implies better experience.\n\n"
        "Instead, the simulator focuses on **explainability**: "
        "making explicit how trust, uncertainty, and attention shape subjective experience over time."
    )

    st.divider()

    st.header("Reading the plots")

    st.markdown(
        "- **ExM** shows how good the observable outcome was under current priorities.\n"
        "- **SE** shows how good the experience felt once trust and uncertainty are taken into account.\n"
        "- **SE − ExM** visualizes epistemic degradation (or, in extended models, amplification).\n"
        "- **Local trust plots** show which evaluative dimensions are considered reliable.\n"
        "- **Weight evolution** shows how attention shifts in response to bad experiences.\n\n"
        "Together, these traces form an *explanatory narrative* of decision-making under uncertainty."
    )


    st.header("1) Core objects")

    st.subheader("Meta-parameter vector")
    st.latex(r"m(\pi)\in[0,1]^K")
    st.latex(r"m(\pi)=\big(m_1(\pi),\dots,m_K(\pi)\big)")
    st.markdown(
        "- **Meaning:** evaluative meta-parameters for policy $\\pi$ (e.g., efficiency, comfort, task completion).\n"
        "- **Range:** each component is normalized to $[0,1]$.\n"
        "- **Dimension:** $K$ is the number of meta-parameters."
    )

    st.subheader("Predicted vs observed")
    st.latex(r"\hat m(\pi)\quad \text{(predicted evaluative meta-parameters)}")
    st.latex(r"m^{obs}(\pi)\quad \text{(observed evaluative meta-parameters)}")
    st.markdown(
        "- $\\hat m(\\pi)$ is what an analogy / prior expects.\n"
        "- $m^{obs}(\\pi)$ is what you actually measure after execution."
    )

    st.divider()
    st.header("2) Meta-utility (observed evaluative score)")

    st.latex(r"\mathrm{ExM}(\pi)=\sum_{k=1}^{K} w_k\,m_k^{obs}(\pi)")
    st.latex(r"w\in[0,1]^K,\qquad \sum_{k=1}^{K} w_k=1")
    st.markdown(
        "- **Meaning:** a weighted evaluation of what happened.\n"
        "- $w$ is the **attention / priority vector** over meta-parameters.\n"
        "- ExM is *not* trust — it is value under observed outcomes."
    )

    st.divider()
    st.header("3) Prediction error and evidence")

    st.subheader("Prediction error per meta-parameter")
    st.latex(r"e_k(\pi)=m_k^{obs}(\pi)-\hat m_k(\pi)")
    st.markdown(
        "- **Meaning:** expectation vs reality along dimension $k$.\n"
        "- In the simulator, evidence is derived from $|e_k|$."
    )

    st.subheader("Evidence extraction (match / mismatch)")
    st.latex(r"z_k=\begin{cases}\text{match} & |e_k|\le\tau_k\\ \text{mismatch} & |e_k|>\tau_k\end{cases}")
    st.markdown(
        "- $\\tau_k$ is a tolerance threshold per dimension.\n"
        "- Evidence drives trust, not value."
    )

    st.divider()
    st.header("4) Local trust per meta-parameter (Markov / HMM)")

    st.subheader("Trust state and belief")
    st.latex(r"X_t^k\in\{T,D\}")
    st.latex(r"b_t^k=\begin{bmatrix}P(X_t^k=T)\\P(X_t^k=D)\end{bmatrix}")
    st.markdown("- Each meta-parameter has its own trust belief (local trust).")

    st.subheader("Trust transition matrix")
    st.latex(r"\mathbf T=\begin{bmatrix}P(T\to T) & P(T\to D)\\P(D\to T) & P(D\to D)\end{bmatrix}")
    st.markdown("- This is a **2-state Markov transition matrix** (a Markov chain over trust states).")

    st.subheader("Prediction step (no evidence)")
    st.latex(r"b_{t}^{k-}=\mathbf T^\top b_{t-1}^k")
    st.markdown("- This propagates belief forward even without observations.")

    st.subheader("Correction step (evidence likelihood + normalization)")
    st.latex(r"\mathbf O(z_k)=\begin{bmatrix}P(z_k\mid T)\\P(z_k\mid D)\end{bmatrix}")
    st.latex(r"b_t^k\propto \mathbf O(z_k)\odot b_t^{k-}")
    st.latex(r"b_t^k=\frac{\mathbf O(z_k)\odot b_t^{k-}}{\mathbf 1^\top\left(\mathbf O(z_k)\odot b_t^{k-}\right)}")
    st.markdown("- $\\odot$ denotes element-wise multiplication; denominator performs normalization.")

    st.divider()
    st.header("5) Global trust over policy")

    st.latex(r"p_T^k(t)=P(X_t^k=T)")
    st.latex(r"p_T(\pi)=\sum_{k=1}^{K} w_k\,p_T^k(t)")
    st.markdown(
        "- **Meaning:** global trust in policy $\\pi$ is attention-weighted local trust.\n"
        "- This gives explainability: which dimensions are reducing trust?"
    )

    st.divider()
    st.header("6) Posterior subjective experience")

    st.latex(r"SE(\pi)=\mathrm{ExM}(\pi)\,p_T(\pi)+\gamma\,\big(1-p_T(\pi)\big),\quad \gamma<0")
    st.markdown(
        "- $\\gamma$ is the **distrust penalty**: how bad it feels to proceed when you don't trust.\n"
        "- If $p_T(\\pi)=1$, then $SE(\\pi)=\\mathrm{ExM}(\\pi)$."
    )

    st.subheader("Distrust penalty signal")
    st.latex(r"SE(\pi)-\mathrm{ExM}(\pi)=(1-p_T(\pi))\big(\gamma-\mathrm{ExM}(\pi)\big)")
    st.markdown(
    "- **Interpretation:** this term measures how much lack of trust *spoiled* the experience.\n"
    "- **ExM** answers: *How good was the result?*\n"
    "- **SE** answers: *How good did it feel overall?*\n"
    "- When trust is high, $SE \\approx \\mathrm{ExM}$ and this term is near zero.\n"
    "- When trust is low, $SE < \\mathrm{ExM}$ because uncertainty and doubt degrade the experience.\n"
    "- When $SE > \\mathrm{ExM}$, high trust and confidence *amplify* the experience, making it feel better than what the outcome alone would justify.\n"
    "- This difference is the **epistemic degradation of experience**: the cost of acting without confidence.\n"
    "- It is **not** an expectation-vs-reality prediction error; it reflects uncertainty, not performance."
)

    st.divider()
    st.header("7) Error attribution (explainability)")

    st.latex(r"c_k(\pi)=w_k\,|e_k(\pi)|")
    st.latex(r"\rho_k(\pi)=\frac{c_k(\pi)}{\sum_j c_j(\pi)}")
    st.markdown(
        "- $c_k$ is **absolute contribution** to mismatch given what you care about.\n"
        "- $\\rho_k$ is **relative attribution** (fraction of total mismatch)."
    )

    st.divider()
    st.header("8) Weight calibration (attention adaptation) used in the simulator")

    st.markdown(
        "- Weight calibration is **event-triggered**, not continuous.\n"
        "- Adaptation is activated only when subjective experience is degraded relative to the observed evaluative outcome:\n"
    )
    st.latex(r"SE(\pi)-\mathrm{ExM}(\pi)<0")
    st.markdown(
        "- This condition indicates that uncertainty or lack of trust has negatively affected the experience.\n"
        "- The system does **not** adapt weights merely because a prediction error exists.\n"
        "- Weight adaptation occurs **only when the error meaningfully impacts experience**, ensuring stability and interpretability."
    )

    st.latex(r"w_k\leftarrow w_k+\phi\,\rho_k(\pi)\,\big(SE(\pi)-\mathrm{ExM}(\pi)\big)\,g_k")
    st.latex(r"g_k=m_k^{obs}(\pi)")
    st.latex(r"w\leftarrow\frac{\max(w,0)}{\mathbf 1^\top \max(w,0)}")

    st.markdown(
        "- $\\phi$ is the learning rate controlling the speed of attention adaptation.\n"
        "- $\\rho_k(\\pi)$ focuses adaptation on the meta-parameters most responsible for the mismatch.\n"
        "- $g_k$ is a simple sensitivity proxy: meta-parameters with higher observed values exert stronger influence.\n"
        "- The final normalization step ensures all attention weights remain nonnegative and sum to one."
    )

    

    st.divider()
    st.header("9) Simulator workflow (conceptual pipeline)")

    st.markdown("At each iteration:")
    st.markdown(
        "1. Compute prediction errors $|e_k|$ from $m^{obs}$ and $\\hat m$.\n"
        "2. Convert errors into evidence $z_k$ using thresholds $\\tau_k$.\n"
        "3. Update local trust beliefs with Markov prediction + evidence correction.\n"
        "4. Aggregate global trust $p_T(\\pi)$ from local trusts using current weights $w$.\n"
        "5. Compute observed meta-utility $\\mathrm{ExM}$.\n"
        "6. Compute subjective experience $SE$ using distrust penalty $\\gamma$.\n"
        "7. Attribute mismatch to dimensions ($c_k$, $\\rho_k$).\n"
        "8. Optionally calibrate weights $w$."
    )


# -----------------------------
# Simulator page
# -----------------------------
def render_simulator():
    st.title("Subjetive experience simulator")
    st.markdown(
        "Edit predicted/observed evaluative meta-parameters in the **Policies JSON** and click **Run simulation ✅**. "
        "Plots are shown **two per row** and traces are expandable."
    )

    # Session state for run control + JSON persistence
    if "run_clicked" not in st.session_state:
        st.session_state.run_clicked = True
    if "policies_json" not in st.session_state:
        st.session_state.policies_json = None

    with st.sidebar:
        st.header("Controls")

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
            if st.session_state.policies_json is None:
                st.session_state.policies_json = default_policies_json(meta_names)

            st.caption("Edit **m_pred** and **m_obs** here. Values must be in [0,1]. Keys must match meta names.")
            policies_json = st.text_area("Policies JSON", value=st.session_state.policies_json, height=280)
            st.session_state.policies_json = policies_json

            c3, c4 = st.columns(2)
            with c3:
                if st.button("Reset JSON to defaults"):
                    st.session_state.policies_json = default_policies_json(meta_names)
                    st.session_state.run_clicked = True
            with c4:
                st.write("")

        run = st.button("Run simulation ✅", type="primary")
        if run:
            st.session_state.run_clicked = True

    if not st.session_state.run_clicked:
        st.stop()

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

    def fig_small():
        return plt.subplots(figsize=(4.8, 2.2))

    def fig_small_tall():
        return plt.subplots(figsize=(4.8, 2.6))

    # Summary table
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

    # Plots: 2 per row
    st.subheader("Plots (2 per row)")

    colA, colB = st.columns(2)

    with colA:
        fig, ax = fig_small_tall()
        for name, df in results.items():
            ax.plot(df["t"], df["SE"], label=f"{name} SE", color=colors[name])
            ax.plot(df["t"], df["ExM"], linestyle="--", alpha=0.6, color=colors[name], label=f"{name} ExM")
        ax.set_title("SE vs ExM")
        ax.set_xlabel("t")
        ax.legend()
        st.pyplot(fig, use_container_width=True)

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

    with colC:
        fig, ax = fig_small()
        for name, df in results.items():
            ax.plot(df["t"], df["SE_minus_ExM"], label=name, color=colors[name])
        ax.axhline(0, color="gray", linestyle=":", linewidth=1)
        ax.set_title("Distrust penalty: SE − ExM")
        ax.set_xlabel("t")
        ax.legend()
        st.pyplot(fig, use_container_width=True)

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

    # Local trust plots (2 per row)
    st.subheader("Local trust per meta-parameter (2 per row)")
    pairs = [meta_names[i:i + 2] for i in range(0, len(meta_names), 2)]
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

    # Attribution bars at final timestep (2 per row, per policy)
    st.subheader("Attribution at final timestep (2 per row)")
    policy_pairs = [policy_names[i:i + 2] for i in range(0, len(policy_names), 2)]
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

    # Detailed traces
    st.subheader("Detailed traces (expandable)")
    for name, df in results.items():
        with st.expander(f"Trace table: {name}", expanded=False):
            st.dataframe(df, use_container_width=True)

    st.caption("Docs are available from the sidebar navigation selector.")


# -----------------------------
# App entry
# -----------------------------
st.set_page_config(page_title="SE & Trust Simulator", layout="wide")

with st.sidebar:
    st.markdown("## Navigation")
    page = st.radio("Go to", ["Simulator", "Documentation"], index=0)

if page == "Documentation":
    render_docs()
else:
    render_simulator()