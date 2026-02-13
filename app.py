# app.py
# Dark, compact, interactive SE + Trust simulator with:
# - JSON policies editor
# - 2 plots per row
# - detailed traces per policy
# - Documentation page with robust math rendering (st.latex everywhere for equations)
# - time-varying observed meta-parameters via optional m_obs_schedule in JSON
# - Examples page with two sections:
#   (A) Stick & Fork (10 variants)
#   (B) Other stories (10 variants; some analogies, some single-policy)

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

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
# Schedule utilities for time-varying observed meta-params
# -----------------------------
def clamp01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))


def sanitize_schedule(raw_schedule: Any, meta_names: List[str]) -> List[Dict[str, Any]]:
    if raw_schedule is None:
        return []
    if not isinstance(raw_schedule, list):
        raise ValueError("m_obs_schedule must be a list of objects like {'t': 30, 'Efficiency': 0.5}.")

    cleaned: List[Dict[str, Any]] = []
    for item in raw_schedule:
        if not isinstance(item, dict) or "t" not in item:
            raise ValueError("Each schedule entry must be an object with at least key 't'.")
        entry: Dict[str, Any] = {"t": int(item["t"])}
        for m in meta_names:
            if m in item:
                entry[m] = clamp01(float(item[m]))
        cleaned.append(entry)

    cleaned.sort(key=lambda d: d["t"])
    return cleaned


def get_m_obs_at_time(m_obs_base: np.ndarray, schedule: List[Dict[str, Any]], meta_names: List[str], t: int) -> np.ndarray:
    m_obs_t = m_obs_base.copy()
    for entry in schedule:
        if entry["t"] <= t:
            for k, m in enumerate(meta_names):
                if m in entry:
                    m_obs_t[k] = float(entry[m])
        else:
            break
    return m_obs_t


def get_change_points(schedule: List[Dict[str, Any]]) -> List[int]:
    return [int(e["t"]) for e in schedule] if schedule else []


def draw_change_lines(ax, change_points: List[int]):
    for cp in change_points:
        ax.axvline(cp, linestyle=":", linewidth=1, alpha=0.6)


# -----------------------------
# Data structures
# -----------------------------
@dataclass
class Policy:
    name: str
    m_pred: np.ndarray
    m_obs_base: np.ndarray
    m_obs_schedule: Optional[List[Dict[str, Any]]] = None


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
    rigidity: float   # NEW
    adapt: bool


# -----------------------------
# Simulation
# -----------------------------
def simulate(policies: List[Policy], model: Model, steps: int) -> Tuple[Dict[str, pd.DataFrame], Dict[str, List[int]]]:
    K = len(model.meta_names)
    results: Dict[str, pd.DataFrame] = {}
    change_points_by_policy: Dict[str, List[int]] = {}

    for pol in policies:
        # per-meta-parameter belief b_k = [p(T), p(D)]
        b = np.tile(np.array([0.5, 0.5], dtype=float), (K, 1))
        w = model.w0.copy()

        schedule = sanitize_schedule(pol.m_obs_schedule, model.meta_names) if pol.m_obs_schedule else []
        change_points_by_policy[pol.name] = get_change_points(schedule)

        rows = []
        for t in range(steps):
            m_obs_t = get_m_obs_at_time(pol.m_obs_base, schedule, model.meta_names, t)

            pT_local = np.zeros(K, dtype=float)
            match_flags = np.zeros(K, dtype=bool)

            for k in range(K):
                b_pred = markov_predict(b[k], model.T)
                err = abs(float(m_obs_t[k] - pol.m_pred[k]))
                match = err <= float(model.taus[k])

                O = np.array([
                    model.p_match_T if match else 1.0 - model.p_match_T,
                    model.p_match_D if match else 1.0 - model.p_match_D
                ], dtype=float)

                b[k] = hmm_correct(b_pred, O)
                pT_local[k] = b[k, 0]
                match_flags[k] = match

            pT_global = float(np.dot(w, pT_local))

            exm_val = exm(m_obs_t, w)
            se_val = se_posterior(exm_val, pT_global, model.gamma)

            e = m_obs_t - pol.m_pred
            c = abs_contrib(e, w)
            rho = rel_attrib(c)

            # Weight calibration (attention adaptation)
            if model.adapt:
                delta = se_val - exm_val

                # Candidate update (your original rule)
                w_candidate = w + model.phi * rho * delta * m_obs_t
                w_candidate = normalize(np.clip(w_candidate, 0.0, None))

                # NEW: rigidity blending
                # rigidity=1 => keep old weights, rigidity=0 => take candidate fully
                r = float(np.clip(model.rigidity, 0.0, 1.0))
                w = r * w + (1.0 - r) * w_candidate
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
                row[f"m_obs_{name}"] = float(m_obs_t[k])
                row[f"e_{name}"] = float(e[k])
                row[f"abs_e_{name}"] = float(abs(e[k]))
                row[f"match_{name}"] = bool(match_flags[k])
                row[f"pT_{name}"] = float(pT_local[k])
                row[f"c_{name}"] = float(c[k])
                row[f"rho_{name}"] = float(rho[k])
                row[f"w_{name}"] = float(w[k])

            rows.append(row)

        results[pol.name] = pd.DataFrame(rows)

    return results, change_points_by_policy


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


def default_policies_json(meta_names: List[str]) -> str:
    e, c, tc = meta_names[0], meta_names[1], meta_names[2] if len(meta_names) >= 3 else "TaskCompletion"
    example = [
        {"name": "Fork",
         "m_pred": {e: 0.6, c: 0.8, tc: 1.0},
         "m_obs":  {e: 0.6, c: 0.8, tc: 1.0},
         "m_obs_schedule": [
             {"t": 0, e: 0.6, c: 0.8, tc: 1.0},
             {"t": 30, e: 0.3}
         ]},
        {"name": "Sticks",
         "m_pred": {e: 0.6, c: 0.8, tc: 1.0},
         "m_obs":  {e: 0.5, c: 0.2, tc: 0.7},
         "m_obs_schedule": [
             {"t": 0, e: 0.5, c: 0.2, tc: 0.7},
             {"t": 20, tc: 0.4}
         ]},
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
        sched = item.get("m_obs_schedule", None)

        m_pred = np.array([clamp01(float(mp.get(m, 0.0))) for m in meta_names], dtype=float)
        m_obs_base = np.array([clamp01(float(mo.get(m, 0.0))) for m in meta_names], dtype=float)
        policies.append(Policy(name=name, m_pred=m_pred, m_obs_base=m_obs_base, m_obs_schedule=sched))
    return policies


def make_policy_colors(policy_names: List[str]) -> Dict[str, Any]:
    cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["#00ffcc", "#ff5555", "#aaaaaa"])
    return {n: cycle[i % len(cycle)] for i, n in enumerate(policy_names)}


# -----------------------------
# Documentation page (ADD ONLY; do not delete existing)
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

    # YOUR EXISTING DOC (kept)
    st.latex(r"w_k\leftarrow w_k+\phi\,\rho_k(\pi)\,\big(SE(\pi)-\mathrm{ExM}(\pi)\big)\,g_k")
    st.latex(r"g_k=m_k^{obs}(\pi)")
    st.latex(r"w\leftarrow\frac{\max(w,0)}{\mathbf 1^\top \max(w,0)}")
    st.markdown(
        "- $\\phi$ is the learning rate.\n"
        "- $g_k$ is a simple sensitivity proxy: higher observed value gets stronger influence.\n"
        "- Final line clips weights nonnegative and renormalizes so $\\sum_k w_k=1$."
    )

    # NEW: add rigidity without deleting old
    st.subheader("Rigidity (inertia) parameter")
    st.latex(r"r\in[0,1]\quad\text{(rigidity)}")
    st.latex(r"w_{\text{cand}} \leftarrow \frac{\max\!\left(w+\phi\,\rho(\pi)\,(SE-\mathrm{ExM})\,g,\;0\right)}{\mathbf 1^\top \max\!\left(w+\phi\,\rho(\pi)\,(SE-\mathrm{ExM})\,g,\;0\right)}")
    st.latex(r"w \leftarrow r\,w + (1-r)\,w_{\text{cand}}")
    st.latex(r"w\leftarrow\frac{\max(w,0)}{\mathbf 1^\top \max(w,0)}")
    st.markdown(
        "- **Meaning:** rigidity controls how reluctant the agent is to change attention weights.\n"
        "- If $r=1$, weights stay essentially fixed (fully rigid).\n"
        "- If $r=0$, the update is fully applied (fully flexible).\n"
        "- This makes adaptation smoother and prevents large weight swings from a single bad episode."
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
        "8. Optionally calibrate weights $w$ (optionally regulated by rigidity $r$)."
    )


# -----------------------------
# Examples library page (TWO SECTIONS)
# -----------------------------
def render_examples():
    st.title("Examples (JSON Library)")
    st.markdown(
        "Three sections:\n"
        "1) **Stick & Fork** (10 variants)\n"
        "2) **Other stories** (10 variants: some analogies, some single-policy baselines)\n"
        "3) **Real data benchamking** (3 variants)\n\n"
        "Click **Load into Simulator** to paste the JSON into the simulator editor."
    )

    if "policies_json" not in st.session_state:
        st.session_state.policies_json = None

    def render_example_block(title: str, story: str, obj: Any, key_prefix: str):
        json_text = json.dumps(obj, indent=2)
        with st.expander(title, expanded=False):
            st.markdown(f"**Story:** {story}")
            c1, c2 = st.columns([1, 1])
            with c1:
                if st.button("Load into Simulator ✅", key=f"{key_prefix}_load"):
                    st.session_state.policies_json = json_text
                    st.success("Loaded! Go to **Simulator** and press **Run simulation ✅**.")
            with c2:
                st.caption("Copy-paste JSON below:")
            st.code(json_text, language="json")

    # ---------- SECTION 1: Stick & Fork (10) ----------
    st.header("Section A — Stick & Fork (10 variants)")
    st.caption("All examples are Fork vs Sticks, but with different prediction quality, schedules, and failure modes.")

    stick_fork_examples: List[Dict[str, Any]] = []

    stick_fork_examples.append({
        "title": "A1) Baseline: Fork strong, Sticks weak, Fork fatigue at t=30",
        "story": "Fork matches prediction, then Efficiency drops at t=30. Sticks start low comfort and completion drops at t=20.",
        "json": [
            {
                "name": "Fork",
                "m_pred": {"Efficiency": 0.6, "Comfort": 0.8, "TaskCompletion": 1.0},
                "m_obs":  {"Efficiency": 0.6, "Comfort": 0.8, "TaskCompletion": 1.0},
                "m_obs_schedule": [
                    {"t": 0, "Efficiency": 0.6, "Comfort": 0.8, "TaskCompletion": 1.0},
                    {"t": 30, "Efficiency": 0.3}
                ]
            },
            {
                "name": "Sticks",
                "m_pred": {"Efficiency": 0.6, "Comfort": 0.8, "TaskCompletion": 1.0},
                "m_obs":  {"Efficiency": 0.5, "Comfort": 0.2, "TaskCompletion": 0.7},
                "m_obs_schedule": [
                    {"t": 0, "Efficiency": 0.5, "Comfort": 0.2, "TaskCompletion": 0.7},
                    {"t": 20, "TaskCompletion": 0.4}
                ]
            }
        ]
    })

    stick_fork_examples.append({
        "title": "A2) Analogy oversells Fork (pred too optimistic)",
        "story": "Fork predicted super high comfort/efficiency, but observed is just 'good'. Trust drops even though outcome is decent.",
        "json": [
            {
                "name": "Fork",
                "m_pred": {"Efficiency": 0.9, "Comfort": 0.95, "TaskCompletion": 1.0},
                "m_obs":  {"Efficiency": 0.65, "Comfort": 0.75, "TaskCompletion": 1.0}
            },
            {
                "name": "Sticks",
                "m_pred": {"Efficiency": 0.6, "Comfort": 0.8, "TaskCompletion": 1.0},
                "m_obs":  {"Efficiency": 0.5, "Comfort": 0.2, "TaskCompletion": 0.7}
            }
        ]
    })

    stick_fork_examples.append({
        "title": "A3) Sticks improve after training at t=25",
        "story": "Sticks start uncomfortable but comfort improves after practice; trust should recover.",
        "json": [
            {
                "name": "Fork",
                "m_pred": {"Efficiency": 0.6, "Comfort": 0.8, "TaskCompletion": 1.0},
                "m_obs":  {"Efficiency": 0.6, "Comfort": 0.8, "TaskCompletion": 1.0}
            },
            {
                "name": "Sticks",
                "m_pred": {"Efficiency": 0.55, "Comfort": 0.50, "TaskCompletion": 0.9},
                "m_obs":  {"Efficiency": 0.50, "Comfort": 0.20, "TaskCompletion": 0.9},
                "m_obs_schedule": [
                    {"t": 0, "Efficiency": 0.50, "Comfort": 0.20, "TaskCompletion": 0.9},
                    {"t": 25, "Comfort": 0.55}
                ]
            }
        ]
    })

    stick_fork_examples.append({
        "title": "A4) Fork breaks at t=12 (TaskCompletion collapses)",
        "story": "Everything is great until the fork bends; completion drops sharply and dominates ExM/SE.",
        "json": [
            {
                "name": "Fork",
                "m_pred": {"Efficiency": 0.6, "Comfort": 0.8, "TaskCompletion": 1.0},
                "m_obs":  {"Efficiency": 0.6, "Comfort": 0.8, "TaskCompletion": 1.0},
                "m_obs_schedule": [
                    {"t": 0, "Efficiency": 0.6, "Comfort": 0.8, "TaskCompletion": 1.0},
                    {"t": 12, "TaskCompletion": 0.3}
                ]
            },
            {
                "name": "Sticks",
                "m_pred": {"Efficiency": 0.55, "Comfort": 0.35, "TaskCompletion": 0.85},
                "m_obs":  {"Efficiency": 0.50, "Comfort": 0.25, "TaskCompletion": 0.80}
            }
        ]
    })

    stick_fork_examples.append({
        "title": "A5) Fork fatigue drift (Efficiency steps down)",
        "story": "Fork’s efficiency decreases in steps; trust drops if prediction didn’t anticipate it.",
        "json": [
            {
                "name": "Fork",
                "m_pred": {"Efficiency": 0.7, "Comfort": 0.8, "TaskCompletion": 1.0},
                "m_obs":  {"Efficiency": 0.7, "Comfort": 0.8, "TaskCompletion": 1.0},
                "m_obs_schedule": [
                    {"t": 0, "Efficiency": 0.7, "Comfort": 0.8, "TaskCompletion": 1.0},
                    {"t": 10, "Efficiency": 0.6},
                    {"t": 20, "Efficiency": 0.5},
                    {"t": 30, "Efficiency": 0.4}
                ]
            },
            {
                "name": "Sticks",
                "m_pred": {"Efficiency": 0.55, "Comfort": 0.40, "TaskCompletion": 0.9},
                "m_obs":  {"Efficiency": 0.50, "Comfort": 0.25, "TaskCompletion": 0.85}
            }
        ]
    })

    stick_fork_examples.append({
        "title": "A6) Sticks stress accumulation (Comfort erodes further)",
        "story": "Sticks start low comfort and it gets worse over time (frustration / stress).",
        "json": [
            {
                "name": "Fork",
                "m_pred": {"Efficiency": 0.6, "Comfort": 0.8, "TaskCompletion": 1.0},
                "m_obs":  {"Efficiency": 0.6, "Comfort": 0.8, "TaskCompletion": 1.0}
            },
            {
                "name": "Sticks",
                "m_pred": {"Efficiency": 0.55, "Comfort": 0.35, "TaskCompletion": 0.9},
                "m_obs":  {"Efficiency": 0.50, "Comfort": 0.30, "TaskCompletion": 0.85},
                "m_obs_schedule": [
                    {"t": 0, "Efficiency": 0.50, "Comfort": 0.30, "TaskCompletion": 0.85},
                    {"t": 15, "Comfort": 0.20},
                    {"t": 30, "Comfort": 0.10}
                ]
            }
        ]
    })

    stick_fork_examples.append({
        "title": "A7) Both succeed, but Fork is smoother (experience gap)",
        "story": "Task completion stays high for both; sticks are less comfortable and slightly less efficient.",
        "json": [
            {
                "name": "Fork",
                "m_pred": {"Efficiency": 0.65, "Comfort": 0.80, "TaskCompletion": 0.95},
                "m_obs":  {"Efficiency": 0.64, "Comfort": 0.82, "TaskCompletion": 0.95}
            },
            {
                "name": "Sticks",
                "m_pred": {"Efficiency": 0.65, "Comfort": 0.80, "TaskCompletion": 0.95},
                "m_obs":  {"Efficiency": 0.55, "Comfort": 0.25, "TaskCompletion": 0.95}
            }
        ]
    })

    stick_fork_examples.append({
        "title": "A8) Fork mismatch in Efficiency only (comfort ok)",
        "story": "Fork comfort is fine, but efficiency is worse than predicted; attribution should point to Efficiency.",
        "json": [
            {
                "name": "Fork",
                "m_pred": {"Efficiency": 0.80, "Comfort": 0.80, "TaskCompletion": 1.0},
                "m_obs":  {"Efficiency": 0.55, "Comfort": 0.80, "TaskCompletion": 1.0}
            },
            {
                "name": "Sticks",
                "m_pred": {"Efficiency": 0.55, "Comfort": 0.40, "TaskCompletion": 0.85},
                "m_obs":  {"Efficiency": 0.50, "Comfort": 0.25, "TaskCompletion": 0.80}
            }
        ]
    })

    stick_fork_examples.append({
        "title": "A9) Sticks predicted pessimistic, observed better (pleasant surprise)",
        "story": "Analogy underestimates sticks, so trust can rise (errors are small or even better than predicted).",
        "json": [
            {
                "name": "Fork",
                "m_pred": {"Efficiency": 0.65, "Comfort": 0.80, "TaskCompletion": 1.0},
                "m_obs":  {"Efficiency": 0.65, "Comfort": 0.80, "TaskCompletion": 1.0}
            },
            {
                "name": "Sticks",
                "m_pred": {"Efficiency": 0.30, "Comfort": 0.20, "TaskCompletion": 0.60},
                "m_obs":  {"Efficiency": 0.50, "Comfort": 0.45, "TaskCompletion": 0.80}
            }
        ]
    })

    stick_fork_examples.append({
        "title": "A10) Sticks unstable skill: completion drops then recovers",
        "story": "Completion dips (bad attempt), then recovers (good attempt). Shows non-stationary reliability.",
        "json": [
            {
                "name": "Fork",
                "m_pred": {"Efficiency": 0.60, "Comfort": 0.80, "TaskCompletion": 1.0},
                "m_obs":  {"Efficiency": 0.60, "Comfort": 0.80, "TaskCompletion": 1.0}
            },
            {
                "name": "Sticks",
                "m_pred": {"Efficiency": 0.55, "Comfort": 0.35, "TaskCompletion": 0.90},
                "m_obs":  {"Efficiency": 0.50, "Comfort": 0.25, "TaskCompletion": 0.90},
                "m_obs_schedule": [
                    {"t": 0, "Efficiency": 0.50, "Comfort": 0.25, "TaskCompletion": 0.90},
                    {"t": 12, "TaskCompletion": 0.40},
                    {"t": 24, "TaskCompletion": 0.85}
                ]
            }
        ]
    })

    for idx, ex in enumerate(stick_fork_examples, start=1):
        render_example_block(ex["title"], ex["story"], ex["json"], key_prefix=f"A{idx}")

    st.divider()

    # ---------- SECTION 2: Other stories (10) ----------
    st.header("Section B — Other stories (10 variants)")
    st.caption("Some are analogies (two policies), others are single-policy cases (no analogy).")

    other_examples: List[Dict[str, Any]] = []

    other_examples.append({
        "title": "B1) Commute analogy: Car vs Bicycle",
        "story": "Car predicted comfy and fast; bicycle predicted slower but pleasant. Observations can flip the story if traffic hits.",
        "json": [
            {
                "name": "Car",
                "m_pred": {"Efficiency": 0.85, "Comfort": 0.80, "TaskCompletion": 1.0},
                "m_obs":  {"Efficiency": 0.85, "Comfort": 0.80, "TaskCompletion": 1.0},
                "m_obs_schedule": [
                    {"t": 0, "Efficiency": 0.85, "Comfort": 0.80, "TaskCompletion": 1.0},
                    {"t": 15, "Efficiency": 0.40, "Comfort": 0.55}
                ]
            },
            {
                "name": "Bicycle",
                "m_pred": {"Efficiency": 0.55, "Comfort": 0.70, "TaskCompletion": 1.0},
                "m_obs":  {"Efficiency": 0.55, "Comfort": 0.70, "TaskCompletion": 1.0}
            }
        ]
    })

    other_examples.append({
        "title": "B2) Travel analogy: Spain vs China (jetlag hits at t=5)",
        "story": "Both trips complete, but comfort drops for long-haul travel due to jetlag and language friction.",
        "json": [
            {
                "name": "SpainTrip",
                "m_pred": {"Efficiency": 0.70, "Comfort": 0.85, "TaskCompletion": 1.0},
                "m_obs":  {"Efficiency": 0.68, "Comfort": 0.83, "TaskCompletion": 1.0}
            },
            {
                "name": "ChinaTrip",
                "m_pred": {"Efficiency": 0.70, "Comfort": 0.85, "TaskCompletion": 1.0},
                "m_obs":  {"Efficiency": 0.70, "Comfort": 0.85, "TaskCompletion": 1.0},
                "m_obs_schedule": [
                    {"t": 0, "Efficiency": 0.70, "Comfort": 0.85, "TaskCompletion": 1.0},
                    {"t": 5, "Comfort": 0.35}
                ]
            }
        ]
    })

    other_examples.append({
        "title": "B3) Single policy: New job onboarding (confidence grows)",
        "story": "One policy only. Starts uncomfortable, then comfort increases after iteration 20 (you get used to it).",
        "json": [
            {
                "name": "Onboarding",
                "m_pred": {"Efficiency": 0.55, "Comfort": 0.60, "TaskCompletion": 0.95},
                "m_obs":  {"Efficiency": 0.55, "Comfort": 0.35, "TaskCompletion": 0.95},
                "m_obs_schedule": [
                    {"t": 0, "Efficiency": 0.55, "Comfort": 0.35, "TaskCompletion": 0.95},
                    {"t": 20, "Comfort": 0.65}
                ]
            }
        ]
    })

    other_examples.append({
        "title": "B4) Single policy: Public speaking (stress accumulation)",
        "story": "Performance is okay, but comfort erodes over repeated exposure (unless you model training).",
        "json": [
            {
                "name": "PublicSpeaking",
                "m_pred": {"Efficiency": 0.65, "Comfort": 0.70, "TaskCompletion": 0.95},
                "m_obs":  {"Efficiency": 0.65, "Comfort": 0.70, "TaskCompletion": 0.95},
                "m_obs_schedule": [
                    {"t": 0, "Efficiency": 0.65, "Comfort": 0.70, "TaskCompletion": 0.95},
                    {"t": 10, "Comfort": 0.55},
                    {"t": 20, "Comfort": 0.40},
                    {"t": 30, "Comfort": 0.30}
                ]
            }
        ]
    })

    other_examples.append({
        "title": "B5) Buying analogy: Mac vs Gaming PC",
        "story": "Mac predicted high comfort, moderate efficiency. PC predicted high efficiency but lower comfort (noise/heat).",
        "json": [
            {
                "name": "Mac",
                "m_pred": {"Efficiency": 0.70, "Comfort": 0.90, "TaskCompletion": 1.0},
                "m_obs":  {"Efficiency": 0.68, "Comfort": 0.92, "TaskCompletion": 1.0}
            },
            {
                "name": "GamingPC",
                "m_pred": {"Efficiency": 0.90, "Comfort": 0.75, "TaskCompletion": 1.0},
                "m_obs":  {"Efficiency": 0.92, "Comfort": 0.55, "TaskCompletion": 1.0}
            }
        ]
    })

    other_examples.append({
        "title": "B6) Robot navigation analogy: SafeDetour vs RiskyShortcut",
        "story": "Shortcut is predicted efficient; observed comfort collapses even if completion stays high.",
        "json": [
            {
                "name": "SafeDetour",
                "m_pred": {"Efficiency": 0.55, "Comfort": 0.80, "TaskCompletion": 1.0},
                "m_obs":  {"Efficiency": 0.52, "Comfort": 0.82, "TaskCompletion": 1.0}
            },
            {
                "name": "RiskyShortcut",
                "m_pred": {"Efficiency": 0.80, "Comfort": 0.80, "TaskCompletion": 1.0},
                "m_obs":  {"Efficiency": 0.78, "Comfort": 0.20, "TaskCompletion": 1.0}
            }
        ]
    })

    other_examples.append({
        "title": "B7) Single policy: Cloud offload (network outage at t=18)",
        "story": "Looks great until network quality collapses; efficiency and completion take a hit at t=18.",
        "json": [
            {
                "name": "CloudOffload",
                "m_pred": {"Efficiency": 0.85, "Comfort": 0.75, "TaskCompletion": 1.0},
                "m_obs":  {"Efficiency": 0.85, "Comfort": 0.75, "TaskCompletion": 1.0},
                "m_obs_schedule": [
                    {"t": 0, "Efficiency": 0.85, "Comfort": 0.75, "TaskCompletion": 1.0},
                    {"t": 18, "Efficiency": 0.35, "TaskCompletion": 0.60}
                ]
            }
        ]
    })

    other_examples.append({
        "title": "B8) Lifestyle analogy: Diet vs Workout",
        "story": "Workout predicted efficient but low comfort initially; comfort improves later with habit formation.",
        "json": [
            {
                "name": "DietPlan",
                "m_pred": {"Efficiency": 0.60, "Comfort": 0.70, "TaskCompletion": 0.90},
                "m_obs":  {"Efficiency": 0.58, "Comfort": 0.65, "TaskCompletion": 0.88}
            },
            {
                "name": "WorkoutPlan",
                "m_pred": {"Efficiency": 0.75, "Comfort": 0.40, "TaskCompletion": 0.90},
                "m_obs":  {"Efficiency": 0.75, "Comfort": 0.25, "TaskCompletion": 0.90},
                "m_obs_schedule": [
                    {"t": 0, "Efficiency": 0.75, "Comfort": 0.25, "TaskCompletion": 0.90},
                    {"t": 25, "Comfort": 0.55}
                ]
            }
        ]
    })

    other_examples.append({
        "title": "B9) Single policy: Meditation (comfort rises slowly)",
        "story": "Predicted modest benefits; observed comfort gradually rises (habit).",
        "json": [
            {
                "name": "Meditation",
                "m_pred": {"Efficiency": 0.45, "Comfort": 0.60, "TaskCompletion": 0.85},
                "m_obs":  {"Efficiency": 0.45, "Comfort": 0.45, "TaskCompletion": 0.85},
                "m_obs_schedule": [
                    {"t": 0, "Efficiency": 0.45, "Comfort": 0.45, "TaskCompletion": 0.85},
                    {"t": 15, "Comfort": 0.55},
                    {"t": 30, "Comfort": 0.65}
                ]
            }
        ]
    })

    other_examples.append({
        "title": "B10) Micro-mobility analogy: E-scooter vs Walking",
        "story": "Scooter predicted efficient and comfortable; comfort drops due to rain and bumps at t=12.",
        "json": [
            {
                "name": "EScooter",
                "m_pred": {"Efficiency": 0.80, "Comfort": 0.75, "TaskCompletion": 1.0},
                "m_obs":  {"Efficiency": 0.80, "Comfort": 0.75, "TaskCompletion": 1.0},
                "m_obs_schedule": [
                    {"t": 0, "Efficiency": 0.80, "Comfort": 0.75, "TaskCompletion": 1.0},
                    {"t": 12, "Comfort": 0.35}
                ]
            },
            {
                "name": "Walking",
                "m_pred": {"Efficiency": 0.55, "Comfort": 0.70, "TaskCompletion": 1.0},
                "m_obs":  {"Efficiency": 0.55, "Comfort": 0.70, "TaskCompletion": 1.0}
            }
        ]
    })

    for idx, ex in enumerate(other_examples, start=1):
        render_example_block(ex["title"], ex["story"], ex["json"], key_prefix=f"B{idx}")

    st.divider()

    # ---------- SECTION 3: Real data benchamking (3) ----------
    st.header("Section C — Real data benchamking (3 variants)")
    st.caption("Examples built from measured values rather than illustrative toy settings.")
    st.markdown(
        "**Data source for benchmarking examples:** U.S. National Household Travel Survey (NHTS), "
        "conducted by the U.S. Department of Transportation, Federal Highway Administration (FHWA).\n\n"
        "- Official site: https://nhts.ornl.gov\n"
        "- 2017 NHTS Trip File (public-use CSV): https://nhts.ornl.gov/assets/2017/download/csv/trippub.csv\n"
        "- 2017 NHTS Household File (public-use CSV): https://nhts.ornl.gov/assets/2017/download/csv/hhpub.csv"
    )
    with st.expander("Local derivation assets used for these benchmarks", expanded=False):
        eq_path = Path("/Users/adrianlendinezibanez/Desktop/meta_params_equations.md")
        xlsx_path = Path("data/benchmarking/Urban_Rural_DistanceBin_Weighted_MetaParams_SPEED_DELAY.xlsx")
        trip_csv_path = Path("data/benchmarking/tripv2pub_with_meta_params.csv")

        if eq_path.exists():
            st.markdown("**Meta-parameter derivation equations (used in Section C):**")
            eq_text = eq_path.read_text(encoding="utf-8")
            cursor = 0
            for match in re.finditer(r"\\\[(.*?)\\\]", eq_text, flags=re.DOTALL):
                text_chunk = eq_text[cursor:match.start()]
                if text_chunk.strip():
                    # Render inline LaTeX of the form \( ... \) inside normal markdown text.
                    text_chunk = re.sub(r"\\\((.*?)\\\)", r"$\1$", text_chunk, flags=re.DOTALL)
                    st.markdown(text_chunk)
                st.latex(match.group(1).strip())
                cursor = match.end()

            tail = eq_text[cursor:]
            if tail.strip():
                tail = re.sub(r"\\\((.*?)\\\)", r"$\1$", tail, flags=re.DOTALL)
                st.markdown(tail)
        else:
            st.warning(f"Could not find equations file: {eq_path}")

        if xlsx_path.exists():
            st.download_button(
                label="Download benchmarking workbook (.xlsx)",
                data=xlsx_path.read_bytes(),
                file_name=xlsx_path.name,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key="section_c_download_xlsx",
            )
            if trip_csv_path.exists():
                st.download_button(
                    label="Download tripv2pub_with_meta_params.csv",
                    data=trip_csv_path.read_bytes(),
                    file_name=trip_csv_path.name,
                    mime="text/csv",
                    key="section_c_download_trip_csv",
                )
            else:
                st.warning(f"Could not find CSV file: {trip_csv_path}")
            st.markdown("**Workbook preview (first 20 rows):**")
            try:
                df_preview = pd.read_excel(xlsx_path)
                st.dataframe(df_preview.head(20), use_container_width=True)
            except Exception as exc:
                st.warning(f"Could not read workbook preview: {exc}")
        else:
            st.warning(f"Could not find workbook file: {xlsx_path}")

    real_data_benchamking_examples: List[Dict[str, Any]] = []

    real_data_benchamking_examples.append({
        "title": "C1) Urban 0-3 miles: Car vs Bicycle",
        "story": "Short urban trip where both options complete the task; observed comfort is high for both with slightly better efficiency for car.",
        "json": [
            {
                "name": "Urban Car 0-3 mi",
                "m_pred": {"Efficiency": 0.20, "Comfort": 0.85, "TaskCompletion": 1.0},
                "m_obs":  {"Efficiency": 0.1536, "Comfort": 0.8763, "TaskCompletion": 1.0}
            },
            {
                "name": "Urban Bicycle 0-3 mi",
                "m_pred": {"Efficiency": 0.15, "Comfort": 0.80, "TaskCompletion": 1.0},
                "m_obs":  {"Efficiency": 0.1226, "Comfort": 0.8051, "TaskCompletion": 1.0}
            }
        ]
    })

    real_data_benchamking_examples.append({
        "title": "C2) Urban 10-30 miles (harder regime)",
        "story": "Harder urban distance regime where both complete the task but observed comfort declines substantially, especially for bicycle.",
        "json": [
            {
                "name": "Urban Car 10-30 mi",
                "m_pred": {"Efficiency": 0.50, "Comfort": 0.65, "TaskCompletion": 1.0},
                "m_obs":  {"Efficiency": 0.6298, "Comfort": 0.5171, "TaskCompletion": 1.0}
            },
            {
                "name": "Urban Bicycle 10-30 mi",
                "m_pred": {"Efficiency": 0.50, "Comfort": 0.60, "TaskCompletion": 1.0},
                "m_obs":  {"Efficiency": 0.5913, "Comfort": 0.3833, "TaskCompletion": 1.0}
            }
        ]
    })

    real_data_benchamking_examples.append({
        "title": "C3) Rural 3-10 miles (high-speed car dominance)",
        "story": "Rural medium-distance regime where higher cruising speed and smoother roads favor car comfort and efficiency over bicycle.",
        "json": [
            {
                "name": "Rural Car 3-10 mi",
                "m_pred": {"Efficiency": 0.40, "Comfort": 0.75, "TaskCompletion": 1.0},
                "m_obs":  {"Efficiency": 0.4723, "Comfort": 0.7973, "TaskCompletion": 1.0}
            },
            {
                "name": "Rural Bicycle 3-10 mi",
                "m_pred": {"Efficiency": 0.40, "Comfort": 0.70, "TaskCompletion": 1.0},
                "m_obs":  {"Efficiency": 0.3940, "Comfort": 0.6584, "TaskCompletion": 1.0}
            }
        ]
    })

    for idx, ex in enumerate(real_data_benchamking_examples, start=1):
        render_example_block(ex["title"], ex["story"], ex["json"], key_prefix=f"C{idx}")


# -----------------------------
# Simulator page
# -----------------------------
def render_simulator():
    st.title("Subjetive experience simulator")

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

            # NEW: Rigidity slider
            rigidity = st.slider("Rigidity r (1=stubborn, 0=flexible)", 0.0, 1.0, 0.50, 0.01)

        with st.expander("Policies (JSON editor)", expanded=True):
            if st.session_state.policies_json is None:
                st.session_state.policies_json = default_policies_json(meta_names)

            st.caption(
                "Edit **m_pred** and **m_obs** here. Values must be in [0,1]. Keys must match meta names.\n"
                "Optional: add **m_obs_schedule** to change observed meta-params over time.\n"
                "Tip: go to **Examples** for copy-paste templates."
            )
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
        rigidity=float(rigidity),  # NEW
        adapt=bool(adapt),
    )

    try:
        policies = parse_policies_from_json(st.session_state.policies_json, meta_names)
    except Exception as e:
        st.error(f"Policies JSON error: {e}")
        st.stop()

    results, change_points_by_policy = simulate(policies, model, int(steps))
    policy_names = list(results.keys())
    colors = make_policy_colors(policy_names)

    def fig_small():
        return plt.subplots(figsize=(4.8, 2.2))

    def fig_small_tall():
        return plt.subplots(figsize=(4.8, 2.6))

    st.subheader("Final summary")
    summary = []
    for name, df in results.items():
        last = df.iloc[-1]
        cps = change_points_by_policy.get(name, [])
        summary.append({
            "Policy": name,
            "ExM_final": round(float(last["ExM"]), 4),
            "SE_final": round(float(last["SE"]), 4),
            "pT_global_final": round(float(last["pT_global"]), 4),
            "SE_minus_ExM_final": round(float(last["SE_minus_ExM"]), 4),
            "change_points": ", ".join(map(str, cps)) if cps else "-"
        })
    st.dataframe(pd.DataFrame(summary).sort_values("SE_final", ascending=False), use_container_width=True)

    colA, colB = st.columns(2)

    with colA:
        fig, ax = fig_small_tall()
        for name, df in results.items():
            ax.plot(df["t"], df["SE"], label=f"{name} SE", color=colors[name])
            ax.plot(df["t"], df["ExM"], linestyle="--", alpha=0.6, color=colors[name], label=f"{name} ExM")
            draw_change_lines(ax, change_points_by_policy.get(name, []))
        ax.set_title("SE vs ExM")
        ax.set_xlabel("t")
        ax.legend()
        st.pyplot(fig, use_container_width=True)

    with colB:
        fig, ax = fig_small_tall()
        for name, df in results.items():
            ax.plot(df["t"], df["pT_global"], label=name, color=colors[name])
            draw_change_lines(ax, change_points_by_policy.get(name, []))
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
            draw_change_lines(ax, change_points_by_policy.get(name, []))
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
        draw_change_lines(ax, change_points_by_policy.get(pick, []))
        ax.set_title(f"Weight evolution (policy: {pick})")
        ax.set_xlabel("t")
        ax.set_ylim(0, 1)
        ax.legend()
        st.pyplot(fig, use_container_width=True)

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
                    draw_change_lines(ax, change_points_by_policy.get(name, []))
                ax.set_ylim(0, 1)
                ax.set_title(f"Local trust in {m}")
                ax.set_xlabel("t")
                ax.legend()
                st.pyplot(fig, use_container_width=True)

    st.subheader("Observed meta-parameters over time (2 per row)")
    pairs = [meta_names[i:i + 2] for i in range(0, len(meta_names), 2)]
    for pair in pairs:
        c1, c2 = st.columns(2)
        for idx, m in enumerate(pair):
            target_col = c1 if idx == 0 else c2
            with target_col:
                fig, ax = fig_small()
                for name, df in results.items():
                    ax.plot(df["t"], df[f"m_obs_{m}"], label=name, color=colors[name])
                    draw_change_lines(ax, change_points_by_policy.get(name, []))
                ax.set_ylim(0, 1)
                ax.set_title(f"Observed {m} (m_obs)")
                ax.set_xlabel("t")
                ax.legend()
                st.pyplot(fig, use_container_width=True)

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
    page = st.radio("Go to", ["Simulator", "Documentation", "Examples (JSON Library)"], index=0)

if page == "Documentation":
    render_docs()
elif page == "Examples (JSON Library)":
    render_examples()
else:
    render_simulator()
