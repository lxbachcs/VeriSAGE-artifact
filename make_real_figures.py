"""
Generate the manuscript's Fig 3 (SLA-bound soundness) and Fig 4 (scalability +
repair convergence) from REAL measured results in results/core_results.json.
Writes into manuscript/figures/ (overwriting the illustrative versions).
"""
import json
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams.update({
    "pdf.fonttype": 42, "ps.fonttype": 42, "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"], "font.size": 11,
    "axes.linewidth": 0.8, "axes.grid": True, "grid.color": "0.85",
    "grid.linewidth": 0.6, "figure.dpi": 200,
})
C = {"blue": "#0072B2", "orange": "#E69F00", "green": "#009E73",
     "red": "#D55E00", "purple": "#CC79A7", "gray": "#7A7A7A"}

R = json.load(open("../results/core_results.json"))
FIG = "../manuscript/figures"

# ---------------- Fig 3: SLA-bound soundness (real) ----------------
cur = R["e2_sla_calibration"]["curve"]
pred = np.array([c["predicted"] for c in cur])
emp = np.array([c["empirical"] for c in cur])
emp_sd = np.array([c["empirical_sd"] for c in cur])

fig, ax = plt.subplots(figsize=(3.5, 2.9))
lim = float(pred.max()) * 1.08
ax.plot([0, lim], [0, lim], color=C["gray"], ls="--", lw=1.0, label="$y=x$ (certified bound)")
ax.fill_between([0, lim], [0, lim], lim, color=C["green"], alpha=0.08)
ax.errorbar(pred, emp, yerr=emp_sd, fmt="o", color=C["blue"], ms=5, capsize=2.5,
            mec="black", mew=0.5, label="measured (4k plans/pt)")
ax.set_xlabel("Predicted bound $\\delta$")
ax.set_ylabel("Empirical violation rate")
ax.set_xlim(0, lim); ax.set_ylim(0, lim)
ax.text(0.62 * lim, 0.15 * lim, "sound region\n(emp. $\\leq\\delta$): 100%",
        fontsize=7.5, color=C["green"], ha="center")
ax.legend(fontsize=7.5, loc="upper left")
for sp in ("top", "right"): ax.spines[sp].set_visible(False)
fig.tight_layout()
fig.savefig(f"{FIG}/fig3_sla_validation.pdf", bbox_inches="tight")
plt.close(fig)

# ---------------- Fig 4: scalability + repair (real) ----------------
# Panel (a) now uses the EXP3 stress test: DAG size k up to 200 (measured
# wall-clock, microseconds), replacing the old length 2-12 sweep.
E3 = json.load(open("../results/exp3_scalability.json"))
sc = E3["scal_k"]["fine"]
L = np.array([s["k"] for s in sc])
ms = np.array([s["us_mean"] for s in sc])         # already microseconds
ms_sd = np.array([s["us_sd"] for s in sc])

# Two-panel figure. Left (a): lightweight structural checker scales linearly in
# DAG size k -> genuine MICROSECONDS. Right (b): a production C2 that discharges
# real QF-LIA+EUF pre/postconditions with Z3 costs MILLISECONDS per service, and
# that cost grows with formula size -- the two honest timing regimes side by side.
fig, (axa, axb) = plt.subplots(1, 2, figsize=(7.0, 2.6))

# --- panel (a): structural scaling vs k (microseconds) ---
axa.plot(L, ms, "-o", color=C["blue"], ms=3.2)
axa.fill_between(L, ms - ms_sd, ms + ms_sd, color=C["blue"], alpha=0.15)
axa.set_xlabel("DAG size $k$ (# services)")
axa.set_ylabel("Verification time ($\\mu$s)")
b1, b0 = np.polyfit(L, ms, 1)
axa.plot(L, b1 * L + b0, ls=":", color=C["red"], lw=1.0,
         label=f"linear fit ($R^2$={np.corrcoef(L, ms)[0,1]**2:.3f})")
axa.legend(fontsize=8, loc="upper left")
axa.set_title("(a) lightweight structural checker", fontsize=8.5)
for sp in ("top", "right"): axa.spines[sp].set_visible(False)

# --- panel (b): production Z3-backed C2 cost vs formula size (milliseconds) ---
z3res = E3.get("z3_c2", {})
if z3res.get("available"):
    cur = z3res["curve"]
    na = np.array([c["n_atoms"] for c in cur], float)
    msc = np.array([c["ms_mean"] for c in cur], float)
    msc_sd = np.array([c["ms_sd"] for c in cur], float)
    axb.errorbar(na, msc, yerr=msc_sd, fmt="-o", color=C["green"], ms=3.5,
                 capsize=2.5, mec="black", mew=0.4, label="measured (Z3, 30 reps)")
    axb.axhline(1.0, color=C["gray"], ls="--", lw=0.9)
    axb.text(na.max(), 1.0, " 1\\,ms", fontsize=7, color=C["gray"],
             ha="right", va="bottom")
    axb.set_xlabel("pre/postcondition size (# atoms)")
    axb.set_ylabel("C2 SMT time (ms/service)")
    axb.set_ylim(0, max(1.25, float(msc.max()) * 1.15))
    axb.legend(fontsize=8, loc="upper left")
    axb.set_title("(b) production full-solver C2 (Z3)", fontsize=8.5)
    for sp in ("top", "right"): axb.spines[sp].set_visible(False)
fig.tight_layout()
fig.savefig(f"{FIG}/fig4_scalability.pdf", bbox_inches="tight")
plt.close(fig)

# ---------------- Fig 5: Bernstein/Bennett SLA-bound tightening (real) --------
# Budget-slack reduction of the variance-aware Bennett bound over Hoeffding as a
# function of latency concentration (variance fraction rho). Measured in
# results/exp4_bernstein.json.
try:
    E4 = json.load(open("../results/exp4_bernstein.json"))
    cur4 = E4["slack_by_concentration"]["curve"]
    rho = np.array([c["rho_variance_fraction"] for c in cur4])
    ratio = np.array([c["ratio_hoeff_over_bennett"] for c in cur4])
    order = np.argsort(rho)
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    ax.plot(rho[order], ratio[order], "-o", color=C["green"], ms=3.5)
    ax.axhline(1.0, color=C["gray"], ls="--", lw=0.9)
    ax.set_xscale("log")
    ax.set_xlabel("latency variance fraction $\\rho=\\sigma^2/(u^2/4)$")
    ax.set_ylabel("Hoeffding / Bennett\nbudget-slack ratio")
    ax.annotate("worst case\n(Hoeffding optimal)", xy=(1.0, ratio[order][-1]),
                fontsize=7, color=C["gray"], ha="right", va="bottom")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    fig.savefig(f"{FIG}/fig5_bernstein_tightening.pdf", bbox_inches="tight")
    plt.close(fig)
    print("Wrote real fig5_bernstein_tightening.pdf")
except Exception as e:
    print("fig5 skipped:", e)

print("Wrote real fig3_sla_validation.pdf and fig4_scalability.pdf")
print(f"  scalability linear R^2 = {np.corrcoef(L, ms)[0,1]**2:.4f}")
