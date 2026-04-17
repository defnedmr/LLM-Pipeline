"""
analysis/analyze.py
-------------------
Pipeline çıktısını (results_all_models.csv) okur, metrikleri hesaplar
ve figures/ klasörüne grafikleri kaydeder.

Kullanım:
    python analysis/analyze.py
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.contingency_tables import mcnemar

RESULTS = Path("results/results_all_models.csv")
FIGURES = Path("figures")
FIGURES.mkdir(exist_ok=True)

# ── Veriyi yükle ──────────────────────────────────────────────────────────────
df = pd.read_csv(RESULTS)
df["score"] = pd.to_numeric(df["score"], errors="coerce")

STAGE_ORDER = ["base", "sft", "rlhf"]
DIM_ORDER   = ["cot", "instruction", "sycophancy"]
LANG_ORDER  = ["tr", "en"]

# ── 1. Grup ortalamaları ──────────────────────────────────────────────────────
summary = (
    df.groupby(["stage", "lang", "dimension"])["score"]
    .agg(mean="mean", std="std", n="count")
    .reset_index()
)
print(summary.to_string(index=False))

# ── 2. Isı haritası: model × dil, her boyut için ─────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for ax, dim in zip(axes, DIM_ORDER):
    pivot = (
        df[df["dimension"] == dim]
        .groupby(["stage", "lang"])["score"]
        .mean()
        .unstack("lang")
        .reindex(STAGE_ORDER)
    )
    sns.heatmap(
        pivot, annot=True, fmt=".2f", vmin=0, vmax=1,
        cmap="viridis", linewidths=0.5, ax=ax
    )
    ax.set_title(dim.upper())
    ax.set_xlabel("Language")
    ax.set_ylabel("Alignment Stage")

plt.tight_layout()
plt.savefig(FIGURES / "heatmap_all_dims.png", dpi=150)
plt.close()
print("[saved] figures/heatmap_all_dims.png")

# ── 3. Çizgi grafik: hizalama ilerledikçe skor nasıl değişiyor? ──────────────
line_data = (
    df.groupby(["stage", "dimension", "lang"])["score"]
    .mean()
    .reset_index()
)
line_data["stage_num"] = line_data["stage"].map({"base": 0, "sft": 1, "rlhf": 2})

fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
for ax, lang in zip(axes, LANG_ORDER):
    subset = line_data[line_data["lang"] == lang]
    for dim in DIM_ORDER:
        s = subset[subset["dimension"] == dim].sort_values("stage_num")
        ax.plot(s["stage_num"], s["score"], marker="o", label=dim)
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["Base", "SFT", "RLHF"])
    ax.set_ylim(0, 1)
    ax.set_title(f"Alignment progression — {lang.upper()}")
    ax.set_xlabel("Alignment Stage")
    ax.set_ylabel("Score")
    ax.legend()
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(FIGURES / "line_alignment_progression.png", dpi=150)
plt.close()
print("[saved] figures/line_alignment_progression.png")

# ── 4. Çubuk grafik: CoT hata hassasiyeti, model × dil ───────────────────────
cot = df[df["dimension"] == "cot"]
bar_data = cot.groupby(["stage", "lang"])["score"].mean().reset_index()

fig, ax = plt.subplots(figsize=(7, 4))
x = range(len(STAGE_ORDER))
width = 0.35
for i, lang in enumerate(LANG_ORDER):
    vals = [
        bar_data.query(f"stage=='{s}' and lang=='{lang}'")["score"].values
        for s in STAGE_ORDER
    ]
    vals = [v[0] if len(v) else 0 for v in vals]
    ax.bar([xi + i * width for xi in x], vals, width, label=lang.upper())

ax.set_xticks([xi + width / 2 for xi in x])
ax.set_xticklabels(["Base", "SFT", "RLHF"])
ax.set_ylim(0, 1)
ax.set_title("CoT Error Sensitivity Rate — Model × Language")
ax.set_ylabel("Sensitivity Rate")
ax.legend()
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(FIGURES / "bar_cot_sensitivity.png", dpi=150)
plt.close()
print("[saved] figures/bar_cot_sensitivity.png")

# ── 5. McNemar testi: TR vs EN anlamlı mı? ───────────────────────────────────
print("\n── McNemar Testi (TR vs EN) ──")
for dim in DIM_ORDER:
    for stage in STAGE_ORDER:
        sub = df[(df["dimension"] == dim) & (df["stage"] == stage)].copy()
        tr = sub[sub["lang"] == "tr"].set_index("question_id")["score"]
        en = sub[sub["lang"] == "en"].set_index("question_id")["score"]
        common = tr.index.intersection(en.index)
        if len(common) < 5:
            continue
        a = tr.loc[common].values
        b = en.loc[common].values
        n00 = ((a == 0) & (b == 0)).sum()
        n01 = ((a == 0) & (b == 1)).sum()
        n10 = ((a == 1) & (b == 0)).sum()
        n11 = ((a == 1) & (b == 1)).sum()
        table = [[n00, n01], [n10, n11]]
        result = mcnemar(table, exact=True)
        sig = "✓ ANLAMLI" if result.pvalue < 0.05 else "✗ anlamsız"
        print(f"  {dim:15s} | {stage:5s} | p={result.pvalue:.4f}  {sig}")

# ── 6. Δ skoru tablosu: RLHF − Base ──────────────────────────────────────────
print("\n── Δ Skoru (RLHF − Base) ──")
delta_rows = []
for dim in DIM_ORDER:
    for lang in LANG_ORDER:
        base_score = df[(df["dimension"]==dim)&(df["stage"]=="base")&(df["lang"]==lang)]["score"].mean()
        rlhf_score = df[(df["dimension"]==dim)&(df["stage"]=="rlhf")&(df["lang"]==lang)]["score"].mean()
        delta = rlhf_score - base_score
        delta_rows.append({"dimension": dim, "lang": lang, "base": base_score, "rlhf": rlhf_score, "delta": delta})

delta_df = pd.DataFrame(delta_rows)
print(delta_df.to_string(index=False))
delta_df.to_csv(FIGURES / "delta_scores.csv", index=False)
print("[saved] figures/delta_scores.csv")
