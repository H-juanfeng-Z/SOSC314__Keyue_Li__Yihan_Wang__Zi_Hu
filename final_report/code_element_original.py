import json
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False


# ============================================================
# Step 1. Load fine_scores.json (0-10 scores)
# ============================================================
with open("fine_scores.json", "r", encoding="utf-8") as f:
    fine = pd.DataFrame(json.load(f))

fine = fine[(fine["status"] == "valid") & (fine["score"].notna())].copy()
print("fine_scores rows:", fine.shape)
print(fine["scoring_version"].value_counts())


# ============================================================
# Step 2. Load questions and merge on question_id
# ============================================================
q = pd.read_parquet("questions_clean.parquet")
q_sub = q[q["question_id"].isin(fine["question_id"].unique())].copy()

merged = fine.merge(
    q_sub[["question_id", "code_element_count"]],
    on="question_id",
    how="left",
)
print("merged:", merged.shape)


# ============================================================
# Step 3. Bin code_element_count
# ============================================================
merged["code_element_bin"] = pd.cut(
    merged["code_element_count"],
    bins=[-1, 0, 2, 5, 10, np.inf],
    labels=["0", "1-2", "3-5", "6-10", "11+"],
)

# ============================================================
# Step 4. Bin 0-10 score into ordered categories
# ============================================================
score_bins = [-0.1, 3, 6, 8, 10]
score_labels = ["low (0-3)", "mid (4-6)", "high (7-8)", "very high (9-10)"]

merged["score_bin"] = pd.cut(
    merged["score"],
    bins=score_bins,
    labels=score_labels,
)


# ============================================================
# Step 5. Plot 1: Percentage stacked bar per scoring version
#          (with count + percentage on each segment,
#           and total count on top of each bar)
# ============================================================
bin_order = ["0", "1-2", "3-5", "6-10", "11+"]
versions = ["candidate_0_10_format", "candidate_0_10_scope"]
colors = ["#C44E52", "#DD8452", "#4C72B0", "#55A868"]  # low -> very high

for ver in versions:
    sub = merged[merged["scoring_version"] == ver].copy()

    # count table and percentage table
    cnt = pd.crosstab(sub["code_element_bin"], sub["score_bin"])
    cnt = cnt.reindex(index=bin_order, columns=score_labels, fill_value=0)

    pct = cnt.div(cnt.sum(axis=1).replace(0, np.nan), axis=0) * 100
    pct = pct.fillna(0)

    print(f"\nCount table for {ver}:")
    print(cnt)
    print(f"\nPercentage table for {ver}:")
    print(pct.round(1))

    ax = pct.plot(
        kind="bar",
        stacked=True,
        figsize=(8, 5),
        color=colors,
        edgecolor="white",
    )

    # ---- annotate each segment with "count (pct%)" ----
    # 遍历每个 x 位置（即每个 code_element_bin），逐段累加高度
    for x_idx, bin_name in enumerate(bin_order):
        cumulative = 0.0
        for score_label in score_labels:
            seg_pct = pct.loc[bin_name, score_label]
            seg_cnt = cnt.loc[bin_name, score_label]
            if seg_pct <= 0:
                continue
            y_center = cumulative + seg_pct / 2.0
            # 只在段高足够时写文字，避免太挤
            if seg_pct >= 4:
                ax.text(
                    x_idx,
                    y_center,
                    f"{seg_cnt}\n({seg_pct:.1f}%)",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="white",
                    fontweight="bold",
                )
            else:
                # 太窄的段只写数量
                ax.text(
                    x_idx,
                    y_center,
                    f"{seg_cnt}",
                    ha="center",
                    va="center",
                    fontsize=6,
                    color="white",
                    fontweight="bold",
                )
            cumulative += seg_pct

        # ---- total count on top of each bar ----
        total_cnt = int(cnt.loc[bin_name].sum())
        ax.text(
            x_idx,
            101.5,
            f"n={total_cnt}",
            ha="center",
            va="bottom",
            fontsize=9,
            color="#2C3E50",
            fontweight="bold",
        )

    ax.set_xlabel("code_element_count")
    ax.set_ylabel("Percentage (%)")
    ax.set_title(f"Score distribution by code_element_count ({ver})")
    ax.legend(title="score range", loc="upper right", frameon=True)
    plt.xticks(rotation=0)
    plt.ylim(0, 110)  # 留点空间放 n=xxx
    plt.tight_layout()

    out_name = f"score_pct_by_code_element_{ver.replace('candidate_0_10_', '')}.png"
    plt.savefig(out_name, dpi=200)
    print("Saved:", out_name)
    plt.show()


# ============================================================
# Step 6. Plot 2: Mean score + SE, format vs scope side by side
#          (with mean and n on top of each bar)
# ============================================================
summary = (
    merged.groupby(["code_element_bin", "scoring_version"])["score"]
    .agg(["mean", "std", "count"])
    .reset_index()
)
summary["se"] = summary["std"] / np.sqrt(summary["count"])

print("\nSummary:")
print(summary.round(2))

fig, ax = plt.subplots(figsize=(9, 6))
positions = np.arange(len(bin_order))
width = 0.35
bar_colors = ["#4C72B0", "#DD8452"]

for i, ver in enumerate(versions):
    sub = summary[summary["scoring_version"] == ver].set_index("code_element_bin")
    means = [sub.loc[b, "mean"]  if b in sub.index else np.nan for b in bin_order]
    ses   = [sub.loc[b, "se"]    if b in sub.index else np.nan for b in bin_order]
    counts= [int(sub.loc[b, "count"]) if b in sub.index else 0 for b in bin_order]

    ax.bar(
        positions + (i - 0.5) * width,
        means,
        width=width,
        yerr=ses,
        capsize=4,
        color=bar_colors[i],
        edgecolor="#2C3E50",
        label=ver,
        alpha=0.85,
    )

    # ---- annotate mean + count on top of each bar ----
    for x, m, n in zip(positions + (i - 0.5) * width, means, counts):
        if not np.isnan(m):
            ax.text(
                x,
                m + 0.15,
                f"{m:.2f}\n(n={n})",
                ha="center",
                va="bottom",
                fontsize=8,
            )

ax.set_xticks(positions)
ax.set_xticklabels(bin_order)
ax.set_xlabel("code_element_count")
ax.set_ylabel("Mean score (0-10)")
ax.set_title("Mean score by code_element_count (format vs scope)")
ax.set_ylim(0, 11.5)  # 稍微抬高一点，防止文字被截断
ax.set_yticks(range(0, 11))
ax.grid(axis="y", linestyle="--", alpha=0.4)
ax.legend(title="scoring_version", loc="upper right", frameon=True)

plt.tight_layout()
plt.savefig("score_by_code_element_bar.png", dpi=200)
plt.show()