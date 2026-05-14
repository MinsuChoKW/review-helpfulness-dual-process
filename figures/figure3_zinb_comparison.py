import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
INPUT_CSV = os.path.join(REPO, "results", "baseline_zinb_coefficients.csv")

# Manuscript row order: Amazon, Booking.com (hotel), Audible, Coursera.
PLATFORM_ORDER  = ["amazon", "hotel", "audible", "coursera"]
PLATFORM_LABELS = {"amazon": "Amazon", "hotel": "Booking.com",
                   "audible": "Audible", "coursera": "Coursera"}

# Within each panel, the four cells are arranged so the conditional and the
# zero-inflation pair are visually adjacent.
COEF_ORDER = [
    ("conditional", "Systematic"),
    ("conditional", "Heuristic"),
    ("zero_inflation", "Systematic"),
    ("zero_inflation", "Heuristic"),
]
COEF_LABELS = [
    "Cond.\nSys.", "Cond.\nHeu.",
    "ZI\nSys.", "ZI\nHeu.",
]

MODEL_LABELS = {"standard": "Standard ZINB", "multilevel": "Multilevel ZINB"}
MODEL_COLORS = {"standard": "#7e7e7e", "multilevel": "#1f4e8c"}


def main():
    df = pd.read_csv(INPUT_CSV)
    # Sanity: 32 rows expected (4 platforms x 2 models x 2 components x 2 constructs).
    assert len(df) == 32, f"unexpected row count: {len(df)}"

    n_panels = len(PLATFORM_ORDER)
    fig, axes = plt.subplots(1, n_panels, figsize=(11.5, 3.6), sharey=False)

    bar_w = 0.38
    x_idx = np.arange(len(COEF_ORDER))

    for i, platform in enumerate(PLATFORM_ORDER):
        ax = axes[i]
        sub = df[df["platform"] == platform]
        for j, model in enumerate(["standard", "multilevel"]):
            ests, ses = [], []
            for comp, con in COEF_ORDER:
                row = sub[(sub.model == model) & (sub.component == comp) & (sub.construct == con)]
                if len(row) == 0:
                    ests.append(np.nan); ses.append(np.nan)
                else:
                    ests.append(row["estimate"].values[0])
                    ses.append(row["se"].values[0])
            offset = (-bar_w / 2) if model == "standard" else (bar_w / 2)
            ax.bar(x_idx + offset, ests, width=bar_w, yerr=ses, capsize=2.4,
                   color=MODEL_COLORS[model], label=MODEL_LABELS[model],
                   edgecolor="black", linewidth=0.4)

        ax.axhline(0, color="black", linewidth=0.6)
        ax.set_xticks(x_idx)
        ax.set_xticklabels(COEF_LABELS, fontsize=8)
        ax.set_title(PLATFORM_LABELS[platform], fontsize=10, weight="bold")
        ax.tick_params(axis="y", labelsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if i == 0:
            ax.set_ylabel("Coefficient (β)", fontsize=9)

    # Shared legend at the top.
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, fontsize=9,
               frameon=False, bbox_to_anchor=(0.5, 1.04))

    fig.suptitle("Standard ZINB vs. Multilevel ZINB — Helpfulness coefficients",
                 fontsize=11, y=1.10)

    plt.tight_layout()

    pdf_out = os.path.join(HERE, "figure3_zinb_comparison.pdf")
    png_out = os.path.join(HERE, "figure3_zinb_comparison.png")
    fig.savefig(pdf_out, bbox_inches="tight")
    fig.savefig(png_out, bbox_inches="tight", dpi=600)
    print(f"Wrote {pdf_out}")
    print(f"Wrote {png_out}")


if __name__ == "__main__":
    main()
