"""
v2: 10M random loan portfolios, plot Female vs Income, and overlay Exercise 1.
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from loguru import logger

# --- Constants ---
BUDGET = 2500000
NUM_PORTFOLIOS = 10000000
INCLUSION_PROB = 0.05  # probability each loan is included in a random portfolio

# CSV column indices (zero based, so 0 = column 1 or A, 10 = column 11 or K ...)
COL_LOAN_NUMBER = 5
COL_LOAN_AMOUNT = 6
COL_FEMALE_FARMERS = 22
COL_EXPECTED_NET_INCOME = 28

# Exercise 1 portfolio names
PORTFOLIO_NAMES = [
    "1 Traditional",
    "2 Positive Screening",
    "3 Integration",
    "4 Impact Investment",
    "5 Philanthropy",
]


def to_float(val):
    """Convert a value to float, stripping commas from strings like '820,000'."""
    if isinstance(val, (int, float)) and not isinstance(val, bool):
        return float(val)
    s = str(val).replace(",", "")
    return float(s)


def load_loan_data(csv_path):
    """Load loan data from CSV. Returns (amounts, incomes, female, dataframe)."""
    df = pd.read_csv(csv_path)
    # Keep only rows with numbers
    loan_num = pd.to_numeric(df.iloc[:, COL_LOAN_NUMBER], errors="coerce")
    data = df[loan_num.notna()].copy().reset_index(drop=True)
    # Limit to first 200 loans, after remving headers there is only 200 rows of data
    data = data.head(200)

    amounts = np.array([to_float(x) for x in data.iloc[:, COL_LOAN_AMOUNT]])
    incomes = np.array([to_float(x) for x in data.iloc[:, COL_EXPECTED_NET_INCOME]])
    female = np.array([to_float(x) for x in data.iloc[:, COL_FEMALE_FARMERS]])

    return amounts, incomes, female, data


def generate_random_portfolios(amounts, incomes, female, n_target=None):
    """Generate random loan combinations within budget. n_target defaults to NUM_PORTFOLIOS."""
    n_target = n_target or NUM_PORTFOLIOS
    incomes_list = []
    female_list = []
    n_loans = len(amounts)

    logger.info("Generating {} valid random portfolios...", n_target)
    with tqdm(total=n_target, unit="portfolios", unit_scale=True, desc="Sampling") as pbar:
        while len(incomes_list) < n_target:
            # Randomly decide which loans to include
            include = np.random.random(n_loans) < INCLUSION_PROB

            total_amount = np.sum(amounts[include])
            if total_amount <= BUDGET:
                total_income = np.sum(incomes[include])
                total_female_reach = np.sum(female[include])
                incomes_list.append(total_income)
                female_list.append(total_female_reach)
                pbar.update(1)

    return np.array(incomes_list), np.array(female_list)


def load_exercise1_portfolios(data, amounts, incomes, female):
    """Load Exercise 1 portfolios from CSV columns (Portfolio 1-5, Yes/No)."""
    results = []
    n_rows = min(len(data), 200)

    for col_idx, name in enumerate(PORTFOLIO_NAMES):
        if col_idx >= 5:
            break
        col_vals = data.iloc[:n_rows, col_idx].astype(str).str.strip().str.lower()
        selected = col_vals == "yes"
        if not selected.any():
            continue

        idx = np.where(selected.values)[0]
        total_amount = np.sum(amounts[idx])
        if total_amount > BUDGET:
            logger.warning("Portfolio '{}' exceeds budget; skipping", name)
            continue

        total_income = np.sum(incomes[idx])
        total_female = np.sum(female[idx])
        results.append((name, total_income, total_female))

    return results


def compute_frontier(incomes, female):
    """Pareto frontier: optimal trade-off from max income (top-left) to max impact (bottom-right).
    Scans from high impact to low impact, keeping max income at each step, so the frontier
    includes Philanthropy (max female reach) through Traditional (max income)."""
    order = np.argsort(female)  # ascending: low to high female
    female_sorted = female[order]
    incomes_sorted = incomes[order]

    n = len(female_sorted)
    # Scan from HIGH female to LOW female (right to left): keep upper envelope of income.
    # This ensures we include max-impact portfolios (Philanthropy) on the frontier.
    max_so_far = np.empty(n)
    max_so_far[-1] = incomes_sorted[-1]  # start from rightmost (highest female)
    for i in range(n - 2, -1, -1):
        max_so_far[i] = max(max_so_far[i + 1], incomes_sorted[i])

    is_frontier = incomes_sorted >= max_so_far
    return female_sorted[is_frontier], incomes_sorted[is_frontier]


# ---- Main function: Runs the samples and plots the results ----
def main():
    parser = argparse.ArgumentParser(description="Efficient Impact Frontier")
    parser.add_argument("--max-points", type=int, default=None, help="Max portfolios (for quick test; default 10M)")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    input_dir = script_dir / "input"
    output_dir = script_dir / "output"
    data_path = input_dir / "Full Set of Possible Loans.csv"

    if not data_path.exists():
        logger.error("CSV not found: {}", data_path)
        raise FileNotFoundError(f"Full Set of Possible Loans.csv not found in {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Load loan data
    amounts, incomes, female, data = load_loan_data(data_path)
    logger.info("Loaded {} loans", len(amounts))

    # Step 2: Generate random portfolios
    random_incomes, random_female = generate_random_portfolios(
        amounts, incomes, female, n_target=args.max_points
    )
    logger.info("Generated {} valid portfolios", len(random_incomes))

    # Step 3: Load Exercise 1 portfolios
    exercise1 = load_exercise1_portfolios(data, amounts, incomes, female)
    logger.info("Loaded {} Exercise 1 portfolios", len(exercise1))

    # Step 4: Compute Efficient Impact Frontier (random + Exercise 1 points)
    all_incomes = np.append(random_incomes, [inc for _, inc, _ in exercise1])
    all_female = np.append(random_female, [fem for _, _, fem in exercise1])
    frontier_female, frontier_incomes = compute_frontier(all_incomes, all_female)

    # Step 5: Create plot
    fig, ax = plt.subplots(figsize=(10, 7))

    # Scatter: all random portfolios
    ax.scatter(
        random_female,
        random_incomes,
        s=0.5,
        alpha=0.15,
        c="steelblue",
        label="Random portfolios",
        rasterized=True,
    )

    # Plot the Efficient Impact Frontier line
    if len(frontier_female) > 0:
        ax.plot(
            frontier_female,
            frontier_incomes,
            color="darkred",
            linewidth=2,
            label="Efficient Impact Frontier",
        )

    # Overlay Exercise 1 portfolios with different colors
    colors = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00"]
    markers = ["o", "s", "D", "^", "v"]
    for i, (name, inc, fem) in enumerate(exercise1):
        c = colors[i % len(colors)]
        m = markers[i % len(markers)]
        ax.scatter(
            fem,
            inc,
            s=120,
            c=c,
            marker=m,
            edgecolors="black",
            linewidths=1.5,
            label=name,
            zorder=5,
        )

    ax.set_xlabel("Total Female Farmers and Employees Reached")
    ax.set_ylabel("Total Expected Net Income (USD)")
    ax.set_title("Efficient Impact Frontier: Loan Portfolio Trade-off (Social vs Financial)")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    out_path = output_dir / "plot.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    logger.info("Saved plot to {}", out_path)
    plt.close()

    # Step 6: Zoom plot on Efficient Impact Frontier + Traditional + Positive Screening
    plot_frontier_zoom(
        random_incomes,
        random_female,
        exercise1,
        frontier_female,
        frontier_incomes,
        output_dir / "plot_zoomed.png",
    )


def plot_frontier_zoom(
    random_incomes,
    random_female,
    exercise1,
    frontier_female,
    frontier_incomes,
    out_path,
):
    """Zoomed plot: Efficient Impact Frontier + Traditional + Positive Screening with value annotations."""
    zoom_names = {"1 Traditional", "2 Positive Screening"}
    zoom_exercise1 = [(n, inc, fem) for n, inc, fem in exercise1 if n in zoom_names]
    if not zoom_exercise1:
        logger.warning("No Traditional or Positive Screening in exercise1; skipping zoom plot")
        return

    fig, ax = plt.subplots(figsize=(10, 7))

    # Compute zoom bounds to show full frontier segment around Traditional & Positive Screening
    fem_min = min(fem for _, _, fem in zoom_exercise1)
    fem_max = max(fem for _, _, fem in zoom_exercise1)
    inc_min = min(inc for _, inc, _ in zoom_exercise1)
    inc_max = max(inc for _, inc, _ in zoom_exercise1)
    # Include frontier points well outside the two portfolios so the line isn't cut off
    pad_x_extra = max((fem_max - fem_min) * 0.5, 800)
    xlow, xhigh = fem_min - pad_x_extra, fem_max + pad_x_extra
    in_range = (frontier_female >= xlow) & (frontier_female <= xhigh)
    if in_range.any():
        inc_min = min(inc_min, frontier_incomes[in_range].min())
        inc_max = max(inc_max, frontier_incomes[in_range].max())
    pad_x = max((fem_max - fem_min) * 0.15, 800)
    pad_y = max((inc_max - inc_min) * 0.2, 25000)
    ax.set_xlim(xlow - pad_x, xhigh + pad_x)
    ax.set_ylim(inc_min - pad_y, inc_max + pad_y)

    # Scatter: only points inside zoom region
    xlo, xhi = ax.get_xlim()
    ylo, yhi = ax.get_ylim()
    in_zoom = (
        (random_female >= xlo) & (random_female <= xhi)
        & (random_incomes >= ylo) & (random_incomes <= yhi)
    )
    if in_zoom.any():
        ax.scatter(
            random_female[in_zoom],
            random_incomes[in_zoom],
            s=2,
            alpha=0.25,
            c="steelblue",
            rasterized=True,
            label="Random portfolios",
        )

    # Frontier line
    if len(frontier_female) > 0:
        ax.plot(
            frontier_female,
            frontier_incomes,
            color="darkred",
            linewidth=2.5,
            label="Efficient Impact Frontier",
        )

    # Traditional and Positive Screening with value annotations
    colors = ["#e41a1c", "#377eb8"]
    markers = ["o", "s"]
    for i, (name, inc, fem) in enumerate(zoom_exercise1):
        c = colors[i % len(colors)]
        m = markers[i % len(markers)]
        ax.scatter(
            fem,
            inc,
            s=180,
            c=c,
            marker=m,
            edgecolors="black",
            linewidths=2,
            label=name,
            zorder=5,
        )
        ax.annotate(
            f"{name}\nFemale: {fem:,.0f}\nIncome: ${inc:,.0f}",
            (fem, inc),
            xytext=(10, 10),
            textcoords="offset points",
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9),
            arrowprops=dict(arrowstyle="->", color=c),
        )

    ax.set_xlabel("Total Female Farmers and Employees Reached")
    ax.set_ylabel("Total Expected Net Income (USD)")
    ax.set_title("Zoom: Efficient Impact Frontier, Traditional & Positive Screening")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    logger.info("Saved zoom plot to {}", out_path)
    plt.close()


if __name__ == "__main__":
    main()
