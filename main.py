"""
Exercise 2 - Efficient Impact Frontier
Root Capital loan portfolio: 10M random portfolios, plot Female vs Income, overlay Exercise 1.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger

# --- Constants ---
BUDGET_USD = 2.5e6
N_TARGET_VALID = 10_000_000
BATCH_SIZE = 500_000
INCLUSION_PROB = 0.05  # Bernoulli p; tuned for ~10-20 loans per portfolio
EXCEL_SHEET = "Full Set of Possible Loans"
COL_LOAN_AMOUNT = 6
COL_FEMALE_FARMERS = 22
COL_EXPECTED_NET_INCOME = 28
COL_LOAN_NUMBER = 5
DATA_START_ROW = 2
N_PORTFOLIO_COLS = 5  # A-E for Portfolio 1-5
PORTFOLIO_NAMES = [
    "Traditional",
    "Positive Screening",
    "Integration",
    "Impact Investment",
    "Philanthropy",
]

# --- EXERCISE 1: Add your portfolios here ---
# When Exercise 1 is done, either:
# (a) Fill "Yes" in columns A-E of the Excel - script will auto-load
# (b) Or paste loan IDs below. Loan numbers are 1-200 (from column F).
EXERCISE1_PORTFOLIOS: dict[str, list[int]] = {
    "Traditional": [],
    "Positive Screening": [],
    "Integration": [],
    "Impact Investment": [],
    "Philanthropy": [],
}


def load_loan_data(excel_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    """Load loan data from Excel. Returns (amounts, incomes, female_reach, full_df)."""
    df = pd.read_excel(excel_path, sheet_name=EXCEL_SHEET, header=None)
    data = df.iloc[DATA_START_ROW:].copy()
    data = data[data[COL_LOAN_NUMBER].notna()].reset_index(drop=True)

    amounts = data[COL_LOAN_AMOUNT].astype(float).values
    incomes = data[COL_EXPECTED_NET_INCOME].astype(float).values
    female = data[COL_FEMALE_FARMERS].astype(float).values

    return amounts, incomes, female, data


def generate_random_portfolios(
    amounts: np.ndarray,
    incomes: np.ndarray,
    female: np.ndarray,
    n_target: int = N_TARGET_VALID,
    batch_size: int = BATCH_SIZE,
    p: float = INCLUSION_PROB,
    max_points: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate n_target valid random portfolios. Returns (incomes_arr, female_arr)."""
    if max_points is not None:
        n_target = min(n_target, max_points)
    n_loans = len(amounts)
    all_incomes: list[np.ndarray] = []
    all_female: list[np.ndarray] = []
    n_valid = 0
    n_attempted = 0

    logger.info("Generating random portfolios (batched Bernoulli sampling)...")

    while n_valid < n_target:
        # Binary matrix: (n_loans, batch_size)
        mask = np.random.random((n_loans, batch_size)) < p
        total_amounts = mask.T @ amounts  # (batch_size,)
        valid_idx = total_amounts <= BUDGET_USD
        n_valid_batch = valid_idx.sum()
        n_attempted += batch_size

        if n_valid_batch > 0:
            valid_mask = mask[:, valid_idx]
            batch_incomes = valid_mask.T @ incomes
            batch_female = valid_mask.T @ female
            all_incomes.append(batch_incomes)
            all_female.append(batch_female)
            n_valid += n_valid_batch

        if n_valid >= n_target:
            # Trim excess
            excess = n_valid - n_target
            if excess > 0:
                all_incomes[-1] = all_incomes[-1][:-excess]
                all_female[-1] = all_female[-1][:-excess]
                n_valid = n_target
            break

        if n_valid > 0 and n_valid % 1_000_000 == 0 and n_valid != len(all_incomes) * batch_size:
            logger.info(f"  Valid portfolios: {n_valid:,} / {n_target:,}")

    incomes_arr = np.concatenate(all_incomes)
    female_arr = np.concatenate(all_female)
    logger.info(f"Generated {n_valid:,} valid portfolios in ~{n_attempted:,} attempts")
    return incomes_arr, female_arr


def load_exercise1_from_excel(df: pd.DataFrame, amounts: np.ndarray, incomes: np.ndarray, female: np.ndarray) -> list[tuple[str, float, float]]:
    """Read Exercise 1 portfolios from Excel columns A-E (Yes/No)."""
    results: list[tuple[str, float, float]] = []
    n_rows = min(len(df), 200)

    for col_idx, name in enumerate(PORTFOLIO_NAMES):
        if col_idx >= N_PORTFOLIO_COLS:
            break
        col_vals = df.iloc[:n_rows, col_idx].astype(str).str.strip().str.lower()
        selected = col_vals == "yes"
        if not selected.any():
            continue
        idx = np.where(selected.values)[0]
        total_amount = amounts[idx].sum()
        if total_amount > BUDGET_USD:
            logger.warning(f"Portfolio '{name}' exceeds budget (${total_amount:,.0f}); skipping")
            continue
        total_income = incomes[idx].sum()
        total_female = female[idx].sum()
        results.append((name, total_income, total_female))

    return results


def load_exercise1_manual(df: pd.DataFrame, amounts: np.ndarray, incomes: np.ndarray, female: np.ndarray) -> list[tuple[str, float, float]]:
    """Use EXERCISE1_PORTFOLIOS dict. Loan IDs are 1-based from column F."""
    results: list[tuple[str, float, float]] = []
    loan_numbers = df[COL_LOAN_NUMBER].astype(int).values  # 1-based

    for name, loan_ids in EXERCISE1_PORTFOLIOS.items():
        if not loan_ids:
            continue
        # Map 1-based loan IDs to 0-based row indices
        row_indices = []
        for lid in loan_ids:
            match = np.where(loan_numbers == lid)[0]
            if len(match) > 0:
                row_indices.append(match[0])
        if not row_indices:
            logger.warning(f"No matching loan IDs for portfolio '{name}'")
            continue
        idx = np.array(row_indices)
        total_amount = amounts[idx].sum()
        if total_amount > BUDGET_USD:
            logger.warning(f"Portfolio '{name}' exceeds budget (${total_amount:,.0f}); skipping")
            continue
        total_income = incomes[idx].sum()
        total_female = female[idx].sum()
        results.append((name, total_income, total_female))

    return results


def load_exercise1_portfolios(
    df: pd.DataFrame,
    amounts: np.ndarray,
    incomes: np.ndarray,
    female: np.ndarray,
) -> list[tuple[str, float, float]]:
    """Try Excel first, then manual. Returns list of (name, income, female_reach)."""
    from_excel = load_exercise1_from_excel(df, amounts, incomes, female)
    if from_excel:
        logger.info(f"Loaded {len(from_excel)} Exercise 1 portfolios from Excel")
        return from_excel
    from_manual = load_exercise1_manual(df, amounts, incomes, female)
    if from_manual:
        logger.info(f"Loaded {len(from_manual)} Exercise 1 portfolios from EXERCISE1_PORTFOLIOS")
        return from_manual
    logger.info(
        "Exercise 1 portfolios not found; add loan IDs to EXERCISE1_PORTFOLIOS when ready, "
        "or fill 'Yes' in Excel columns A-E."
    )
    return []


def compute_frontier(incomes: np.ndarray, female: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Pareto frontier (upper envelope): max income for each female level."""
    order = np.argsort(female)
    female_sorted = female[order]
    incomes_sorted = incomes[order]

    # Upper envelope: for each x, keep max y seen when scanning left-to-right
    n = len(female_sorted)
    max_so_far = np.empty(n)
    max_so_far[0] = incomes_sorted[0]
    for i in range(1, n):
        max_so_far[i] = max(max_so_far[i - 1], incomes_sorted[i])

    # Frontier points: where we achieve a new max
    is_frontier = incomes_sorted >= max_so_far
    return female_sorted[is_frontier], incomes_sorted[is_frontier]


def plot_frontier(
    incomes: np.ndarray,
    female: np.ndarray,
    exercise1: list[tuple[str, float, float]],
    frontier_female: np.ndarray,
    frontier_incomes: np.ndarray,
    out_path: Path,
) -> None:
    """Create scatter + Exercise 1 overlay + frontier line."""
    fig, ax = plt.subplots(figsize=(10, 7))

    # 10M random points
    ax.scatter(
        female, incomes,
        s=0.5, alpha=0.15, c="steelblue", rasterized=True, label="Random portfolios",
    )

    # Frontier line
    if len(frontier_female) > 0:
        ax.plot(
            frontier_female, frontier_incomes,
            color="darkred", linewidth=2, label="Efficient Impact Frontier",
        )

    # Exercise 1 overlay
    if exercise1:
        colors = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00"]
        markers = ["o", "s", "D", "^", "v"]
        for i, (name, inc, fem) in enumerate(exercise1):
            c = colors[i % len(colors)]
            m = markers[i % len(markers)]
            ax.scatter(fem, inc, s=120, c=c, marker=m, edgecolors="black", linewidths=1.5, label=name, zorder=5)

    ax.set_xlabel("Total Female Farmers and Employees Reached")
    ax.set_ylabel("Total Expected Net Income (USD)")
    ax.set_title("Efficient Impact Frontier: Loan Portfolio Trade-off (Social vs Financial)")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    logger.info(f"Saved plot to {out_path}")
    plt.close()


def write_analysis_notes(
    out_path: Path,
    exercise1: list[tuple[str, float, float]],
    frontier_female: np.ndarray,
    frontier_incomes: np.ndarray,
) -> None:
    """Write interpretation notes for the report."""
    has_ex1 = len(exercise1) > 0
    with open(out_path, "w") as f:
        f.write("EFFICIENT IMPACT FRONTIER – INTERPRETATION NOTES\n")
        f.write("=" * 50 + "\n\n")
        f.write("1. TRADE-OFF BETWEEN SOCIAL AND FINANCIAL VALUE\n")
        f.write("-" * 40 + "\n")
        f.write(
            "The Efficient Impact Frontier traces the optimal trade-off between Total Expected "
            "Net Income (financial value) and Total Female Farmers and Employees Reached "
            "(social value). Portfolios on the frontier represent Pareto-optimal combinations: "
            "for a given level of social impact, no other feasible portfolio yields higher "
            "financial return; and for a given financial return, no other portfolio reaches "
            "more female farmers and employees.\n\n"
        )
        f.write("2. SHAPE OF THE FRONTIER\n")
        f.write("-" * 40 + "\n")
        f.write(
            "The upward-sloping frontier illustrates that achieving higher social impact "
            "typically requires accepting lower financial returns (or vice versa). "
            "The cloud of random portfolios below the frontier shows that most random "
            "combinations are dominated—i.e., they achieve neither the best financial nor "
            "social outcomes. Strategic portfolio construction (as in Exercise 1) aims to "
            "select combinations that lie on or near this frontier.\n\n"
        )
        if has_ex1:
            f.write("3. POSITION OF EXERCISE 1 PORTFOLIOS\n")
            f.write("-" * 40 + "\n")
            f.write(
                "When Exercise 1 portfolios are overlaid, their position relative to the "
                "frontier indicates how well each ESG strategy performs:\n"
            )
            f.write(
                "- Portfolios on the frontier: optimally trade off impact and return.\n"
            )
            f.write(
                "- Portfolios inside the frontier: there exists another combination that "
                "dominates them (higher income and/or higher female reach).\n"
            )
            f.write(
                "- The Traditional portfolio typically maximizes income but minimizes "
                "female reach; Philanthropy maximizes female reach but sacrifices income; "
                "Integration and Impact Investment aim for intermediate trade-offs.\n\n"
            )
        else:
            f.write("3. EXERCISE 1 PORTFOLIOS\n")
            f.write("-" * 40 + "\n")
            f.write(
                "Exercise 1 portfolios were not loaded. Once added (via Excel or "
                "EXERCISE1_PORTFOLIOS), re-run and compare their positions to the frontier "
                "to assess how each strategy performs relative to the efficient trade-off.\n\n"
            )
        f.write("4. IMPLICATIONS FOR ROOT CAPITAL\n")
        f.write("-" * 40 + "\n")
        f.write(
            "The frontier helps Root Capital communicate the trade-off to stakeholders: "
            "pursuing greater impact (e.g., reaching more female farmers) generally comes "
            "at the cost of lower expected financial return. The Efficient Impact Frontier "
            "identifies the set of portfolios that optimally balance these objectives "
            "within the USD 2.5 million budget constraint.\n"
        )
    logger.info(f"Saved analysis notes to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Exercise 2: Efficient Impact Frontier")
    parser.add_argument(
        "--max-points",
        type=int,
        default=None,
        help="Max random portfolios to generate (for quick iterations; default 10M)",
    )
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    input_dir = script_dir / "input"
    output_dir = script_dir / "output"
    excel_path = input_dir / "Loan_Portfolio.xlsx"
    if not excel_path.exists():
        logger.error(f"Excel not found: {excel_path}")
        raise FileNotFoundError(f"Loan_Portfolio.xlsx not found in {input_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Load data
    amounts, incomes, female, data_df = load_loan_data(excel_path)
    logger.info(f"Loaded {len(amounts)} loans")

    # Step 2: Random portfolios
    random_incomes, random_female = generate_random_portfolios(
        amounts, incomes, female, max_points=args.max_points
    )

    # Step 3: Exercise 1 (boilerplate)
    exercise1 = load_exercise1_portfolios(data_df, amounts, incomes, female)

    # Step 4: Frontier
    all_incomes = random_incomes.copy()
    all_female = random_female.copy()
    for _, inc, fem in exercise1:
        all_incomes = np.append(all_incomes, inc)
        all_female = np.append(all_female, fem)
    frontier_female, frontier_incomes = compute_frontier(all_incomes, all_female)

    # Save frontier CSV
    frontier_path = output_dir / "frontier_points.csv"
    pd.DataFrame({"female_reached": frontier_female, "expected_net_income": frontier_incomes}).to_csv(
        frontier_path, index=False
    )
    logger.info(f"Saved frontier points to {frontier_path}")

    # Step 5: Visualization
    out_path = output_dir / "efficient_impact_frontier.png"
    plot_frontier(
        random_incomes, random_female,
        exercise1,
        frontier_female, frontier_incomes,
        out_path,
    )

    # Step 6: Analysis notes for report
    analysis_path = output_dir / "analysis_notes.txt"
    write_analysis_notes(analysis_path, exercise1, frontier_female, frontier_incomes)


if __name__ == "__main__":
    main()
