#!/usr/bin/env python3
"""
Tracksuit DS Interview — Survey Allocation Optimisation
"""

import json
import numpy as np
import pandas as pd
import pulp
from pathlib import Path
from typing import Dict, List, Tuple
from itertools import combinations

# ── Seed & Constants ───────────────────────────────────────────────────────────
RNG_SEED               = 42
TARGET_QUALIFIED       = 200
TIME_BUDGET_S          = 480.0
MAX_BUNDLE_SIZE        = 3
SAMPLE_SIZE            = 2_400
MAX_OVERDELIVERY_RATIO = 1.3   # max (expected qualified / target) for any category in a bundle
DATA_PATH              = Path("data/fake_category_data.csv")
CONFIDENCE       = 0.95   # one-sided probability of meeting the qualified target
_Z               = 1.6449  # norm.ppf(CONFIDENCE) — z-score for 95th percentile


def _prob_respondents(p: float, target: int = TARGET_QUALIFIED) -> int:
    """
    Minimum respondents n so that P(Binomial(n, p) >= target) >= CONFIDENCE.

    Derivation (normal approximation, closed-form quadratic):
      We need  np - z*sqrt(np(1-p)) >= target
      Let w = sqrt(np):  w^2 - z*sqrt(1-p)*w - target >= 0
      Solving:  w = (z*sqrt(q) + sqrt(z^2*q + 4*target)) / 2
                n = ceil(w^2 / p)

    This is strictly larger than ceil(target/p) for all p < 1, since it
    inflates the allocation to absorb Bernoulli variance rather than
    relying purely on the expected value hitting the target.
    """
    q = 1.0 - p
    w = (_Z * np.sqrt(q) + np.sqrt(_Z * _Z * q + 4.0 * target)) / 2.0
    return int(np.ceil(w * w / p))


def _prob_exp_qualified(p: float, target: int = TARGET_QUALIFIED) -> float:
    """
    Expected qualified count E[X] = n*p required so P(Bin(n,p) >= target) >= CONFIDENCE.
    Used as the RHS of each LP coverage constraint instead of the bare `target`.
    """
    q = 1.0 - p
    w = (_Z * np.sqrt(q) + np.sqrt(_Z * _Z * q + 4.0 * target)) / 2.0
    return w * w

# ── NZ Demographics ────────────────────────────────────────────────────────────
# Gender: Stats NZ, 31 Dec 2025
NZ_GENDER = {
    "Female": 0.4965,
    "Male":   0.5035,
}

# Age brackets: Infometrics NZ, 30 Jun 2025.
# Two census 5-year bands straddle the bracket boundaries, so they are
# split proportionally by the number of single years that fall in each bracket:
#   10-14 band  →  10-13 (4 years) goes to "<14",   14 (1 year) goes to "14-17"
#   15-19 band  →  15-17 (3 years) goes to "14-17", 18-19 (2 years) goes to "18-24"
_NZ_AGE_COUNTS: Dict[str, int] = {
    "<14":   297_580 + 325_440 + int(4 / 5 * 348_240),   # 0–13
    "14-17": int(1 / 5 * 348_240) + int(3 / 5 * 354_350),  # 14–17
    "18-24": int(2 / 5 * 354_350) + 327_760,               # 18–24
    "25-39": 344_080 + 401_820 + 401_290,                   # 25–39
    "40-54": 357_420 + 316_820 + 326_640,                   # 40–54
    "55-64": 313_260 + 309_870,                              # 55–64
    "65+":   272_530 + 225_340 + 185_530 + 117_040 + 65_300 + 34_400,
}
_total_pop = sum(_NZ_AGE_COUNTS.values())
NZ_AGE: Dict[str, float] = {
    bracket: count / _total_pop for bracket, count in _NZ_AGE_COUNTS.items()
}


# ══════════════════════════════════════════════════════════════════════════════
# Environment Setup
# ══════════════════════════════════════════════════════════════════════════════

def setup_environment(
    seed: int = RNG_SEED,
) -> Tuple[np.random.Generator, pd.DataFrame]:
    """
    Seed the random environment and load the category data.

    Returns
    -------
    rng : seeded NumPy Generator — pass this everywhere so all randomness
          is reproducible from a single seed.
    df  : category DataFrame with an added expected_time column (p_i × l_i),
          representing the expected seconds one respondent spends on category i.
    """
    rng = np.random.default_rng(seed)

    df = pd.read_csv(DATA_PATH)
    df["expected_time"] = df["incidence_rate"] * df["category_length_seconds"]

    return rng, df


# ══════════════════════════════════════════════════════════════════════════════
# Respondent Generation
# ══════════════════════════════════════════════════════════════════════════════

def generate_respondents(n: int, rng: np.random.Generator,) -> pd.DataFrame:
    """
    Generate n synthetic respondents stratified to NZ census demographics.

    Sampling is proportional to the NZ population (uniform in the sense
    that every NZ resident has an equal probability of selection), so the
    returned pool is nationally representative across gender and age.

    Sources
    -------
    Gender : Stats NZ, national population estimates, 31 Dec 2025
    Age    : Infometrics NZ age composition estimates, 30 Jun 2025
    """
    return pd.DataFrame({
        "gender": rng.choice(
            list(NZ_GENDER.keys()), size=n, p=list(NZ_GENDER.values())
        ),
        "age": rng.choice(
            list(NZ_AGE.keys()), size=n, p=list(NZ_AGE.values())
        ),
    })


# ══════════════════════════════════════════════════════════════════════════════
# Target Setup
# ══════════════════════════════════════════════════════════════════════════════

def target_setup(
    df: pd.DataFrame,
    target: int = TARGET_QUALIFIED,
) -> Tuple[pd.DataFrame, Dict[int, Dict[str, float]]]:
    """
    Compute per-category targets and build a category lookup dictionary.

    naive_respondents : ceil(target / p_i) — respondents needed if the
                        category sits alone in its own bundle.
    seconds_surveyed  : category_length_seconds — time (s) each *qualified*
                        respondent spends completing the category survey.

    The naive total (sum of naive_respondents) is the baseline the LP must beat.

    Returns
    -------
    targets : DataFrame sorted by incidence_rate ascending, with columns
                  category_id, category_name, incidence_rate,
                  seconds_surveyed, expected_time, naive_respondents
    lookup  : dict  { category_id -> {"incidence_rate": float,
                                      "seconds_surveyed": float} }
              Fast per-category access used throughout bundle generation
              and the LP formulation.
    """
    targets = df.copy()
    targets = targets.rename(columns={"category_length_seconds": "seconds_surveyed"})
    # naive_respondents as assuming if 0.8% incidence rate and 200 required, 250 people ttl. surveyed in that category.
    targets["naive_respondents"] = np.ceil(
        target / targets["incidence_rate"]
    ).astype(int)
    targets = targets.sort_values("incidence_rate").reset_index(drop=True)

    lookup: Dict[int, Dict[str, float]] = {
        int(row["category_id"]): {
            "incidence_rate":  row["incidence_rate"],
            "seconds_surveyed": row["seconds_surveyed"],
        }
        for _, row in targets.iterrows()
    }

    return targets, lookup

def _n_respondents(p: float, target: int = TARGET_QUALIFIED, probabilistic: bool = True) -> int:
    """Return respondent count either with or without the 95% confidence buffer."""
    return _prob_respondents(p, target) if probabilistic else int(np.ceil(target / p))


def greedy_approach(
    pool: pd.DataFrame,
    lookup: Dict[int, Dict[str, float]],
    time_budget: float = TIME_BUDGET_S,
    probabilistic: bool = True,
    overage: float = 0.0,
) -> List[Dict]:
    """
    Greedy bundle allocation.

    Gender-specific categories are handled first — only respondents of the
    matching gender are assigned to those bundles.  All remaining categories
    are then bundled greedily using the full pool, seeding each new bundle
    with the lowest-incidence unassigned category and filling up to the
    time budget.

    Returns a list of bundle dicts, each containing:
        categories        — list of category_ids in the bundle
        n_respondents     — total respondents needed (ceil(200 / min_p))
        gender_counts     — {"Female": n, "Male": n} breakdown
        expected_time     — Σ p_i · l_i across categories in the bundle
        gender_restricted — "Female" | "Male" | None
    overage inflates the recruitment target (e.g. 0.10 → size for 220 instead of 200).
    """
    effective_target  = int(round(TARGET_QUALIFIED * (1 + overage)))
    female_categories = [4, 35, 45]
    male_categories   = [31]
    gender_cat_map    = {"Female": female_categories, "Male": male_categories}
    gender_specific   = set(female_categories + male_categories)

    bundles: List[Dict] = []

    # ── Step 1: gender-restricted bundles — one independent bundle per category ─
    # Each category is sized for its own incidence rate; bundling them all into one
    # bundle sized for the rarest causes 2-3× over-delivery for the more common ones.
    for gender in NZ_GENDER:
        for cat_id in gender_cat_map[gender]:
            if cat_id not in lookup:
                continue
            p       = lookup[cat_id]["incidence_rate"]
            n_total = _n_respondents(p, effective_target, probabilistic)
            bundles.append({
                "categories":        [cat_id],
                "n_respondents":     n_total,
                "gender_counts":     {g: (n_total if g == gender else 0) for g in NZ_GENDER},
                "worst_case_time":   lookup[cat_id]["seconds_surveyed"],
                "expected_time":     p * lookup[cat_id]["seconds_surveyed"],
                "gender_restricted": gender,
            })

    # ── Step 2: remaining categories — greedy fill with full pool ─────────────
    # Sort by incidence_rate ascending so the most expensive categories seed
    # their own bundles rather than being left as orphans.
    remaining = sorted(
        [c for c in lookup if c not in gender_specific],
        key=lambda c: lookup[c]["incidence_rate"],
    )
    assigned = set(gender_specific)

    for cat_id in remaining:
        if cat_id in assigned:
            continue

        seed_p   = lookup[cat_id]["incidence_rate"]
        # Estimate bundle size now so we can check over-delivery for candidates.
        # remaining is sorted ascending so seed_p is always the min IR in the bundle.
        n_est    = _n_respondents(seed_p, effective_target, probabilistic)
        bundle   = [cat_id]
        assigned.add(cat_id)
        capacity = time_budget - lookup[cat_id]["seconds_surveyed"]

        for cand_id in remaining:
            if len(bundle) >= MAX_BUNDLE_SIZE:
                break
            if cand_id in assigned:
                continue
            cand_p      = lookup[cand_id]["incidence_rate"]
            survey_time = lookup[cand_id]["seconds_surveyed"]
            # Skip if adding this category would cause >2× over-delivery:
            # n_est respondents × cand_p qualified > MAX_OVERDELIVERY_RATIO × target.
            if n_est * cand_p > effective_target * MAX_OVERDELIVERY_RATIO:
                continue
            if survey_time <= capacity:
                bundle.append(cand_id)
                assigned.add(cand_id)
                capacity -= survey_time

        min_p   = min(lookup[c]["incidence_rate"] for c in bundle)
        n_total = _n_respondents(min_p, effective_target, probabilistic)

        gender_counts = {}
        for gender, prop in NZ_GENDER.items():
            gender_counts[gender] = int(np.round(n_total * prop))

        bundles.append({
            "categories":        bundle,
            "n_respondents":     n_total,
            "gender_counts":     gender_counts,
            "worst_case_time":   sum(lookup[c]["seconds_surveyed"] for c in bundle),
            "expected_time":     sum(lookup[c]["incidence_rate"] * lookup[c]["seconds_surveyed"] for c in bundle),
            "gender_restricted": None,
        })

    return bundles



# ══════════════════════════════════════════════════════════════════════════════
# Naive Allocation (baseline)
# ══════════════════════════════════════════════════════════════════════════════

def naive_allocation(
    lookup: Dict[int, Dict[str, float]],
    target: int = TARGET_QUALIFIED,
    probabilistic: bool = True,
    overage: float = 0.0,
) -> List[Dict]:
    """
    Baseline: every category gets its own dedicated bundle of respondents.
    No sharing — each category pays its full recruitment cost independently.
    Total respondents = Σ ceil(target / p_i) across all categories.
    overage inflates the recruitment target (e.g. 0.10 → size for 220 instead of 200).
    """
    # overage for effective target to essentially inflate the target to account for Bernoulli 
    # variance, so we don't just meet the target on expectation but with 95% confidence.
    effective_target = int(round(target * (1 + overage)))
    bundles: List[Dict] = []
    for cat_id, info in lookup.items():
        n = _n_respondents(info["incidence_rate"], effective_target, probabilistic)
        bundles.append({
            "categories":        [cat_id],
            "n_respondents":     n,
            "gender_counts":     {g: int(np.round(n * p)) for g, p in NZ_GENDER.items()},
            "worst_case_time":   info["seconds_surveyed"],
            "expected_time":     info["incidence_rate"] * info["seconds_surveyed"],
            "gender_restricted": None,
        })
    return bundles


# ══════════════════════════════════════════════════════════════════════════════
# LP-Optimal Allocation
# ══════════════════════════════════════════════════════════════════════════════

def _gender_bundles(lookup: Dict[int, Dict[str, float]], time_budget: float, probabilistic: bool = True, overage: float = 0.0) -> Tuple[List[Dict], set]:
    """
    Pre-assign gender-specific categories as independent single-category bundles.

    Each gender-specific category gets its own bundle sized to its own incidence
    rate so that every category receives approximately the target number of
    qualified respondents.  Packing all female (or all male) categories into a
    single bundle sized for the rarest one causes 2–3× over-delivery for the
    more common ones, which wastes budget and misleads the simulation.
    """
    effective_target  = int(round(TARGET_QUALIFIED * (1 + overage)))
    female_categories = [4, 35, 45]
    male_categories   = [31]
    gender_cat_map    = {"Female": female_categories, "Male": male_categories}
    gender_specific   = set(female_categories + male_categories)
    bundles: List[Dict] = []

    for gender in NZ_GENDER:
        for cat_id in gender_cat_map[gender]:
            if cat_id not in lookup:
                continue
            p       = lookup[cat_id]["incidence_rate"]
            n_total = _n_respondents(p, effective_target, probabilistic)
            bundles.append({
                "categories":        [cat_id],
                "n_respondents":     n_total,
                "gender_counts":     {g: (n_total if g == gender else 0) for g in NZ_GENDER},
                "worst_case_time":   lookup[cat_id]["seconds_surveyed"],
                "expected_time":     p * lookup[cat_id]["seconds_surveyed"],
                "gender_restricted": gender,
            })
    return bundles, gender_specific


def lp_optimal(
    lookup: Dict[int, Dict[str, float]],
    time_budget: float = TIME_BUDGET_S,
    target: int = TARGET_QUALIFIED,
    probabilistic: bool = True,
    overage: float = 0.0,
) -> List[Dict]:
    """
    LP-optimal bundle allocation via PuLP (CBC solver).

    Formulation
    -----------
    Decision variables : x_b ≥ 0  (respondents allocated to bundle b)
    Objective          : minimise Σ x_b
    Constraints        : for each category i,
                           Σ_{b ∋ i} p_i · x_b ≥ effective_target
    Bundle feasibility : Σ l_i ≤ time_budget  (worst-case: respondent qualifies for all)
                         |bundle| ≤ MAX_BUNDLE_SIZE

    Gender-specific categories are pre-assigned before the LP runs.
    The LP relaxation is solved continuously then rounded up to integers.
    overage inflates the recruitment target (e.g. 0.10 → size for 220 instead of 200).
    """
    effective_target = int(round(target * (1 + overage)))
    fixed_bundles, gender_specific = _gender_bundles(lookup, time_budget, probabilistic, overage)
    remaining = [c for c in lookup if c not in gender_specific]

    # Generate all feasible candidate bundles for remaining categories.
    # Two feasibility checks per candidate combo:
    #   1. Worst-case time (Σ l_i) ≤ time_budget
    #   2. No category in the bundle would receive more than MAX_OVERDELIVERY_RATIO × target
    #      qualified respondents (prevents pairing very different incidence rates).
    candidates: List[Tuple[int, ...]] = []
    for size in range(1, MAX_BUNDLE_SIZE + 1):
        for combo in combinations(remaining, size):
            worst_case = sum(lookup[c]["seconds_surveyed"] for c in combo)
            if worst_case > time_budget:
                continue
            if size > 1:
                ps     = [lookup[c]["incidence_rate"] for c in combo]
                min_p  = min(ps)
                max_p  = max(ps)
                if max_p / min_p > MAX_OVERDELIVERY_RATIO:
                    continue
            candidates.append(combo)

    # Solve LP
    prob = pulp.LpProblem("survey_lp", pulp.LpMinimize)
    x    = [pulp.LpVariable(f"x{i}", lowBound=0) for i in range(len(candidates))]
    prob += pulp.lpSum(x)

    for cat in remaining:
        p   = lookup[cat]["incidence_rate"]
        idx = [i for i, b in enumerate(candidates) if cat in b]
        rhs = _prob_exp_qualified(p, effective_target) if probabilistic else float(effective_target)
        prob += pulp.lpSum(p * x[i] for i in idx) >= rhs, f"cov_{cat}"

    prob.solve(pulp.PULP_CBC_CMD(msg=0))

    lp_bundles: List[Dict] = []
    for i, b in enumerate(candidates):
        v = pulp.value(x[i])
        if v and v > 1e-6:
            n_total = int(np.ceil(v))
            lp_bundles.append({
                "categories":        list(b),
                "n_respondents":     n_total,
                "gender_counts":     {g: int(np.round(n_total * p)) for g, p in NZ_GENDER.items()},
                "worst_case_time":   sum(lookup[c]["seconds_surveyed"] for c in b),
                "expected_time":     sum(lookup[c]["incidence_rate"] * lookup[c]["seconds_surveyed"] for c in b),
                "gender_restricted": None,
            })

    return fixed_bundles + lp_bundles


# ══════════════════════════════════════════════════════════════════════════════
# Monte Carlo Simulation
# ══════════════════════════════════════════════════════════════════════════════

def simulate_allocation(
    bundles:  List[Dict],
    lookup:   Dict[int, Dict[str, float]],
    n_months: int = 1_000,
    rng:      np.random.Generator = None,
) -> Dict:
    """
    Simulate n_months months under the given bundle allocation.

    Each month, each bundle draws its n_respondents; every respondent
    independently qualifies for each category in the bundle (Bernoulli p_i)
    and completes its survey (l_i seconds) if qualified.

    Returns
    -------
    qualified      — (n_months, n_categories) array of qualified counts
    mean_times     — (n_months,) mean respondent time per month
    pct_over       — (n_months,) % respondents exceeding TIME_BUDGET_S
    cat_ids        — ordered list of category_ids matching axis-1 of qualified
    total_respondents — total unique respondents recruited per month
    """
    if rng is None:
        rng = np.random.default_rng(RNG_SEED)

    cat_ids = list(lookup.keys())
    cat_idx = {cid: i for i, cid in enumerate(cat_ids)}

    specs = [
        (
            b["categories"],
            np.array([lookup[c]["incidence_rate"]  for c in b["categories"]]),
            np.array([lookup[c]["seconds_surveyed"] for c in b["categories"]]),
            b["n_respondents"],
        )
        for b in bundles
    ]

    qualified    = np.zeros((n_months, len(cat_ids)), dtype=np.int32)
    mean_times   = np.zeros(n_months)
    median_times = np.zeros(n_months)
    pct_over     = np.zeros(n_months)

    for m in range(n_months):
        month_times = []
        for cats, ps, ls, n in specs:
            qual  = rng.random((n, len(ps))) < ps
            times = (qual * ls).sum(axis=1)
            month_times.append(times)
            for j, cid in enumerate(cats):
                qualified[m, cat_idx[cid]] += int(qual[:, j].sum())
        all_times      = np.concatenate(month_times)
        mean_times[m]  = float(all_times.mean())
        median_times[m] = float(np.median(all_times))
        pct_over[m]    = 100.0 * (all_times > TIME_BUDGET_S).sum() / len(all_times)

    return {
        "qualified":          qualified,
        "mean_times":         mean_times,
        "median_times":       median_times,
        "pct_over":           pct_over,
        "cat_ids":            cat_ids,
        "total_respondents":  sum(b["n_respondents"] for b in bundles),
    }


# ══════════════════════════════════════════════════════════════════════════════
# Output Persistence
# ══════════════════════════════════════════════════════════════════════════════

def save_outputs(
    models:  Dict[str, Dict],
    targets: pd.DataFrame,
    lookup:  Dict[int, Dict[str, float]],
    output_dir: str = "model_outputs",
    n_months_sim: int = 100,
) -> None:
    """
    Persist bundle configurations and simulation summaries to output_dir/.

    Files written
    -------------
    bundles_{model}.json     — one record per bundle with category metadata
    simulation_summary.csv   — per-category mean qualified and % months met
    run_summary.json         — top-level metrics for each model
    """
    out = Path(output_dir)
    out.mkdir(exist_ok=True)
    cat_names = {int(r["category_id"]): r["category_name"] for _, r in targets.iterrows()}

    for model_name, data in models.items():
        bundles = data["bundles"]
        sim     = data["sim"]

        # ── bundles_{model}.json ──────────────────────────────────────────────
        bundle_records = []
        for i, b in enumerate(bundles):
            bundle_records.append({
                "bundle_id":        i,
                "size":             len(b["categories"]),
                "n_respondents":    b["n_respondents"],
                "worst_case_time_s": round(b["worst_case_time"], 2),
                "expected_time_s":  round(b["expected_time"], 2),
                "gender_restricted": b["gender_restricted"],
                "categories": [
                    {
                        "category_id":   c,
                        "category_name": cat_names.get(c, str(c)),
                        "incidence_rate": round(lookup[c]["incidence_rate"], 6),
                    }
                    for c in b["categories"]
                ],
            })
        with open(out / f"bundles_{model_name}.json", "w") as fh:
            json.dump(bundle_records, fh, indent=2)

        # ── run_summary.json ──────────────────────────────────────────────────
        q          = sim["qualified"]
        size_count = {1: 0, 2: 0, 3: 0}
        size_resp  = {1: 0, 2: 0, 3: 0}
        for b in bundles:
            s = len(b["categories"])
            size_count[s] = size_count.get(s, 0) + 1
            size_resp[s]  = size_resp.get(s, 0) + b["n_respondents"]

        summary_entry = {
            "total_respondents":              sim["total_respondents"],
            "n_bundles":                      len(bundles),
            "bundle_size_breakdown": {
                "singles":  {"count": size_count[1], "respondents": size_resp[1]},
                "pairs":    {"count": size_count[2], "respondents": size_resp[2]},
                "triples":  {"count": size_count[3], "respondents": size_resp[3]},
            },
            "pct_categories_meeting_target":  round(float((q >= TARGET_QUALIFIED).mean() * 100), 2),
            "mean_qualified_per_category":    round(float(q.mean()), 1),
            "avg_mean_respondent_time_s":     round(float(sim["mean_times"].mean()), 2),
            "avg_median_respondent_time_s":   round(float(sim["median_times"].mean()), 2),
            "mean_pct_over_budget":           round(float(sim["pct_over"].mean()), 4),
        }
        existing_summary: Dict = {}
        summary_path = out / "run_summary.json"
        if summary_path.exists():
            with open(summary_path) as fh:
                existing_summary = json.load(fh)
        existing_summary[model_name] = summary_entry
        with open(summary_path, "w") as fh:
            json.dump(existing_summary, fh, indent=2)

    # ── simulation_summary.csv ────────────────────────────────────────────────
    rows = []
    for _, row in targets.iterrows():
        cid  = int(row["category_id"])
        rec  = {
            "category_id":   cid,
            "category_name": row["category_name"],
            "incidence_rate": round(float(row["incidence_rate"]), 6),
        }
        for model_name, data in models.items():
            sim     = data["sim"]
            cat_ids = sim["cat_ids"]
            if cid in cat_ids:
                idx = cat_ids.index(cid)
                q   = sim["qualified"][:, idx]
                rec[f"{model_name}_mean_qualified"]  = round(float(q.mean()), 1)
                rec[f"{model_name}_pct_months_met"]  = round(float((q >= TARGET_QUALIFIED).mean() * 100), 1)
                rec[f"{model_name}_min_qualified"]   = int(q.min())
                rec[f"{model_name}_max_qualified"]   = int(q.max())
        rows.append(rec)

    pd.DataFrame(rows).to_csv(out / "simulation_summary.csv", index=False)
    print(f"Outputs saved → {out}/  ({', '.join(f'bundles_{m}.json' for m in models)}, simulation_summary.csv, run_summary.json)")


if __name__ == "__main__":
    rng, df         = setup_environment()
    pool            = generate_respondents(SAMPLE_SIZE, rng)
    targets, lookup = target_setup(df)

    naive   = naive_allocation(lookup)
    greedy  = greedy_approach(pool, lookup)
    lp      = lp_optimal(lookup)

    sim_naive  = simulate_allocation(naive,  lookup, rng=np.random.default_rng(RNG_SEED))
    sim_greedy = simulate_allocation(greedy, lookup, rng=np.random.default_rng(RNG_SEED))
    sim_lp     = simulate_allocation(lp,     lookup, rng=np.random.default_rng(RNG_SEED))

    save_outputs(
        models={
            "naive":      {"bundles": naive,  "sim": sim_naive},
            "greedy":     {"bundles": greedy, "sim": sim_greedy},
            "lp_optimal": {"bundles": lp,     "sim": sim_lp},
        },
        targets=targets,
        lookup=lookup,
    )
