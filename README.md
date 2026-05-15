# Survey Allocation Optimisation — Tracksuit DS Take-Home

A constrained optimisation solution for minimising the total number of survey respondents needed to guarantee every Tracksuit category receives its monthly qualified sample, while respecting an 8-minute respondent time budget and national demographic representativeness requirements.

---

## Problem Statement

Tracksuit runs monthly surveys across ~80 product categories. Each category requires **n = 200 qualified respondents per month** (≥ 2,400 per year). A respondent "qualifies" for a category by answering its screening question affirmatively — a Bernoulli draw with probability equal to that category's incidence rate.

The challenge is to design **survey bundles** — assignments of one, two, or three categories to a single respondent — and determine how many respondents to allocate to each bundle, such that every category's qualification target is met at minimum total recruiting cost.

Three constraints make this non-trivial:

| Constraint | Detail |
|---|---|
| **Qualification target** | Every category must receive ≥ 200 qualified respondents per month in expectation, with 95% confidence. Annual contract guarantees ≥ 2,400. |
| **Time budget** | Each respondent has a hard limit of **480 seconds (8 minutes)** of total interview time. Bundling is only feasible if the worst-case combined survey length (i.e. a respondent who qualifies for every category in the bundle) stays under this limit. |
| **Demographic representation** | The pool **exposed** to each category's qualifier must be nationally representative across gender, age, and region. Qualification skew is expected and acceptable; exposure skew is not. |

The objective is to minimise the total number of respondents surveyed across all bundles.

---

## Data

### `data/fake_category_data.csv`
The primary dataset. 77 categories with the following fields:

| Field | Description |
|---|---|
| `category_id` | Unique integer identifier |
| `category_name` | Human-readable category name |
| `incidence_rate` | Estimated proportion of NZ population that qualifies for the category (Bernoulli p) |
| `category_length_seconds` | Expected survey completion time (seconds) for a qualified respondent |

### `data/fake_category_data_and_toy_problem.xlsx`

**Use this spreadsheet as a sandbox before touching any code.** It contains the full category data alongside a toy-problem tab scaled to ~10 categories and a target of 10 qualified respondents per category. Working through manual bundle assignments in the spreadsheet builds the intuition needed to understand which constraints bind first (time vs. incidence vs. coverage) and which pairing strategies are naturally efficient — low-incidence categories paired with shorter surveys, similar-incidence categories grouped together to avoid over-delivery. The algorithmic solutions in `main.py` directly encode the intuitions developed here.

---

## Project Stages

The solution was built in six stages, following the project plan in `tracksuit_project_plan.html`.

### Stage 1 — Toy Problem and Manual Intuition (~1 hour)

Scale the problem to 10 categories with a target of 10 qualified per category and manually allocate respondents to bundles in `fake_category_data_and_toy_problem.xlsx`. Test pairing strategies: mix high and low incidence rates, mix short and long surveys, respect the time budget. Track target coverage and time utilisation per respondent across manual iterations to build intuition before writing any code.

**Outputs:** A working baseline on the toy problem, intuition for which constraints bind first, and an understanding of why pairing dissimilar incidence rates causes over-delivery for the higher-incidence category.

### Stage 2 — Naive Baseline (~1 hour)

Implement a deterministic "no bundling" baseline where every category gets its own pool of respondents sized to `ceil(target / incidence_rate)`. This establishes the upper-bound respondent cost that optimisation must beat and provides a reference for measuring percentage savings.

**Naive total: ~44,790 respondents.** All further strategies are benchmarked against this.

### Stage 3 — Bundle Generation and LP Optimisation (~2 hours)

**Bundle generation.** Enumerate all feasible single, pair, and triple-category combinations from the 73 non-gender-restricted categories. A bundle is feasible if:
1. Its **worst-case combined survey length** (sum of all category lengths) ≤ 480 seconds — guarantees no respondent exceeds the time budget even if they qualify for every category in the bundle.
2. The **maximum-to-minimum incidence rate ratio** within the bundle ≤ 1.3 — prevents over-delivery where the rarest category drives up the respondent count and floods the more common categories with excess qualified respondents.

Gender-specific categories (female-only: 4, 35, 45; male-only: 31) are pre-assigned as independent single-category bundles sized for their own incidence rates, not pooled into one gender bundle. Pooling them caused 2–3× over-delivery for higher-incidence categories in the same gender group.

**LP formulation (PuLP / CBC solver).**

- **Decision variables:** `x_b ≥ 0` — respondents allocated to bundle `b`
- **Objective:** minimise `Σ x_b`
- **Coverage constraint per category:** `Σ_{b ∋ i} p_i · x_b ≥ rhs_i`
  - `rhs_i` is the inflated expected qualified count required to achieve 95% confidence: `rhs_i = _prob_exp_qualified(p_i, 200)` (derived from a normal-approximation closed-form quadratic)
- **Time feasibility:** enforced at candidate generation — infeasible bundles never enter the LP
- The LP is solved as a continuous relaxation; each non-zero `x_b` is rounded up to the nearest integer

**Probabilistic guarantee.** The `_prob_respondents(p, target)` function solves for the minimum `n` such that `P(Binomial(n, p) ≥ target) ≥ 0.95`. This uses the one-sided z-score `z = 1.6449` (i.e. `norm.ppf(0.95)`), not the two-sided `1.96`. The closed-form solution: let `w = sqrt(n·p)`, solve `w² - z·sqrt(1-p)·w - target ≥ 0`, giving `w = (z·sqrt(q) + sqrt(z²·q + 4·target)) / 2`, `n = ceil(w²/p)`.

**Outputs:** Optimal bundle set and respondent allocations. LP-optimal total: ~17,500–20,000 respondents (55–60% reduction vs. naive).

### Stage 4 — Greedy Heuristic (~1 hour)

Implement a greedy algorithm as a faster, interpretable alternative to the LP. Categories are sorted by incidence rate ascending. The rarest unassigned category always seeds a new bundle. The inner loop scans remaining unassigned categories and adds the first one that passes both the time budget guard and the over-delivery guard (`n_est × cand_p ≤ target × 1.3`). Bundle size is capped at 3.

The greedy algorithm is substantially faster than the LP solver (no combinatorial candidate enumeration or solver runtime) but produces suboptimal pairings for categories with large gaps in their incidence rates — particularly rare singletons that cannot pair with anything within the IR ratio limit.

**Outputs:** Greedy bundle set. Greedy total: ~20,000–21,000 respondents (~54% reduction vs. naive).

### Stage 5 — Simulation and Validation (~1.5 hours)

Monte Carlo simulation validates the pre-computed allocation under probabilistic conditions. For each simulated month:

1. Each bundle draws its `n_respondents` from the respondent pool
2. Each respondent independently qualifies for each category in their bundle (Bernoulli draw with `p = incidence_rate`)
3. Qualified respondents complete the category survey (consuming `category_length_seconds`)
4. Aggregates: qualified count per category, mean and median interview time per respondent, percentage exceeding 480 seconds

The simulation runs for 1–24 months (configurable in the dashboard) with a fresh unseeded RNG on every run so results are stochastic and no two runs are identical.

**Key validation checks:**
- Every category meets ≥ 200 qualified respondents per month in ≥ 90% of simulated months
- Zero respondents exceed the 480-second budget (worst-case time constraint is hard)
- Mean ≈ median interview time per respondent — a large gap signals unbalanced bundling where some respondents qualify for multiple surveys while others qualify for none
- Demographic representativeness is maintained by construction: respondents are stratified to NZ census proportions before allocation

### Stage 6 — Dashboard, Documentation, and Delivery (~1 hour)

Interactive Streamlit dashboard (`app.py`) with six tabs for comparative analysis. Outputs are persisted to `model_outputs/` on every `python main.py` run.

---

## Architecture

```
ds-survey-george-luther/
├── main.py                     # Core algorithm — all allocation logic and simulation
├── app.py                      # Streamlit dashboard
├── data/
│   ├── fake_category_data.csv              # Primary dataset (77 categories)
│   └── fake_category_data_and_toy_problem.xlsx   # Manual sandbox — start here
├── model_outputs/              # Generated on python main.py (gitignored or committed)
│   ├── bundles_naive.json
│   ├── bundles_greedy.json
│   ├── bundles_lp_optimal.json
│   ├── simulation_summary.csv  # Per-category mean qualified, % months met, min/max
│   └── run_summary.json        # Top-level metrics: respondents, time, bundle breakdown
├── project_instructions/
│   ├── Instructions.md
│   └── README.md
└── tracksuit_project_plan.html # Full project plan with staged approach
```

### `main.py` — Key Functions

| Function | Purpose |
|---|---|
| `setup_environment()` | Load CSV, seed RNG, compute expected time column |
| `generate_respondents(n, rng)` | Synthetic respondent pool stratified to NZ census (Stats NZ gender, Infometrics age) |
| `target_setup(df)` | Build `lookup` dict `{category_id: {incidence_rate, seconds_surveyed}}` |
| `naive_allocation(lookup)` | Baseline: one bundle per category, no sharing |
| `greedy_approach(pool, lookup)` | Greedy heuristic: seed by rarest, fill with IR-ratio and time guards |
| `lp_optimal(lookup)` | LP via PuLP/CBC: enumerate candidates, solve coverage LP, round up |
| `simulate_allocation(bundles, lookup, n_months, rng)` | Monte Carlo simulation, returns qualified counts, mean/median times, % over budget |
| `save_outputs(models, targets, lookup)` | Persist bundles and simulation summaries to `model_outputs/` |

### `app.py` — Dashboard Tabs

| Tab | Content |
|---|---|
| Respondent Cost | Total respondents by strategy; respondents saved vs. naive baseline |
| Yield Distribution | Per-category qualified count histogram (mean vs. median check); mean/median alignment table |
| Time Budget | Mean interview time distribution across simulated months; mean vs. median time per respondent (alignment diagnostic) |
| Bundle Breakdown | Respondent breakdown by bundle size (singles/pairs/triples); per-bundle horizontal bar chart with hover details |
| Pool and Demographics | NZ census gender and age distributions; incidence rate and expected time scatter |
| Category Fill | Heatmap and bar chart showing per-category fill rates (% months meeting target, or mean qualified as % of target) |

---

## Setup

```bash
# 1. Create and activate virtual environment
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the core algorithm and save model outputs
python main.py

# 4. Launch the dashboard
streamlit run app.py
```

`requirements.txt`:
```
numpy
pandas
matplotlib
pulp
streamlit
plotly
```

---

## Sidebar Controls (Dashboard)

| Control | Effect |
|---|---|
| Models to run | Select any combination of Naive, Greedy, LP-Optimal |
| Simulation months | 1–24 months; 12 months = one annual cycle, 24 months = two-year stress test |
| Probabilistic guarantee (95%) | ON: inflates respondent counts via the closed-form confidence interval formula. OFF: deterministic `ceil(200/p)` baseline |
| Safety buffer (overage) | +0/5/10/15/20% above the 200 target — additional headroom for stochastic shortfalls. Scoring still uses the original 200 threshold |
| Run / Refresh | Each press produces a fresh stochastic simulation (new unseeded RNG) |

---

## Model Outputs (`model_outputs/`)

Generated by `python main.py`. Three models × three file types:

**`bundles_{model}.json`** — one record per bundle:
```json
{
  "bundle_id": 12,
  "size": 3,
  "n_respondents": 730,
  "worst_case_time_s": 421.3,
  "expected_time_s": 96.8,
  "gender_restricted": null,
  "categories": [
    {"category_id": 27, "category_name": "Dark Spirits", "incidence_rate": 0.302},
    {"category_id": 28, "category_name": "Men's Online Health Providers", "incidence_rate": 0.305},
    {"category_id": 30, "category_name": "Car Modifications", "incidence_rate": 0.305}
  ]
}
```

**`simulation_summary.csv`** — one row per category, columns for each model's mean qualified, % months meeting target, min, and max.

**`run_summary.json`** — top-level metrics per model: total respondents, bundle size breakdown, % categories meeting target, average mean and median interview time.

---

## Key Design Decisions

**Worst-case time constraint, not expected time.** Bundle feasibility is checked against `Σ l_i` (sum of raw survey lengths), not `Σ p_i · l_i` (expected time). This guarantees that even the rare respondent who qualifies for every survey in the bundle never exceeds 480 seconds. Using expected time would allow structurally infeasible bundles that occasionally violate the hard limit.

**One bundle per gender-specific category.** Categories labelled female-only (4, 35, 45) or male-only (31) are each given independent single-category bundles. Pooling all female categories into one bundle sized for the rarest causes the more common female categories to over-deliver by 2–3×, inflating their qualified counts without reducing cost.

**Incidence ratio cap of 1.3 within bundles.** When a rare category (e.g. IR = 0.096) is bundled with a more common one (e.g. IR = 0.131), the bundle is sized for the rare category — which over-delivers for the common one. A cap of 1.3× prevents the worst pairings: no category in a bundle receives more than ~30% more qualified respondents than the seed category's probabilistic target.

**One-sided z-score for the 95% guarantee.** The probabilistic allocation uses `z = 1.6449 = norm.ppf(0.95)`, not `1.96`. This is correct because the constraint is one-sided — we only care about the lower tail (failing to reach 200), not the upper tail. Using 1.96 would over-allocate by approximately 10% for no benefit.

---

## Submission

Per the project instructions:

1. Add `tracksuit-technical-test` as a collaborator on your GitHub repository
2. Share the repository link with the Tracksuit talent manager

All results are fully reproducible: `python main.py` runs all three models with a fixed seed and saves outputs to `model_outputs/`. `streamlit run app.py` launches the interactive dashboard.

---

## Data Sources

- **Gender split:** Stats NZ, National Population Estimates at 31 December 2025. Ratio: 49.65% Female, 50.35% Male.  
  https://www.stats.govt.nz/information-releases/national-population-estimates-at-31-december-2025/

- **Age composition:** Infometrics NZ, Age Composition Estimates at 30 June 2025. Seven brackets from <14 to 65+, two census bands split proportionally at bracket boundaries.  
  https://regions.infometrics.co.nz/new-zealand/population/age-composition
