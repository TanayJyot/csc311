"""missing_values.py - step-by-step trace of every parseability / missing-value
number quoted in the report.

Each step prints where a number comes from, so a reader can check it by hand.
All cleaning and parsing rules are imported from features.py (the same code
eda.py and pred.py use), so the numbers here match eda.py exactly.

Run:    python missing_values.py
Needs:  csc311_project_training_202609.csv, split_ids.csv, features.py,
        numpy and pandas (no matplotlib / sklearn).
Writes: eda_outputs/missing_by_feature.csv        training-split table
        eda_outputs/missing_by_feature_full.csv   same table, all respondents
"""
import os
from collections import Counter

import numpy as np
import pandas as pd

from features import (FEATURE_COLS, LABELS, LIKERT_COLS, MULTI_OPTIONS, TEXT_COLS,
                      _RANGE, _SINGLE, _ZERO, _word_number, blank_rows,
                      is_non_answer, load_survey, norm_text, parse_price,
                      structured_features)

DATA = "csc311_project_training_202609.csv"
SPLITS = "split_ids.csv"
OUT_DIR = "eda_outputs"                      # same output folder as eda.py
pd.set_option("display.width", 200, "display.max_columns", 30, "display.max_rows", 200)


def section(title):
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def chi2_2x3(missing, labels):
    """Chi-square test of 'missing' vs label for a 2x3 table (df = 2, p = exp(-x/2)).

    Copied from eda.py (eda.py imports matplotlib at module level, so it cannot be
    imported here). For df = 2 the chi-square survival function is exactly exp(-x/2).
    """
    obs = pd.crosstab(missing, labels).reindex(columns=LABELS, fill_value=0).to_numpy(float)
    if obs.shape[0] < 2:                     # nothing missing (or everything) -> no test
        return 0.0, 1.0
    exp = obs.sum(1, keepdims=True) * obs.sum(0, keepdims=True) / obs.sum()
    stat = float(((obs - exp) ** 2 / exp).sum())
    return stat, float(np.exp(-stat / 2))


# ----------------------------------------------------------------------------
# Missingness rules (identical to eda.py section 3)
# ----------------------------------------------------------------------------
def missing_mask(df):
    """One boolean column per feature: True where the answer counts as missing.

    ease / Likert : the parsed value is NaN (blank or not a valid option)
    multi-selects : no option ticked
    free text     : features.is_non_answer (blank, 'n/a', 'none', 'idk', '#NAME?', ...)
    price         : features.parse_price returns NaN (blank or unparseable)
    """
    X = structured_features(df)
    miss = pd.DataFrame(index=df.index)
    miss["ease"] = X["ease"].isna()
    for col in LIKERT_COLS:
        miss[col] = X[col].isna()
    for col in MULTI_OPTIONS:
        miss[col] = X.filter(like=f"{col}=").sum(axis=1) == 0
    for col in TEXT_COLS:
        miss[col] = df[col].map(is_non_answer)
    miss["price"] = X["price"].isna()
    return miss[FEATURE_COLS]


def missing_table(df, miss):
    y = df["label"]
    rows = []
    for col in FEATURE_COLS:
        stat, p = chi2_2x3(miss[col], y)
        r = {"feature": col, "rows": len(df), "missing": int(miss[col].sum()),
             "missing_pct": 100 * miss[col].mean()}
        for s in LABELS:
            r[f"{s}_missing"] = int(miss.loc[y == s, col].sum())
            r[f"{s}_pct"] = 100 * miss.loc[y == s, col].mean()
        r["chi2"], r["p_value"] = stat, p
        rows.append(r)
    return pd.DataFrame(rows)


def raw_forms(df, miss, col, top=None):
    """Counter of the raw strings (repr, whitespace kept) that were counted as missing."""
    c = Counter(df.loc[miss[col], col].map(lambda s: s if s.strip() else ""))
    return c.most_common(top)


# ----------------------------------------------------------------------------
# Price parse-rule classification (mirrors the order of rules in parse_price)
# ----------------------------------------------------------------------------
def price_rule(s):
    t = norm_text(s)
    if t == "":
        return "blank"
    if "scale" in t:
        return "unparseable ('scale' answer)"
    m = _RANGE.search(t)
    if m:
        return "range -> midpoint"
    m = _SINGLE.search(t)
    if m:
        if m.group(2):
            return "k suffix (x1000)"
        if "," in m.group(1) or " " in m.group(1):
            return "thousands separator"
        if any(k in t for k in ("$", "cad", "usd", "dollar", "buck")):
            return "number with currency"
        return "bare number"
    if _ZERO.match(t):
        return "refusal -> 0"
    if _word_number(t) is not None:
        return "spelled-out amount"
    return "unparseable"


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # ---- 1. Rows, respondents, split ------------------------------------------
    section("1. Rows, respondents and the split")
    raw = load_survey(DATA)
    print(f"raw rows in the CSV (pandas, multi-line quoted fields handled): {len(raw)}")
    print(f"raw respondents: {raw['id'].nunique()}  "
          f"(rows per respondent: {sorted(raw.groupby('id').size().unique().tolist())})")
    blank = blank_rows(raw)
    blank_ids = sorted(raw.loc[blank, "id"].astype(int).unique().tolist())
    print(f"entirely blank rows (all {len(FEATURE_COLS)} answers empty): {int(blank.sum())} "
          f"from {len(blank_ids)} respondents {blank_ids}")
    print("  blank rows per respondent:",
          {int(k): int(v) for k, v in raw.loc[blank].groupby("id").size().items()})
    df = raw.loc[~blank].reset_index(drop=True)
    print(f"cleaned rows: {len(raw)} - {int(blank.sum())} = {len(df)}; "
          f"respondents: {df['id'].nunique()}")

    split = pd.read_csv(SPLITS, dtype=str).set_index("id")["split"]
    df["split"] = df["id"].map(split)
    if df["split"].isna().any():
        raise SystemExit(f"ids missing from {SPLITS}: {sorted(df.loc[df.split.isna(), 'id'])}")
    print(f"ids in {SPLITS}: {len(split)}")
    summary = (df.groupby("split").agg(respondents=("id", "nunique"), rows=("id", "size"))
               .loc[["train", "val", "test"]])
    summary.loc["total"] = summary.sum()
    print(summary)

    tr = df[df["split"] == "train"].reset_index(drop=True)
    print("\ntraining rows per sport:",
          {k: int(v) for k, v in tr["label"].value_counts().reindex(LABELS).items()})

    # ---- 2. Per-feature missing table (training split) -------------------------
    section("2. Missing values per feature (TRAINING split, "
            f"{tr['id'].nunique()} respondents, {len(tr)} rows)")
    miss = missing_mask(tr)
    table = missing_table(tr, miss)
    show = table[["feature", "rows", "missing", "missing_pct"]].copy()
    show["missing_pct"] = show["missing_pct"].round(1)
    print(show.to_string(index=False))

    price_blank = tr["price"].str.strip() == ""
    price_unparse = miss["price"] & ~price_blank
    print(f"\nprice breakdown: blank {int(price_blank.sum())} "
          f"+ present-but-unparseable {int(price_unparse.sum())} "
          f"= {int(miss['price'].sum())} missing "
          f"({100 * miss['price'].mean():.1f}% of {len(tr)})")

    print("\nRaw forms counted as missing, per feature (repr of the raw cell; '' = blank):")
    for col in FEATURE_COLS:
        forms = raw_forms(tr, miss, col)
        print(f"  {col:<10} {int(miss[col].sum()):>3}: "
              + (", ".join(f"{f!r}: {n}" for f, n in forms) if forms else "-"))

    print("\nPlaceholder categories across the four free-text questions "
          "(after norm_text, trailing ' .!' stripped):")
    cats = Counter()
    for col in TEXT_COLS:
        for s in tr.loc[miss[col], col]:
            t = norm_text(s).rstrip(" .!")
            key = ("blank" if t == "" else "#NAME?" if t.startswith("#name") else
                   "n/a" if t.startswith(("n/a", "na")) else
                   "none" if t.startswith("none") else "idk" if t == "idk" else "other")
            cats[(col, key)] += 1
    cat_df = (pd.Series(cats).unstack(fill_value=0)
              .reindex(index=TEXT_COLS, fill_value=0))
    cat_df["total"] = cat_df.sum(axis=1)
    print(cat_df)
    print("('other' = longer placeholders such as 'no particular song', 'i don't know')")

    # ---- 3. Missingness by sport + chi-square ----------------------------------
    section("3. Missing % by sport and chi-square test of missingness vs label (df = 2)")
    by_sport = table[["feature"] + [f"{s}_pct" for s in LABELS] + ["chi2", "p_value"]].copy()
    for s in LABELS:
        by_sport[f"{s}_pct"] = by_sport[f"{s}_pct"].round(1)
    by_sport["chi2"] = by_sport["chi2"].round(2)
    by_sport["p_value"] = by_sport["p_value"].map(lambda p: f"{p:.2g}")
    print(by_sport.to_string(index=False))
    print("\ncounts behind the percentages (missing / rows):")
    n = tr["label"].value_counts()
    for col in FEATURE_COLS:
        print(f"  {col:<11}" + "  ".join(
            f"{s} {int(table.loc[table.feature == col, f'{s}_missing'].iloc[0])}/{n[s]}"
            for s in LABELS))
    print("p = exp(-chi2/2); p = 1 with chi2 = 0 means nothing was missing (no test).")

    table.round(4).to_csv(f"{OUT_DIR}/missing_by_feature.csv", index=False)

    # ---- 4. Price parsing audit ------------------------------------------------
    section("4. Willingness-to-pay parsing audit (training split)")
    price = tr["price"].map(parse_price)
    rules = tr["price"].map(price_rule)
    print(f"parsed {int(price.notna().sum())} / {len(tr)}, "
          f"missing {int(price.isna().sum())} (blank {int(price_blank.sum())}, "
          f"unparseable {int(price_unparse.sum())})")
    print("\nrows per parse rule:")
    print(rules.value_counts().to_string())
    consistent = (price.isna() == rules.str.startswith(("blank", "unparseable"))).all()
    print(f"rule labels agree with parse_price NaN / non-NaN: {consistent}")

    print("\nall present-but-unparseable raw strings:")
    for s in tr.loc[price_unparse, "price"]:
        print(f"  {s!r}")

    print("\nexamples of each parse rule (raw -> parsed $):")
    for rule in rules.value_counts().index:
        if rule.startswith(("blank", "unparseable")):
            continue
        ex = tr.loc[rules == rule, "price"].drop_duplicates().head(5)
        print(f"  {rule}:")
        for s in ex:
            print(f"      {s!r:<60} -> {parse_price(s):,.2f}")

    p = price.dropna()
    q1, q3 = p.quantile([.25, .75])
    lq1, lq3 = np.log1p(p).quantile([.25, .75])
    fence = np.expm1(lq3 + 1.5 * (lq3 - lq1))
    print("\nsummary of parsed prices ($):")
    print(f"  n {len(p)}, zero {int((p == 0).sum())} ({100 * (p == 0).mean():.1f}%)")
    print(f"  median {p.median():,.0f}, mean {p.mean():,.0f}, "
          f"IQR {q1:,.0f}-{q3:,.0f}, min {p.min():,.0f}, max {p.max():,.0f}")
    print(f"  p95 {p.quantile(.95):,.0f}, p99 {p.quantile(.99):,.0f} (99th-percentile clip)")
    print(f"  log1p scale: Q1 {lq1:.3f}, Q3 {lq3:.3f}, upper fence Q3 + 1.5 IQR = "
          f"{lq3 + 1.5 * (lq3 - lq1):.3f} -> ${fence:,.0f}")
    print(f"  values above the fence: {int((p > fence).sum())} "
          f"{sorted(p[p > fence].astype(int).tolist())}")
    print(f"  values above the p99 clip: {int((p > p.quantile(.99)).sum())}")

    # ---- 5. Full cleaned dataset (reference) -----------------------------------
    section(f"5. Missing values per feature, FULL cleaned data "
            f"({df['id'].nunique()} respondents, {len(df)} rows) - reference only")
    miss_all = missing_mask(df)
    full = missing_table(df, miss_all)
    out = full[["feature", "rows", "missing", "missing_pct"]
               + [f"{s}_pct" for s in LABELS]].copy()
    for c in ["missing_pct"] + [f"{s}_pct" for s in LABELS]:
        out[c] = out[c].round(1)
    out["train_pct"] = table["missing_pct"].round(1).to_numpy()
    print(out.to_string(index=False))
    pb = df["price"].str.strip() == ""
    print(f"\nprice: blank {int(pb.sum())}, present but unparseable "
          f"{int((miss_all['price'] & ~pb).sum())}")
    full.round(4).to_csv(f"{OUT_DIR}/missing_by_feature_full.csv", index=False)

    print(f"\nwrote {OUT_DIR}/missing_by_feature.csv and {OUT_DIR}/missing_by_feature_full.csv")


if __name__ == "__main__":
    main()
