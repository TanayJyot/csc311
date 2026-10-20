"""eda.py - data exploration for the CSC311 sport-survey project.

Everything is computed on the TRAINING split only. Respondents are assigned to
train / validation / test by id first (50/20/30, seed 42, as in the instructors'
project_baseline.py), and the validation and test rows are never inspected here.

Run:    python eda.py
Needs:  csc311_project_training_202609.csv and features.py in this folder,
        plus numpy, pandas and matplotlib.
Writes: split_ids.csv          respondent id -> split (shared by the whole team)
        eda_outputs/*.csv      every table quoted in the proposal
        figures/*.png          the exploration figures
"""
import os
from collections import Counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from features import (LABELS, LIKERT_COLS, MULTI_OPTIONS, TEXT_COLS,  # noqa: E402
                      blank_rows, is_non_answer, load_survey, norm_text,
                      respondent_split, structured_features, tokenize)

DATA = "csc311_project_training_202609.csv"
OUT_DIR, FIG_DIR = "eda_outputs", "figures"
SEED = 42
SPORT_COLOURS = {"Soccer": "#1baf7a", "Baseball": "#2a78d6", "Basketball": "#eb6834"}
QUESTION_NAMES = {"places": "Where", "seasons": "Season", "music": "Music",
                  "food": "Food", "colour": "Colour", "song": "Song", "feeling": "Feeling"}
# Short display names for the charts
OPTION_LABELS = {"At home or nearby": "At home / nearby",
                 "Community recreation facility": "Community facility",
                 "Private sports facility": "Private facility",
                 "School or university facility": "School / university",
                 "Others": "Other place", "Hip-hop or rap": "Hip-hop / rap",
                 "Electronic or dance": "Electronic / dance",
                 "Folk or traditional": "Folk / traditional",
                 "Classical or jazz": "Classical / jazz", "Other": "Other music"}
TOKEN_LABELS = {"wakawaka": "waka waka", "ballgame": "ball game", "take": "take (me out...)",
                "hotdog": "hot dog", "wing": "wings", "peanut": "peanuts", "nacho": "nachos",
                "chip": "chips", "ph_competition": "[a competition]", "ph_team": "[a team]",
                "ph_this_sport": "[this sport]"}
TOKENS_PER_QUESTION = 5
pd.set_option("display.width", 200, "display.max_columns", 30, "display.max_rows", 200)


# ----------------------------------------------------------------------------
# Small statistics helpers
# ----------------------------------------------------------------------------
def section(title):
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def separation(x, y):
    """(largest class mean - smallest class mean) / pooled within-class SD."""
    d = pd.DataFrame({"x": np.asarray(x, float), "y": np.asarray(y)}).dropna()
    means = d.groupby("y")["x"].transform("mean")
    pooled_var = ((d["x"] - means) ** 2).sum() / (len(d) - d["y"].nunique())
    m = d.groupby("y")["x"].mean()
    return (m.max() - m.min()) / np.sqrt(pooled_var)


def mutual_info_bits(x, y):
    """Mutual information (bits) between a discrete feature and the label."""
    joint = pd.crosstab(pd.Series(x).astype(str).to_numpy(), pd.Series(y).to_numpy(),
                        normalize=True).to_numpy()
    px, py = joint.sum(1, keepdims=True), joint.sum(0, keepdims=True)
    nz = joint > 0
    return float((joint[nz] * np.log2(joint[nz] / (px @ py)[nz])).sum())


def chi2_2x3(missing, labels):
    """Chi-square test of 'missing' vs label for a 2x3 table (df = 2, p = exp(-x/2))."""
    obs = pd.crosstab(missing, labels).reindex(columns=LABELS, fill_value=0).to_numpy(float)
    if obs.shape[0] < 2:
        return 0.0, 1.0
    exp = obs.sum(1, keepdims=True) * obs.sum(0, keepdims=True) / obs.sum()
    stat = float(((obs - exp) ** 2 / exp).sum())
    return stat, float(np.exp(-stat / 2))


def icc_after_sport(values, ids, labels):
    """Intraclass correlation of a numeric answer across a respondent's three rows,
    after subtracting each sport's mean (so it measures personal answering style)."""
    d = pd.DataFrame({"v": values, "id": ids, "y": labels}).dropna()
    d["r"] = d["v"] - d.groupby("y")["v"].transform("mean")
    d = d[d.groupby("id")["r"].transform("size") == 3]          # complete respondents
    g, k = d["id"].nunique(), 3
    grand = d["r"].mean()
    ss_between = k * ((d.groupby("id")["r"].mean() - grand) ** 2).sum()
    ss_within = ((d["r"] - d.groupby("id")["r"].transform("mean")) ** 2).sum()
    msb, msw = ss_between / (g - 1), ss_within / (len(d) - g)
    return (msb - msw) / (msb + (k - 1) * msw)


def skewness(x):
    x = pd.Series(x).dropna()
    return float(((x - x.mean()) ** 3).mean() / x.std(ddof=0) ** 3)


def class_rates(tokens, labels):
    """Share of each class's rows whose answer contains each token."""
    counts = {c: Counter() for c in LABELS}
    for toks, lab in zip(tokens, labels):
        counts[lab].update(set(toks))
    n = pd.Series(labels).value_counts()
    vocab = sorted(set().union(*counts.values()))
    rates = pd.DataFrame({c: [counts[c][t] / n[c] for t in vocab] for c in LABELS}, index=vocab)
    rates["rows"] = [sum(counts[c][t] for c in LABELS) for t in vocab]
    return rates


# ----------------------------------------------------------------------------
# Figures (matplotlib versions of the proposal's exhibits)
# ----------------------------------------------------------------------------
def _dot_panel(ax, long_df, title):
    """One row per item (grouped by question), one dot per sport at its rate."""
    y, ticks, labels = 0.0, [], []
    for q, block in long_df.groupby("question", sort=False):
        ax.text(-0.02, y, q, transform=ax.get_yaxis_transform(), ha="right", va="center",
                fontsize=8, fontweight="bold")
        y += 1
        for item, rows in block.groupby("item", sort=False):
            vals = rows.set_index("sport")["pct"].reindex(LABELS)
            ax.plot([vals.min(), vals.max()], [y, y], color="#d6d5d0", lw=2, zorder=1)
            for s in LABELS:
                ax.scatter(vals[s], y, s=30, color=SPORT_COLOURS[s], edgecolor="white",
                           linewidth=1, zorder=2)
            ticks.append(y)
            labels.append(item)
            y += 1
        y += 0.4
    ax.set_yticks(ticks, labels, fontsize=7.5)
    ax.set_ylim(y - 0.4, -0.8)
    ax.set_xlim(0, 100)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel("% of the sport's training rows", fontsize=8)
    ax.grid(axis="x", color="#ecebe7", lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", labelsize=7.5)
    ax.set_title(title, loc="left", fontsize=9)


def answers_figure(multi_long, token_long, path):
    """Left: multi-select options; right: most informative words, by sport."""
    n_left = multi_long["item"].nunique() + multi_long["question"].nunique()
    n_right = token_long["item"].nunique() + token_long["question"].nunique()
    fig, (left, right) = plt.subplots(
        1, 2, figsize=(8.5, 0.2 * max(n_left, n_right) + 1.3),
        gridspec_kw={"wspace": 0.75, "height_ratios": [1]})
    _dot_panel(left, multi_long, "Multi-select options")
    _dot_panel(right, token_long, "Most informative words")
    handles = [plt.Line2D([], [], marker="o", ls="", color=SPORT_COLOURS[s], label=s)
               for s in LABELS]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False, fontsize=8)
    fig.subplots_adjust(left=0.2, right=0.98, top=0.9, bottom=0.08)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def numeric_figure(tr, X, path):
    """Ease and price by sport (box plots) and the three Likert items (stacked shares)."""
    fig, axes = plt.subplots(1, 5, figsize=(11, 2.8))
    for ax, col, label in ((axes[0], "ease", "Ease to start (0-10)"),
                           (axes[4], "log_price", "log(1 + price in $)")):
        data = [X.loc[tr["label"] == s, col].dropna() for s in LABELS]
        bp = ax.boxplot(data, patch_artist=True, widths=0.55, medianprops={"color": "black"})
        for patch, s in zip(bp["boxes"], LABELS):
            patch.set_facecolor(SPORT_COLOURS[s])
            patch.set_alpha(0.55)
        ax.set_xticks([1, 2, 3], LABELS, fontsize=7.5)
        ax.set_title(label, fontsize=9)
    shades = ["#c94a3c", "#eba091", "#e0dfda", "#9ec5f4", "#2a78d6"]
    names = ["Strongly disagree", "Disagree", "Neutral", "Agree", "Strongly agree"]
    for ax, col in zip(axes[1:4], LIKERT_COLS):
        left = np.zeros(3)
        for level, (shade, name) in enumerate(zip(shades, names), start=1):
            share = np.array([(X.loc[tr["label"] == s, col] == level).mean() * 100 for s in LABELS])
            ax.barh(LABELS, share, left=left, color=shade, edgecolor="white", label=name)
            left += share
        ax.set_xlim(0, 100)
        ax.set_title(col.capitalize() + " (% of rows)", fontsize=9)
        ax.tick_params(labelsize=7.5)
        ax.invert_yaxis()
    handles, names_ = axes[1].get_legend_handles_labels()
    fig.legend(handles, names_, loc="lower center", ncol=5, frameon=False, fontsize=7.5)
    fig.subplots_adjust(left=0.05, right=0.99, top=0.88, bottom=0.25, wspace=0.55)
    fig.savefig(path, dpi=200)
    plt.close(fig)


# ----------------------------------------------------------------------------
# Main exploration
# ----------------------------------------------------------------------------
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)

    # ---- 1. Load, drop all-blank rows, split by respondent -----------------
    section("1. Rows, respondents and the split")
    raw = load_survey(DATA)
    print(f"raw rows: {len(raw)}, respondents: {raw['id'].nunique()}, "
          f"rows per respondent: {raw.groupby('id').size().unique().tolist()}")
    order = raw.groupby("id", sort=False)["label"].agg(tuple).value_counts()
    print("label order within respondents:", dict(order))
    blank = blank_rows(raw)
    print(f"all-blank rows dropped: {int(blank.sum())} "
          f"(respondents {sorted(raw.loc[blank, 'id'].astype(int).unique().tolist())})")
    df = raw.loc[~blank].reset_index(drop=True)
    df["split"] = respondent_split(df["id"])
    df.drop_duplicates("id")[["id", "split"]].to_csv("split_ids.csv", index=False)
    summary = df.groupby("split").agg(respondents=("id", "nunique"), rows=("id", "size"))
    print(summary.loc[["train", "val", "test"]])

    tr = df[df["split"] == "train"].reset_index(drop=True)      # <- the only data explored
    y = tr["label"]
    X = structured_features(tr)

    # ---- 2. Class balance ----------------------------------------------------
    section("2. Class balance (training split)")
    print((y.value_counts().reindex(LABELS).to_frame("rows")
           .assign(pct=lambda d: (100 * d.rows / d.rows.sum()).round(1))))

    # ---- 3. Missing values ---------------------------------------------------
    section("3. Missing values after normalising blanks / 'n/a' / 'none' / 'idk' ...")
    miss = pd.DataFrame(index=tr.index)
    miss["ease"] = X["ease"].isna()
    for col in LIKERT_COLS:
        miss[col] = X[col].isna()
    for col in MULTI_OPTIONS:
        miss[col] = X.filter(like=f"{col}=").sum(axis=1) == 0
    for col in TEXT_COLS:
        miss[col] = tr[col].map(is_non_answer)
    miss["price"] = X["price"].isna()
    rows = []
    for col in miss.columns:
        stat, p = chi2_2x3(miss[col], y)
        r = {"feature": col, "missing_pct": 100 * miss[col].mean()}
        for s in LABELS:
            r[f"{s}_pct"] = 100 * miss.loc[y == s, col].mean()
        r["chi2"], r["p_value"] = stat, p
        rows.append(r)
    missing = pd.DataFrame(rows).round(3)
    missing.to_csv(f"{OUT_DIR}/missing_values.csv", index=False)
    print(missing.to_string(index=False))
    price_blank = tr["price"].str.strip() == ""
    print(f"price: blank {int(price_blank.sum())}, present but unparseable "
          f"{int((X['price'].isna() & ~price_blank).sum())}")
    print("Excel artefacts ('#NAME?'):",
          int(tr[TEXT_COLS].apply(lambda c: c.str.strip() == "#NAME?").to_numpy().sum()))

    # ---- 4. Numeric and Likert features -------------------------------------
    section("4. Numeric and Likert features by sport")
    num_rows = []
    for col in ["ease"] + LIKERT_COLS + ["log_price"]:
        r = {"feature": col}
        for s in LABELS:
            v = X.loc[y == s, col]
            r[f"{s}_mean"], r[f"{s}_sd"] = v.mean(), v.std()
        r["separation"] = separation(X[col], y)
        num_rows.append(r)
    numeric = pd.DataFrame(num_rows).round(2)
    numeric.to_csv(f"{OUT_DIR}/numeric_by_sport.csv", index=False)
    print(numeric.to_string(index=False))
    e = X["ease"].dropna()
    print(f"ease: range {e.min()}-{e.max()}, non-integer values {int((e % 1 != 0).sum())}")
    for col in LIKERT_COLS:
        dist = pd.crosstab(y, X[col], normalize="index").reindex(LABELS).mul(100).round(0)
        print(f"\n{col} (% of rows, 1 = strongly disagree ... 5 = strongly agree)\n{dist}")

    # ---- 5. Willingness to pay ----------------------------------------------
    section("5. Willingness to pay (parsed from free text)")
    p = X["price"].dropna()
    print(f"parsed {len(p)} / {len(tr)}; zero: {int((p == 0).sum())} ({100 * (p == 0).mean():.1f}%)")
    print(f"median {p.median():.0f}, IQR {p.quantile(.25):.0f}-{p.quantile(.75):.0f}, "
          f"mean {p.mean():.0f}, p95 {p.quantile(.95):.0f}, p99 {p.quantile(.99):.0f}, max {p.max():.0f}")
    print(f"skewness raw {skewness(p):.1f} vs log1p {skewness(np.log1p(p)):.2f}")
    q1, q3 = np.log1p(p).quantile([.25, .75])
    hi = np.expm1(q3 + 1.5 * (q3 - q1))
    print(f"outliers above Q3 + 1.5 IQR on the log scale (> ${hi:,.0f}): {int((p > hi).sum())}; "
          f"values >= $2,000: {sorted(p[p >= 2000].astype(int).tolist())}")
    by_sport = X.groupby(y)["price"].quantile([.25, .5, .75]).unstack().reindex(LABELS)
    by_sport.columns = ["q1", "median", "q3"]
    by_sport.to_csv(f"{OUT_DIR}/price_by_sport.csv")
    print("price quartiles by sport ($):\n", by_sport)
    pd.DataFrame({"stat": ["parsed", "zero", "median", "q1", "q3", "p95", "p99", "max",
                           "skew_raw", "skew_log1p", "log_iqr_upper_fence"],
                  "value": [len(p), int((p == 0).sum()), p.median(), p.quantile(.25),
                            p.quantile(.75), p.quantile(.95), p.quantile(.99), p.max(),
                            skewness(p), skewness(np.log1p(p)), hi]}
                 ).round(2).to_csv(f"{OUT_DIR}/price_summary.csv", index=False)

    # ---- 6. Respondent effects (why we split by id) --------------------------
    section("6. How strongly a respondent's three rows resemble each other")
    icc = {col: icc_after_sport(X[col], tr["id"], y) for col in
           ["ease"] + LIKERT_COLS + ["log_price"]}
    print({k: round(v, 2) for k, v in icc.items()})
    # Simpler view: correlation between the same person's answers for two sports
    wide = X[["ease", "log_price"]].assign(id=tr["id"], label=y)
    pair_r = {}
    for col in ["ease", "log_price"]:
        w = wide.pivot(index="id", columns="label", values=col)
        pair_r[col] = np.mean([w[a].corr(w[b]) for a, b in
                               (("Soccer", "Baseball"), ("Soccer", "Basketball"),
                                ("Baseball", "Basketball"))])
    print("mean correlation of a person's answers across sport pairs:",
          {k: round(v, 2) for k, v in pair_r.items()})
    pd.Series(icc, name="icc").round(3).to_csv(f"{OUT_DIR}/respondent_icc.csv")

    # ---- 7. Multi-select questions -------------------------------------------
    section("7. Multi-select options: % of each sport's rows selecting the option")
    long_rows = []
    for col in MULTI_OPTIONS:
        block = X.filter(like=f"{col}=")
        for c in block.columns:
            option = c.split("=", 1)[1]
            for s in LABELS:
                long_rows.append({"question": col, "option": option, "sport": s,
                                  "pct": 100 * block.loc[y == s, c].mean(),
                                  "overall_pct": 100 * block[c].mean()})
    ms = pd.DataFrame(long_rows)
    spread = ms.groupby(["question", "option"])["pct"].agg(lambda v: v.max() - v.min())
    ms["spread"] = ms.set_index(["question", "option"]).index.map(spread)
    order = {**{q: i for i, q in enumerate(MULTI_OPTIONS)}, **{s: i for i, s in enumerate(LABELS)}}
    ms = ms.sort_values(["question", "spread", "option", "sport"],
                        key=lambda s: s.map(order) if s.name in ("question", "sport") else s,
                        ascending=[True, False, True, True])
    ms = ms.assign(question=ms["question"].map(QUESTION_NAMES),
                   item=ms["option"].map(lambda o: OPTION_LABELS.get(o, o))).round(1)
    ms[["question", "item", "option", "sport", "pct", "overall_pct", "spread"]].to_csv(
        f"{OUT_DIR}/multiselect_rates.csv", index=False)
    wide = ms.pivot_table(index=["question", "option"], columns="sport", values="pct",
                          sort=False)[LABELS]
    wide["overall"] = ms.drop_duplicates(["question", "option"]).set_index(
        ["question", "option"])["overall_pct"]
    print(wide.round(0))
    print("\nnear-constant options (chosen by > 85% or < 5% of rows):")
    print(wide[(wide["overall"] > 85) | (wide["overall"] < 5)]["overall"].round(1))
    print("options selected per row (mean):",
          {q: round(X.filter(like=f"{q}=").sum(axis=1).mean(), 2) for q in MULTI_OPTIONS})

    # ---- 8. Correlated features ------------------------------------------------
    section("8. Correlations")
    ordinal = X[["ease"] + LIKERT_COLS + ["log_price"]]
    print("Spearman (pairwise complete):\n", ordinal.rank().corr().round(2))
    num = X.drop(columns=["price"])
    corr = num.corr()
    pairs = (corr.where(np.triu(np.ones(corr.shape, bool), 1)).stack()
             .rename("r").reset_index())
    pairs = pairs.reindex(pairs["r"].abs().sort_values(ascending=False).index)
    pairs.head(10).round(3).to_csv(f"{OUT_DIR}/top_correlations.csv", index=False)
    print("\nlargest |r| among all structured features:\n", pairs.head(10).round(2).to_string(index=False))

    # ---- 9. Free text -----------------------------------------------------------
    section("9. Free-text questions")
    tokens = {col: tr[col].map(tokenize) for col in TEXT_COLS}
    text_rows, token_long = [], []
    for col in TEXT_COLS:
        rates = class_rates(tokens[col], y)
        answered = ~tr[col].map(is_non_answer)
        text_rows.append({
            "question": col,
            "distinct_answers": tr.loc[answered, col].map(norm_text).nunique(),
            "vocab_min_df_1": len(rates), "vocab_min_df_3": int((rates["rows"] >= 3).sum()),
            "mean_tokens": tokens[col][answered].map(len).mean()})
        # Most informative words: highest mutual information with the label among
        # words used in >= 10 rows, skipping a word that mostly co-occurs with one
        # already chosen (e.g. 'shakira' with 'waka waka').
        present = {t: tokens[col].map(lambda ts, t=t: t in ts) for t in rates.index[rates["rows"] >= 10]}
        ranked = sorted(present, key=lambda t: mutual_info_bits(present[t], y), reverse=True)
        chosen = []
        for t in ranked:
            overlap = [(present[t] & present[c]).sum() / min(present[t].sum(), present[c].sum())
                       for c in chosen]
            if all(o < 0.6 for o in overlap):
                chosen.append(t)
            if len(chosen) == TOKENS_PER_QUESTION:
                break
        for rank, t in enumerate(chosen):
            for s in LABELS:
                token_long.append({"question": QUESTION_NAMES[col], "item": TOKEN_LABELS.get(t, t),
                                   "token": t, "rank": rank, "sport": s,
                                   "pct": 100 * rates.loc[t, s],
                                   "mi_bits": mutual_info_bits(present[t], y)})
    text_summary = pd.DataFrame(text_rows).round(2)
    text_summary.to_csv(f"{OUT_DIR}/text_summary.csv", index=False)
    print(text_summary.to_string(index=False))
    token_long = pd.DataFrame(token_long).round(3)
    token_long.to_csv(f"{OUT_DIR}/text_token_rates.csv", index=False)
    print("\nmost informative words (% of each sport's rows containing the word):")
    print(token_long.pivot_table(index=["question", "item"], columns="sport", values="pct",
                                 sort=False)[LABELS].round(1))

    # Words never seen before: grouped 5-fold check inside the training split
    ids = tr["id"].unique()
    rng = np.random.default_rng(SEED)
    fold_of = dict(zip(rng.permutation(ids), np.arange(len(ids)) % 5))
    fold = tr["id"].map(fold_of)
    unseen, total, rows_with_unseen = 0, 0, 0
    for k in range(5):
        for col in TEXT_COLS:
            vocab = set(t for toks in tokens[col][fold != k] for t in toks)
            for toks in tokens[col][fold == k]:
                miss_tok = [t for t in toks if t not in vocab]
                unseen += len(miss_tok)
                total += len(toks)
    for k in range(5):
        vocabs = {col: set(t for toks in tokens[col][fold != k] for t in toks) for col in TEXT_COLS}
        held = tr.index[fold == k]
        rows_with_unseen += sum(any(t not in vocabs[col] for col in TEXT_COLS
                                    for t in tokens[col][i]) for i in held)
    print(f"\nunseen-word rate in a held-out fold: {100 * unseen / total:.1f}% of tokens; "
          f"{100 * rows_with_unseen / len(tr):.1f}% of rows contain at least one unseen word")

    n_struct = X.drop(columns=["price"]).shape[1]
    n_text1 = int(text_summary["vocab_min_df_1"].sum())
    n_text3 = int(text_summary["vocab_min_df_3"].sum())
    print(f"feature count: {n_struct} structured + {n_text1} words (min_df=1) "
          f"= {n_struct + n_text1}; with min_df=3: {n_struct + n_text3}; rows = {len(tr)}")

    # ---- 10. Which features are predictive (mutual information) ----------------
    section("10. Mutual information with the label (bits; max = log2(3) = 1.585)")
    feats = {}
    feats["ease (binned)"] = pd.cut(X["ease"], [-0.1, 3, 5, 7, 10]).astype(str)
    for col in LIKERT_COLS:
        feats[col] = X[col].fillna(-1)
    feats["price (quartiles)"] = pd.qcut(X["price"], 4, duplicates="drop").astype(str)
    for c in X.columns:
        if "=" in c:
            feats[c.replace("places=", "where: ").replace("seasons=", "season: ")
                  .replace("music=", "music: ")] = X[c]
    for col in TEXT_COLS:
        rates = class_rates(tokens[col], y)
        for tok in rates.index[rates["rows"] >= 10]:
            feats[f"{col}: '{tok}'"] = tokens[col].map(lambda t, tok=tok: tok in t)
        feats[f"{col}: missing"] = tr[col].map(is_non_answer)
    mi = pd.Series({k: mutual_info_bits(v, y) for k, v in feats.items()}).sort_values(ascending=False)
    mi.round(3).to_csv(f"{OUT_DIR}/mutual_information.csv", header=["mi_bits"])
    print(mi.head(25).round(3).to_string())
    print("\nseparation (largest gap between sport means / pooled within-sport SD) of binary features:")
    binsep = pd.Series({c: separation(X[c], y) for c in X.columns if "=" in c}).sort_values(ascending=False)
    print(binsep.head(8).round(2).to_string())

    # ---- 11. Figures --------------------------------------------------------------
    answers_figure(ms, token_long, f"{FIG_DIR}/fig1_answers_by_sport.png")
    numeric_figure(tr, X, f"{FIG_DIR}/fig2_numeric_likert.png")
    print(f"\nwrote {OUT_DIR}/*.csv, {FIG_DIR}/*.png and split_ids.csv")


if __name__ == "__main__":
    main()
