"""
eda.py — expanded, self-contained EDA for the CSC311 sport-survey project.

Analyses only the training split. Respondents (IDs) are split as groups so all
three sport rows from a respondent stay together. Default split: 50/20/30, seed 42.

Input:
    csc311_project_training_202609.csv

Outputs:
    split_ids.csv
    eda_outputs/*.csv       summary tables and machine-readable associations
    eda_outputs/eda_report.md  automatically generated interpretation of distributions
    figures/*.png           plots

Requirements:
    pip install pandas numpy matplotlib
Run:
    python eda.py
    python eda.py --input path/to/data.csv --out-dir results --seed 42
"""
from pathlib import Path
import argparse
import re
import math
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
DATA = str(HERE.parent / "csc311_project_training_202609.csv")
OUT_DIR = HERE / "eda_outputs"
FIG_DIR = HERE / "figures"
SEED = 42
LABELS = ["Soccer", "Baseball", "Basketball"]
COLORS = {"Soccer": "#1baf7a", "Baseball": "#2a78d6", "Basketball": "#eb6834"}
LIKERT_COLS = [
    "Success in this sport depends heavily on strategy and planning.",
    "Players must be comfortable with a high risk of injury.",
    "Individual brilliance can outweigh teamwork in this sport.",
]
MULTI_COLS = [
    "Where might you play this sport?",
    "During which seasons would you be most likely to play this sport recreationally in Toronto?",
    "What types of music do you most associate with this sport?",
]
TEXT_COLS = [
    "What foods and drinks do you associate with watching this sport in the place or culture you know best?",
    "What colour do you associate with this sport?",
    "Which song do you associate with this sport?",
    "How do you feel when you watch this sport?",
]
EASE_COL = "How easy do you think it would be for someone with no prior experience to start playing this sport recreationally? Rate it on a scale from 0 to 10, where 0 means extremely difficult and 10 means extremely easy."
PRICE_COL = "How much would you be willing to pay for a decent seat at a professional competition of this sport?"
LIKERT_MAP = {
    "strongly disagree": 1, "disagree": 2, "neutral": 3,
    "agree": 4, "strongly agree": 5
}
MISSING_STRINGS = {"", "nan", "n/a", "na", "none", "none provided", "idk", "i don't know", "don't know", "#name?"}

def clean_cell(x):
    if pd.isna(x):
        return np.nan
    s = re.sub(r"\s+", " ", str(x)).strip()
    if s.casefold() in MISSING_STRINGS:
        return np.nan
    return s

def norm_answer(x):
    s = clean_cell(x)
    if pd.isna(s):
        return np.nan
    return re.sub(r"\s+", " ", s).casefold()

def tokenize(x):
    s = norm_answer(x)
    if pd.isna(s):
        return []
    # Preserve common multiword entities as token bigrams later; strip punctuation.
    return re.findall(r"[a-z]+(?:'[a-z]+)?|\d+", s)

def parse_price(x):
    s = clean_cell(x)
    if pd.isna(s):
        return np.nan
    z = s.casefold().replace(",", "")
    if z in {"free", "zero", "$0", "0 dollars"}:
        return 0.0
    m = re.search(r"-?\d+(?:\.\d+)?", z)
    return float(m.group()) if m else np.nan

def respondent_split(ids, seed=42, val_frac=0.2, test_frac=0.3):
    """Same split as split_by_student in the instructors' project_baseline.py:
    permute the unique IDs, then take the first 30% as test, the next 20% as
    validation and the rest as training (50/20/30)."""
    students = np.random.default_rng(seed).permutation(pd.Series(ids).unique())
    n_test = int(len(students) * test_frac)
    n_val = int(len(students) * val_frac)
    split = {x: "test" for x in students[:n_test]}
    split.update({x: "val" for x in students[n_test:n_test + n_val]})
    split.update({x: "dev" for x in students[n_test + n_val:]})
    return pd.Series(ids).map(split).to_numpy()

def mutual_info_bits(x, y):
    tab = pd.crosstab(pd.Series(x).fillna("[MISSING]").astype(str),
                      pd.Series(y).astype(str), normalize=True).to_numpy()
    if tab.size == 0:
        return 0.0
    px, py = tab.sum(axis=1, keepdims=True), tab.sum(axis=0, keepdims=True)
    expected = px @ py
    mask = tab > 0
    return float(np.sum(tab[mask] * np.log2(tab[mask] / expected[mask])))

def cramers_v(tab):
    obs = np.asarray(tab, dtype=float)
    if obs.size == 0 or obs.sum() == 0:
        return np.nan
    total = obs.sum()
    expected = np.outer(obs.sum(1), obs.sum(0)) / total
    valid = expected > 0
    chi2 = (((obs - expected) ** 2)[valid] / expected[valid]).sum()
    denom = total * min(obs.shape[0]-1, obs.shape[1]-1)
    return float(np.sqrt(chi2 / denom)) if denom > 0 else np.nan

def savefig(name):
    plt.tight_layout()
    plt.savefig(FIG_DIR / name, dpi=180, bbox_inches="tight")
    plt.close()

def short(s, n=48):
    return s if len(s) <= n else s[:n-3] + "..."

def main(input_path=DATA, out_dir=OUT_DIR, fig_dir=FIG_DIR, seed=SEED):
    global OUT_DIR, FIG_DIR
    OUT_DIR, FIG_DIR = Path(out_dir), Path(fig_dir)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(input_path)
    raw.columns = [str(c).strip() for c in raw.columns]
    required = {"id", "label", EASE_COL, PRICE_COL, *LIKERT_COLS, *MULTI_COLS, *TEXT_COLS}
    missing_cols = sorted(required - set(raw.columns))
    if missing_cols:
        raise ValueError("Missing expected survey columns:\n" + "\n".join(missing_cols))

    df = raw.copy()
    for col in df.columns:
        if col != "id":
            df[col] = df[col].map(clean_cell)
    df["id"] = pd.to_numeric(df["id"], errors="coerce")
    df["label"] = df["label"].astype("string").str.strip().str.title()
    df = df[df["label"].isin(LABELS)].copy()
    # Remove rows with no answers in any feature column, but retain valid partial responses.
    feature_cols = [c for c in df.columns if c not in {"id", "label"}]
    blank = df[feature_cols].isna().all(axis=1)
    blank_count = int(blank.sum())
    df = df.loc[~blank].drop_duplicates().reset_index(drop=True)
    # Drop respondents who no longer have one row per sport, then split 70/30 by respondent.
    rows_per_id = df.groupby("id")["id"].transform("size")
    partial_count = int((rows_per_id < len(LABELS)).sum())
    df = df.loc[rows_per_id == len(LABELS)].reset_index(drop=True)
    df["split"] = respondent_split(df["id"], seed, val_frac=0.0)

    # Clean and engineer analysis-only columns. Original answer columns remain available.
    df["ease_numeric"] = pd.to_numeric(df[EASE_COL], errors="coerce")
    df.loc[~df["ease_numeric"].between(0, 10), "ease_numeric"] = np.nan
    for col in LIKERT_COLS:
        df[col + " [ordinal]"] = df[col].astype("string").str.casefold().map(LIKERT_MAP).astype(float)
    df["price_numeric"] = df[PRICE_COL].map(parse_price)
    df.loc[df["price_numeric"] < 0, "price_numeric"] = np.nan
    df["log_price"] = np.log1p(df["price_numeric"])

    tr = df[df["split"] == "dev"].copy().reset_index(drop=True)
    y = tr["label"]
    # Use clean, split files with the helper columns explicitly named.
    df.drop(columns=["split"]).to_csv(OUT_DIR / "cleaned_survey.csv", index=False)
    df[["id", "split"]].drop_duplicates().sort_values("id").to_csv(HERE / "split_ids.csv", index=False)
    for name in ["dev", "test"]:
        df[df["split"] == name].drop(columns=["split"]).to_csv(OUT_DIR / f"{name}.csv", index=False)

    report = []
    def say(s=""):
        print(s)
        report.append(str(s))
    say("# Sport-survey exploratory data analysis")
    say("")
    say("All label comparisons and plots below use the **training split only**. Respondent IDs are split as groups to avoid putting the same person's answers in more than one split.")
    say("")
    say("## 1. Dataset and split integrity")
    say(f"- Raw rows: {len(raw)}; retained rows: {len(df)}; all-answer-blank rows removed: {blank_count}.")
    say(f"- Unique respondents: {df['id'].nunique()}; training respondents: {tr['id'].nunique()}; training rows: {len(tr)}.")
    label_orders = df.groupby("id")["label"].agg(lambda x: tuple(x.tolist()))
    say(f"- Most common label sequences per respondent: {label_orders.value_counts().head(5).to_dict()}.")
    say("- The ID is an identifier, not a predictor; exclude it from model inputs.")
    say("")
    split_summary = df.groupby("split").agg(respondents=("id", "nunique"), rows=("id", "size"))
    split_summary.to_csv(OUT_DIR / "split_summary.csv")
    say("Split summary:")
    say(split_summary.to_string())
    counts = y.value_counts().reindex(LABELS, fill_value=0)
    proportions = (counts / counts.sum() * 100).round(2)
    pd.DataFrame({"rows": counts, "percent": proportions}).to_csv(OUT_DIR / "label_balance.csv")
    say("")
    say("## 2. Label balance")
    say(pd.DataFrame({"rows": counts, "percent": proportions}).to_string())
    say("Interpretation: the class balance is " + ("approximately even, so accuracy is a reasonable baseline metric." if proportions.max()-proportions.min() < 5 else "not perfectly even; compare accuracy with macro-F1 and per-class recall."))

    # Label counts and split counts.
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(LABELS, [counts[s] for s in LABELS], color=[COLORS[s] for s in LABELS])
    ax.set_ylabel("Training rows"); ax.set_title("Training examples by sport")
    for i, s in enumerate(LABELS): ax.text(i, counts[s], f"{counts[s]} ({proportions[s]:.1f}%)", ha="center", va="bottom")
    savefig("01_label_balance.png")

    # Numeric distributions and boxplots.
    say("")
    say("## 3. Numeric feature distributions and label relationships")
    numeric_rows = []
    numeric_specs = [("ease_numeric", "Ease rating (0–10)"), ("price_numeric", "Willingness to pay ($)"), ("log_price", "log(1 + willingness to pay)")]
    for col, label in numeric_specs:
        row = {"feature": label}
        vals_all = tr[col].dropna()
        row.update({"n_nonmissing": len(vals_all), "missing_pct": 100*tr[col].isna().mean(),
                    "mean": vals_all.mean(), "std": vals_all.std(), "median": vals_all.median(),
                    "q1": vals_all.quantile(.25), "q3": vals_all.quantile(.75),
                    "min": vals_all.min(), "max": vals_all.max(),
                    "skewness": vals_all.skew()})
        numeric_rows.append(row)
        by = tr.groupby("label")[col].agg(["count", "mean", "std", "median", lambda z: z.quantile(.25), lambda z: z.quantile(.75)]).reindex(LABELS)
        by.columns = ["n", "mean", "sd", "median", "q1", "q3"]
        by.to_csv(OUT_DIR / f"{col}_by_sport.csv")
        say(f"- **{label}:** n={len(vals_all)}/{len(tr)} non-missing; median={vals_all.median():.2f}; IQR={vals_all.quantile(.25):.2f}–{vals_all.quantile(.75):.2f}; range={vals_all.min():.2f}–{vals_all.max():.2f}; skewness={vals_all.skew():.2f}.")
        say("  By sport (mean / median): " + "; ".join(f"{s}: {by.loc[s,'mean']:.2f} / {by.loc[s,'median']:.2f}" for s in LABELS))
        means = by["mean"].dropna()
        if len(means):
            say(f"  Largest difference in sport means: {means.max()-means.min():.2f}. This is descriptive and does not establish causation.")
    pd.DataFrame(numeric_rows).to_csv(OUT_DIR / "numeric_distributions.csv", index=False)

    # Boxplots by sport for ease, raw price, and log price.
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, (col, title) in zip(axes, numeric_specs):
        vals = [tr.loc[tr.label == s, col].dropna().values for s in LABELS]
        bp = ax.boxplot(vals, patch_artist=True, tick_labels=LABELS, showfliers=True)
        for patch, s in zip(bp["boxes"], LABELS):
            patch.set_facecolor(COLORS[s]); patch.set_alpha(.6)
        ax.set_title(title); ax.tick_params(axis="x", rotation=20)
        ax.grid(axis="y", alpha=.25)
    savefig("02_numeric_boxplots_by_sport.png")

    # Histograms and ECDF-ish cumulative distributions.
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for s in LABELS:
        z = tr.loc[tr.label == s, "ease_numeric"].dropna()
        axes[0].hist(z, bins=np.arange(-.5, 11.5, 1), density=True, histtype="step", linewidth=2, label=s, color=COLORS[s])
        z = tr.loc[tr.label == s, "log_price"].dropna().sort_values()
        if len(z): axes[1].plot(z, np.arange(1, len(z)+1)/len(z), label=s, color=COLORS[s], linewidth=2)
    axes[0].set_title("Ease rating distributions"); axes[0].set_xlabel("Rating"); axes[0].set_ylabel("Density")
    axes[1].set_title("Cumulative distribution of log(1 + price)"); axes[1].set_xlabel("log(1 + price)"); axes[1].set_ylabel("Cumulative share")
    for ax in axes: ax.legend(frameon=False); ax.grid(alpha=.2)
    savefig("03_numeric_distributions.png")

    # Price outliers and skewness discussion.
    p = tr["price_numeric"].dropna()
    if len(p):
        lp = np.log1p(p)
        q1, q3 = lp.quantile([.25, .75]); fence = np.expm1(q3 + 1.5*(q3-q1))
        say(f"- Price is typically right-skewed: raw-price skewness={p.skew():.2f}, log-price skewness={lp.skew():.2f}. The log transform {'reduces' if abs(lp.skew()) < abs(p.skew()) else 'does not reduce'} the absolute skewness.")
        say(f"- On the log-IQR rule, {int((p > fence).sum())} values exceed approximately ${fence:,.0f}. Inspect extreme values; do not automatically delete genuine high bids.")
        pd.DataFrame({"metric":["raw_skewness","log1p_skewness","log_iqr_upper_fence","n_values_above_fence"],
                      "value":[p.skew(),lp.skew(),fence,int((p>fence).sum())]}).to_csv(OUT_DIR/"price_outlier_diagnostics.csv",index=False)

    # Likert distributions.
    say("")
    say("## 4. Ordinal (Likert) responses")
    likert_long = []
    for col in LIKERT_COLS:
        ord_col = col + " [ordinal]"
        tab = pd.crosstab(tr["label"], tr[ord_col], normalize="index").reindex(index=LABELS, columns=[1,2,3,4,5], fill_value=0)*100
        tab.to_csv(OUT_DIR / f"likert_{LIKERT_COLS.index(col)+1}_percent_by_sport.csv")
        means = tr.groupby("label")[ord_col].mean().reindex(LABELS)
        say(f"- **{col}** Mean ordinal score: " + "; ".join(f"{s} {means[s]:.2f}" if pd.notna(means[s]) else f"{s} n/a" for s in LABELS) + ".")
        say("  Within-sport response percentages (columns 1=strongly disagree to 5=strongly agree):")
        say(tab.round(1).to_string())
        for s in LABELS:
            for level in [1,2,3,4,5]:
                likert_long.append({"question":short(col,60),"sport":s,"score":level,"percent":tab.loc[s,level]})
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    level_colors = ["#b23b35","#e59c8f","#dedbd5","#92b8e8","#2867b2"]
    for ax, col in zip(axes, LIKERT_COLS):
        ord_col = col + " [ordinal]"
        tab = pd.crosstab(tr["label"], tr[ord_col], normalize="index").reindex(index=LABELS, columns=[1,2,3,4,5], fill_value=0)*100
        left = np.zeros(len(LABELS))
        for level, color in zip([1,2,3,4,5], level_colors):
            ax.barh(LABELS, tab[level].values, left=left, color=color, edgecolor="white", label=str(level))
            left += tab[level].values
        ax.set_title(short(col, 42)); ax.set_xlim(0,100); ax.grid(axis="x", alpha=.15)
    fig.legend(["Strongly disagree","Disagree","Neutral","Agree","Strongly agree"], loc="lower center", ncol=5, frameon=False)
    fig.subplots_adjust(bottom=.25, wspace=.4)
    savefig("04_likert_distributions_by_sport.png")
    pd.DataFrame(likert_long).to_csv(OUT_DIR/"likert_long.csv",index=False)

    # Multi-select option frequencies.
    say("")
    say("## 5. Multi-select answer choices")
    multi_rows = []
    for col in MULTI_COLS:
        option_set = sorted(set(opt.strip() for v in tr[col].dropna() for opt in str(v).split(",") if opt.strip()))
        for option in option_set:
            rates = {}
            for s in LABELS:
                sub = tr.loc[tr.label == s, col].fillna("")
                rates[s] = 100 * sub.map(lambda z: option.casefold() in [x.strip().casefold() for x in z.split(",")]).mean()
            multi_rows.append({"question":col,"option":option,**{f"{s}_pct":rates[s] for s in LABELS},
                               "spread_percentage_points":max(rates.values())-min(rates.values())})
    multi = pd.DataFrame(multi_rows)
    if not multi.empty:
        multi.to_csv(OUT_DIR/"multiselect_option_rates.csv",index=False)
        say("Options with the largest between-sport differences (percentage-point spread):")
        top_multi = multi.sort_values("spread_percentage_points",ascending=False).head(15)
        say(top_multi[["question","option",*[f"{s}_pct" for s in LABELS],"spread_percentage_points"]].round(1).to_string(index=False))
        fig, axes = plt.subplots(1, len(MULTI_COLS), figsize=(17, max(5, min(8, len(multi)/6))))
        for ax, col in zip(axes, MULTI_COLS):
            sub = multi[multi.question == col].sort_values("spread_percentage_points",ascending=False).head(10)
            yy = np.arange(len(sub))
            for s in LABELS:
                ax.scatter(sub[f"{s}_pct"], yy, label=s, color=COLORS[s], s=35)
            ax.set_yticks(yy, [short(x,25) for x in sub.option], fontsize=7)
            ax.set_xlim(0,100); ax.set_xlabel("% of sport rows selecting"); ax.set_title(short(col,35)); ax.grid(axis="x",alpha=.2)
            ax.invert_yaxis()
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles,labels,loc="lower center",ncol=3,frameon=False)
        fig.subplots_adjust(bottom=.15,wspace=.8)
        savefig("05_multiselect_rates_by_sport.png")

    # Missingness patterns.
    say("")
    say("## 6. Missing answers and response patterns")
    missing_rows = []
    miss_cols = [EASE_COL, PRICE_COL] + LIKERT_COLS + MULTI_COLS + TEXT_COLS
    for col in miss_cols:
        def is_missing_answer(v):
            if pd.isna(v): return True
            return str(v).strip().casefold() in MISSING_STRINGS
        overall = 100*tr[col].map(is_missing_answer).mean()
        rates = {s:100*tr.loc[tr.label==s,col].map(is_missing_answer).mean() for s in LABELS}
        missing_rows.append({"question":col,"overall_missing_pct":overall,**{f"{s}_missing_pct":rates[s] for s in LABELS},
                             "max_min_gap_pp":max(rates.values())-min(rates.values())})
    missing_df = pd.DataFrame(missing_rows).sort_values("overall_missing_pct",ascending=False)
    missing_df.to_csv(OUT_DIR/"missingness_by_question.csv",index=False)
    say(missing_df[["question","overall_missing_pct",*[f"{s}_missing_pct" for s in LABELS],"max_min_gap_pp"]].round(1).to_string(index=False))
    say("Interpretation: label-dependent missingness can itself become predictive, but may also reflect survey-taking behaviour. Compare a model with and without missingness indicators.")
    fig, ax = plt.subplots(figsize=(10,6))
    plot_miss = missing_df.sort_values("overall_missing_pct")
    yy = np.arange(len(plot_miss))
    ax.barh(yy, plot_miss.overall_missing_pct, color="#8c8c8c")
    ax.set_yticks(yy,[short(x,55) for x in plot_miss.question],fontsize=8)
    ax.set_xlabel("Missing / non-answer (%)"); ax.set_title("Missingness by survey question")
    savefig("06_missingness_by_question.png")

    # Free-response analysis: lengths, distinct answers, token prevalence and MI.
    say("")
    say("## 7. Free-response answers: vocabulary and label associations")
    text_summary, token_rows, answer_rows = [], [], []
    for col in TEXT_COLS:
        valid = tr[col].dropna()
        toks = tr[col].map(tokenize)
        answered = toks.map(bool)
        words = toks[answered].map(len)
        normalized = tr.loc[answered,col].map(norm_answer)
        text_summary.append({"question":col,"answered_n":int(answered.sum()),"answer_rate_pct":100*answered.mean(),
                             "distinct_normalized_answers":int(normalized.nunique()),
                             "mean_words":words.mean(),"median_words":words.median(),
                             "vocabulary_size":len(set(t for ts in toks for t in ts))})
        say(f"- **{col}** Answer rate={100*answered.mean():.1f}%; distinct normalized answers={normalized.nunique()}; mean words={words.mean():.2f}; median words={words.median():.1f}; vocabulary size={len(set(t for ts in toks for t in ts))}.")
        # Top normalized exact answers per sport.
        for s in LABELS:
            counts = tr.loc[(tr.label==s)&answered,col].map(norm_answer).value_counts().head(10)
            for rank,(ans,n) in enumerate(counts.items(),1):
                answer_rows.append({"question":col,"sport":s,"rank":rank,"normalized_answer":ans,"count":n,
                                    "within_sport_pct":100*n/max(1,(tr.label==s).sum())})
        # Token prevalence and mutual information. Use only tokens in >= 5 rows for stable display.
        vocab = sorted(set(t for ts in toks for t in ts))
        token_presence = {t: toks.map(lambda ts, t=t: t in ts) for t in vocab}
        for t, present in token_presence.items():
            n = int(present.sum())
            if n < 3: continue
            rates = {s:100*present[tr.label==s].mean() for s in LABELS}
            token_rows.append({"question":col,"token":t,"n_rows":n,**{f"{s}_pct":rates[s] for s in LABELS},
                               "mi_bits":mutual_info_bits(present.astype(int),y)})
        # Length distribution plot for each question.
        fig, ax = plt.subplots(figsize=(7,4))
        lengths = pd.DataFrame({"sport":y,"word_count":toks.map(len),"answered":answered})
        lengths = lengths[lengths.answered]
        data = [lengths.loc[lengths.sport==s,"word_count"] for s in LABELS]
        bp = ax.boxplot(data,patch_artist=True,tick_labels=LABELS,showfliers=False)
        for patch,s in zip(bp["boxes"],LABELS): patch.set_facecolor(COLORS[s]); patch.set_alpha(.6)
        ax.set_ylabel("Words per response"); ax.set_title("Response length: "+short(col,60)); ax.grid(axis="y",alpha=.2)
        savefig(f"07_text_length_{TEXT_COLS.index(col)+1}.png")

    text_df = pd.DataFrame(text_summary)
    text_df.to_csv(OUT_DIR/"text_summary.csv",index=False)
    answer_df = pd.DataFrame(answer_rows)
    answer_df.to_csv(OUT_DIR/"top_free_response_answers.csv",index=False)
    token_df = pd.DataFrame(token_rows)
    if not token_df.empty:
        token_df.to_csv(OUT_DIR/"text_token_label_associations.csv",index=False)
        for col in TEXT_COLS:
            sub = token_df[token_df.question==col].sort_values("mi_bits",ascending=False).head(12)
            say(f"\nHighest-MI tokens for **{col}** (exploratory; not causal):")
            say(sub[["token","n_rows",*[f"{s}_pct" for s in LABELS],"mi_bits"]].round(3).to_string(index=False))
        # Top token prevalence dot chart.
        top_tokens = token_df.sort_values("mi_bits",ascending=False).groupby("question").head(8).copy()
        fig, axes = plt.subplots(len(TEXT_COLS),1,figsize=(10,3.0*len(TEXT_COLS)))
        for ax,col in zip(axes,TEXT_COLS):
            sub=top_tokens[top_tokens.question==col].sort_values("mi_bits")
            yy=np.arange(len(sub))
            for s in LABELS: ax.scatter(sub[f"{s}_pct"],yy,label=s,color=COLORS[s],s=36)
            ax.set_yticks(yy,sub.token,fontsize=8); ax.set_xlim(0,100)
            ax.set_xlabel("% of sport rows containing token"); ax.set_title(short(col,65),loc="left")
            ax.grid(axis="x",alpha=.2)
        handles, labels=axes[0].get_legend_handles_labels()
        fig.legend(handles,labels,loc="upper right",ncol=3,frameon=False)
        fig.subplots_adjust(top=.96,hspace=.65,right=.98)
        savefig("08_text_tokens_by_sport.png")
    say("Free-response interpretation: exact-answer counts reveal recurring associations; token prevalence and mutual information reveal terms whose presence differs across labels. Rare terms can appear highly distinctive by chance, so inspect support counts and validate any selected text features using cross-validation grouped by respondent.")

    # Correlations among numeric/ordinal items.
    say("")
    say("## 8. Relationships among numeric and ordinal features")
    corr_cols = ["ease_numeric"] + [c+" [ordinal]" for c in LIKERT_COLS] + ["log_price"]
    corr = tr[corr_cols].corr(method="spearman")
    corr.to_csv(OUT_DIR/"spearman_correlations.csv")
    say("Spearman correlation matrix (monotonic associations; pooled across sports):")
    say(corr.round(2).to_string())
    pairs=[]
    for i,a in enumerate(corr.columns):
        for b in corr.columns[i+1:]:
            pairs.append({"feature_1":a,"feature_2":b,"spearman_rho":corr.loc[a,b]})
    pair_df=pd.DataFrame(pairs)
    pair_df["abs_rho"]=pair_df.spearman_rho.abs()
    pair_df.sort_values("abs_rho",ascending=False).to_csv(OUT_DIR/"correlation_pairs.csv",index=False)
    if len(pair_df):
        top=pair_df.sort_values("abs_rho",ascending=False).iloc[0]
        say(f"Strongest absolute pooled Spearman association: {top.feature_1} vs {top.feature_2}, rho={top.spearman_rho:.2f}. Pooled correlations can be driven by label differences; inspect within-sport correlations before claiming respondent-level relationships.")
    fig,ax=plt.subplots(figsize=(8,6))
    im=ax.imshow(corr.values,vmin=-1,vmax=1,cmap="coolwarm")
    ax.set_xticks(range(len(corr.columns)),[short(c.replace(" [ordinal]",""),24) for c in corr.columns],rotation=45,ha="right")
    ax.set_yticks(range(len(corr.columns)),[short(c.replace(" [ordinal]",""),24) for c in corr.columns])
    for i in range(len(corr.columns)):
        for j in range(len(corr.columns)):
            if pd.notna(corr.iloc[i,j]): ax.text(j,i,f"{corr.iloc[i,j]:.2f}",ha="center",va="center",fontsize=8)
    fig.colorbar(im,ax=ax,label="Spearman rho")
    ax.set_title("Correlations among numeric and ordinal features")
    savefig("09_spearman_correlation_heatmap.png")

    # Feature-label association ranking. Discretize numeric variables; use one-hot options and token-presence indicators.
    say("")
    say("## 9. Exploratory feature-label association ranking")
    features={}
    features["Ease (binned)"]=pd.cut(tr["ease_numeric"],[-.01,2,4,6,8,10],include_lowest=True).astype(str)
    features["Price (quartiles)"]=pd.qcut(tr["price_numeric"],q=4,duplicates="drop").astype(str)
    for col in LIKERT_COLS:
        features[short(col,70)]=tr[col].fillna("[MISSING]")
    for col in MULTI_COLS:
        options=sorted(set(o.strip() for v in tr[col].dropna() for o in str(v).split(",") if o.strip()))
        for opt in options:
            features[f"{short(col,22)}: {opt}"]=tr[col].fillna("").map(lambda z,opt=opt: int(opt.casefold() in [a.strip().casefold() for a in z.split(",")]))
    for col in TEXT_COLS:
        toks=tr[col].map(tokenize)
        vocab=Counter(t for ts in toks for t in set(ts))
        for tok,n in vocab.items():
            if n>=5:
                features[f"{short(col,18)}: {tok}"]=toks.map(lambda ts,tok=tok:int(tok in ts))
        features[f"{short(col,18)}: [missing]"]=~toks.map(bool)
    mi_rows=[]
    for name,values in features.items():
        try: score=mutual_info_bits(values,y)
        except Exception: continue
        mi_rows.append({"feature":name,"mutual_information_bits":score})
    mi_df=pd.DataFrame(mi_rows).sort_values("mutual_information_bits",ascending=False)
    mi_df.to_csv(OUT_DIR/"feature_label_mutual_information.csv",index=False)
    fig,ax=plt.subplots(figsize=(10,7))
    top=mi_df.head(20).sort_values("mutual_information_bits")
    ax.barh(np.arange(len(top)),top.mutual_information_bits,color="#596d9c")
    ax.set_yticks(np.arange(len(top)),[short(s,55) for s in top.feature],fontsize=8)
    ax.set_xlabel("Mutual information (bits)"); ax.set_title("Features most associated with the sport label")
    savefig("10_feature_label_mutual_information.png")

    # Missingness chart by label, for answer-pattern effects.
    miss_rates=[]
    for col in TEXT_COLS + [PRICE_COL] + [EASE_COL]:
        vals=tr[col]
        m=vals.isna()
        for s in LABELS:
            miss_rates.append({"question":col,"sport":s,"missing_pct":100*m[tr.label==s].mean()})
    misslab=pd.DataFrame(miss_rates)
    misslab.to_csv(OUT_DIR/"missingness_by_label_long.csv",index=False)
    fig,ax=plt.subplots(figsize=(10,6))
    pivot=misslab.pivot(index="question",columns="sport",values="missing_pct").reindex(columns=LABELS)
    pivot.index=[short(s,48) for s in pivot.index]
    pivot.plot(kind="barh",ax=ax,color=[COLORS[s] for s in LABELS])
    ax.set_xlabel("Missing / non-answer (%)"); ax.set_title("Missingness by label for selected questions")
    ax.legend(frameon=False)
    savefig("11_missingness_by_label.png")

    # Final concise recommendations.
    say("")
    say("## 10. Implications for model development")
    say("1. Start with a structured-only baseline (one-hot categorical options + ordinal/numeric features) and multinomial logistic regression.")
    say("2. Compare against TF-IDF on free-response columns, then a combined structured + text model.")
    say("3. Fit imputers, scalers, encoders, and text vocabularies on training folds only. Use respondent-grouped cross-validation.")
    say("4. Use macro-F1, accuracy, per-class precision/recall, and a confusion matrix. Keep the test set untouched until final model selection.")
    say("5. Remove respondent ID from predictors. Consider a sensitivity analysis with and without missingness indicators.")
    say("6. Treat all rankings here as exploratory. They describe association, not causation, and small samples/rare words can create unstable apparent signals.")
    say("")
    say("## Output inventory")
    say("- Tables: `data_exploration/eda_outputs/`")
    say("- Figures: `data_exploration/figures/`")
    say("- Main plots: label balance; numeric boxplots/distributions; Likert distributions; multi-select option rates; missingness; free-text response lengths and token rates; correlation heatmap; mutual-information ranking.")
    (OUT_DIR/"eda_report.md").write_text("\n".join(report)+"\n",encoding="utf-8")
    print("\nWrote report: data_exploration/eda_outputs/eda_report.md")
    print("Wrote figures to: data_exploration/figures/")

if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--input",default=DATA)
    parser.add_argument("--out-dir",default=str(OUT_DIR))
    parser.add_argument("--fig-dir",default=str(FIG_DIR))
    parser.add_argument("--seed",type=int,default=SEED)
    args=parser.parse_args()
    main(args.input,args.out_dir,args.fig_dir,args.seed)
