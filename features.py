"""features.py - cleaning and feature extraction for the CSC311 sport-survey data.

Only uses the Python standard library, numpy and pandas, so pred.py can import
these exact functions at prediction time (pred.py may not import sklearn).
"""
import re

import numpy as np
import pandas as pd

LABELS = ["Soccer", "Baseball", "Basketball"]

# Short name -> keyword that identifies the survey question (case-insensitive),
# so the code does not depend on the column order of the CSV.
QUESTION_KEYWORDS = {
    "ease": "how easy",
    "strategy": "strategy and planning",
    "injury": "risk of injury",
    "individual": "individual brilliance",
    "places": "where might you play",
    "seasons": "which seasons",
    "music": "types of music",
    "food": "foods and drinks",
    "colour": "what colour",
    "song": "which song",
    "feeling": "how do you feel",
    "price": "willing to pay",
}

LIKERT = {"strongly disagree": 1, "disagree": 2, "neutral": 3,
          "agree": 4, "strongly agree": 5}
LIKERT_COLS = ["strategy", "injury", "individual"]

MULTI_OPTIONS = {
    "places": ["Indoor facility", "Outdoor public space",
               "School or university facility", "Community recreation facility",
               "Private sports facility", "At home or nearby", "Others"],
    "seasons": ["Winter", "Spring", "Summer", "Fall"],
    "music": ["Pop", "Rock", "Hip-hop or rap", "Electronic or dance", "Latin",
              "Folk or traditional", "Country", "Classical or jazz", "Other",
              "No particular type"],
}
TEXT_COLS = ["food", "colour", "song", "feeling"]
FEATURE_COLS = ["ease"] + LIKERT_COLS + list(MULTI_OPTIONS) + TEXT_COLS + ["price"]


# ----------------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------------
def load_survey(path):
    """Read the CSV as strings (blank cells -> '') and shorten the column names."""
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    mapping = {}
    for col in df.columns:
        low = col.strip().lower()
        if low in ("id", "label"):
            mapping[col] = low
            continue
        for short, kw in QUESTION_KEYWORDS.items():
            if kw in low:
                mapping[col] = short
                break
    return df.rename(columns=mapping)


def blank_rows(df):
    """Boolean mask of rows where every survey answer is blank."""
    present = [c for c in FEATURE_COLS if c in df.columns]
    return df[present].apply(lambda col: col.str.strip() == "").all(axis=1)


# ----------------------------------------------------------------------------
# Text normalisation and missing values
# ----------------------------------------------------------------------------
def norm_text(s):
    """Lower-case, unify quotes and collapse whitespace."""
    s = "" if s is None else str(s)
    for a, b in (("’", "'"), ("‘", "'"), ("“", '"'), ("”", '"'),
                 ("，", ","), ("。", "."), ("—", "-"), ("–", "-")):
        s = s.replace(a, b)
    return re.sub(r"\s+", " ", s).strip().lower()


# Placeholder answers that carry no content ("n/a", "idk", "no particular song", ...)
_NON_ANSWER = re.compile(
    r"^(?:"
    r"|[-.*_/]+"                                       # blank or punctuation only
    r"|n/?a\b.*|nil|none\b.*|nothing|no|idk|unsure|not sure|#name\??"
    r"|(?:really )?(?:i )?(?:have )?no idea\b.*"
    r"|i (?:do not|don'?t) (?:know|no)(?: any songs?)?"
    r"|i (?:do not|don'?t) (?:associate|have)\b.*"
    r"|don'?t (?:really )?have\b.*"
    r"|(?:no|nothing) (?:particular|specific|in particular|really)\b.*"
    r"|no (?:\w+ )?(?:songs?|colou?rs?|food)\b.*"
    r"|there'?s no\b.*|can'?t think of\b.*"
    r")$"
)


def is_non_answer(s):
    """True for blank cells and placeholder answers such as 'n/a', 'idk', 'none'."""
    return bool(_NON_ANSWER.match(norm_text(s).rstrip(" .!")))


# ----------------------------------------------------------------------------
# Numeric, Likert and multi-select questions
# ----------------------------------------------------------------------------
def parse_ease(df):
    """0-10 ease rating as a float (NaN if blank or not a number)."""
    return pd.to_numeric(df["ease"].str.strip(), errors="coerce").clip(0, 10)


def parse_likert(df, col):
    """Strongly disagree..Strongly agree -> 1..5 (NaN if blank)."""
    return df[col].map(lambda s: LIKERT.get(norm_text(s), np.nan)).astype(float)


def multi_hot(df, col):
    """One 0/1 column per option of a multi-select question."""
    chosen = df[col].map(lambda s: {o.strip() for o in str(s).split(",") if o.strip()})
    return pd.DataFrame({f"{col}={o}": chosen.map(lambda c, o=o: int(o in c))
                         for o in MULTI_OPTIONS[col]}, index=df.index)


# ----------------------------------------------------------------------------
# Willingness to pay (free text that should contain a dollar amount)
# ----------------------------------------------------------------------------
_NUM = r"\d{1,3}(?:[, ]\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?"       # 1,000 / 10 000 / 150 / 7.5
_K = r"(k(?![a-z]))?"                                        # 5k -> 5000 (not "kr...")
_RANGE = re.compile(rf"({_NUM})\s*{_K}\s*(?:\$|cad|usd|dollars?)?\s*(?:-|to)\s*\$?\s*({_NUM})\s*{_K}")
_SINGLE = re.compile(rf"({_NUM})\s*{_K}")
_ZERO = re.compile(r"^(?:zero|none|nothing|no\b|not (?:really )?willing|unwilling|"
                   r"i would(?: not|n'?t) pay|you need to pay me)")
_UNITS = {"one": 1, "two": 2, "couple": 2, "three": 3, "four": 4, "five": 5, "six": 6,
          "seven": 7, "eight": 8, "nine": 9, "ten": 10, "fifteen": 15, "twenty": 20,
          "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70,
          "eighty": 80, "ninety": 90}
_SCALES = {"hundred": 100, "thousand": 1000}


def _number(tok, k):
    value = float(tok.replace(",", "").replace(" ", ""))
    return value * 1000 if k else value


def _word_number(t):
    """'twenty five dollars' -> 25, 'couple hundred bucks' -> 200; else None."""
    if not re.search(r"dollar|buck|cad", t):
        return None
    total, seen, scaled = 0, False, False
    for w in re.findall(r"[a-z]+", t):
        if w in _UNITS:
            total, seen = total + _UNITS[w], True
        elif w in _SCALES and seen:
            total, scaled = total * _SCALES[w], True
        elif seen:
            break
    if not seen or (total == 2 and not scaled):      # a bare "couple" is not an amount
        return None
    return float(total)


def parse_price(s):
    """Dollar amount from a free-text answer; ranges -> midpoint; NaN if unparseable.

    Order of rules: (1) answers that rate on a '1 to 10 scale' are unparseable;
    (2) the first range ('$80-$120', '50 to 150') -> its midpoint; (3) the first
    number ('1,000', '5k', '200CAD'); (4) refusals ('none', 'not willing') -> 0;
    (5) spelled-out amounts ('twenty five dollars'); otherwise NaN.
    """
    t = norm_text(s)
    if t == "" or "scale" in t:
        return np.nan
    m = _RANGE.search(t)
    if m:
        return (_number(m.group(1), m.group(2)) + _number(m.group(3), m.group(4))) / 2
    m = _SINGLE.search(t)
    if m:
        return _number(m.group(1), m.group(2))
    if _ZERO.match(t):
        return 0.0
    value = _word_number(t)
    return np.nan if value is None else value


# ----------------------------------------------------------------------------
# Free text -> tokens (bag of words)
# ----------------------------------------------------------------------------
# The survey replaced sport-specific names with placeholders such as
# "[a competition]"; each placeholder becomes one token because it is informative.
_PLACEHOLDERS = [("competition", "ph_competition"), ("league", "ph_league"),
                 ("team", "ph_team"), ("player", "ph_player"),
                 ("organization", "ph_organization"), ("another sport", "ph_another_sport"),
                 ("this sport", "ph_this_sport")]
_MERGES = [(r"hot[\s-]*dogs?", "hotdog"), (r"pop[\s-]*corns?", "popcorn"),
           (r"ball[\s-]*game", "ballgame"), (r"coca[\s-]*cola|\bcokes?\b", "coke"),
           (r"chicken wings?", "wings"), (r"french fries", "fries"),
           (r"hip[\s-]*hop", "hiphop"), (r"waka[\s-]*waka", "wakawaka"),
           (r"fish (?:and|&|n) chips", "fishandchips"), (r"\bgrey\b", "gray"),
           (r"\bcolour", "color"), (r"\b(?:boring|boredom)\b", "bored"),
           (r"\b(?:exited|exicted|excitement|exciting)\b", "excited"),
           (r"\b(?:confusing|confusion|confusions)\b", "confused"),
           (r"\benerg\w*", "energetic"), (r"\brelax\w*", "relaxed"),
           (r"\bnerv\w*", "nervous"), (r"\b(?:tension|tensed)\b", "tense")]
STOP_WORDS = set("""a an the and or of to in for with my i it is its it's this that be at on as by
but so if me we you your our are was were very really just like some any also maybe would
when while because from there their they them than then too can i'm im usually mostly
more most much will feel get gets especially something""".split())


def _placeholder(match):
    inner = match.group(1)
    for key, token in _PLACEHOLDERS:
        if key in inner:
            return f" {token} "
    return " "


def tokenize(s):
    """Normalised tokens of one free-text answer ([] for a non-answer)."""
    if is_non_answer(s):
        return []
    t = norm_text(s)
    t = re.sub(r"\[+([^\[\]]*)\]+", _placeholder, t)
    for pattern, repl in _MERGES:
        t = re.sub(pattern, repl, t)
    tokens = []
    for tok in re.findall(r"[a-z0-9_]+|[一-鿿]+|[぀-ヿ]+", t):
        if tok in STOP_WORDS or (len(tok) < 2 and tok.isascii()):
            continue
        if (len(tok) > 3 and tok.endswith("s") and not tok.startswith("ph_")
                and not tok.endswith(("ss", "us", "is", "ies", "ous"))):
            tok = tok[:-1]                                  # light plural stripping
        tokens.append(tok)
    return tokens


def structured_features(df):
    """Numeric, Likert, price and multi-hot columns (before imputation/scaling).

    Imputation medians, clipping thresholds and scaling statistics are NOT
    computed here: they must be learned on the training split (or training
    folds) and then applied to validation/test rows.
    """
    out = pd.DataFrame(index=df.index)
    out["ease"] = parse_ease(df)
    for col in LIKERT_COLS:
        out[col] = parse_likert(df, col)
    out["price"] = df["price"].map(parse_price)
    out["log_price"] = np.log1p(out["price"])
    for col in MULTI_OPTIONS:
        out = out.join(multi_hot(df, col))
    return out


def respondent_split(ids, seed=311, fractions=(0.70, 0.15, 0.15)):
    """Assign each respondent id to train / val / test (all rows of an id together)."""
    unique = np.array(sorted(set(ids), key=int))
    rng = np.random.default_rng(seed)
    rng.shuffle(unique)
    n_train = int(round(fractions[0] * len(unique)))
    n_val = int(round(fractions[1] * len(unique)))
    split = {}
    for i, rid in enumerate(unique):
        split[rid] = "train" if i < n_train else ("val" if i < n_train + n_val else "test")
    return pd.Series(ids).map(split).to_numpy()
