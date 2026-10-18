# Adapted from Yihan Wang; path arguments and per-case feature display fixed by Codex.

from pathlib import Path
import re
import pandas as pd

# ============================================================
# 1. File paths
# ============================================================
import argparse
parser = argparse.ArgumentParser(description="Review existing technical labels and keyword features")
parser.add_argument("--clean", type=Path, required=True, help="questions_clean.parquet")
parser.add_argument("--types", type=Path, required=True, help="questions_with_types.csv")
args = parser.parse_args()
DATA_DIR = Path.cwd()

CLEAN_FILE = args.clean
TYPES_FILE = args.types

# Excel workbook: multiple sheets require an .xlsx file
OUTPUT_FILE = DATA_DIR / "technical_question_case_review.xlsx"

# Number of examples to display per category
N_PER_TYPE = 15
RANDOM_SEED = 42


# ============================================================
# 2. Clean text for Excel
# ============================================================
def clean_excel_text(value):
    """
    Remove characters forbidden in Excel cells.
    Preserve normal text, HTML tags, code and punctuation.
    """
    if isinstance(value, str):
        # Remove XML/Excel-illegal control characters
        value = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F]", "", value)

        # Excel cells have a maximum length of 32,767 characters.
        # Truncate exceptionally long text to prevent export errors.
        value = value[:32767]

    return value


# ============================================================
# 3. Read files
# ============================================================
print("Reading files...")

if not CLEAN_FILE.exists():
    raise FileNotFoundError(f"Cannot find file: {CLEAN_FILE}")

if not TYPES_FILE.exists():
    raise FileNotFoundError(f"Cannot find file: {TYPES_FILE}")

questions_clean = pd.read_parquet(CLEAN_FILE)
questions_types = pd.read_csv(TYPES_FILE)

print("questions_clean:", questions_clean.shape)
print("questions_with_types:", questions_types.shape)
print("Clean columns:", questions_clean.columns.tolist())
print("Type columns:", questions_types.columns.tolist())


# ============================================================
# 4. Identify columns
# ============================================================
def find_column(df, candidates, description):
    normalized = {
        re.sub(r"[_\s]+", "", str(col).strip().lower()): col
        for col in df.columns
    }

    for candidate in candidates:
        key = re.sub(r"[_\s]+", "", candidate.strip().lower())
        if key in normalized:
            return normalized[key]

    raise ValueError(
        f"Cannot identify {description}.\n"
        f"Available columns: {df.columns.tolist()}"
    )


# Actual column names in questions_clean.parquet
title_col = "title"
id_col = "question_id"
body_col = "body_html"

type_id_col = find_column(
    questions_types,
    ["question_id", "id", "post_id"],
    "question ID in questions_with_types"
)

type_col = find_column(
    questions_types,
    ["question_type", "type_name", "type",
     "technical_type", "topic"],
    "question type"
)

print("\nIdentified columns:")
print("Clean ID:", id_col)
print("Title:", title_col)
print("Body:", body_col)
print("Type-table ID:", type_id_col)
print("Question type:", type_col)


# ============================================================
# 5. Normalize IDs and merge
# ============================================================
def normalize_id(series):
    return (
        series.astype("string")
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
    )


clean = questions_clean.copy()
types = questions_types.copy()

clean["_join_id"] = normalize_id(clean[id_col])
types["_join_id"] = normalize_id(types[type_id_col])

print(
    "\nDuplicate IDs in questions_clean:",
    clean["_join_id"].duplicated().sum()
)

print(
    "Duplicate IDs in questions_with_types:",
    types["_join_id"].duplicated().sum()
)

type_conflicts = (
    types.groupby("_join_id")[type_col]
    .nunique(dropna=True)
)

conflict_ids = type_conflicts[type_conflicts > 1].index

print("IDs assigned multiple types:", len(conflict_ids))

clean_selected = clean[
    ["_join_id", title_col, body_col]
].drop_duplicates("_join_id", keep="first")

types_selected = types[["_join_id", type_col]].copy()

merged = types_selected.merge(
    clean_selected,
    on="_join_id",
    how="left",
    indicator=True,
    validate="many_to_one"
)

matched = merged[merged["_merge"] == "both"].copy()
unmatched = merged[merged["_merge"] == "left_only"].copy()

print("\nMatched rows:", len(matched))
print("Unmatched rows:", len(unmatched))
print(
    "Match rate:",
    round(100 * len(matched) / max(len(merged), 1), 2),
    "%"
)

matched = matched.rename(columns={
    "_join_id": "question_id",
    type_col: "question_type",
    title_col: "title",
    body_col: "body"
})

matched["title"] = matched["title"].fillna("").astype(str)
matched["body"] = matched["body"].fillna("").astype(str)


# ============================================================
# 6. Select existing categories
# ============================================================
print("\nAvailable question types:")
print(
    matched["question_type"]
    .value_counts(dropna=False)
    .to_string()
)

# These patterns select existing labels.
# They do not create new classifications.
target_patterns = {
    "API_server": r"api|server",
    "NumPy": r"numpy",
    "Matplotlib": r"matplotlib",
    "Python_basics": r"python basics|python syntax|python fundamental",
    "Django": r"django"
}

matched["type_text"] = matched["question_type"].fillna("").astype(str)

category_results = {}

for category, pattern in target_patterns.items():
    subset = matched[
        matched["type_text"].str.contains(
            pattern,
            case=False,
            regex=True,
            na=False
        )
    ].copy()

    category_results[category] = subset
    print(f"\n{category}: {len(subset)} questions")


# ============================================================
# 7. Sample real examples for manual review
# ============================================================
sampled_frames = []

for category, subset in category_results.items():
    if subset.empty:
        continue

    n = min(N_PER_TYPE, len(subset))

    sample = subset.sample(
        n=n,
        random_state=RANDOM_SEED
    ).copy()

    sample.insert(0, "analysis_group", category)
    sampled_frames.append(sample)

if sampled_frames:
    case_samples = pd.concat(sampled_frames, ignore_index=True)
else:
    case_samples = pd.DataFrame(
        columns=[
            "analysis_group", "question_id",
            "question_type", "title", "body"
        ]
    )


# ============================================================
# Define exploratory text feature patterns
# ============================================================
context_patterns = {
    "mentions_code": (
        r"<code|<pre>|```|"
        r"\bdef\s+\w+|import\s+\w+|"
        r"\bnp\.\w+|plt\.\w+"
    ),
    "mentions_version": (
        r"\bversion\b|\bv\d+(?:\.\d+)+\b|"
        r"python\s*3\.\d+"
    ),
    "mentions_auth": (
        r"auth(?:entication|orization)?|"
        r"token|oauth|api key"
    ),
    "mentions_request_response": (
        r"\brequest\b|\bresponse\b|"
        r"status code|HTTP\s*[45]\d\d"
    ),
    "mentions_error": (
        r"traceback|exception|\berror\b|"
        r"failed|doesn't work|not working"
    ),
    "mentions_expected_output": (
        r"expected output|expected result|"
        r"should return|instead returns"
    ),
    "mentions_environment": (
        r"operating system|windows|linux|macos|"
        r"environment|configuration"
    )
}

feature_columns = list(context_patterns.keys())


# ============================================================
# 8. Summarize text features by category
# ============================================================

# ============================================================
# Calculate text features BEFORE calculating feature_summary
# ============================================================
feature_columns = list(context_patterns.keys())

# Generate the seven feature columns in case_samples first
for column, pattern in context_patterns.items():
    case_samples[column] = case_samples["body"].str.contains(
        pattern,
        case=False,
        regex=True,
        na=False
    )

# Now calculate percentages by category
if not case_samples.empty:
    feature_summary = (
        case_samples.groupby("analysis_group")[feature_columns]
        .mean()
        .mul(100)
        .round(1)
    )
else:
    feature_summary = pd.DataFrame(columns=feature_columns)



# ============================================================
# 9. Print category counts
# ============================================================
print("\n" + "=" * 90)
print("QUESTION COUNTS BY TYPE")
print("=" * 90)

for category, subset in category_results.items():
    print(f"{category:20s}: {len(subset):,} questions")


# ============================================================
# 10. Print text feature percentages
# ============================================================
print("\n" + "=" * 90)
print("TEXT FEATURE PERCENTAGES IN RANDOM SAMPLES")
print("Each percentage = proportion of sampled questions matching the feature")
print("=" * 90)

if not feature_summary.empty:
    print(feature_summary.to_string())
else:
    print("No samples available.")

print("\nNote: These are keyword-based indicators, not definitive measures")
print("of question quality or whether a question can be answered.")


# ============================================================
# 11. Print up to 15 REAL question examples per category
# ============================================================
print("\n" + "=" * 90)
print("REAL QUESTION EXAMPLES")
print("=" * 90)

for category, subset in category_results.items():

    if subset.empty:
        print(f"\n[{category}] No matching questions.")
        continue

    sample = case_samples.loc[case_samples["analysis_group"] == category].copy()

    print("\n" + "#" * 90)
    print(f"CATEGORY: {category} | Showing {len(sample)} examples")
    print("#" * 90)

    for i, (_, row) in enumerate(sample.iterrows(), start=1):

        print(f"\n--- Example {i} ---")
        print(f"Question ID: {row['question_id']}")
        print(f"Original type: {row['question_type']}")
        print(f"Title: {row['title']}")

        # Display a readable version of the HTML body
        body = str(row["body"])

        # Remove script/style blocks and HTML tags for console display
        body = re.sub(
            r"<(script|style)\b[^>]*>.*?</\1>",
            " ",
            body,
            flags=re.IGNORECASE | re.DOTALL
        )
        body = re.sub(r"<[^>]+>", " ", body)
        body = (
            body.replace("&lt;", "<")
                .replace("&gt;", ">")
                .replace("&amp;", "&")
                .replace("&quot;", '"')
                .replace("&#39;", "'")
                .replace("&nbsp;", " ")
        )
        body = re.sub(r"\s+", " ", body).strip()

        # Keep the console output manageable
        max_chars = 1200
        if len(body) > max_chars:
            body = body[:max_chars] + "... [truncated]"

        print("Body:")
        print(body)

        # Print the exploratory indicators for this question
        matched_features = [
            feature for feature in feature_columns
            if bool(row.get(feature, False))
        ]

        print(
            "Detected text features:",
            ", ".join(matched_features)
            if matched_features else "None detected"
        )


# ============================================================
# 12. Direct comparison: NumPy/Matplotlib vs API/Django
# ============================================================
print("\n" + "=" * 90)
print("COMPARISON: LIBRARY QUESTIONS VS API/DJANGO QUESTIONS")
print("=" * 90)

comparison_groups = [
    "NumPy",
    "Matplotlib",
    "API_server",
    "Django"
]

available_groups = [
    group for group in comparison_groups
    if group in category_results and not category_results[group].empty
]

if available_groups:
    comparison_rows = []

    for group in available_groups:
        subset = category_results[group].copy()

        row = {
            "category": group,
            "n_questions": len(subset)
        }

        for feature in feature_columns:
            row[feature + "_pct"] = round(
                subset["body"].str.contains(
                    context_patterns[feature],
                    case=False,
                    regex=True,
                    na=False
                ).mean() * 100,
                1
            )

        comparison_rows.append(row)

    comparison_df = pd.DataFrame(comparison_rows).set_index("category")
    print(comparison_df.to_string())

else:
    print("No comparison groups have matching questions.")


print("\n" + "=" * 90)
print("Finished. Results were printed to the console.")
print("No output file was created.")
print("=" * 90)
