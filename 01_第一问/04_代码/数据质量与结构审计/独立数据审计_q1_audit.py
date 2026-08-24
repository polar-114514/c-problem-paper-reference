import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = SCRIPT_DIR.parents[2]
parser = argparse.ArgumentParser(description="第一问男胎数据独立质量审计")
parser.add_argument(
    "--input",
    type=Path,
    default=WORKSPACE_ROOT / "00_题目与原始资料" / "02_原始数据" / "附件.xlsx",
)
parser.add_argument(
    "--output",
    type=Path,
    default=WORKSPACE_ROOT / "99_临时中转" / "第一问数据审计复现" / "independent_audit",
)
args = parser.parse_args()
INPUT = args.input.resolve()
OUT = args.output.resolve()
OUT.mkdir(parents=True, exist_ok=True)

COLS = [
    "序号", "孕妇代码", "年龄", "身高", "体重", "末次月经", "IVF妊娠", "检测日期", "检测抽血次数", "检测孕周",
    "孕妇BMI", "原始读段数", "在参考基因组上比对的比例", "重复读段的比例", "唯一比对的读段数", "GC含量",
    "13号染色体的Z值", "18号染色体的Z值", "21号染色体的Z值", "X染色体的Z值", "Y染色体的Z值", "Y染色体浓度",
    "X染色体浓度", "13号染色体的GC含量", "18号染色体的GC含量", "21号染色体的GC含量", "被过滤掉读段数的比例",
    "染色体的非整倍体", "怀孕次数", "生产次数", "胎儿是否健康",
]


def gest_days(value):
    if pd.isna(value):
        return np.nan
    match = re.fullmatch(r"\s*(\d+)\s*[wW](?:\s*\+\s*(\d+))?\s*", str(value))
    if not match:
        return np.nan
    weeks = int(match.group(1))
    days = int(match.group(2) or 0)
    return np.nan if days > 6 else weeks * 7 + days


def parse_lmp(value):
    if pd.isna(value):
        return pd.NaT
    if isinstance(value, (pd.Timestamp, np.datetime64)) or hasattr(value, "year"):
        return pd.Timestamp(value).normalize()
    text = str(value).strip()
    if not text:
        return pd.NaT
    try:
        number = int(float(text))
        if 1 <= number <= 200000:
            return pd.Timestamp("1899-12-30") + pd.to_timedelta(number, unit="D")
    except Exception:
        pass
    return pd.to_datetime(text, errors="coerce")


def parse_detection(value):
    if pd.isna(value):
        return pd.NaT
    if isinstance(value, (pd.Timestamp, np.datetime64)) or hasattr(value, "year"):
        return pd.Timestamp(value).normalize()
    try:
        number = int(float(value))
        if 19000101 <= number <= 22001231:
            return pd.to_datetime(str(number), format="%Y%m%d", errors="coerce")
        if 1 <= number <= 200000:
            return pd.Timestamp("1899-12-30") + pd.to_timedelta(number, unit="D")
    except Exception:
        pass
    return pd.to_datetime(str(value).strip(), errors="coerce")


def js(value):
    if value is None or value is pd.NA:
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if np.isnan(value) else float(value)
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, set):
        return sorted(value)
    raise TypeError(type(value).__name__)


source = pd.read_excel(INPUT, sheet_name="男胎检测数据", dtype=object)
original_headers = list(source.columns)
if len(original_headers) != 31:
    raise RuntimeError(f"Expected 31 columns, got {len(original_headers)}")
source.columns = COLS
df = source.copy()

raw_types = {
    column: {kind.__name__: int(count) for kind, count in df[column].map(type).value_counts().items()}
    for column in ["末次月经", "检测日期"]
}

numeric_cols = [
    "序号", "年龄", "身高", "体重", "检测抽血次数", "孕妇BMI", "原始读段数", "在参考基因组上比对的比例",
    "重复读段的比例", "唯一比对的读段数", "GC含量", "13号染色体的Z值", "18号染色体的Z值",
    "21号染色体的Z值", "X染色体的Z值", "Y染色体的Z值", "Y染色体浓度", "X染色体浓度",
    "13号染色体的GC含量", "18号染色体的GC含量", "21号染色体的GC含量", "被过滤掉读段数的比例", "生产次数",
]
for column in numeric_cols:
    df[column] = pd.to_numeric(df[column], errors="coerce")

df["gest_days"] = source["检测孕周"].map(gest_days)
df["gest_weeks"] = df["gest_days"] / 7.0
df["lmp_date"] = source["末次月经"].map(parse_lmp)
df["detection_date"] = source["检测日期"].map(parse_detection)
df["date_gest_delta_days"] = (df["detection_date"] - df["lmp_date"]).dt.days - df["gest_days"]
df["bmi_calculated"] = df["体重"] / (df["身高"] / 100.0) ** 2
df["bmi_abs_error"] = (df["孕妇BMI"] - df["bmi_calculated"]).abs()
df["row_id"] = df["序号"].astype("Int64").astype(str)
df["draw_id"] = df["孕妇代码"].astype(str) + "#" + df["检测抽血次数"].astype("Int64").astype(str)
df["detection_date_key"] = df["detection_date"].dt.strftime("%Y-%m-%d").fillna("MISSING_DATE")
df["date_session_id"] = df["孕妇代码"].astype(str) + "#" + df["detection_date_key"]
df["assay_session_id"] = df["draw_id"] + "#" + df["detection_date_key"]
df["strict_metadata_id"] = df["assay_session_id"] + "#" + df["gest_days"].astype("Int64").astype(str)

draw_sizes = df.groupby("draw_id", sort=False).size()
df["draw_assay_count"] = df["draw_id"].map(draw_sizes)
df["same_draw_multitest"] = df["draw_assay_count"] > 1
session_sizes = df.groupby("assay_session_id", sort=False).size()
df["assay_session_record_count"] = df["assay_session_id"].map(session_sizes)
df["same_session_multirecord"] = df["assay_session_record_count"] > 1
draw_gestation_unique = df.groupby("draw_id", sort=False)["gest_days"].nunique(dropna=False)
df["draw_gestation_conflict"] = df["draw_id"].map(draw_gestation_unique).gt(1)

L = df["原始读段数"]
M = df["在参考基因组上比对的比例"]
N = df["重复读段的比例"]
O = df["唯一比对的读段数"]
AA = df["被过滤掉读段数的比例"]
df["expected_unique_reads"] = L * M * (1.0 - N) * (1.0 - AA)
df["unique_read_formula_abs_error"] = (O - df["expected_unique_reads"]).abs()

filter_p99 = float(AA.quantile(0.99))
y_q1, y_q3 = df["Y染色体浓度"].quantile([0.25, 0.75])
y_iqr = y_q3 - y_q1
y_lo, y_hi = float(y_q1 - 1.5 * y_iqr), float(y_q3 + 1.5 * y_iqr)

flag_definitions = {
    "core_missing_or_invalid": (
        df["孕妇代码"].isna()
        | df["gest_days"].isna()
        | df["孕妇BMI"].isna()
        | df["Y染色体浓度"].isna()
        | ~df["Y染色体浓度"].between(0, 1, inclusive="neither")
        | (df["孕妇BMI"] <= 0)
    ),
    "outside_10w0_25w0": ~df["gest_days"].between(70, 175, inclusive="both"),
    "outside_10w0_25w6": ~df["gest_days"].between(70, 181, inclusive="both"),
    "batch683": df["序号"] >= 683,
    "storage687": df["序号"] >= 687,
    "lmp_unusable": df["lmp_date"].isna(),
    "date_delta_negative": df["date_gest_delta_days"] < 0,
    "date_delta_abs_gt21": df["date_gest_delta_days"].abs() > 21,
    "bmi_formula_abs_gt_0_01": df["bmi_abs_error"] > 0.01,
    "unique_reads_gt_raw": O > L,
    "read_formula_abs_error_gt2": df["unique_read_formula_abs_error"] > 2,
    "gc_below_40pct": df["GC含量"] < 0.40,
    "gc_below_39pct": df["GC含量"] < 0.39,
    "filter_ratio_above_p99": AA > filter_p99,
    "y_tukey_outlier": (df["Y染色体浓度"] < y_lo) | (df["Y染色体浓度"] > y_hi),
    "y_below_4pct": df["Y染色体浓度"] < 0.04,
}
for name, values in flag_definitions.items():
    df[name] = values.fillna(False).astype(bool)

df["primary_include"] = (
    ~df["core_missing_or_invalid"]
    & ~df["batch683"]
    & ~df["outside_10w0_25w6"]
)
df["sensitivity_through_25w0_include"] = (
    ~df["core_missing_or_invalid"]
    & ~df["batch683"]
    & ~df["outside_10w0_25w0"]
)

def exclusion_reasons(row):
    reasons = []
    if row["core_missing_or_invalid"]:
        reasons.append("hard_core_invalid")
    if row["batch683"]:
        reasons.append("primary_batch683")
    if row["outside_10w0_25w6"]:
        reasons.append("primary_outside_10w0_25w6")
    return ";".join(reasons)


def mark_reasons(row):
    names = [
        "same_draw_multitest", "same_session_multirecord", "draw_gestation_conflict", "lmp_unusable", "date_delta_negative", "date_delta_abs_gt21",
        "bmi_formula_abs_gt_0_01", "unique_reads_gt_raw", "read_formula_abs_error_gt2",
        "gc_below_40pct", "gc_below_39pct", "filter_ratio_above_p99", "y_tukey_outlier", "y_below_4pct",
    ]
    return ";".join(name for name in names if bool(row[name]))


df["primary_exclusion_reasons"] = df.apply(exclusion_reasons, axis=1)
df["mark_only_reasons"] = df.apply(mark_reasons, axis=1)
df["audit_action"] = np.select(
    [
        df["core_missing_or_invalid"],
        ~df["primary_include"],
        df["mark_only_reasons"].ne(""),
    ],
    ["HARD_EXCLUDE", "EXCLUDE_PRIMARY_KEEP_SENSITIVITY", "KEEP_PRIMARY_MARKED"],
    default="KEEP_PRIMARY_CLEAN",
)


def distribution(series):
    return {str(key): int(value) for key, value in series.value_counts(dropna=False).sort_index().items()}


def numeric_summary(frame, column):
    values = pd.to_numeric(frame[column], errors="coerce")
    return {
        "n": int(values.notna().sum()),
        "missing": int(values.isna().sum()),
        "min": float(values.min()),
        "p01": float(values.quantile(0.01)),
        "median": float(values.median()),
        "mean": float(values.mean()),
        "p99": float(values.quantile(0.99)),
        "max": float(values.max()),
    }


def cohort_summary(frame):
    repeated_draws = frame.groupby("draw_id").size()
    repeated_sessions = frame.groupby("assay_session_id").size()
    return {
        "rows": int(len(frame)),
        "women": int(frame["孕妇代码"].nunique()),
        "draws": int(frame["draw_id"].nunique()),
        "multitest_draw_groups": int((repeated_draws > 1).sum()),
        "assay_sessions": int(frame["assay_session_id"].nunique()),
        "multirecord_assay_sessions": int((repeated_sessions > 1).sum()),
        "gest_weeks": numeric_summary(frame, "gest_weeks"),
        "bmi": numeric_summary(frame, "孕妇BMI"),
        "y_concentration": numeric_summary(frame, "Y染色体浓度"),
        "y_at_or_above_4pct": int((frame["Y染色体浓度"] >= 0.04).sum()),
        "y_at_or_above_4pct_rate": float((frame["Y染色体浓度"] >= 0.04).mean()),
        "unique_reads_gt_raw": int(frame["unique_reads_gt_raw"].sum()),
        "read_formula_ok_within2": int((frame["unique_read_formula_abs_error"] <= 2).sum()),
    }


person_sizes = df.groupby("孕妇代码").size()
person_draws = df.groupby("孕妇代码")["draw_id"].nunique()
multi_draw_details = []
varying_gest_draws = []
for draw_id, group in df.groupby("draw_id", sort=False):
    if len(group) > 1:
        detail = {
            "draw_id": draw_id,
            "woman": str(group["孕妇代码"].iloc[0]),
            "draw_number": int(group["检测抽血次数"].iloc[0]),
            "assays": int(len(group)),
            "serials": [int(x) for x in group["序号"].tolist()],
            "unique_detection_dates": int(group["detection_date"].nunique(dropna=False)),
            "unique_gestational_ages": int(group["gest_days"].nunique(dropna=False)),
            "y_min": float(group["Y染色体浓度"].min()),
            "y_max": float(group["Y染色体浓度"].max()),
        }
        multi_draw_details.append(detail)
        if detail["unique_gestational_ages"] > 1:
            varying_gest_draws.append(detail)


within_person_variation = {}
for column in ["年龄", "身高", "末次月经", "IVF妊娠", "怀孕次数", "生产次数", "胎儿是否健康"]:
    counts = source.groupby("孕妇代码")[column].nunique(dropna=False)
    within_person_variation[column] = {
        "varying_women": int((counts > 1).sum()),
        "max_unique_per_woman": int(counts.max()),
    }


ratio_columns = [
    "在参考基因组上比对的比例", "重复读段的比例", "GC含量", "Y染色体浓度",
    "13号染色体的GC含量", "18号染色体的GC含量", "21号染色体的GC含量", "被过滤掉读段数的比例",
]
ratio_checks = {
    column: {
        "below_zero": int((df[column] < 0).sum()),
        "above_one": int((df[column] > 1).sum()),
    }
    for column in ratio_columns
}


outlier_columns = [
    "年龄", "身高", "体重", "gest_weeks", "孕妇BMI", "原始读段数", "在参考基因组上比对的比例",
    "重复读段的比例", "唯一比对的读段数", "GC含量", "Y染色体的Z值", "Y染色体浓度",
    "X染色体浓度", "被过滤掉读段数的比例",
]
outliers = {}
for column in outlier_columns:
    values = pd.to_numeric(df[column], errors="coerce")
    q1, q3 = values.quantile([0.25, 0.75])
    iqr = q3 - q1
    low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    mask = (values < low) | (values > high)
    outliers[column] = {
        "tukey_low": float(low),
        "tukey_high": float(high),
        "rows": int(mask.sum()),
        "serial_examples": [int(x) for x in df.loc[mask, "序号"].head(20).tolist()],
        "note": "Statistical flag only; not an automatic exclusion.",
    }


bmi_bands = pd.IntervalIndex.from_tuples([(20, 28), (28, 32), (32, 36), (36, 40), (40, np.inf)], closed="left")
bmi_labels = ["[20,28)", "[28,32)", "[32,36)", "[36,40)", "[40,+inf)"]
df["bmi_band"] = pd.cut(df["孕妇BMI"], bins=[20, 28, 32, 36, 40, np.inf], right=False, labels=bmi_labels)
first_person = df.sort_values(["gest_days", "序号"], kind="stable").drop_duplicates("孕妇代码")

support = {
    "row_bmi_bands_full": {str(k): int(v) for k, v in df["bmi_band"].value_counts(sort=False, dropna=False).items()},
    "baseline_bmi_bands_full": {str(k): int(v) for k, v in first_person["bmi_band"].value_counts(sort=False, dropna=False).items()},
    "baseline_bmi_bands_pre683": {
        str(k): int(v)
        for k, v in first_person[first_person["序号"] < 683]["bmi_band"].value_counts(sort=False, dropna=False).items()
    },
    "baseline_bmi_bands_post683": {
        str(k): int(v)
        for k, v in first_person[first_person["序号"] >= 683]["bmi_band"].value_counts(sort=False, dropna=False).items()
    },
}

df["gest_floor_week"] = np.floor(df["gest_weeks"]).astype("Int64")
gest_support = []
for week, group in df.groupby("gest_floor_week", dropna=False):
    gest_support.append({
        "week_bin": None if pd.isna(week) else int(week),
        "rows": int(len(group)),
        "women": int(group["孕妇代码"].nunique()),
        "draws": int(group["draw_id"].nunique()),
        "pre683_rows": int((group["序号"] < 683).sum()),
        "post683_rows": int((group["序号"] >= 683).sum()),
    })

main = df[df["primary_include"]].copy()
main_cross_rows = pd.crosstab(main["bmi_band"], main["gest_floor_week"], dropna=False)
main_cross_women = pd.crosstab(main["bmi_band"], main["gest_floor_week"], values=main["孕妇代码"], aggfunc=pd.Series.nunique, dropna=False).fillna(0).astype(int)
support_cell_counts = {
    "cells": int(main_cross_women.size),
    "zero_woman_cells": int((main_cross_women == 0).to_numpy().sum()),
    "cells_with_fewer_than_5_women": int((main_cross_women < 5).to_numpy().sum()),
    "cells_with_fewer_than_10_women": int((main_cross_women < 10).to_numpy().sum()),
}


cohorts = {
    "full_male": cohort_summary(df),
    "clinical_10w0_25w0": cohort_summary(df[df["gest_days"].between(70, 175)]),
    "clinical_10w0_25w6": cohort_summary(df[df["gest_days"].between(70, 181)]),
    "pre683_all": cohort_summary(df[df["序号"] < 683]),
    "pre683_clinical_10w0_25w6_primary": cohort_summary(main),
    "pre683_clinical_through_25w0_sensitivity": cohort_summary(df[df["sensitivity_through_25w0_include"]]),
    "post683_all": cohort_summary(df[df["序号"] >= 683]),
    "post683_clinical_10w0_25w0": cohort_summary(df[(df["序号"] >= 683) & df["gest_days"].between(70, 175)]),
}


sequence_missing = {
    column: int(source[column].isna().sum())
    for column in COLS[11:27]
}

action_counts = {str(key): int(value) for key, value in df["audit_action"].value_counts().items()}
flag_counts = {name: int(df[name].sum()) for name in flag_definitions}
flag_counts["same_draw_multitest"] = int(df["same_draw_multitest"].sum())
flag_counts["same_session_multirecord"] = int(df["same_session_multirecord"].sum())
flag_counts["draw_gestation_conflict"] = int(df["draw_gestation_conflict"].sum())


findings = [
    {
        "severity": "CRITICAL",
        "finding": "Male-data generation mechanism break begins at serial 683 / woman A168; laboratory-batch cause is unconfirmed.",
        "count": 400,
        "location_rule": "序号 >= 683",
        "evidence": "Rows 1-682 all satisfy the empirical unique-read identity within 2 reads; rows 683-1082 none do. A168-A267 each have exactly four rows. Y>=4% changes from 540/682 to 397/400.",
        "action": "Use pre-683 as the recommended primary cohort and retain post-683 as a mandatory separate sensitivity cohort; do not claim a confirmed laboratory batch without external metadata.",
    },
    {
        "severity": "CRITICAL",
        "finding": "Rows are nested repeated measurements, not independent observations.",
        "count": 1082,
        "location_rule": "woman=B; draw=B+I; assay session=B+I+H; row=A",
        "evidence": "267 women, 1021 operational draw events, 1063 assay sessions and 1082 rows; 40 draw events contain multiple detections covering 101 rows, while 19 same-day sessions cover 38 rows.",
        "action": "Retain all four hierarchical IDs; never use row-independence for inference or splitting, and do not treat H as a blood-draw timestamp.",
    },
    {
        "severity": "HIGH",
        "finding": "Unique mapped reads exceed raw reads in a subset.",
        "count": int(df["unique_reads_gt_raw"].sum()),
        "location_rule": "O > L; all affected rows have 序号 >= 683",
        "evidence": "Logical contradiction under the supplied column definitions.",
        "action": "Mark; exclude from analyses that use sequencing-count covariates. Already outside the recommended primary cohort.",
    },
    {
        "severity": "HIGH",
        "finding": "Date-derived gestational age is not interchangeable with column J.",
        "count": int(df["date_delta_abs_gt21"].sum()),
        "location_rule": "abs((H-F)-J_days) > 21 days",
        "evidence": f"{int((df['date_gest_delta_days']==0).sum())} exact matches; {int(df['lmp_unusable'].sum())} unusable LMP values; {int(df['date_delta_negative'].sum())} negative deltas.",
        "action": "Use J as primary gestational age. Date flags are diagnostic only.",
    },
    {
        "severity": "HIGH",
        "finding": "Clinical support is strongly concentrated in high BMI and irregularly sampled weeks.",
        "count": int((first_person["孕妇BMI"] >= 28).sum()),
        "location_rule": "first chronological record BMI >= 28 kg/m^2",
        "evidence": f"{int((first_person['孕妇BMI']>=28).sum())}/267 women; only {int((first_person['孕妇BMI']<28).sum())} below 28 and none below 20.",
        "action": "Keep, but restrict interpretation to observed support and report sparse cells.",
    },
    {
        "severity": "HIGH",
        "finding": "Sequencing failures are not directly identifiable in the male sheet.",
        "count": 0,
        "location_rule": "No explicit failure flag and no row with missing L-AA sequencing metrics.",
        "evidence": "All 1082 rows contain the sequencing numeric block. Repeated assays may reflect retesting, but no failed attempt is labelled.",
        "action": "Do not estimate a failure rate or delete a row as failed solely from retesting/GC. Treat complete observed records as potentially selected successful outputs.",
    },
    {
        "severity": "MEDIUM",
        "finding": "Reported gestational ages are parseable but extend beyond the stated 10-25 week window.",
        "count": int(df["outside_10w0_25w6"].sum()),
        "location_rule": "J_days < 70 or J_days > 181",
        "evidence": f"{int(df.loc[df['outside_10w0_25w6'],'孕妇代码'].nunique())} women affected; no record is below 10 weeks.",
        "action": "Exclude from the approved [10,26) primary window; retain for sensitivity.",
    },
    {
        "severity": "MEDIUM",
        "finding": "Global GC is technically below 40% in many rows, mostly just below the boundary.",
        "count": int(df["gc_below_40pct"].sum()),
        "location_rule": "P < 0.40",
        "evidence": f"Only {int(df['gc_below_39pct'].sum())} rows are below 39%; the remainder lie in [39%,40%).",
        "action": "Mark only; do not label all as failed or delete without a validated assay rule.",
    },
    {
        "severity": "MEDIUM",
        "finding": "Baseline-like fields are not perfectly constant within woman.",
        "count": int(sum(x["varying_women"] for x in within_person_variation.values())),
        "location_rule": "nunique(field within B) > 1",
        "evidence": "Age varies in 12 women, height in 5, LMP in 5; IVF is constant.",
        "action": "Mark and predefine baseline/first-record or time-varying treatment by field.",
    },
    {
        "severity": "LOW",
        "finding": "Date storage types change at serial 687 / A169.",
        "count": 396,
        "location_rule": "序号 >= 687",
        "evidence": "H changes from integer YYYYMMDD to date objects; F changes from date objects to strings.",
        "action": "Normalize types and retain; do not treat storage format as biology.",
    },
    {
        "severity": "LOW",
        "finding": "BMI is arithmetically consistent with height and weight.",
        "count": int((df["bmi_abs_error"] <= 0.01).sum()),
        "location_rule": "abs(K - E/(D/100)^2) <= 0.01",
        "evidence": f"Maximum absolute discrepancy {float(df['bmi_abs_error'].max()):.6f} kg/m^2.",
        "action": "Retain K as supplied; no BMI imputation or correction required in the male sheet.",
    },
]


def key_count_summary(frame, key):
    sizes = frame.groupby(key, dropna=False).size()
    repeated = sizes[sizes > 1]
    return {
        "events": int(len(sizes)),
        "multi_groups": int(len(repeated)),
        "multi_rows": int(repeated.sum()),
        "size_distribution": distribution(sizes),
    }


def repeated_measurement_metrics(frame, key):
    groups = [group for _, group in frame.groupby(key, sort=False) if len(group) > 1]
    denominator = sum(len(group) - 1 for group in groups)
    numerator = sum(
        (len(group) - 1) * float(group["Y染色体浓度"].var(ddof=1))
        for group in groups
    )
    return {
        "multi_groups": int(len(groups)),
        "multi_rows": int(sum(len(group) for group in groups)),
        "pooled_within_y_sd": float(math.sqrt(numerator / denominator)) if denominator else 0.0,
        "groups_crossing_4pct": int(sum(
            group["Y染色体浓度"].min() < 0.04 <= group["Y染色体浓度"].max()
            for group in groups
        )),
    }


event_scope_frames = {
    "全男胎1082条": df,
    "683前682条": df[df["序号"] < 683],
    "主参考674条": df[df["primary_include"]],
    "敏感性670条": df[df["sensitivity_through_25w0_include"]],
}
event_key_columns = {
    "B+I": "draw_id",
    "B+H": "date_session_id",
    "B+I+H": "assay_session_id",
    "B+I+H+J": "strict_metadata_id",
}
event_hierarchy = {
    scope: {
        notation: {**key_count_summary(frame, column), **repeated_measurement_metrics(frame, column)}
        for notation, column in event_key_columns.items()
    }
    for scope, frame in event_scope_frames.items()
}
repeat_definition_sensitivity = {
    "B+I抽血事件主口径": repeated_measurement_metrics(df, "draw_id"),
    "B+I+H同日检测会话口径": repeated_measurement_metrics(df, "assay_session_id"),
    "B+I+H+J严格元数据口径": repeated_measurement_metrics(df, "strict_metadata_id"),
}


summary = {
    "source": str(INPUT),
    "scope": "Q1 male-sheet audit only; no formal relationship model fitted",
    "workbook_structure": {
        "sheets": ["男胎检测数据", "女胎检测数据"],
        "male_used_range": "A1:AE1083",
        "male_data_rows": int(len(df)),
        "columns": int(len(COLS)),
        "original_headers": original_headers,
        "normalized_headers": COLS,
        "male_O_header_has_trailing_spaces": bool(str(original_headers[14]) != "唯一比对的读段数"),
    },
    "field_types": raw_types,
    "missing_by_column": {column: int(source[column].isna().sum()) for column in COLS},
    "sequence_missing_by_column_L_to_AA": sequence_missing,
    "event_hierarchy": event_hierarchy,
    "repeat_definition_sensitivity": repeat_definition_sensitivity,
    "id_hierarchy": {
        "women": int(df["孕妇代码"].nunique()),
        "draws": int(df["draw_id"].nunique()),
        "date_sessions_BH": int(df["date_session_id"].nunique()),
        "assay_sessions_BIH": int(df["assay_session_id"].nunique()),
        "strict_metadata_groups_BIHJ": int(df["strict_metadata_id"].nunique()),
        "assays": int(len(df)),
        "records_per_woman_distribution": distribution(person_sizes),
        "draws_per_woman_distribution": distribution(person_draws),
        "multitest_draw_groups": int((draw_sizes > 1).sum()),
        "multitest_rows": int(draw_sizes[draw_sizes > 1].sum()),
        "extra_assay_rows": int((draw_sizes[draw_sizes > 1] - 1).sum()),
        "multitest_group_size_distribution": distribution(draw_sizes[draw_sizes > 1]),
        "multitest_details": multi_draw_details,
        "draws_with_varying_gestational_age": varying_gest_draws,
        "exact_duplicate_rows_ignoring_serial": int(source.duplicated([c for c in COLS if c != "序号"]).sum()),
    },
    "gestational_age": {
        "parsed": int(df["gest_days"].notna().sum()),
        "unparsed": int(df["gest_days"].isna().sum()),
        "range_days": [int(df["gest_days"].min()), int(df["gest_days"].max())],
        "range_weeks": [float(df["gest_weeks"].min()), float(df["gest_weeks"].max())],
        "outside_10w0_25w0_rows": int(df["outside_10w0_25w0"].sum()),
        "outside_10w0_25w0_women": int(df.loc[df["outside_10w0_25w0"], "孕妇代码"].nunique()),
        "outside_10w0_25w6_rows": int(df["outside_10w0_25w6"].sum()),
        "support_by_floor_week": gest_support,
    },
    "date_consistency": {
        "lmp_usable": int(df["lmp_date"].notna().sum()),
        "lmp_unusable": int(df["lmp_date"].isna().sum()),
        "detection_dates_usable": int(df["detection_date"].notna().sum()),
        "exact_delta_rows": int((df["date_gest_delta_days"] == 0).sum()),
        "negative_delta_rows": int(df["date_delta_negative"].sum()),
        "abs_delta_gt21_rows": int(df["date_delta_abs_gt21"].sum()),
        "delta_min": float(df["date_gest_delta_days"].min()),
        "delta_median": float(df["date_gest_delta_days"].median()),
        "delta_max": float(df["date_gest_delta_days"].max()),
        "severe_serials": [int(x) for x in df.loc[df["date_delta_abs_gt21"], "序号"].tolist()],
    },
    "bmi_consistency": {
        "complete_triplets": int(df[["身高", "体重", "孕妇BMI"]].notna().all(axis=1).sum()),
        "max_abs_error": float(df["bmi_abs_error"].max()),
        "abs_error_gt_0_01": int(df["bmi_formula_abs_gt_0_01"].sum()),
        "within_person_variation": within_person_variation,
    },
    "logical_ranges": {
        "age": numeric_summary(df, "年龄"),
        "height_cm": numeric_summary(df, "身高"),
        "weight_kg": numeric_summary(df, "体重"),
        "bmi": numeric_summary(df, "孕妇BMI"),
        "y_concentration": numeric_summary(df, "Y染色体浓度"),
        "ratio_checks": ratio_checks,
        "x_concentration_negative_rows_allowed_by_appendix": int((df["X染色体浓度"] < 0).sum()),
        "raw_reads_nonpositive": int((L <= 0).sum()),
        "unique_reads_nonpositive": int((O <= 0).sum()),
        "unique_reads_gt_raw": int(df["unique_reads_gt_raw"].sum()),
    },
    "sequencing_quality": {
        "global_gc_bands": {
            "below_38pct": int((df["GC含量"] < 0.38).sum()),
            "38_to_39pct": int(((df["GC含量"] >= 0.38) & (df["GC含量"] < 0.39)).sum()),
            "39_to_40pct": int(((df["GC含量"] >= 0.39) & (df["GC含量"] < 0.40)).sum()),
            "40_to_60pct": int(df["GC含量"].between(0.40, 0.60).sum()),
            "above_60pct": int((df["GC含量"] > 0.60).sum()),
        },
        "filter_ratio_p99": filter_p99,
        "filter_ratio_above_p99_rows": int(df["filter_ratio_above_p99"].sum()),
        "empirical_unique_read_relation": {
            "formula": "O ~= L*M*(1-N)*(1-AA)",
            "serial_1_682_ok_within2": int(((df["序号"] < 683) & (df["unique_read_formula_abs_error"] <= 2)).sum()),
            "serial_1_682_total": int((df["序号"] < 683).sum()),
            "serial_683_1082_ok_within2": int(((df["序号"] >= 683) & (df["unique_read_formula_abs_error"] <= 2)).sum()),
            "serial_683_1082_total": int((df["序号"] >= 683).sum()),
        },
    },
    "statistical_outliers": outliers,
    "cohorts": cohorts,
    "support_domain": support,
    "main_support_cell_counts": support_cell_counts,
    "main_support_cross_table_rows": {
        str(index): {str(column): int(value) for column, value in row.items()}
        for index, row in main_cross_rows.to_dict(orient="index").items()
    },
    "main_support_cross_table_women": {
        str(index): {str(column): int(value) for column, value in row.items()}
        for index, row in main_cross_women.to_dict(orient="index").items()
    },
    "flag_counts": flag_counts,
    "action_counts": action_counts,
    "findings": findings,
    "modelability_without_fitting": {
        "continuous_outcome_inside_open_unit_interval": bool(df["Y染色体浓度"].between(0, 1, inclusive="neither").all()),
        "y_zero_rows": int((df["Y染色体浓度"] == 0).sum()),
        "y_one_rows": int((df["Y染色体浓度"] == 1).sum()),
        "y_skewness": float(df["Y染色体浓度"].skew()),
        "primary_rows": int(df["primary_include"].sum()),
        "primary_women": int(df.loc[df["primary_include"], "孕妇代码"].nunique()),
        "primary_draws": int(df.loc[df["primary_include"], "draw_id"].nunique()),
        "hard_core_invalid_rows": int(df["core_missing_or_invalid"].sum()),
        "warning": "Nominal row count is not the independent sample size; inference must respect woman/draw nesting and the 683 batch split.",
    },
    "lists": {
        "hard_exclude": {
            "rows": int((df["audit_action"] == "HARD_EXCLUDE").sum()),
            "rule": "Missing/invalid B, J, K or V; V outside (0,1); BMI<=0.",
        },
        "exclude_from_recommended_primary_keep_for_sensitivity": {
            "rows": int((df["audit_action"] == "EXCLUDE_PRIMARY_KEEP_SENSITIVITY").sum()),
            "rule": "序号>=683 or J outside 10w0d-25w6d; retain in separate sensitivity cohorts.",
            "batch_only_rows": int((df["batch683"] & ~df["outside_10w0_25w6"]).sum()),
            "window_only_rows": int((~df["batch683"] & df["outside_10w0_25w6"]).sum()),
            "both_batch_and_window_rows": int((df["batch683"] & df["outside_10w0_25w6"]).sum()),
        },
        "mark_only": {
            "rule": "Repeated assays, unusable LMP/date inconsistency, GC boundary, sequencing-count flags, Tukey outliers and V<4% are retained with explicit flags.",
            "rows_with_any_mark": int(df["mark_only_reasons"].ne("").sum()),
        },
        "keep_primary": {
            "rows": int(df["primary_include"].sum()),
            "women": int(df.loc[df["primary_include"], "孕妇代码"].nunique()),
            "draws": int(df.loc[df["primary_include"], "draw_id"].nunique()),
        },
    },
}


flag_columns = list(dict.fromkeys([
    "序号", "row_id", "孕妇代码", "检测抽血次数", "draw_id", "draw_assay_count", "date_session_id", "assay_session_id",
    "assay_session_record_count", "strict_metadata_id", "检测孕周", "gest_days", "gest_weeks", "孕妇BMI",
    "bmi_calculated", "bmi_abs_error", "Y染色体浓度", "date_gest_delta_days", "batch683", "storage687", "primary_include",
    "sensitivity_through_25w0_include",
    *flag_definitions.keys(), "same_draw_multitest", "same_session_multirecord", "draw_gestation_conflict",
    "primary_exclusion_reasons", "mark_only_reasons", "audit_action",
]))
df[flag_columns].to_csv(OUT / "q1_row_flags.csv", index=False, encoding="utf-8-sig")
with open(OUT / "q1_audit_summary.json", "w", encoding="utf-8") as stream:
    json.dump(summary, stream, ensure_ascii=False, indent=2, default=js)


# Diagnostic-only plots; no relationship model or fitted smooth is produced.
try:
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, axes = plt.subplots(2, 1, figsize=(11, 8), constrained_layout=True)
    axes[0].scatter(df["序号"], np.log10(df["unique_read_formula_abs_error"] + 1), s=9, alpha=0.65, c=np.where(df["batch683"], "#C2410C", "#2563EB"))
    axes[0].axvline(683, color="black", linestyle="--", linewidth=1.2)
    axes[0].set_ylabel("log10(|O-经验关系|+1)")
    axes[0].set_title("男胎数据序号683处的读段关系断点（诊断图）")
    axes[1].scatter(df["序号"], df["Y染色体浓度"] * 100, s=9, alpha=0.65, c=np.where(df["batch683"], "#C2410C", "#2563EB"))
    axes[1].axvline(683, color="black", linestyle="--", linewidth=1.2)
    axes[1].axhline(4, color="#666666", linestyle=":", linewidth=1)
    axes[1].set_xlabel("序号")
    axes[1].set_ylabel("Y浓度 (%)")
    fig.savefig(OUT / "q1_batch_break_diagnostic.svg", format="svg")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    colors = np.where(df["batch683"], "#C2410C", "#2563EB")
    ax.scatter(df["gest_weeks"], df["孕妇BMI"], s=12, alpha=0.45, c=colors)
    ax.axvline(25, color="#666666", linestyle="--", linewidth=1)
    ax.set_xlabel("检测孕周（连续周）")
    ax.set_ylabel("BMI (kg/m²)")
    ax.set_title("孕周-BMI支持域：蓝=序号1-682，橙=序号683以后")
    fig.savefig(OUT / "q1_support_domain_diagnostic.svg", format="svg")
    plt.close(fig)
except Exception as exc:
    summary["plot_warning"] = repr(exc)
    with open(OUT / "q1_audit_summary.json", "w", encoding="utf-8") as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2, default=js)

print(json.dumps({
    "summary_path": str(OUT / "q1_audit_summary.json"),
    "flags_path": str(OUT / "q1_row_flags.csv"),
    "rows": len(df),
    "actions": action_counts,
    "primary": cohorts["pre683_clinical_10w0_25w6_primary"],
    "findings": findings,
}, ensure_ascii=False, indent=2, default=js))
