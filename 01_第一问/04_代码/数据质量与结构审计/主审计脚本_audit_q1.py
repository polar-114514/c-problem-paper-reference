from __future__ import annotations

import argparse
import itertools
import json
import math
import re
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = SCRIPT_DIR.parents[2]
parser = argparse.ArgumentParser(description="第一问男胎数据质量与结构审计（不拟合正式关系模型）")
parser.add_argument(
    "--source-json",
    type=Path,
    default=WORKSPACE_ROOT / "99_临时中转" / "第一问数据审计复现" / "source_snapshot" / "01_男胎检测数据.json",
    help="只读提取脚本生成的男胎工作表JSON快照",
)
parser.add_argument(
    "--output",
    type=Path,
    default=WORKSPACE_ROOT / "99_临时中转" / "第一问数据审计复现" / "outputs",
    help="审计结果输出目录",
)
args = parser.parse_args()
SOURCE_JSON = args.source_json.resolve()
OUTPUT = args.output.resolve()
OUTPUT.mkdir(parents=True, exist_ok=True)

COLS = [
    "seq",
    "woman_id",
    "age",
    "height_cm",
    "weight_kg",
    "lmp_raw",
    "conception",
    "test_date_raw",
    "draw_no",
    "gest_raw",
    "bmi",
    "reads_total",
    "align_rate",
    "duplicate_rate",
    "unique_reads",
    "gc_rate",
    "z13",
    "z18",
    "z21",
    "zx",
    "zy",
    "y_conc",
    "x_conc",
    "gc13",
    "gc18",
    "gc21",
    "filtered_rate",
    "aneuploidy",
    "gravidity",
    "parity",
    "healthy",
]

CN = {
    "seq": "序号",
    "woman_id": "孕妇代码",
    "age": "年龄",
    "height_cm": "身高(cm)",
    "weight_kg": "体重(kg)",
    "lmp_raw": "末次月经",
    "conception": "受孕方式",
    "test_date_raw": "检测日期",
    "draw_no": "检测抽血次数",
    "gest_raw": "检测孕周",
    "bmi": "BMI(kg/m²)",
    "reads_total": "原始读段数",
    "align_rate": "比对率",
    "duplicate_rate": "重复读段率",
    "unique_reads": "唯一比对读段数",
    "gc_rate": "GC含量",
    "z13": "13号染色体Z值",
    "z18": "18号染色体Z值",
    "z21": "21号染色体Z值",
    "zx": "X染色体Z值",
    "zy": "Y染色体Z值",
    "y_conc": "Y染色体浓度",
    "x_conc": "X染色体浓度",
    "gc13": "13号染色体GC含量",
    "gc18": "18号染色体GC含量",
    "gc21": "21号染色体GC含量",
    "filtered_rate": "过滤读段率",
    "aneuploidy": "非整倍体结果",
    "gravidity": "怀孕次数",
    "parity": "生产次数",
    "healthy": "胎儿是否健康",
}

NUMERIC = [
    "seq",
    "age",
    "height_cm",
    "weight_kg",
    "draw_no",
    "bmi",
    "reads_total",
    "align_rate",
    "duplicate_rate",
    "unique_reads",
    "gc_rate",
    "z13",
    "z18",
    "z21",
    "zx",
    "zy",
    "y_conc",
    "x_conc",
    "gc13",
    "gc18",
    "gc21",
    "filtered_rate",
    "parity",
]


def is_blank(value) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    return isinstance(value, str) and value.strip() == ""


def parse_gestation(value):
    if is_blank(value):
        return np.nan, np.nan, "blank"
    text = str(value).strip()
    match = re.fullmatch(r"(?i)(\d+)\s*[w周]\s*(?:\+\s*(\d+))?", text)
    if not match:
        return np.nan, np.nan, "invalid"
    weeks = int(match.group(1))
    days = int(match.group(2) or 0)
    if days < 0 or days > 6:
        return np.nan, np.nan, "invalid_day"
    return weeks + days / 7.0, weeks * 7 + days, "ok"


def parse_date(value):
    if is_blank(value):
        return pd.NaT
    if isinstance(value, (int, float, np.integer, np.floating)):
        number = int(value)
        if 19000101 <= number <= 21001231:
            try:
                return pd.Timestamp(datetime.strptime(str(number), "%Y%m%d"))
            except ValueError:
                return pd.NaT
        if 1 <= float(value) <= 100000:
            return pd.Timestamp(datetime(1899, 12, 30) + timedelta(days=float(value)))
        return pd.NaT
    text = str(value).strip()
    if re.fullmatch(r"\d{8}", text):
        try:
            return pd.Timestamp(datetime.strptime(text, "%Y%m%d"))
        except ValueError:
            return pd.NaT
    try:
        stamp = pd.to_datetime(text, errors="raise")
        if getattr(stamp, "tzinfo", None) is not None:
            stamp = stamp.tz_localize(None)
        return pd.Timestamp(stamp).normalize()
    except Exception:
        return pd.NaT


def quantile_summary(series: pd.Series):
    values = pd.to_numeric(series, errors="coerce").dropna().astype(float)
    if values.empty:
        return {"n": 0}
    return {
        "n": int(values.size),
        "min": float(values.min()),
        "q1": float(values.quantile(0.25)),
        "median": float(values.median()),
        "mean": float(values.mean()),
        "q3": float(values.quantile(0.75)),
        "max": float(values.max()),
        "sd": float(values.std(ddof=1)) if values.size > 1 else 0.0,
    }


def sample_count(frame: pd.DataFrame, mask):
    part = frame.loc[mask]
    return {
        "records": int(len(part)),
        "draws": int(part["draw_id"].nunique()),
        "women": int(part["woman_id"].nunique()),
    }


def correlation(x, y, method="pearson"):
    pair = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(pair) < 3 or pair["x"].nunique() < 2 or pair["y"].nunique() < 2:
        return {"n": int(len(pair)), "r": None, "p": None}
    if method == "spearman":
        result = stats.spearmanr(pair["x"], pair["y"])
        return {"n": int(len(pair)), "r": float(result.statistic), "p": float(result.pvalue)}
    result = stats.pearsonr(pair["x"], pair["y"])
    return {"n": int(len(pair)), "r": float(result.statistic), "p": float(result.pvalue)}


def json_ready(value):
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if np.isnan(value) else float(value)
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if pd.isna(value):
        return None
    return value


def markdown_table(headers, rows):
    def esc(value):
        if value is None:
            return ""
        return str(value).replace("|", "\\|").replace("\n", " ")

    lines = ["| " + " | ".join(map(esc, headers)) + " |"]
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")
    lines.extend("| " + " | ".join(map(esc, row)) + " |" for row in rows)
    return "\n".join(lines)


with SOURCE_JSON.open("r", encoding="utf-8") as handle:
    source = json.load(handle)

raw_headers = source["values"][0]
raw_rows = source["values"][1:]
raw_types = source["types"][1:]
if len(raw_headers) != 31 or len(raw_rows) != 1082:
    raise RuntimeError(f"unexpected male sheet shape: headers={len(raw_headers)}, rows={len(raw_rows)}")

df = pd.DataFrame(raw_rows, columns=COLS)
type_df = pd.DataFrame(raw_types, columns=COLS)
for col in NUMERIC:
    df[col] = pd.to_numeric(df[col], errors="coerce")

gest = df["gest_raw"].apply(parse_gestation)
df["gest_week"] = [item[0] for item in gest]
df["gest_days"] = [item[1] for item in gest]
df["gest_parse_status"] = [item[2] for item in gest]
df["lmp_date"] = df["lmp_raw"].apply(parse_date)
df["test_date"] = df["test_date_raw"].apply(parse_date)
df["row_id"] = df["seq"].astype("Int64").astype(str)
df["assay_id"] = df["row_id"]  # 向后兼容：一行即一条检测记录
df["draw_id"] = df["woman_id"].astype(str) + "|" + df["draw_no"].astype("Int64").astype(str)
df["test_date_key"] = df["test_date"].dt.strftime("%Y-%m-%d").fillna("MISSING_DATE")
df["date_session_id"] = df["woman_id"].astype(str) + "|" + df["test_date_key"]
df["assay_session_id"] = df["draw_id"] + "|" + df["test_date_key"]
df["strict_metadata_id"] = df["assay_session_id"] + "|" + df["gest_days"].astype("Int64").astype(str)
df["batch683"] = np.where(df["seq"] >= 683, "683后", "683前")
df["clinical_10_25w6"] = (df["gest_week"] >= 10) & (df["gest_week"] < 26)
df["clinical_through_25w0"] = (df["gest_week"] >= 10) & (df["gest_week"] <= 25)

header_audit = {
    "raw": raw_headers,
    "trimmed": [str(value).strip() if value is not None else None for value in raw_headers],
    "trailing_space_columns": [
        {"position": index + 1, "header": value}
        for index, value in enumerate(raw_headers)
        if isinstance(value, str) and value != value.strip()
    ],
    "duplicate_trimmed_headers": [
        key for key, count in Counter(str(value).strip() for value in raw_headers if value is not None).items() if count > 1
    ],
}

missing_rows = []
for col in COLS:
    count = int(df[col].apply(is_blank).sum())
    missing_rows.append(
        {"column": col, "label": CN[col], "missing": count, "pct": count / len(df)}
    )
missing_df = pd.DataFrame(missing_rows)
missing_df.to_csv(OUTPUT / "missingness.csv", index=False, encoding="utf-8-sig")

storage_type_counts = {
    col: {key: int(value) for key, value in type_df[col].value_counts(dropna=False).items()}
    for col in COLS
}


def date_storage_class(value):
    if is_blank(value):
        return "blank"
    if isinstance(value, str):
        return "iso_string"
    if isinstance(value, (int, float, np.integer, np.floating)):
        return "yyyymmdd_integer" if int(value) >= 19000101 else "excel_serial"
    return type(value).__name__


df["lmp_storage_class"] = df["lmp_raw"].apply(date_storage_class)
df["test_date_storage_class"] = df["test_date_raw"].apply(date_storage_class)

basic = {
    "records": int(len(df)),
    "women": int(df["woman_id"].nunique()),
    "draws": int(df["draw_id"].nunique()),
    "columns": int(len(df.columns)),
    "seq_unique": bool(df["seq"].is_unique),
    "seq_consecutive": bool(np.array_equal(df["seq"].to_numpy(dtype=int), np.arange(1, len(df) + 1))),
    "woman_code_pattern_failures": int((~df["woman_id"].astype(str).str.fullmatch(r"A\d{3}")).sum()),
    "exact_duplicates_excluding_seq": int(df.drop(columns=["seq", "row_id", "assay_id"]).astype(str).duplicated().sum()),
    "gestation_parse_failures": int((df["gest_parse_status"] != "ok").sum()),
    "core_missing": {
        "gestation": int(df["gest_week"].isna().sum()),
        "bmi": int(df["bmi"].isna().sum()),
        "y_conc": int(df["y_conc"].isna().sum()),
    },
}

samples = {
    "全男胎": sample_count(df, pd.Series(True, index=df.index)),
    "题干窗口10w0d-25w6d": sample_count(df, df["clinical_10_25w6"]),
    "至25w0d敏感性": sample_count(df, df["clinical_through_25w0"]),
    "683前全部": sample_count(df, df["batch683"] == "683前"),
    "683前题干窗口": sample_count(df, (df["batch683"] == "683前") & df["clinical_10_25w6"]),
    "683后全部": sample_count(df, df["batch683"] == "683后"),
    "683后题干窗口": sample_count(df, (df["batch683"] == "683后") & df["clinical_10_25w6"]),
}
pd.DataFrame([{"sample": key, **value} for key, value in samples.items()]).to_csv(
    OUTPUT / "sample_counts.csv", index=False, encoding="utf-8-sig"
)


def grouped_key_counts(frame: pd.DataFrame, key: str):
    sizes = frame.groupby(key, dropna=False).size()
    repeated = sizes[sizes > 1]
    return {
        "events": int(len(sizes)),
        "multi_groups": int(len(repeated)),
        "multi_rows": int(repeated.sum()),
        "size_distribution": {str(int(size)): int(count) for size, count in sizes.value_counts().sort_index().items()},
    }


def grouped_repeat_metrics(frame: pd.DataFrame, key: str):
    grouped = [(group_id, group) for group_id, group in frame.groupby(key, sort=False) if len(group) > 1]
    denominator = sum(len(group) - 1 for _, group in grouped)
    pooled_sd = math.sqrt(
        sum((len(group) - 1) * float(group["y_conc"].var(ddof=1)) for _, group in grouped) / denominator
    ) if denominator else 0.0
    return {
        "pooled_within_y_sd": float(pooled_sd),
        "groups_crossing_4pct": int(sum(
            (group["y_conc"].min() < 0.04) and (group["y_conc"].max() >= 0.04)
            for _, group in grouped
        )),
    }


event_scopes = {
    "全男胎1082条": pd.Series(True, index=df.index),
    "683前682条": df["seq"] < 683,
    "主参考674条": (df["seq"] < 683) & df["clinical_10_25w6"],
    "敏感性670条": (df["seq"] < 683) & df["clinical_through_25w0"],
}
event_keys = {
    "B+I": ("draw_id", "孕妇代码＋检测抽血次数", "抽血事件主口径"),
    "B+H": ("date_session_id", "孕妇代码＋检测日期", "检测日期会话；不是抽血事件"),
    "B+I+H": ("assay_session_id", "孕妇代码＋检测抽血次数＋检测日期", "同次抽血的检测会话"),
    "B+I+H+J": ("strict_metadata_id", "再加记录孕周", "严格元数据一致组；只作冲突诊断"),
}
event_hierarchy_rows = []
event_hierarchy_audit = {}
for scope_name, scope_mask in event_scopes.items():
    part = df.loc[scope_mask]
    event_hierarchy_audit[scope_name] = {}
    for notation, (key, definition, role) in event_keys.items():
        counts = {**grouped_key_counts(part, key), **grouped_repeat_metrics(part, key)}
        event_hierarchy_audit[scope_name][notation] = counts
        event_hierarchy_rows.append({
            "scope": scope_name,
            "key_notation": notation,
            "definition": definition,
            "recommended_role": role,
            "records": int(len(part)),
            "unique_events": counts["events"],
            "multi_groups": counts["multi_groups"],
            "multi_rows": counts["multi_rows"],
            "pooled_within_y_sd": counts["pooled_within_y_sd"],
            "groups_crossing_4pct": counts["groups_crossing_4pct"],
            "size_distribution": json.dumps(counts["size_distribution"], ensure_ascii=False, sort_keys=True),
        })
pd.DataFrame(event_hierarchy_rows).to_csv(
    OUTPUT / "event_hierarchy_counts.csv", index=False, encoding="utf-8-sig"
)

record_count_distribution = {
    str(int(key)): int(value)
    for key, value in df.groupby("woman_id").size().value_counts().sort_index().items()
}
draw_frame = (
    df.groupby("draw_id", as_index=False)
    .agg(
        woman_id=("woman_id", "first"),
        draw_no=("draw_no", "first"),
        batch683=("batch683", "first"),
        gest_week=("gest_week", "first"),
        gest_week_nunique=("gest_week", "nunique"),
        bmi=("bmi", "mean"),
        y_conc=("y_conc", "mean"),
        test_date=("test_date", "min"),
        assay_records=("seq", "size"),
    )
)
# 同一抽血事件若记录了多个孕周（A055#3），不得用均值/中位数静默制造唯一孕周。
draw_frame["gest_week_conflict"] = draw_frame["gest_week_nunique"] > 1
draw_frame.loc[draw_frame["gest_week_conflict"], "gest_week"] = np.nan
draw_count_distribution = {
    str(int(key)): int(value)
    for key, value in draw_frame.groupby("woman_id").size().value_counts().sort_index().items()
}

multi_groups = df.groupby("draw_id").filter(lambda group: len(group) > 1).groupby("draw_id")
repeat_sizes = {str(int(key)): int(value) for key, value in multi_groups.size().value_counts().sort_index().items()}
repeat_ranges = multi_groups["y_conc"].agg(lambda values: float(values.max() - values.min()))
repeat_sds = multi_groups["y_conc"].std(ddof=1)
pooled_repeat_sd = math.sqrt(
    sum((len(group) - 1) * float(group["y_conc"].var(ddof=1)) for _, group in multi_groups)
    / sum(len(group) - 1 for _, group in multi_groups)
)
pairwise_y_diffs = []
for _, group in multi_groups:
    pairwise_y_diffs.extend(abs(a - b) for a, b in itertools.combinations(group["y_conc"].astype(float), 2))
flip_groups = int(
    sum((group["y_conc"].min() < 0.04) and (group["y_conc"].max() >= 0.04) for _, group in multi_groups)
)

repeat_inconsistency_fields = ["gest_days", "test_date", "weight_kg", "bmi", "age", "height_cm", "conception"]
repeat_inconsistency = []
for draw_id, group in multi_groups:
    row = {"draw_id": draw_id, "records": int(len(group)), "seqs": ",".join(map(str, group["seq"].astype(int)))}
    any_issue = False
    for field in repeat_inconsistency_fields:
        count = group[field].nunique(dropna=False)
        row[f"{field}_nunique"] = int(count)
        any_issue |= count > 1
    row["crosses_4pct"] = bool((group["y_conc"].min() < 0.04) and (group["y_conc"].max() >= 0.04))
    row["y_range"] = float(group["y_conc"].max() - group["y_conc"].min())
    row["any_metadata_inconsistency"] = bool(any_issue)
    repeat_inconsistency.append(row)
repeat_inconsistency_df = pd.DataFrame(repeat_inconsistency)
repeat_inconsistency_df.to_csv(OUTPUT / "technical_repeat_groups.csv", index=False, encoding="utf-8-sig")

repeat_audit = {
    "operational_definition": "同一孕妇代码B与检测抽血次数I；题干中一次采血可多次检测",
    "multi_record_draw_groups": int(multi_groups.ngroups),
    "records_in_multi_groups": int(sum(len(group) for _, group in multi_groups)),
    "women_with_multi_groups": int(df[df["draw_id"].isin(list(multi_groups.groups))]["woman_id"].nunique()),
    "group_size_distribution": repeat_sizes,
    "all_before_683": bool(df[df["draw_id"].isin(list(multi_groups.groups))]["seq"].max() < 683),
    "pooled_within_draw_y_sd": float(pooled_repeat_sd),
    "within_draw_y_range": quantile_summary(repeat_ranges),
    "pairwise_abs_y_diff": quantile_summary(pd.Series(pairwise_y_diffs)),
    "groups_crossing_4pct": flip_groups,
    "groups_with_different_test_dates": int((repeat_inconsistency_df["test_date_nunique"] > 1).sum()),
    "groups_with_different_gestation": int((repeat_inconsistency_df["gest_days_nunique"] > 1).sum()),
    "groups_with_different_bmi": int((repeat_inconsistency_df["bmi_nunique"] > 1).sum()),
}


def repeat_metrics(frame: pd.DataFrame, key: str):
    grouped = [(group_id, group) for group_id, group in frame.groupby(key, sort=False) if len(group) > 1]
    return {
        "multi_groups": int(len(grouped)),
        "multi_rows": int(sum(len(group) for _, group in grouped)),
        "size_distribution": {
            str(int(size)): int(count)
            for size, count in pd.Series([len(group) for _, group in grouped], dtype="int64").value_counts().sort_index().items()
        },
        **grouped_repeat_metrics(frame, key),
    }


repeat_sensitivity = {
    "B+I抽血事件主口径（含跨日复检）": {
        "definition": "孕妇代码＋检测抽血次数",
        **repeat_metrics(df, "draw_id"),
    },
    "B+I+H同日检测会话口径": {
        "definition": "孕妇代码＋检测抽血次数＋检测日期；仅识别同日重复，是主口径严格子集",
        **repeat_metrics(df, "assay_session_id"),
    },
    "B+I+H+J严格元数据口径": {
        "definition": "再要求记录孕周相同；只用于发现A055元数据冲突，不用于拆分抽血",
        **repeat_metrics(df, "strict_metadata_id"),
    },
}

df["date_gest_error_days"] = (df["test_date"] - df["lmp_date"]).dt.days - df["gest_days"]
date_error = df["date_gest_error_days"].dropna()
date_outliers = df.loc[df["date_gest_error_days"].abs() > 14, [
    "seq", "woman_id", "draw_no", "gest_raw", "lmp_raw", "test_date_raw", "date_gest_error_days", "batch683"
]].copy()
date_outliers.to_csv(OUTPUT / "date_gestation_outliers_gt14d.csv", index=False, encoding="utf-8-sig")

progression_rows = []
for woman, group in draw_frame.groupby("woman_id"):
    group = group.sort_values(["draw_no", "gest_week"])
    previous = None
    for _, row in group.iterrows():
        if previous is not None:
            date_delta = (row["test_date"] - previous["test_date"]).days if pd.notna(row["test_date"]) and pd.notna(previous["test_date"]) else np.nan
            gest_delta = (row["gest_week"] - previous["gest_week"]) * 7
            progression_rows.append({
                "woman_id": woman,
                "from_draw": previous["draw_no"],
                "to_draw": row["draw_no"],
                "date_delta_days": date_delta,
                "gest_delta_days": gest_delta,
                "delta_mismatch_days": date_delta - gest_delta if pd.notna(date_delta) else np.nan,
                "gestation_nonincreasing": bool(gest_delta <= 0),
                "date_nonincreasing": bool(pd.notna(date_delta) and date_delta <= 0),
            })
        previous = row
progression_df = pd.DataFrame(progression_rows)
progression_df.to_csv(OUTPUT / "longitudinal_progression_checks.csv", index=False, encoding="utf-8-sig")

draw_gaps = []
for woman, group in draw_frame.groupby("woman_id"):
    numbers = sorted(set(int(value) for value in group["draw_no"].dropna()))
    expected = list(range(min(numbers), max(numbers) + 1)) if numbers else []
    if numbers != expected:
        draw_gaps.append({"woman_id": woman, "observed": ",".join(map(str, numbers)), "expected": ",".join(map(str, expected))})
pd.DataFrame(draw_gaps).to_csv(OUTPUT / "draw_number_gaps.csv", index=False, encoding="utf-8-sig")

date_audit = {
    "parsed_lmp": int(df["lmp_date"].notna().sum()),
    "parsed_test_date": int(df["test_date"].notna().sum()),
    "date_gestation_error": quantile_summary(date_error),
    "abs_error_gt7": int((date_error.abs() > 7).sum()),
    "abs_error_gt14": int((date_error.abs() > 14).sum()),
    "by_batch": {
        batch: quantile_summary(group["date_gest_error_days"].dropna())
        for batch, group in df.groupby("batch683")
    },
    "gestation_nonincreasing_draw_transitions": int(progression_df["gestation_nonincreasing"].sum()),
    "date_nonincreasing_draw_transitions": int(progression_df["date_nonincreasing"].sum()),
    "transition_abs_mismatch_gt14": int((progression_df["delta_mismatch_days"].abs() > 14).sum()),
    "women_with_draw_number_gaps": int(len(draw_gaps)),
    "first_lmp_iso_string_seq": int(df.loc[df["lmp_storage_class"] == "iso_string", "seq"].min()) if (df["lmp_storage_class"] == "iso_string").any() else None,
    "first_test_date_excel_serial_seq": int(df.loc[df["test_date_storage_class"] == "excel_serial", "seq"].min()) if (df["test_date_storage_class"] == "excel_serial").any() else None,
    "lmp_storage_types": storage_type_counts["lmp_raw"],
    "lmp_storage_classes": {key: int(value) for key, value in df["lmp_storage_class"].value_counts().items()},
    "test_date_storage_classes": {key: int(value) for key, value in df["test_date_storage_class"].value_counts().items()},
}

df["bmi_calc"] = df["weight_kg"] / (df["height_cm"] / 100.0) ** 2
df["bmi_formula_abs_error"] = (df["bmi"] - df["bmi_calc"]).abs()
within_bmi_range = df.groupby("woman_id")["bmi"].agg(lambda values: float(values.max() - values.min()))
bin_edges = [-np.inf, 20, 28, 32, 36, 40, np.inf]
df["bmi_example_bin"] = pd.cut(df["bmi"], bins=bin_edges, right=False)
women_cross_bins = int((df.groupby("woman_id")["bmi_example_bin"].nunique() > 1).sum())

woman_stability = {}
for field in ["age", "height_cm", "lmp_date", "conception", "gravidity", "parity", "healthy"]:
    woman_stability[field] = int((df.groupby("woman_id")[field].nunique(dropna=False) > 1).sum())

body_corr = df[["height_cm", "weight_kg", "bmi"]].corr(method="pearson")
try:
    body_vif_values = np.diag(np.linalg.inv(body_corr.to_numpy()))
    body_vif = {column: float(value) for column, value in zip(body_corr.columns, body_vif_values)}
except np.linalg.LinAlgError:
    body_vif = {column: None for column in body_corr.columns}

body_audit = {
    "age": quantile_summary(df["age"]),
    "height_cm": quantile_summary(df["height_cm"]),
    "weight_kg": quantile_summary(df["weight_kg"]),
    "bmi": quantile_summary(df["bmi"]),
    "bmi_formula_abs_error": quantile_summary(df["bmi_formula_abs_error"]),
    "within_woman_bmi_range": quantile_summary(within_bmi_range),
    "women_crossing_example_bmi_bins": women_cross_bins,
    "woman_level_inconsistency_counts": woman_stability,
    "body_correlations": body_corr.to_dict(),
    "body_vif": body_vif,
}

rate_columns = ["align_rate", "duplicate_rate", "gc_rate", "gc13", "gc18", "gc21", "filtered_rate"]
quality_ranges = {col: quantile_summary(df[col]) for col in ["reads_total", "unique_reads", *rate_columns]}
quality_bounds = {
    col: int(((df[col] < 0) | (df[col] > 1)).sum()) for col in rate_columns
}
df["expected_unique_reads"] = (
    df["reads_total"] * df["align_rate"] * (1 - df["duplicate_rate"]) * (1 - df["filtered_rate"])
)
df["unique_read_relation_residual"] = df["unique_reads"] - df["expected_unique_reads"]
df["unique_read_relation_match_2"] = df["unique_read_relation_residual"].abs() <= 2
df["unique_reads_gt_total"] = df["unique_reads"] > df["reads_total"]

read_relation_by_batch = {}
for batch, group in df.groupby("batch683"):
    read_relation_by_batch[batch] = {
        "records": int(len(group)),
        "match_within_2_reads": int(group["unique_read_relation_match_2"].sum()),
        "match_rate": float(group["unique_read_relation_match_2"].mean()),
        "residual": quantile_summary(group["unique_read_relation_residual"]),
        "unique_reads_gt_total": int(group["unique_reads_gt_total"].sum()),
    }

quality_audit = {
    "ranges": quality_ranges,
    "rate_values_outside_0_1": quality_bounds,
    "nonpositive_total_reads": int((df["reads_total"] <= 0).sum()),
    "nonpositive_unique_reads": int((df["unique_reads"] <= 0).sum()),
    "gc_outside_40_60pct": int(((df["gc_rate"] < 0.40) | (df["gc_rate"] > 0.60)).sum()),
    "unique_reads_gt_total": int(df["unique_reads_gt_total"].sum()),
    "read_relation_by_batch": read_relation_by_batch,
}

def centered_relation(group, x="bmi", y="y_conc"):
    x_centered = group[x] - group.groupby("woman_id")[x].transform("mean")
    y_centered = group[y] - group.groupby("woman_id")[y].transform("mean")
    valid = x_centered.notna() & y_centered.notna()
    xv = x_centered[valid].to_numpy(dtype=float)
    yv = y_centered[valid].to_numpy(dtype=float)
    if np.dot(xv, xv) == 0:
        return {"n": int(valid.sum()), "slope": None, "r2": None, "residual_sd": None}
    slope = float(np.dot(xv, yv) / np.dot(xv, xv))
    residual = yv - slope * xv
    sst = float(np.dot(yv, yv))
    return {
        "n": int(valid.sum()),
        "slope": slope,
        "r2": float(1 - np.dot(residual, residual) / sst) if sst else None,
        "residual_sd": float(np.std(residual, ddof=1)) if len(residual) > 1 else 0.0,
        "max_abs_residual": float(np.max(np.abs(residual))) if len(residual) else None,
    }


def monotonic_women(group):
    counts = {"women": 0, "y_nondecreasing": 0, "bmi_nondecreasing": 0, "both_nondecreasing": 0}
    slopes = []
    for _, woman_group in group.groupby("woman_id"):
        trajectory = (
            woman_group.groupby("draw_id", as_index=False)
            .agg(
                gest_week=("gest_week", "first"),
                gest_week_nunique=("gest_week", "nunique"),
                y_conc=("y_conc", "mean"),
                bmi=("bmi", "mean"),
            )
            .loc[lambda frame: frame["gest_week_nunique"] == 1]
            .sort_values("gest_week")
        )
        if len(trajectory) < 2:
            continue
        counts["women"] += 1
        y_mon = bool(np.all(np.diff(trajectory["y_conc"].to_numpy()) >= -1e-12))
        bmi_mon = bool(np.all(np.diff(trajectory["bmi"].to_numpy()) >= -1e-12))
        counts["y_nondecreasing"] += int(y_mon)
        counts["bmi_nondecreasing"] += int(bmi_mon)
        counts["both_nondecreasing"] += int(y_mon and bmi_mon)
        if trajectory["bmi"].nunique() > 1:
            slopes.append(float(np.polyfit(trajectory["bmi"], trajectory["y_conc"], 1)[0]))
    counts["y_nondecreasing_rate"] = counts["y_nondecreasing"] / counts["women"] if counts["women"] else None
    counts["bmi_nondecreasing_rate"] = counts["bmi_nondecreasing"] / counts["women"] if counts["women"] else None
    counts["slope_summary"] = quantile_summary(pd.Series(slopes))
    return counts


batch_rows = []
batch_audit = {}
for batch, group in df.groupby("batch683"):
    batch_row = {
        "batch": batch,
        "records": int(len(group)),
        "women": int(group["woman_id"].nunique()),
        "draws": int(group["draw_id"].nunique()),
        "week_mean": float(group["gest_week"].mean()),
        "week_min": float(group["gest_week"].min()),
        "week_max": float(group["gest_week"].max()),
        "bmi_mean": float(group["bmi"].mean()),
        "bmi_min": float(group["bmi"].min()),
        "bmi_max": float(group["bmi"].max()),
        "y_mean": float(group["y_conc"].mean()),
        "y_min": float(group["y_conc"].min()),
        "y_max": float(group["y_conc"].max()),
        "multi_record_draws": int((group.groupby("draw_id").size() > 1).sum()),
    }
    batch_rows.append(batch_row)
    batch_audit[batch] = {
        **batch_row,
        "records_per_woman": quantile_summary(group.groupby("woman_id").size()),
        "draws_per_woman": quantile_summary(group.groupby("woman_id")["draw_id"].nunique()),
        "centered_bmi_y_relation": centered_relation(group),
        "trajectory_monotonicity": monotonic_women(group),
        "lmp_storage_types": {key: int(value) for key, value in type_df.loc[group.index, "lmp_raw"].value_counts().items()},
    }
batch_df = pd.DataFrame(batch_rows)
batch_df.to_csv(OUTPUT / "batch_comparison.csv", index=False, encoding="utf-8-sig")
batch_audit["woman_overlap"] = int(
    len(set(df.loc[df["batch683"] == "683前", "woman_id"]) & set(df.loc[df["batch683"] == "683后", "woman_id"]))
)
post_counts = df.loc[df["batch683"] == "683后"].groupby("woman_id").size()
batch_audit["post_women_exactly_4_records"] = int((post_counts == 4).sum())
batch_audit["post_women_total"] = int(len(post_counts))

correlation_rows = []
correlation_summary = {}
for batch_label, group in [("全部", df), ("683前", df[df["batch683"] == "683前"]), ("683后", df[df["batch683"] == "683后"])]:
    correlation_summary[batch_label] = {"raw": {}, "within": {}, "between": {}}
    for variable in ["gest_week", "bmi", "weight_kg", "height_cm", "age", "gc_rate", "filtered_rate"]:
        pear = correlation(group[variable], group["y_conc"], "pearson")
        spear = correlation(group[variable], group["y_conc"], "spearman")
        correlation_summary[batch_label]["raw"][variable] = {"pearson": pear, "spearman": spear}
        correlation_rows.append({"batch": batch_label, "level": "raw", "variable": variable, "method": "pearson", **pear})
        correlation_rows.append({"batch": batch_label, "level": "raw", "variable": variable, "method": "spearman", **spear})

        centered_x = group[variable] - group.groupby("woman_id")[variable].transform("mean")
        centered_y = group["y_conc"] - group.groupby("woman_id")["y_conc"].transform("mean")
        within = correlation(centered_x, centered_y, "pearson")
        correlation_summary[batch_label]["within"][variable] = within
        correlation_rows.append({"batch": batch_label, "level": "within", "variable": variable, "method": "pearson", **within})

        means = group.groupby("woman_id")[[variable, "y_conc"]].mean(numeric_only=True)
        between = correlation(means[variable], means["y_conc"], "pearson")
        correlation_summary[batch_label]["between"][variable] = between
        correlation_rows.append({"batch": batch_label, "level": "between", "variable": variable, "method": "pearson", **between})
correlation_df = pd.DataFrame(correlation_rows)
correlation_df.to_csv(OUTPUT / "correlation_decomposition.csv", index=False, encoding="utf-8-sig")

support = {}
for batch, group in df.groupby("batch683"):
    support[batch] = {}
    for variable in ["gest_week", "bmi", "y_conc"]:
        support[batch][variable] = {
            f"q{int(q * 100):02d}": float(group[variable].quantile(q))
            for q in [0, 0.05, 0.25, 0.5, 0.75, 0.95, 1]
        }
robust_common = {}
for variable in ["gest_week", "bmi"]:
    low = max(df.loc[df["batch683"] == batch, variable].quantile(0.05) for batch in ["683前", "683后"])
    high = min(df.loc[df["batch683"] == batch, variable].quantile(0.95) for batch in ["683前", "683后"])
    robust_common[variable] = {"low": float(low), "high": float(high)}
support["robust_common_q05_q95"] = robust_common
support["records_in_common_rectangle"] = {
    batch: int(
        (
            (group["gest_week"] >= robust_common["gest_week"]["low"])
            & (group["gest_week"] <= robust_common["gest_week"]["high"])
            & (group["bmi"] >= robust_common["bmi"]["low"])
            & (group["bmi"] <= robust_common["bmi"]["high"])
        ).sum()
    )
    for batch, group in df.groupby("batch683")
}
support["dense_human_readable_common_domain"] = {
    "rule": "11<=gest_week<20 and 28<=bmi<36",
    "by_batch": {
        batch: {
            "records": int(len(group.loc[(group["gest_week"] >= 11) & (group["gest_week"] < 20) & (group["bmi"] >= 28) & (group["bmi"] < 36)])),
            "women": int(group.loc[(group["gest_week"] >= 11) & (group["gest_week"] < 20) & (group["bmi"] >= 28) & (group["bmi"] < 36), "woman_id"].nunique()),
        }
        for batch, group in df.groupby("batch683")
    },
}

first_draw = draw_frame.sort_values(["woman_id", "draw_no"]).groupby("woman_id", as_index=False).head(1).copy()
followup_audit = {
    "draw_count_distribution": draw_count_distribution,
    "record_count_distribution": record_count_distribution,
    "draw_count_vs_first_week": correlation(
        draw_frame.groupby("woman_id").size(),
        first_draw.set_index("woman_id")["gest_week"],
        "pearson",
    ),
    "first_week_single_draw_women": quantile_summary(
        first_draw.loc[first_draw["woman_id"].isin(draw_frame.groupby("woman_id").size().loc[lambda s: s == 1].index), "gest_week"]
    ),
    "first_week_multi_draw_women": quantile_summary(
        first_draw.loc[first_draw["woman_id"].isin(draw_frame.groupby("woman_id").size().loc[lambda s: s > 1].index), "gest_week"]
    ),
}

outlier_records = []
for variable in ["gest_week", "bmi", "y_conc", "age", "height_cm", "weight_kg", "reads_total", "unique_reads", "gc_rate", "filtered_rate"]:
    values = df[variable].astype(float)
    median = values.median()
    mad = (values - median).abs().median()
    if not mad or pd.isna(mad):
        continue
    score = 0.67448975 * (values - median) / mad
    flagged = df.loc[score.abs() > 5, ["seq", "woman_id", "batch683", variable]].copy()
    flagged["variable"] = variable
    flagged["robust_z"] = score.loc[flagged.index]
    flagged = flagged.rename(columns={variable: "value"})
    outlier_records.extend(flagged.to_dict("records"))
outlier_df = pd.DataFrame(outlier_records)
if not outlier_df.empty:
    outlier_df = outlier_df.sort_values("robust_z", key=lambda values: values.abs(), ascending=False)
outlier_df.to_csv(OUTPUT / "robust_univariate_flags.csv", index=False, encoding="utf-8-sig")

y_audit = {
    "distribution": quantile_summary(df["y_conc"]),
    "outside_open_0_1": int(((df["y_conc"] <= 0) | (df["y_conc"] >= 1)).sum()),
    "at_or_above_4pct": int((df["y_conc"] >= 0.04).sum()),
    "below_4pct": int((df["y_conc"] < 0.04).sum()),
    "within_0_2_percentage_points_of_4pct": int((df["y_conc"].sub(0.04).abs() <= 0.002).sum()),
    "by_batch": {batch: quantile_summary(group["y_conc"]) for batch, group in df.groupby("batch683")},
}

same_draw_conflict = repeat_inconsistency_df.loc[repeat_inconsistency_df["gest_days_nunique"] > 1]
critical_examples = {
    "same_draw_gestation_conflicts": same_draw_conflict.to_dict("records"),
    "A055_draw3_rows": df.loc[(df["woman_id"] == "A055") & (df["draw_no"] == 3), [
        "seq", "woman_id", "draw_no", "gest_raw", "test_date_raw", "weight_kg", "bmi", "y_conc"
    ]].to_dict("records"),
    "unique_reads_gt_total_first_rows": df.loc[df["unique_reads_gt_total"], [
        "seq", "woman_id", "reads_total", "unique_reads", "batch683"
    ]].head(10).to_dict("records"),
}

summary = {
    "source": {
        "sheet": source["sheetName"],
        "used_range": source["usedRange"],
        "source_rows_including_header": source["rowCount"],
        "source_columns": source["columnCount"],
        "formula_cells": int(sum(
            1 for row in source["formulas"] for value in row if isinstance(value, str) and value.startswith("=")
        )),
    },
    "header_audit": header_audit,
    "basic": basic,
    "samples": samples,
    "missingness": missing_rows,
    "storage_type_counts": storage_type_counts,
    "event_hierarchy": event_hierarchy_audit,
    "repeat_audit": repeat_audit,
    "repeat_definition_sensitivity": repeat_sensitivity,
    "date_audit": date_audit,
    "body_audit": body_audit,
    "quality_audit": quality_audit,
    "batch_audit": batch_audit,
    "correlations": correlation_summary,
    "support": support,
    "followup_audit": followup_audit,
    "y_audit": y_audit,
    "robust_outlier_counts": outlier_df["variable"].value_counts().to_dict() if not outlier_df.empty else {},
    "critical_examples": critical_examples,
}


# Candidate charts. These remain in the temporary audit area until user approval.
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
colors = {"683前": "#2563EB", "683后": "#EA580C"}

fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
for ax, variable, label, scale in [
    (axes[0], "gest_week", "检测孕周（周）", 1),
    (axes[1], "bmi", "BMI（kg/m²）", 1),
    (axes[2], "y_conc", "Y染色体浓度（%）", 100),
]:
    data = [df.loc[df["batch683"] == batch, variable].dropna().to_numpy() * scale for batch in ["683前", "683后"]]
    box = ax.boxplot(data, tick_labels=["683前", "683后"], patch_artist=True, showfliers=True)
    for patch, batch in zip(box["boxes"], ["683前", "683后"]):
        patch.set_facecolor(colors[batch])
        patch.set_alpha(0.55)
    ax.set_ylabel(label)
    ax.grid(axis="y", alpha=0.25)
fig.suptitle("图1  第683行前后主要变量分布")
fig.tight_layout()
fig.savefig(OUTPUT / "01_batch_distributions.svg", format="svg", bbox_inches="tight")
plt.close(fig)

fig, ax = plt.subplots(figsize=(8.5, 5.8))
for batch in ["683前", "683后"]:
    group = df[df["batch683"] == batch]
    ax.scatter(group["gest_week"], group["bmi"], s=16, alpha=0.38, c=colors[batch], label=f"{batch}（n={len(group)}）")
ax.add_patch(plt.Rectangle((11, 28), 9, 8, facecolor="#16A34A", edgecolor="#16A34A", alpha=0.08, linewidth=1.5, label="密集公共域候选"))
ax.set_xlabel("检测孕周（周）")
ax.set_ylabel("BMI（kg/m²）")
ax.set_title("图2  两批次孕周-BMI支持域")
ax.legend()
ax.grid(alpha=0.2)
fig.tight_layout()
fig.savefig(OUTPUT / "02_week_bmi_support.svg", format="svg", bbox_inches="tight")
plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
for ax, batch in zip(axes, ["683前", "683后"]):
    group = df[df["batch683"] == batch]
    ax.scatter(group["gest_week"], group["y_conc"] * 100, s=13, alpha=0.28, c=colors[batch])
    bins = np.arange(math.floor(group["gest_week"].min()), math.ceil(group["gest_week"].max()) + 1)
    bucket = pd.cut(group["gest_week"], bins=bins, right=False)
    med = group.assign(bucket=bucket).groupby("bucket", observed=True).agg(
        week=("gest_week", "median"), y=("y_conc", "median"), n=("seq", "size")
    )
    ax.plot(med["week"], med["y"] * 100, color="#111827", marker="o", linewidth=1.7, label="分箱中位数")
    ax.set_title(f"{batch}（{group['woman_id'].nunique()}名孕妇）")
    ax.set_xlabel("检测孕周（周）")
    ax.grid(alpha=0.2)
    ax.legend()
axes[0].set_ylabel("Y染色体浓度（%）")
fig.suptitle("图3  Y染色体浓度与孕周（按批次）")
fig.tight_layout()
fig.savefig(OUTPUT / "03_y_week_by_batch.svg", format="svg", bbox_inches="tight")
plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=False)
for ax, batch in zip(axes, ["683前", "683后"]):
    group = df[df["batch683"] == batch].copy()
    group["bmi_wc"] = group["bmi"] - group.groupby("woman_id")["bmi"].transform("mean")
    group["y_wc"] = group["y_conc"] - group.groupby("woman_id")["y_conc"].transform("mean")
    relation = batch_audit[batch]["centered_bmi_y_relation"]
    ax.scatter(group["bmi_wc"], group["y_wc"] * 100, s=14, alpha=0.32, c=colors[batch])
    low, high = group["bmi_wc"].min(), group["bmi_wc"].max()
    xline = np.linspace(low, high, 100)
    ax.plot(xline, relation["slope"] * xline * 100, color="#111827", linewidth=2)
    ax.set_xlabel("个体内BMI偏差（kg/m²）")
    ax.set_ylabel("个体内Y浓度偏差（百分点）")
    ax.set_title(f"{batch}: slope={relation['slope']:.8f}, R²={relation['r2']:.6f}")
    ax.grid(alpha=0.2)
fig.suptitle("图4  个体内BMI变化与Y浓度变化")
fig.tight_layout()
fig.savefig(OUTPUT / "04_within_bmi_y_by_batch.svg", format="svg", bbox_inches="tight")
plt.close(fig)

fig, ax = plt.subplots(figsize=(10, 5.2))
for batch in ["683前", "683后"]:
    group = df[df["batch683"] == batch]
    ax.scatter(group["seq"], group["unique_read_relation_residual"], s=12, alpha=0.45, c=colors[batch], label=batch)
ax.axvline(682.5, color="#111827", linestyle="--", linewidth=1.5, label="序号683断点")
ax.axhline(0, color="#6B7280", linewidth=1)
ax.set_yscale("symlog", linthresh=10)
ax.set_xlabel("样本序号")
ax.set_ylabel("唯一读段实际值 - 逻辑计算值（条，symlog）")
ax.set_title("图5  读段内部逻辑关系在序号683处断裂")
ax.legend()
ax.grid(alpha=0.2)
fig.tight_layout()
fig.savefig(OUTPUT / "05_read_relation_break.svg", format="svg", bbox_inches="tight")
plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), sharey=True)
for ax, key, title_text, color, line_color in [
    (axes[0], "draw_id", "同一抽血编号内多次检测", "#2563EB", "#93C5FD"),
    (axes[1], "assay_session_id", "同日检测会话内重复记录", "#EA580C", "#FDBA74"),
]:
    repeat_plot = []
    for group_id, group in df.groupby(key):
        if len(group) > 1:
            repeat_plot.append((group_id, group.sort_values("seq")["y_conc"].to_numpy() * 100, group["y_conc"].mean()))
    repeat_plot.sort(key=lambda item: item[2])
    for index, (_, values, _) in enumerate(repeat_plot, start=1):
        ax.plot([index] * len(values), values, "o", color=color, alpha=0.75, markersize=4)
        ax.plot([index, index], [values.min(), values.max()], color=line_color, linewidth=1)
    ax.axhline(4, color="#DC2626", linestyle="--", linewidth=1.3, label="4%参考线")
    ax.set_xlabel("多记录组（按组均值排序）")
    ax.set_title(f"{title_text}：{len(repeat_plot)}组")
    ax.legend()
    ax.grid(axis="y", alpha=0.2)
axes[0].set_ylabel("Y染色体浓度（%）")
fig.suptitle("图6  复测离散度的两种操作性定义")
fig.tight_layout()
fig.savefig(OUTPUT / "06_technical_repeats.svg", format="svg", bbox_inches="tight")
plt.close(fig)

fig, ax = plt.subplots(figsize=(9.5, 5.5))
bar_rows = correlation_df[
    (correlation_df["method"] == "pearson")
    & (correlation_df["variable"].isin(["gest_week", "bmi"]))
    & (correlation_df["batch"] == "全部")
]
labels = []
values = []
bar_colors = []
for variable, variable_label in [("gest_week", "孕周"), ("bmi", "BMI")]:
    for level, level_label, color in [("raw", "逐行", "#6B7280"), ("between", "个体间", "#2563EB"), ("within", "个体内", "#EA580C")]:
        row = bar_rows[(bar_rows["variable"] == variable) & (bar_rows["level"] == level)].iloc[0]
        labels.append(f"{variable_label}-{level_label}")
        values.append(row["r"])
        bar_colors.append(color)
ax.bar(labels, values, color=bar_colors)
ax.axhline(0, color="#111827", linewidth=1)
ax.set_ylabel("Pearson相关系数")
ax.set_title("图7  逐行、个体间与个体内相关并不等价")
ax.tick_params(axis="x", rotation=25)
ax.grid(axis="y", alpha=0.2)
fig.tight_layout()
fig.savefig(OUTPUT / "07_correlation_decomposition.svg", format="svg", bbox_inches="tight")
plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2))
for ax, batch in zip(axes, ["683前", "683后"]):
    values = df.loc[df["batch683"] == batch, "date_gest_error_days"].dropna()
    ax.hist(values, bins=40 if batch == "683前" else np.arange(-0.5, 1.0, 1.0), alpha=0.75, color=colors[batch])
    ax.axvline(0, color="#111827", linewidth=1)
    ax.axvline(-14, color="#DC2626", linestyle="--", linewidth=1)
    ax.axvline(14, color="#DC2626", linestyle="--", linewidth=1)
    ax.set_xlabel("日期孕周 - J列孕周（天）")
    ax.set_title(f"{batch}（n={len(values)}）")
    ax.grid(axis="y", alpha=0.2)
    if batch == "683后":
        ax.text(0, len(values) * 0.92, "400/400条差值均为0天", ha="center", va="top", fontsize=11)
axes[0].set_ylabel("记录数")
fig.suptitle("图8  日期与孕周一致性在批次间显著不同")
fig.tight_layout()
fig.savefig(OUTPUT / "08_date_gestation_consistency.svg", format="svg", bbox_inches="tight")
plt.close(fig)


issues = [
    {
        "severity": "否决性",
        "issue": "683数据生成机制结构断点",
        "evidence": f"前682行{batch_audit['683前']['women']}人，后400行{batch_audit['683后']['women']}人且无重叠；后段{batch_audit['post_women_exactly_4_records']}/{batch_audit['post_women_total']}人恰4条记录。",
        "action": "按683前后分层并在共同支持域比较；无外部实验元数据时不得断言为已确认实验室批次。",
    },
    {
        "severity": "否决性",
        "issue": "683后BMI-Y确定性关系",
        "evidence": f"个体内斜率={batch_audit['683后']['centered_bmi_y_relation']['slope']:.10f}，R²={batch_audit['683后']['centered_bmi_y_relation']['r2']:.12f}。",
        "action": "后段个体内BMI效应只能作为数据机制特征，不能作生物学解释。",
    },
    {
        "severity": "否决性",
        "issue": "重复测量非独立",
        "evidence": f"1082条记录仅来自267人；B+I主口径为1021次抽血、{repeat_audit['multi_record_draw_groups']}个多检测抽血组，B+I+H口径为1063个检测会话、{repeat_sensitivity['B+I+H同日检测会话口径']['multi_groups']}个同日重复会话。",
        "action": "后续模型必须按孕妇分组，并显式保留孕妇→抽血事件→检测会话→检测记录四层标识。",
    },
    {
        "severity": "严重",
        "issue": "读段内部关系断裂",
        "evidence": f"683前关系式容差2条内通过{read_relation_by_batch['683前']['match_within_2_reads']}/{read_relation_by_batch['683前']['records']}，683后通过{read_relation_by_batch['683后']['match_within_2_reads']}/{read_relation_by_batch['683后']['records']}；另有{quality_audit['unique_reads_gt_total']}条唯一读段数大于总读段数。",
        "action": "测序质量变量只作批次诊断和敏感性调整，不直接合并解释。",
    },
    {
        "severity": "严重",
        "issue": "多记录抽血事件与同日重复检测均存在离散度",
        "evidence": f"B+I主口径组内SD={pooled_repeat_sd:.6f}、{flip_groups}/{repeat_audit['multi_record_draw_groups']}组跨4%，且39/40组跨检测日期；B+I+H同日口径组内SD={repeat_sensitivity['B+I+H同日检测会话口径']['pooled_within_y_sd']:.6f}、{repeat_sensitivity['B+I+H同日检测会话口径']['groups_crossing_4pct']}/{repeat_sensitivity['B+I+H同日检测会话口径']['multi_groups']}组跨4%。",
        "action": "主分析按B+I抽血事件聚合或嵌套建模；同时报告B+I+H同日口径敏感性。B+I结果称综合离散度，不全部解释为纯技术误差。",
    },
    {
        "severity": "主要",
        "issue": "日期/孕周不一致与存储类型切换",
        "evidence": f"可核记录{len(date_error)}条，|误差|>14天共{int((date_error.abs()>14).sum())}条；末次月经和检测日期编码均从序号{date_audit['first_lmp_iso_string_seq']}起切换。",
        "action": "J列作为主孕周，日期只做核验；冲突记录设标志并做敏感性。",
    },
    {
        "severity": "主要",
        "issue": "A055同次抽血孕周冲突",
        "evidence": f"检测到{repeat_audit['groups_with_different_gestation']}个复测组孕周不一致。",
        "action": "主分析对冲突抽血设歧义标志；聚合时不得把两个孕周当不同抽血。",
    },
    {
        "severity": "主要",
        "issue": "总体、个体间、个体内关系混淆",
        "evidence": "逐行相关不能代表同一孕妇随孕周/BMI变化的关系。",
        "action": "描述阶段必须同时报告raw/between/within相关。",
    },
    {
        "severity": "一般",
        "issue": "BMI、身高和体重非独立",
        "evidence": f"BMI重算最大绝对误差={df['bmi_formula_abs_error'].max():.6f}。",
        "action": "主模型不同时将三者解释为独立效应；BMI连续建模。",
    },
    {
        "severity": "一般",
        "issue": "非随机随访与稀疏支持域",
        "evidence": f"抽血次数与首测孕周相关r={followup_audit['draw_count_vs_first_week']['r']:.3f}；完整支持域并不均匀。",
        "action": "验证按孕妇划分，效应只在共同支持域解释。",
    },
]
issues_df = pd.DataFrame(issues)
issues_df.to_csv(OUTPUT / "audit_issues.csv", index=False, encoding="utf-8-sig")

report = []
report.append("# 第一问数据分析与质量结构审计（可复现审计输出）")
report.append("")
report.append("> 本报告只做数据与结构审计，不包含第一问正式关系模型拟合。正式归档前仍需按一键验收流程复核。")
report.append("")
report.append("## 一、数据范围与四层结构")
report.append("")
report.append(f"男胎表共有 {basic['records']} 条检测记录、{basic['draws']} 次操作性抽血事件、{basic['women']} 名孕妇。核心变量孕周、BMI和Y浓度均无缺失；序号唯一且连续。")
report.append("")
report.append("事件层级固定为：孕妇B → 抽血事件(B+I) → 检测会话(B+I+H) → 检测记录A。H列是检测时间而非采血时间；B+H只能表示检测日期会话。B+I+H+J只用于检查元数据一致性，不用于拆分抽血事件。")
report.append("")
report.append(markdown_table(["样本口径", "记录", "抽血", "孕妇"], [[key, value['records'], value['draws'], value['women']] for key, value in samples.items()]))
report.append("")
display_scopes = {"全男胎1082条", "主参考674条"}
display_hierarchy = [row for row in event_hierarchy_rows if row["scope"] in display_scopes]
report.append(markdown_table(
    ["范围", "事件键", "定义", "唯一组", "多记录组", "组内记录"],
    [[row["scope"], row["key_notation"], row["definition"], row["unique_events"], row["multi_groups"], row["multi_rows"]] for row in display_hierarchy],
))
report.append("")
report.append("## 二、主审计问题")
report.append("")
report.append(markdown_table(["等级", "问题", "证据", "处理原则"], [[row['severity'], row['issue'], row['evidence'], row['action']] for row in issues]))
report.append("")
report.append("## 三、重复检测口径与离散度")
report.append("")
report.append("按题目字段语义，B+I是抽血事件主口径；B+I+H是同日检测会话敏感性口径。两种口径同时报告，以避免把检测日期误当成采血日期。")
report.append("")
report.append(markdown_table(
    ["口径", "多记录组", "组内记录", "经验组内Y浓度SD", "跨4%组"],
    [[name, item["multi_groups"], item["multi_rows"], f"{item['pooled_within_y_sd']:.8f}", item["groups_crossing_4pct"]] for name, item in repeat_sensitivity.items()],
))
report.append("")
report.append("40个多记录抽血事件中有39个跨检测日期，因此B+I口径反映同一抽血编号下多次检测的综合离散度，不等同于严格同日技术误差。")
report.append("")
report.append("A055第3次抽血的两条记录同日、同抽血次数，但孕周分别为21w+1和20w+3，应标记为孕周元数据冲突；不得通过把J加入事件主键而静默拆成两次抽血，也不得取均值或中位数人为生成一个孕周。")
report.append("")
report.append("## 四、683数据生成机制结构断点")
report.append("")
report.append(markdown_table(["批次", "记录", "孕妇", "抽血", "平均孕周", "平均BMI", "平均Y浓度"], [[row['batch'], row['records'], row['women'], row['draws'], f"{row['week_mean']:.3f}", f"{row['bmi_mean']:.3f}", f"{row['y_mean']*100:.3f}%"] for row in batch_rows]))
report.append("")
post_relation = batch_audit["683后"]["centered_bmi_y_relation"]
report.append(f"后400行在孕妇内部满足近似确定性关系：斜率 {post_relation['slope']:.10f}，R²={post_relation['r2']:.12f}，残差SD={post_relation['residual_sd']:.3e}。这应判定为数据机制特征，而非直接的生物学证据。")
report.append("")
report.append("因此可将683作为候选数据生成机制分段边界，但附件不足以确认其一定对应实验室、仪器、试剂或软件批次切换。日期单元格存储格式实际在序号687附近切换，不应把格式变化写成683断点的唯一原因。")
report.append("")
report.append("## 五、日期、BMI与测序质量")
report.append("")
report.append(f"孕周字符串全部可解析。末次月经与检测日期可联合核验 {len(date_error)} 条，其中绝对偏差超过14天的记录为 {int((date_error.abs()>14).sum())} 条。BMI与体重/身高公式最大绝对误差为 {df['bmi_formula_abs_error'].max():.6f}。GC含量不在40%-60%题面正常范围内的记录为 {quality_audit['gc_outside_40_60pct']} 条，但这些记录只应标记，不宜直接批量删除。")
report.append("")
report.append("## 六、相关结构的初步分解（只作数据审计，不是正式模型）")
report.append("")
corr_rows_report = []
for variable, label in [("gest_week", "孕周"), ("bmi", "BMI")]:
    corr_rows_report.append([
        label,
        f"{correlation_summary['全部']['raw'][variable]['pearson']['r']:.4f}",
        f"{correlation_summary['全部']['between'][variable]['r']:.4f}",
        f"{correlation_summary['全部']['within'][variable]['r']:.4f}",
        f"{correlation_summary['683前']['within'][variable]['r']:.4f}",
        f"{correlation_summary['683后']['within'][variable]['r']:.4f}",
    ])
report.append(markdown_table(["变量", "逐行", "个体间", "个体内", "683前个体内", "683后个体内"], corr_rows_report))
report.append("")
report.append("## 七、候选图表")
report.append("")
for name in [
    "01_batch_distributions.svg",
    "02_week_bmi_support.svg",
    "03_y_week_by_batch.svg",
    "04_within_bmi_y_by_batch.svg",
    "05_read_relation_break.svg",
    "06_technical_repeats.svg",
    "07_correlation_decomposition.svg",
    "08_date_gestation_consistency.svg",
]:
    report.append(f"- {name}")
report.append("")
report.append("## 八、进入正式关系分析前的硬门槛")
report.append("")
report.extend([
    "1. 不把1082行当作1082个独立个体。",
    "2. 不在忽略683数据生成机制结构断点的情况下报告统一BMI效应，也不把该断点无依据地断言为实验室批次。",
    "3. 不静默删除或平均多记录抽血事件；必须保留孕妇、抽血事件、检测会话和检测记录四层标识，并报告B+I综合离散度与B+I+H同日离散度。",
    "4. 不用日期覆盖J列孕周；所有冲突进入显式质量标志。",
    "5. 不把Y-Z、非整倍体结果、出生健康或序号当作第一问临床预测变量。",
    "6. 训练/验证必须按孕妇划分，并在正式调参前锁定。",
    "7. 所有效应只在683前后两段共同支持域内解释；无外部元数据时不称已确认实验室批次。",
])

(OUTPUT / "candidate_audit_report.md").write_text("\n".join(report), encoding="utf-8")
(OUTPUT / "audit_summary.json").write_text(
    json.dumps(json_ready(summary), ensure_ascii=False, indent=2), encoding="utf-8"
)

print(json.dumps({
    "outputs": str(OUTPUT),
    "records": basic["records"],
    "women": basic["women"],
    "draws": basic["draws"],
    "critical_issue_count": int((issues_df["severity"] == "否决性").sum()),
    "major_issue_count": int(issues_df["severity"].isin(["严重", "主要"]).sum()),
    "charts": 8,
}, ensure_ascii=False, indent=2))
