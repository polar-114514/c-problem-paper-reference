from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = SCRIPT_DIR.parents[2]
parser = argparse.ArgumentParser(description="为第一问数据审计图导出确定性绘图数据（不建模）")
parser.add_argument("--source-json", type=Path, required=True)
parser.add_argument("--audit-summary", type=Path, required=True)
parser.add_argument(
    "--output",
    type=Path,
    default=WORKSPACE_ROOT / "99_临时中转" / "第一问数据审计复现" / "matlab_plot_data",
)
args = parser.parse_args()
source_path = args.source_json.resolve()
summary_path = args.audit_summary.resolve()
output = args.output.resolve()
output.mkdir(parents=True, exist_ok=True)


def blank(value) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    return isinstance(value, str) and value.strip() == ""


def gestation_week(value):
    if blank(value):
        return np.nan
    match = re.fullmatch(r"(?i)\s*(\d+)\s*[w周]\s*(?:\+\s*(\d+))?\s*", str(value))
    if not match:
        return np.nan
    week = int(match.group(1))
    day = int(match.group(2) or 0)
    return np.nan if day > 6 else week + day / 7.0


def parse_date(value):
    if blank(value):
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
    try:
        return pd.Timestamp(pd.to_datetime(str(value).strip(), errors="raise")).normalize()
    except Exception:
        return pd.NaT


def through_origin_relation(x, y):
    frame = pd.DataFrame({"x": x, "y": y}).dropna()
    xv = frame["x"].to_numpy(float)
    yv = frame["y"].to_numpy(float)
    slope = float(np.dot(xv, yv) / np.dot(xv, xv))
    residual = yv - slope * xv
    sst = float(np.sum((yv - yv.mean()) ** 2))
    r2 = float(1 - np.sum(residual**2) / sst)
    return slope, r2, len(frame)


def box_summary(values):
    values = np.sort(pd.Series(values).dropna().to_numpy(float))
    q1, median, q3 = np.quantile(values, [0.25, 0.5, 0.75])
    iqr = q3 - q1
    low_limit, high_limit = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    inside = values[(values >= low_limit) & (values <= high_limit)]
    outliers = values[(values < low_limit) | (values > high_limit)]
    return {
        "样本数": int(len(values)),
        "下须值": float(inside.min()),
        "第一四分位数": float(q1),
        "中位数": float(median),
        "第三四分位数": float(q3),
        "上须值": float(inside.max()),
        "异常值数量": int(len(outliers)),
        "异常值列表（与指标同单位，分号分隔）": ";".join(f"{value:.15g}" for value in outliers),
    }


source = json.loads(source_path.read_text(encoding="utf-8"))
summary = json.loads(summary_path.read_text(encoding="utf-8"))
rows = source["values"][1:]
columns = [
    "seq", "woman", "age", "height", "weight", "lmp", "conception", "test_date", "draw_no", "gest_raw", "bmi",
    "reads_total", "align_rate", "duplicate_rate", "unique_reads", "gc", "z13", "z18", "z21", "zx", "zy", "y",
    "x", "gc13", "gc18", "gc21", "filtered_rate", "aneuploidy", "gravidity", "parity", "healthy",
]
df = pd.DataFrame(rows, columns=columns)
for column in ["seq", "draw_no", "bmi", "reads_total", "align_rate", "duplicate_rate", "unique_reads", "filtered_rate", "y"]:
    df[column] = pd.to_numeric(df[column], errors="coerce")
df["gest_week"] = df["gest_raw"].map(gestation_week)
df["gest_days"] = df["gest_week"] * 7
df["lmp_date"] = df["lmp"].map(parse_date)
df["test_date_parsed"] = df["test_date"].map(parse_date)
df["date_error"] = (df["test_date_parsed"] - df["lmp_date"]).dt.days - df["gest_days"]
df["batch"] = np.where(df["seq"] < 683, "683前", "683后")
df["draw_id"] = df["woman"].astype(str) + "#" + df["draw_no"].astype("Int64").astype(str)
df["test_date_key"] = df["test_date_parsed"].dt.strftime("%Y-%m-%d").fillna("MISSING_DATE")
df["assay_session_id"] = df["draw_id"] + "#" + df["test_date_key"]
df["read_expected"] = df["reads_total"] * df["align_rate"] * (1 - df["duplicate_rate"]) * (1 - df["filtered_rate"])
df["read_residual"] = df["unique_reads"] - df["read_expected"]
df["bmi_wc"] = df["bmi"] - df.groupby("woman")["bmi"].transform("mean")
df["y_wc"] = df["y"] - df.groupby("woman")["y"].transform("mean")

base = df[["seq", "woman", "draw_id", "assay_session_id", "batch", "gest_week", "bmi", "y", "bmi_wc", "y_wc", "read_residual", "date_error"]].rename(columns={
    "seq": "序号",
    "woman": "孕妇代码",
    "draw_id": "抽血事件标识",
    "assay_session_id": "检测会话标识（B+I+H）",
    "batch": "分析批次（683断点口径）",
    "gest_week": "解析检测孕周（周）",
    "bmi": "孕妇BMI（kg/m²）",
    "y": "Y染色体浓度（比例，0–1）",
    "bmi_wc": "个体内BMI偏差（kg/m²）",
    "y_wc": "个体内Y染色体浓度偏差（比例）",
    "read_residual": "唯一比对读段逻辑残差（条）",
    "date_error": "日期推算孕周－记录孕周（天）",
})
base.to_csv(output / "绘图基础数据.csv", index=False, encoding="utf-8-sig")

box_rows = []
for variable, label, scale in [("gest_week", "解析检测孕周", 1), ("bmi", "孕妇BMI", 1), ("y", "Y染色体浓度", 100)]:
    for batch in ["683前", "683后"]:
        row = {"指标": label, "分析批次（683断点口径）": batch, "显示倍率": scale}
        row.update(box_summary(df.loc[df["batch"] == batch, variable] * scale))
        box_rows.append(row)
pd.DataFrame(box_rows).to_csv(output / "图01_箱线摘要.csv", index=False, encoding="utf-8-sig")

bin_rows = []
bin_coverage = {}
for batch in ["683前", "683后"]:
    group = df[df["batch"] == batch]
    first_edge = math.floor(group["gest_week"].min())
    last_edge = math.floor(group["gest_week"].max()) + 1
    edges = np.arange(first_edge, last_edge + 1, dtype=float)
    bucket = pd.cut(group["gest_week"], bins=edges, right=False)
    if bucket.isna().any():
        raise RuntimeError(f"{batch}孕周分箱仍有{int(bucket.isna().sum())}条未覆盖")
    med = group.assign(bucket=bucket).groupby("bucket", observed=True).agg(
        week=("gest_week", "median"), y=("y", "median"), n=("seq", "size")
    )
    covered = int(med["n"].sum())
    if covered != len(group):
        raise RuntimeError(f"{batch}孕周分箱仅覆盖{covered}/{len(group)}条")
    bin_coverage[batch] = covered
    for interval, values in med.iterrows():
        bin_rows.append({
            "分析批次（683断点口径）": batch,
            "分箱左边界（周）": float(interval.left),
            "分箱右边界（周）": float(interval.right),
            "箱内孕周中位数（周）": float(values["week"]),
            "箱内Y染色体浓度中位数（%）": float(values["y"] * 100),
            "箱内记录数（条）": int(values["n"]),
        })
pd.DataFrame(bin_rows).to_csv(output / "图03_孕周分箱中位数.csv", index=False, encoding="utf-8-sig")

relation_rows = []
for batch in ["683前", "683后"]:
    group = df[df["batch"] == batch]
    slope, r2, n = through_origin_relation(group["bmi_wc"], group["y_wc"])
    relation_rows.append({"分析批次（683断点口径）": batch, "过原点斜率": slope, "决定系数R²": r2, "有效记录数（条）": n})
pd.DataFrame(relation_rows).to_csv(output / "图04_个体内关系参数.csv", index=False, encoding="utf-8-sig")

repeat_rows = []
repeat_groups_by_definition = {}
for definition, key in [
    ("同一抽血编号（B+I）", "draw_id"),
    ("同日检测会话（B+I+H）", "assay_session_id"),
]:
    groups = [(group_id, group.sort_values("seq")) for group_id, group in df.groupby(key) if len(group) > 1]
    groups.sort(key=lambda item: (float(item[1]["y"].mean()), item[0]))
    repeat_groups_by_definition[definition] = groups
    for order, (group_id, group) in enumerate(groups, start=1):
        crosses = bool(group["y"].min() < 0.04 <= group["y"].max())
        for assay_order, (_, row) in enumerate(group.iterrows(), start=1):
            repeat_rows.append({
                "复测口径": definition,
                "复测组绘图顺序": order,
                "复测组标识": group_id,
                "组内检测顺序": assay_order,
                "序号": int(row["seq"]),
                "Y染色体浓度（%）": float(row["y"] * 100),
                "组内Y染色体浓度均值（%）": float(group["y"].mean() * 100),
                "组内是否跨越4%阈值": crosses,
            })
pd.DataFrame(repeat_rows).to_csv(output / "图06_重复检测离散度明细.csv", index=False, encoding="utf-8-sig")

corr_rows = []
for variable, label in [("gest_week", "孕周"), ("bmi", "BMI")]:
    for level, level_label in [("raw", "逐行"), ("between", "个体间"), ("within", "个体内")]:
        item = summary["correlations"]["全部"][level][variable]
        if level == "raw":
            item = item["pearson"]
        corr_rows.append({"变量": label, "相关结构层级": level_label, "Pearson相关系数": float(item["r"])})
pd.DataFrame(corr_rows).to_csv(output / "图07_相关结构摘要.csv", index=False, encoding="utf-8-sig")

hist_rows = []
hist_coverage = {}
for batch in ["683前", "683后"]:
    batch_values = df.loc[df["batch"] == batch, "date_error"]
    values = batch_values.dropna().to_numpy(float)
    total = int(len(batch_values))
    valid = int(len(values))
    missing = total - valid
    edges = np.histogram_bin_edges(values, bins=40) if batch == "683前" else np.array([-0.5, 0.5])
    counts, edges = np.histogram(values, bins=edges)
    if int(counts.sum()) != valid:
        raise RuntimeError(f"{batch}日期孕周直方图仅覆盖{int(counts.sum())}/{valid}条有效记录")
    hist_coverage[batch] = {"总记录数": total, "有效记录数": valid, "缺失记录数": missing}
    for left, right, count in zip(edges[:-1], edges[1:], counts):
        hist_rows.append({
            "分析批次（683断点口径）": batch,
            "分箱左边界（天）": float(left),
            "分箱右边界（天）": float(right),
            "记录数（条）": int(count),
            "批次总记录数（条）": total,
            "有效日期孕周差记录数（条）": valid,
            "日期孕周差缺失记录数（条）": missing,
        })
pd.DataFrame(hist_rows).to_csv(output / "图08_日期孕周直方图.csv", index=False, encoding="utf-8-sig")

manifest = {
    "用途": "第一问数据质量与结构审计SVG的确定性绘图输入；不包含正式模型拟合",
    "源快照SHA256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
    "审计汇总SHA256": hashlib.sha256(summary_path.read_bytes()).hexdigest(),
    "记录数": int(len(df)),
    "孕妇数": int(df["woman"].nunique()),
    "抽血事件数": int(df["draw_id"].nunique()),
    "检测会话数": int(df["assay_session_id"].nunique()),
    "同一抽血编号多检测组数": int(len(repeat_groups_by_definition["同一抽血编号（B+I）"])),
    "同日多记录检测会话数": int(len(repeat_groups_by_definition["同日检测会话（B+I+H）"])),
    "主样本口径": "序号小于683且孕周位于[10,26)，共674条",
    "图03分箱覆盖记录数": bin_coverage,
    "图08日期孕周差覆盖": hist_coverage,
    "颜色": {"683前": "#2563EB", "683后": "#EA580C", "公共域": "#16A34A"},
    "图件": [f"{index:02d}" for index in range(1, 9)],
}
(output / "绘图数据清单.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps({"output": str(output), "files": 8, "manifest": manifest}, ensure_ascii=False, indent=2))
