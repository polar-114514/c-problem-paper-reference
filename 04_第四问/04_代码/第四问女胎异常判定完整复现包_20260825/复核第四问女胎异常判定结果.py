from __future__ import annotations

import hashlib
import json
import math
import re
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.special import logit
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")

脚本路径 = Path(__file__).resolve()
项目目录 = 脚本路径.parent
工作区 = 项目目录.parents[2]
输出根目录 = 项目目录 / "正式候选输出"
独立复核目录 = 项目目录 / "独立复核"
独立复核目录.mkdir(parents=True, exist_ok=True)

工作簿路径 = 工作区 / "00_题目与原始资料/02_原始数据/附件.xlsx"
题目路径 = 工作区 / "00_题目与原始资料/01_题目原文/C题.pdf"
合同路径 = 项目目录 / "00_第四问题意合同与候选设计.md"
建模脚本路径 = 项目目录 / "第四问女胎异常判定建模.py"
一键脚本路径 = 项目目录 / "一键运行第四问女胎异常判定.ps1"

正则强度倒数候选 = np.logspace(-4, 4, 10)
内层折数 = 5
最大迭代次数 = 5000
标准分数规则阈值 = 3.0
概率下限 = np.finfo(float).eps
整簇自助次数 = 400

特征名称 = [
    "孕周数",
    "孕妇年龄",
    "孕妇体质指数",
    "原始读段数_对数",
    "参考基因组比对比例",
    "重复读段比例",
    "唯一比对读段数_对数",
    "总GC含量",
    "13号染色体Z值",
    "18号染色体Z值",
    "21号染色体Z值",
    "X染色体Z值",
    "X染色体浓度",
    "13号染色体GC含量",
    "18号染色体GC含量",
    "21号染色体GC含量",
    "过滤读段比例",
]


def 文件哈希(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def 读JSON(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def 读CSV(path: Path, **kwargs):
    kwargs.setdefault("float_precision", "round_trip")
    return pd.read_csv(path, **kwargs)


def 解析孕周(value) -> float:
    match = re.fullmatch(r"(\d+)\s*w(?:\s*\+?\s*(\d+)\s*)?", str(value).strip().lower())
    if not match:
        raise ValueError(f"不能解析孕周：{value!r}")
    week, day = int(match.group(1)), int(match.group(2) or 0)
    if not 0 <= day <= 6:
        raise ValueError(f"孕周天数错误：{value!r}")
    return week + day / 7.0


def 解析日期(value) -> str:
    if isinstance(value, (int, float, np.integer, np.floating)):
        if not np.isfinite(float(value)):
            raise ValueError(f"检测日期不是有限数值：{value!r}")
        integer_value = int(value)
        if float(value) == integer_value and re.fullmatch(r"\d{8}", str(integer_value)):
            timestamp = pd.to_datetime(str(integer_value), format="%Y%m%d", errors="raise")
        else:
            timestamp = pd.Timestamp("1899-12-30") + pd.to_timedelta(float(value), unit="D")
    else:
        timestamp = pd.to_datetime(value, errors="raise")
    return timestamp.strftime("%Y-%m-%d")


def 解析标签(value) -> frozenset[str]:
    if pd.isna(value) or str(value).strip() == "":
        return frozenset()
    tokens = re.findall(r"T(?:13|18|21)", str(value).upper())
    if not tokens:
        raise ValueError(f"未知标签：{value!r}")
    return frozenset(tokens)


def 随机种子() -> int:
    return int(文件哈希(工作簿路径)[:8], 16)


def 重建数据():
    raw = pd.read_excel(工作簿路径, sheet_name="女胎检测数据", engine="openpyxl")
    labels = raw["染色体的非整倍体"].map(解析标签)
    dates = raw["检测日期"].map(解析日期)
    data = pd.DataFrame(
        {
            "记录行号": np.arange(1, len(raw) + 1),
            "原始序号": raw["序号"],
            "孕妇代码": raw["孕妇代码"].astype(str),
            "检测日期": dates,
            "检测孕周原文": raw["检测孕周"].astype(str),
            "抽血事件键": raw["孕妇代码"].astype(str) + "|" + dates + "|" + raw["检测抽血次数"].astype(str) + "|" + raw["检测孕周"].astype(str),
            "AB原始标签": raw["染色体的非整倍体"].fillna("空白=无异常").astype(str),
            "AE出生健康结果": raw["胎儿是否健康"].astype(str),
            "任一异常标志": labels.map(bool).astype(int),
            "T13异常标志": labels.map(lambda value: int("T13" in value)),
            "T18异常标志": labels.map(lambda value: int("T18" in value)),
            "T21异常标志": labels.map(lambda value: int("T21" in value)),
            "孕周数": raw["检测孕周"].map(解析孕周),
            "孕妇年龄": pd.to_numeric(raw["年龄"], errors="coerce"),
            "孕妇体质指数": pd.to_numeric(raw["孕妇BMI"], errors="coerce"),
            "原始读段数_对数": np.log1p(pd.to_numeric(raw["原始读段数"], errors="coerce")),
            "参考基因组比对比例": pd.to_numeric(raw["在参考基因组上比对的比例"], errors="coerce"),
            "重复读段比例": pd.to_numeric(raw["重复读段的比例"], errors="coerce"),
            "唯一比对读段数_对数": np.log1p(pd.to_numeric(raw["唯一比对的读段数"], errors="coerce")),
            "总GC含量": pd.to_numeric(raw["GC含量"], errors="coerce"),
            "13号染色体Z值": pd.to_numeric(raw["13号染色体的Z值"], errors="coerce"),
            "18号染色体Z值": pd.to_numeric(raw["18号染色体的Z值"], errors="coerce"),
            "21号染色体Z值": pd.to_numeric(raw["21号染色体的Z值"], errors="coerce"),
            "X染色体Z值": pd.to_numeric(raw["X染色体的Z值"], errors="coerce"),
            "X染色体浓度": pd.to_numeric(raw["X染色体浓度"], errors="coerce"),
            "13号染色体GC含量": pd.to_numeric(raw["13号染色体的GC含量"], errors="coerce"),
            "18号染色体GC含量": pd.to_numeric(raw["18号染色体的GC含量"], errors="coerce"),
            "21号染色体GC含量": pd.to_numeric(raw["21号染色体的GC含量"], errors="coerce"),
            "过滤读段比例": pd.to_numeric(raw["被过滤掉读段数的比例"], errors="coerce"),
        }
    )
    data["Z规则连续分数"] = data[["13号染色体Z值", "18号染色体Z值", "21号染色体Z值"]].max(axis=1)
    return data


def 等孕妇权重(frame: pd.DataFrame):
    counts = frame["孕妇代码"].value_counts()
    return frame["孕妇代码"].map(lambda value: 1.0 / counts[value]).to_numpy(float)


def 分层分组器(target: str, folds: int = 内层折数):
    offset = sum(ord(character) for character in target)
    return StratifiedGroupKFold(
        n_splits=folds,
        shuffle=True,
        random_state=(随机种子() + offset + folds) % (2**32 - 1),
    )


def 新建逻辑(c_value: float):
    return LogisticRegression(
        l1_ratio=1.0,
        solver="liblinear",
        C=float(c_value),
        max_iter=最大迭代次数,
        random_state=随机种子(),
    )


def 混淆(y_true, y_pred, weight):
    y_true, y_pred, weight = np.asarray(y_true, int), np.asarray(y_pred, int), np.asarray(weight, float)
    tp = float(np.sum(weight[(y_true == 1) & (y_pred == 1)]))
    fp = float(np.sum(weight[(y_true == 0) & (y_pred == 1)]))
    fn = float(np.sum(weight[(y_true == 1) & (y_pred == 0)]))
    tn = float(np.sum(weight[(y_true == 0) & (y_pred == 0)]))
    divide = lambda a, b: float(a / b) if b > 0 else np.nan
    sensitivity = divide(tp, tp + fn)
    specificity = divide(tn, tn + fp)
    precision = divide(tp, tp + fp)
    f1 = divide(2 * precision * sensitivity, precision + sensitivity) if np.isfinite(precision) else np.nan
    denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = divide(tp * tn - fp * fn, denominator)
    return {
        "真阳性": tp,
        "假阳性": fp,
        "假阴性": fn,
        "真阴性": tn,
        "灵敏度": sensitivity,
        "特异度": specificity,
        "精确率": precision,
        "F1分数": f1,
        "马修斯相关系数": mcc,
        "平衡准确率": float(np.nanmean([sensitivity, specificity])),
        "准确率": divide(tp + tn, tp + fp + fn + tn),
    }


def 选择阈值(y_true, probability, weight):
    candidates = np.r_[np.nextafter(np.max(probability), np.inf), np.unique(probability)[::-1]]
    rows = []
    for threshold in candidates:
        rows.append({"阈值": float(threshold), **混淆(y_true, probability >= threshold, weight)})
    curve = pd.DataFrame(rows)
    ranking = curve[["马修斯相关系数", "F1分数", "平衡准确率"]].fillna(-np.inf)
    best = max(map(tuple, ranking.to_numpy(float)))
    tied = curve.loc[ranking.apply(lambda row: tuple(row.to_numpy(float)) == best, axis=1)]
    return float(np.median(tied["阈值"]))


def 拟合逻辑(train: pd.DataFrame, target: str, folds: int = 内层折数):
    cross = np.full((len(正则强度倒数候选), len(train)), np.nan)
    validation_weight = np.zeros(len(train))
    for fit_index, validation_index in 分层分组器(target, folds).split(train, train[target], train["孕妇代码"]):
        fit_frame, validation_frame = train.iloc[fit_index], train.iloc[validation_index]
        imputer = SimpleImputer(strategy="median")
        scaler = StandardScaler()
        x_fit = scaler.fit_transform(imputer.fit_transform(fit_frame[特征名称]))
        x_validation = scaler.transform(imputer.transform(validation_frame[特征名称]))
        validation_weight[validation_index] = 等孕妇权重(validation_frame)
        for index, c_value in enumerate(正则强度倒数候选):
            model = 新建逻辑(c_value)
            model.fit(x_fit, fit_frame[target], sample_weight=等孕妇权重(fit_frame))
            cross[index, validation_index] = model.predict_proba(x_validation)[:, 1]
    losses = [log_loss(train[target], values, sample_weight=validation_weight, labels=[0, 1]) for values in cross]
    selected_index = int(np.argmin(losses))
    selected_c = float(正则强度倒数候选[selected_index])
    threshold = 选择阈值(train[target].to_numpy(int), cross[selected_index], validation_weight)
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    x_train = scaler.fit_transform(imputer.fit_transform(train[特征名称]))
    model = 新建逻辑(selected_c)
    model.fit(x_train, train[target], sample_weight=等孕妇权重(train))
    return model, imputer, scaler, selected_c, threshold, float(losses[selected_index])


def 拟合树(train: pd.DataFrame, target: str):
    imputer = SimpleImputer(strategy="median")
    x = imputer.fit_transform(train[特征名称])
    weight = 等孕妇权重(train)
    base = DecisionTreeClassifier(random_state=随机种子())
    alphas = np.unique(base.cost_complexity_pruning_path(x, train[target], sample_weight=weight).ccp_alphas)
    choices = []
    for alpha in alphas:
        model = DecisionTreeClassifier(random_state=随机种子(), ccp_alpha=float(alpha))
        model.fit(x, train[target], sample_weight=weight)
        probability = np.clip(model.predict_proba(x)[:, 1], 概率下限, 1 - 概率下限)
        y = train[target].to_numpy(int)
        log_likelihood = float(np.sum(weight * (y * np.log(probability) + (1 - y) * np.log(1 - probability))))
        leaves = int(model.get_n_leaves())
        bic = float(-2 * log_likelihood + leaves * np.log(train["孕妇代码"].nunique()))
        choices.append((bic, leaves, float(alpha), model))
    bic, leaves, alpha, model = min(choices, key=lambda value: (value[0], value[1], value[2]))
    return model, imputer, bic, leaves, alpha


def 路线指标(frame: pd.DataFrame, weight, probability_route: bool):
    y = frame["实际异常标志"].to_numpy(int)
    score = frame["连续评分"].to_numpy(float)
    metrics = 混淆(y, frame["预测异常标志"].to_numpy(int), weight)
    metrics["ROC曲线下面积"] = float(roc_auc_score(y, score, sample_weight=weight))
    metrics["PR曲线下面积"] = float(average_precision_score(y, score, sample_weight=weight))
    metrics["实际异常比例"] = float(np.average(y, weights=weight))
    metrics["预测异常比例"] = float(np.average(frame["预测异常标志"], weights=weight))
    if probability_route:
        probability = np.clip(frame["预测概率"].to_numpy(float), 概率下限, 1 - 概率下限)
        metrics["布里尔分数"] = float(np.average((probability - y) ** 2, weights=weight))
        metrics["对数损失"] = float(log_loss(y, probability, sample_weight=weight, labels=[0, 1]))
        design = sm.add_constant(logit(probability), has_constant="add")
        fit = sm.GLM(y, design, family=sm.families.Binomial(), freq_weights=weight).fit()
        metrics["校准截距"] = float(fit.params[0])
        metrics["校准斜率"] = float(fit.params[1])
        metrics["校准实际异常比例"] = float(np.average(y, weights=weight))
        metrics["校准平均预测概率"] = float(np.average(probability, weights=weight))
    else:
        for column in ["布里尔分数", "对数损失", "校准截距", "校准斜率", "校准实际异常比例", "校准平均预测概率"]:
            metrics[column] = np.nan
    return metrics


def 最大数值差(left, right, columns):
    differences = []
    for column in columns:
        a = pd.to_numeric(left[column], errors="coerce").to_numpy(float)
        b = pd.to_numeric(right[column], errors="coerce").to_numpy(float)
        valid = np.isfinite(a) & np.isfinite(b)
        if np.any(valid):
            differences.append(float(np.max(np.abs(a[valid] - b[valid]))))
        if not np.array_equal(np.isnan(a), np.isnan(b)):
            return np.inf
    return max(differences, default=0.0)


def main():
    checks = []

    def 验收(name: str, condition, evidence: str):
        passed = bool(condition)
        checks.append({"验收项": name, "通过标志": passed, "状态": "通过" if passed else "失败", "独立证据": evidence})

    manifest_path = 输出根目录 / "04_复现/第四问运行清单.json"
    self_pass_path = 输出根目录 / "04_复现/第四问自审PASS记录.json"
    hash_table_path = 输出根目录 / "04_复现/第四问结果文件哈希.csv"
    manifest = 读JSON(manifest_path)
    self_pass = 读JSON(self_pass_path)

    source_hashes = {
        "工作簿SHA256": 文件哈希(工作簿路径),
        "题目SHA256": 文件哈希(题目路径),
        "合同SHA256": 文件哈希(合同路径),
        "建模脚本SHA256": 文件哈希(建模脚本路径),
        "一键脚本SHA256": 文件哈希(一键脚本路径),
    }
    source_difference = {key: (manifest.get(key), value) for key, value in source_hashes.items() if manifest.get(key) != value}
    验收("输入和代码哈希", not source_difference, f"核对5个源对象；差异={source_difference}")
    验收(
        "自审哈希链",
        self_pass.get("状态") == "PASS"
        and self_pass.get("运行清单SHA256") == 文件哈希(manifest_path)
        and self_pass.get("结果哈希表SHA256") == 文件哈希(hash_table_path),
        "自审状态、运行清单与结果哈希表反向核对",
    )

    reported_hashes = 读CSV(hash_table_path)
    hash_differences = []
    for _, row in reported_hashes.iterrows():
        path = 输出根目录 / str(row["相对路径"])
        if not path.exists() or 文件哈希(path) != str(row["安全散列值_SHA256"]):
            hash_differences.append(str(row["相对路径"]))
    验收("结果逐文件哈希", len(reported_hashes) == self_pass.get("哈希覆盖文件数") and not hash_differences, f"复算{len(reported_hashes)}个文件；差异={hash_differences}")

    csv_paths = list(输出根目录.rglob("*.csv"))
    bad_headers = []
    duplicate_headers = []
    for path in csv_paths:
        columns = 读CSV(path, nrows=0).columns.tolist()
        if any(re.search(r"[\u4e00-\u9fff]", str(column)) is None for column in columns):
            bad_headers.append(str(path.relative_to(输出根目录)))
        if len(columns) != len(set(columns)):
            duplicate_headers.append(str(path.relative_to(输出根目录)))
    验收("中文表头", not bad_headers and not duplicate_headers, f"检查{len(csv_paths)}个CSV；无中文={bad_headers}；重复={duplicate_headers}")

    data = 重建数据()
    exported = 读CSV(输出根目录 / "01_数据/第四问女胎记录级建模表.csv")
    exported = exported.sort_values("记录行号").reset_index(drop=True)
    data = data.sort_values("记录行号").reset_index(drop=True)
    numeric_columns = ["记录行号", "任一异常标志", "T13异常标志", "T18异常标志", "T21异常标志"] + 特征名称
    data_difference = 最大数值差(data, exported, numeric_columns)
    text_match = data[["孕妇代码", "检测日期", "检测孕周原文", "抽血事件键", "AB原始标签", "AE出生健康结果"]].astype(str).equals(
        exported[["孕妇代码", "检测日期", "检测孕周原文", "抽血事件键", "AB原始标签", "AE出生健康结果"]].astype(str)
    )
    验收("原始Excel独立重建", len(data) == 605 and data_difference < 1e-12 and text_match, f"605条逐行重建；数值最大差={data_difference:.3e}；文本一致={text_match}")
    date_ok = data["检测日期"].str.fullmatch(r"\d{4}-\d{2}-\d{2}").all() and data["检测日期"].min() == "2023-03-05" and data["检测日期"].max() == "2024-07-08"
    验收("检测日期独立解析", date_ok, f"数据实得范围={data['检测日期'].min()}至{data['检测日期'].max()}；全部为四位年份={bool(data['检测日期'].str.fullmatch(r'\d{4}-\d{2}-\d{2}').all())}")

    event = data.groupby("抽血事件键").agg(记录数=("记录行号", "size"), AB状态数=("任一异常标志", "nunique"))
    mixed_women = data.groupby("孕妇代码")["任一异常标志"].agg(["min", "max"])
    label_ok = (
        data["孕妇代码"].nunique() == 147
        and data["任一异常标志"].sum() == 67
        and data.loc[data["任一异常标志"].eq(1), "孕妇代码"].nunique() == 44
        and ((mixed_women["min"] == 0) & (mixed_women["max"] == 1)).sum() == 43
        and data["AE出生健康结果"].eq("是").all()
        and event.shape[0] == 590
        and event["记录数"].gt(1).sum() == 15
        and event.loc[event["记录数"].gt(1), "AB状态数"].gt(1).sum() == 2
    )
    验收("标签、AE与层级", label_ok, "AB异常67条/44人；43名混合；AE全是；590事件、15个复测事件、2组AB不一致")
    type_counts = [int(data[column].sum()) for column in ["T13异常标志", "T18异常标志", "T21异常标志"]]
    type_women = [int(data.loc[data[column].eq(1), "孕妇代码"].nunique()) for column in ["T13异常标志", "T18异常标志", "T21异常标志"]]
    验收("分型标签", type_counts == [23, 46, 13] and type_women == [18, 30, 12], f"记录={type_counts}；孕妇={type_women}")

    roles = 读CSV(输出根目录 / "01_数据/第四问变量角色与泄漏禁用表.csv")
    allowed = set(roles.loc[roles["允许进入模型"].eq("是"), "变量"])
    forbidden = set(roles.loc[roles["允许进入模型"].eq("否"), "变量"])
    验收("特征和泄漏", allowed == set(特征名称) and {"AB染色体非整倍体", "AE胎儿是否健康", "孕妇代码", "样本序号", "检测日期", "检测抽血次数"}.issubset(forbidden), f"允许17项；禁用项={sorted(forbidden)}")

    candidate_predictions = 读CSV(输出根目录 / "03_验证/第四问候选路线逐孕妇留一逐记录.csv")
    candidate_summary = 读CSV(输出根目录 / "02_模型结果/第四问候选路线统一比较.csv")
    route_counts = candidate_predictions.groupby("路线").agg(记录数=("记录行号", "size"), 唯一记录=("记录行号", "nunique"), 孕妇数=("孕妇代码", "nunique"))
    验收("外层验证覆盖与组隔离", route_counts[["记录数", "唯一记录"]].eq(605).all().all() and route_counts["孕妇数"].eq(147).all() and candidate_predictions["外层留出孕妇"].eq(candidate_predictions["孕妇代码"]).all(), "三路线各605条/147人且留出组逐行一致")

    baseline = candidate_predictions.loc[candidate_predictions["路线"].eq("三标准差Z值规则基准")].sort_values("记录行号")
    baseline_score = data["Z规则连续分数"].to_numpy(float)
    baseline_ok = (
        np.max(np.abs(baseline["连续评分"].to_numpy(float) - baseline_score)) < 1e-12
        and np.array_equal(baseline["预测异常标志"].to_numpy(int), (baseline_score >= 标准分数规则阈值).astype(int))
        and baseline["预测概率"].isna().all()
    )
    验收("Z值规则基准", baseline_ok, f"独立复算正向Z>=3；预测阳性={(baseline_score>=3).sum()}，真阳性={int(((baseline_score>=3)&data['任一异常标志'].eq(1)).sum())}")

    metric_columns = [
        "真阳性", "假阳性", "假阴性", "真阴性", "灵敏度", "特异度", "精确率", "F1分数", "马修斯相关系数",
        "平衡准确率", "准确率", "ROC曲线下面积", "PR曲线下面积", "实际异常比例", "预测异常比例", "布里尔分数", "对数损失",
        "校准截距", "校准斜率", "校准实际异常比例", "校准平均预测概率",
    ]
    independent_summary_rows = []
    for route, frame in candidate_predictions.groupby("路线", sort=False):
        metrics = 路线指标(frame.reset_index(drop=True), 等孕妇权重(frame), bool(frame["概率输出标志"].eq(1).all()))
        independent_summary_rows.append({"路线": route, **metrics})
    independent_summary = pd.DataFrame(independent_summary_rows).sort_values("路线").reset_index(drop=True)
    reported_summary = candidate_summary[["路线"] + metric_columns].sort_values("路线").reset_index(drop=True)
    summary_difference = 最大数值差(independent_summary, reported_summary, metric_columns)
    验收("候选指标独立复算", summary_difference < 1e-12, f"三路线21类指标最大绝对差={summary_difference:.3e}")

    print("独立复核：重跑主L1逻辑回归147名孕妇逐人留一", flush=True)
    groups = sorted(data["孕妇代码"].unique().tolist())
    independent_probability = np.full(len(data), np.nan)
    independent_threshold = {}
    independent_c = {}
    independent_coefficients = []
    for number, held_out in enumerate(groups, start=1):
        test_mask = data["孕妇代码"].eq(held_out).to_numpy()
        train = data.loc[~test_mask].reset_index(drop=True)
        test = data.loc[test_mask]
        model, imputer, scaler, c_value, threshold, _ = 拟合逻辑(train, "任一异常标志")
        independent_probability[test_mask] = model.predict_proba(scaler.transform(imputer.transform(test[特征名称])))[:, 1]
        independent_threshold[held_out] = threshold
        independent_c[held_out] = c_value
        for feature, coefficient in zip(特征名称, model.coef_[0]):
            independent_coefficients.append({"外层留出孕妇": held_out, "特征": feature, "标准化系数": float(coefficient), "正则强度倒数": c_value})
        if number % 30 == 0 or number == len(groups):
            print(f"独立主L1逐孕妇 {number}/{len(groups)}", flush=True)
    reported_logistic = candidate_predictions.loc[candidate_predictions["路线"].eq("L1正则多因素逻辑回归")].sort_values("记录行号")
    probability_difference = float(np.max(np.abs(independent_probability - reported_logistic["预测概率"].to_numpy(float))))
    threshold_table = 读CSV(输出根目录 / "03_验证/第四问候选路线训练内阈值逐孕妇.csv")
    threshold_table = threshold_table.loc[threshold_table["路线"].eq("L1正则多因素逻辑回归")].sort_values("外层留出孕妇")
    threshold_difference = float(np.max(np.abs(threshold_table["训练内阈值"].to_numpy(float) - np.array([independent_threshold[group] for group in sorted(independent_threshold)]))))
    c_difference = float(np.max(np.abs(threshold_table["正则强度倒数"].to_numpy(float) - np.array([independent_c[group] for group in sorted(independent_c)]))))
    reported_coefficients = 读CSV(输出根目录 / "03_验证/第四问逻辑回归留一系数逐次.csv").sort_values(["外层留出孕妇", "特征"]).reset_index(drop=True)
    independent_coefficients = pd.DataFrame(independent_coefficients).sort_values(["外层留出孕妇", "特征"]).reset_index(drop=True)
    coefficient_difference = 最大数值差(independent_coefficients, reported_coefficients, ["标准化系数", "正则强度倒数"])
    class_match = np.array_equal(
        (independent_probability >= data["孕妇代码"].map(independent_threshold).to_numpy(float)).astype(int),
        reported_logistic["预测异常标志"].to_numpy(int),
    )
    验收("入选路线147次独立重拟合", max(probability_difference, threshold_difference, c_difference, coefficient_difference) < 1e-10 and class_match, f"概率差={probability_difference:.3e}；阈值差={threshold_difference:.3e}；C差={c_difference:.3e}；系数差={coefficient_difference:.3e}")

    print("独立复核：重跑剪枝树147个外层概率模型", flush=True)
    independent_tree_probability = np.full(len(data), np.nan)
    tree_rows = []
    for number, held_out in enumerate(groups, start=1):
        test_mask = data["孕妇代码"].eq(held_out).to_numpy()
        train, test = data.loc[~test_mask].reset_index(drop=True), data.loc[test_mask]
        model, imputer, bic, leaves, alpha = 拟合树(train, "任一异常标志")
        independent_tree_probability[test_mask] = model.predict_proba(imputer.transform(test[特征名称]))[:, 1]
        tree_rows.append({"外层留出孕妇": held_out, "剪枝复杂度": alpha, "叶节点数": leaves, "贝叶斯信息准则": bic})
        if number % 50 == 0 or number == len(groups):
            print(f"独立剪枝树 {number}/{len(groups)}", flush=True)
    reported_tree = candidate_predictions.loc[candidate_predictions["路线"].eq("贝叶斯信息准则剪枝决策树")].sort_values("记录行号")
    tree_probability_difference = float(np.max(np.abs(independent_tree_probability - reported_tree["预测概率"].to_numpy(float))))
    reported_tree_detail = 读CSV(输出根目录 / "03_验证/第四问剪枝树留一结构逐次.csv").sort_values("外层留出孕妇").reset_index(drop=True)
    independent_tree_detail = pd.DataFrame(tree_rows).sort_values("外层留出孕妇").reset_index(drop=True)
    tree_structure_difference = 最大数值差(independent_tree_detail, reported_tree_detail, ["剪枝复杂度", "叶节点数", "贝叶斯信息准则"])
    验收("非线性路线独立重拟合", max(tree_probability_difference, tree_structure_difference) < 1e-10, f"147个树概率差={tree_probability_difference:.3e}；结构/BIC差={tree_structure_difference:.3e}")

    print("独立复核：复算400次孕妇整簇候选指标", flush=True)
    reported_bootstrap = 读CSV(输出根目录 / "03_验证/第四问候选路线指标整簇自助逐次.csv")
    rng = np.random.default_rng(随机种子())
    group_array = np.array(groups, dtype=object)
    bootstrap_rows = []
    for repeat in range(1, 整簇自助次数 + 1):
        sampled = rng.choice(group_array, size=len(group_array), replace=True)
        multiplicity = pd.Series(sampled).value_counts()
        for route, frame in candidate_predictions.groupby("路线", sort=False):
            frame = frame.reset_index(drop=True)
            record_counts = frame["孕妇代码"].value_counts()
            weight = frame["孕妇代码"].map(lambda group: float(multiplicity.get(group, 0)) / record_counts[group]).to_numpy(float)
            metrics = 路线指标(frame, weight, bool(frame["概率输出标志"].eq(1).all()))
            bootstrap_rows.append({"自助序号": repeat, "路线": route, **metrics})
    independent_bootstrap = pd.DataFrame(bootstrap_rows).sort_values(["自助序号", "路线"]).reset_index(drop=True)
    reported_bootstrap = reported_bootstrap.sort_values(["自助序号", "路线"]).reset_index(drop=True)
    bootstrap_columns = [
        "灵敏度", "特异度", "精确率", "F1分数", "马修斯相关系数", "ROC曲线下面积", "PR曲线下面积", "布里尔分数", "对数损失", "校准截距", "校准斜率"
    ]
    bootstrap_difference = 最大数值差(independent_bootstrap, reported_bootstrap, bootstrap_columns)
    验收("400次候选整簇自助", len(reported_bootstrap) == 1200 and bootstrap_difference < 1e-10, f"1200行核心指标最大差={bootstrap_difference:.3e}")

    difference = 读CSV(输出根目录 / "03_验证/第四问候选路线差异整簇自助逐次.csv")
    independent_pivot = independent_bootstrap.pivot(index="自助序号", columns="路线", values="PR曲线下面积")
    independent_pr_difference = independent_pivot["贝叶斯信息准则剪枝决策树"] - independent_pivot["L1正则多因素逻辑回归"]
    reported_pr_difference = difference.sort_values("自助序号")["剪枝树减逻辑回归_PR曲线下面积差"].to_numpy(float)
    pr_difference_error = float(np.max(np.abs(independent_pr_difference.to_numpy(float) - reported_pr_difference)))
    pr_interval = np.quantile(independent_pr_difference, [0.025, 0.975])
    selected_rows = candidate_summary.loc[candidate_summary["主模型标志"].eq(1), "路线"].tolist()
    验收("主路线选择裁决", pr_difference_error < 1e-12 and pr_interval[1] < 0 and selected_rows == ["L1正则多因素逻辑回归"], f"树减逻辑PR差95%区间={pr_interval.tolist()}；逐次差误差={pr_difference_error:.3e}")

    model, imputer, scaler, full_c, full_threshold, full_loss = 拟合逻辑(data.reset_index(drop=True), "任一异常标志")
    parameters = 读CSV(输出根目录 / "02_模型结果/第四问入选模型全样本参数表.csv")
    parameter_map = parameters.set_index("参数")["标准化模型估计值"]
    parameter_differences = [float(model.intercept_[0]) - float(parameter_map["截距"])]
    parameter_differences.extend(float(value) - float(parameter_map[feature]) for feature, value in zip(特征名称, model.coef_[0]))
    parameter_differences.extend([full_c - float(parameter_map["正则强度倒数"]), full_threshold - float(parameter_map["训练内MCC参考阈值"])])
    reference = 读CSV(输出根目录 / "02_模型结果/第四问入选模型预处理参照表.csv").set_index("特征").loc[特征名称]
    reference_difference = max(
        float(np.max(np.abs(reference["训练中位数填补值"].to_numpy(float) - imputer.statistics_))),
        float(np.max(np.abs(reference["训练标准化均值"].to_numpy(float) - scaler.mean_))),
        float(np.max(np.abs(reference["训练标准化尺度"].to_numpy(float) - scaler.scale_))),
    )
    验收("全样本入选模型独立重拟合", max(max(abs(value) for value in parameter_differences), reference_difference) < 1e-10, f"参数最大差={max(abs(value) for value in parameter_differences):.3e}；预处理差={reference_difference:.3e}；内层损失={full_loss:.6f}")

    reported_threshold_curve = 读CSV(输出根目录 / "02_模型结果/第四问入选模型阈值性能完整曲线.csv")
    selected_frame = candidate_predictions.loc[candidate_predictions["路线"].eq("L1正则多因素逻辑回归")].sort_values("记录行号").reset_index(drop=True)
    probability = selected_frame["预测概率"].to_numpy(float)
    candidates = np.r_[np.nextafter(np.max(probability), np.inf), np.unique(probability)[::-1]]
    threshold_rows_independent = []
    selected_weight = 等孕妇权重(selected_frame)
    for threshold in candidates:
        threshold_rows_independent.append({"概率阈值": threshold, **混淆(selected_frame["实际异常标志"], probability >= threshold, selected_weight), "阳性判定比例": float(np.average(probability >= threshold, weights=selected_weight))})
    independent_threshold_curve = pd.DataFrame(threshold_rows_independent)
    threshold_difference = 最大数值差(
        independent_threshold_curve,
        reported_threshold_curve,
        ["概率阈值", "灵敏度", "特异度", "精确率", "F1分数", "马修斯相关系数", "阳性判定比例"],
    )
    验收("阈值完整曲线独立复算", len(reported_threshold_curve) == len(candidates) == 606 and threshold_difference < 1e-12, f"606个经验阈值；最大差={threshold_difference:.3e}")

    reported_calibration = 读CSV(输出根目录 / "02_模型结果/第四问入选模型校准十分位表.csv")
    calibration_work = selected_frame.copy()
    calibration_work["校准组"] = pd.qcut(calibration_work["预测概率"].rank(method="first"), q=10, labels=False) + 1
    calibration_work["孕妇等权"] = 等孕妇权重(calibration_work)
    calibration_rows = []
    for group, part in calibration_work.groupby("校准组"):
        calibration_rows.append(
            {
                "校准十分位组": int(group),
                "记录数": len(part),
                "孕妇数": part["孕妇代码"].nunique(),
                "概率下限": part["预测概率"].min(),
                "概率上限": part["预测概率"].max(),
                "等孕妇权平均预测概率": np.average(part["预测概率"], weights=part["孕妇等权"]),
                "等孕妇权实际异常比例": np.average(part["实际异常标志"], weights=part["孕妇等权"]),
            }
        )
    independent_calibration = pd.DataFrame(calibration_rows)
    calibration_difference = 最大数值差(independent_calibration, reported_calibration, ["记录数", "孕妇数", "概率下限", "概率上限", "等孕妇权平均预测概率", "等孕妇权实际异常比例"])
    验收("校准十分位独立复算", len(reported_calibration) == 10 and calibration_difference < 1e-12, f"10组覆盖605条；最大差={calibration_difference:.3e}")

    coefficient_stability = 读CSV(输出根目录 / "02_模型结果/第四问入选模型留一系数稳定性.csv")
    recomputed_stability = []
    for feature, frame in reported_coefficients.groupby("特征", sort=False):
        values = frame["标准化系数"].to_numpy(float)
        recomputed_stability.append(
            {
                "特征": feature,
                "外层拟合次数": len(values),
                "非零比例": np.mean(values != 0),
                "正系数比例": np.mean(values > 0),
                "负系数比例": np.mean(values < 0),
                "标准化系数中位数": np.median(values),
                "标准化系数2.5%分位": np.quantile(values, 0.025),
                "标准化系数97.5%分位": np.quantile(values, 0.975),
                "正则强度倒数中位数": np.median(frame["正则强度倒数"]),
            }
        )
    recomputed_stability = pd.DataFrame(recomputed_stability).sort_values("特征").reset_index(drop=True)
    coefficient_stability = coefficient_stability.sort_values("特征").reset_index(drop=True)
    stability_difference = 最大数值差(recomputed_stability, coefficient_stability, [column for column in recomputed_stability.columns if column != "特征"])
    验收("留一系数稳定性复算", len(coefficient_stability) == 17 and stability_difference < 1e-12, f"17特征×147次；最大差={stability_difference:.3e}")

    type_predictions = 读CSV(输出根目录 / "03_验证/第四问各异常类型逐孕妇留一预测.csv")
    type_summary = 读CSV(输出根目录 / "02_模型结果/第四问各异常类型识别指标.csv")
    target_mapping = {"T13": "T13异常标志", "T18": "T18异常标志", "T21": "T21异常标志"}
    type_checks = []
    type_metric_rows = []
    for target_label, target_column in target_mapping.items():
        for route, frame in type_predictions.loc[type_predictions["目标"].eq(target_label)].groupby("路线", sort=False):
            frame = frame.sort_values("记录行号").reset_index(drop=True)
            actual_match = np.array_equal(frame["实际异常标志"].to_numpy(int), data[target_column].to_numpy(int))
            if "Z值规则" in route:
                score = data[f"{target_label.replace('T','')}号染色体Z值"].to_numpy(float)
                rule_match = np.max(np.abs(frame["连续评分"].to_numpy(float) - score)) < 1e-12 and np.array_equal(frame["预测异常标志"].to_numpy(int), (score >= 3).astype(int))
            else:
                rule_match = frame["预测概率"].between(0, 1).all()
            type_checks.append(actual_match and rule_match and frame["孕妇代码"].nunique() == 147 and frame["记录行号"].nunique() == 605)
            type_metric_rows.append({"异常类型": target_label, "路线": route, **路线指标(frame, 等孕妇权重(frame), "Z值规则" not in route)})
    independent_type_summary = pd.DataFrame(type_metric_rows).sort_values(["异常类型", "路线"]).reset_index(drop=True)
    reported_type_summary = type_summary.sort_values(["异常类型", "路线"]).reset_index(drop=True)
    type_metric_columns = [column for column in metric_columns if column in independent_type_summary.columns]
    type_metric_difference = 最大数值差(independent_type_summary, reported_type_summary, type_metric_columns)
    验收("三类异常外层结果与指标", all(type_checks) and type_metric_difference < 1e-12, f"T13/T18/T21各两路线覆盖605条；指标差={type_metric_difference:.3e}")

    print("独立复核：重拟合三个分型全样本模型", flush=True)
    type_parameters = 读CSV(输出根目录 / "02_模型结果/第四问各异常类型全样本参数表.csv")
    type_parameter_differences = []
    for target_label, target_column in target_mapping.items():
        type_model, _, _, type_c, type_threshold, _ = 拟合逻辑(data.reset_index(drop=True), target_column)
        table = type_parameters.loc[type_parameters["异常类型"].eq(target_label)].set_index("参数")["标准化模型估计值"]
        type_parameter_differences.append(float(type_model.intercept_[0]) - float(table["截距"]))
        type_parameter_differences.extend(float(value) - float(table[feature]) for feature, value in zip(特征名称, type_model.coef_[0]))
        type_parameter_differences.extend([type_c - float(table["正则强度倒数"]), type_threshold - float(table["训练内MCC参考阈值"])])
    验收("三个分型全样本模型重拟合", max(abs(value) for value in type_parameter_differences) < 1e-10, f"T13/T18/T21参数最大差={max(abs(value) for value in type_parameter_differences):.3e}")

    type_intervals = 读CSV(输出根目录 / "03_验证/第四问各异常类型指标95%区间.csv")
    t21_rule = type_intervals.loc[(type_intervals["异常类型"].eq("T21")) & (type_intervals["路线"].str.contains("Z值规则"))].set_index("统计量")
    undefined_ok = (
        int(t21_rule.loc["精确率", "有效次数"]) == 0
        and int(t21_rule.loc["F1分数", "有效次数"]) == 0
        and int(t21_rule.loc["马修斯相关系数", "有效次数"]) == 0
        and pd.isna(t21_rule.loc["精确率", "中位数"])
        and pd.isna(t21_rule.loc["F1分数", "中位数"])
        and pd.isna(t21_rule.loc["马修斯相关系数", "中位数"])
    )
    验收("未定义指标诚实保留", undefined_ok, "T21 Z>=3从不报阳性；精确率/F1/MCC有效次数为0且区间为空，未伪造0")

    sensitivity = 读CSV(输出根目录 / "03_验证/第四问数据与评价口径敏感性.csv")
    main_sensitivity = sensitivity.loc[sensitivity["敏感性口径"].eq("主口径_等孕妇权")].iloc[0]
    selected_summary = candidate_summary.loc[candidate_summary["路线"].eq("L1正则多因素逻辑回归")].iloc[0]
    sensitivity_main_difference = max(abs(float(main_sensitivity[column]) - float(selected_summary[column])) for column in ["灵敏度", "特异度", "精确率", "F1分数", "马修斯相关系数", "ROC曲线下面积", "PR曲线下面积", "布里尔分数", "对数损失"])
    sensitivity_counts = dict(zip(sensitivity["敏感性口径"], sensitivity["记录数"]))
    fold_sensitivity = 读CSV(输出根目录 / "03_验证/第四问内层折数敏感性.csv")
    sensitivity_ok = (
        len(sensitivity) == 8
        and sensitivity_main_difference < 1e-12
        and int(sensitivity_counts["仅10至25周"]) == 594
        and int(sensitivity_counts["仅总GC在40%至60%"]) == 385
        and int(sensitivity_counts["每个抽血事件仅首条记录"]) == 590
        and int(sensitivity_counts["仅BMI完整记录"]) == 604
        and set(fold_sensitivity["内层分层分组折数"].astype(int)) == {4, 5, 10}
        and fold_sensitivity["正则强度倒数"].nunique() == 1
    )
    验收("敏感性口径", sensitivity_ok, f"主指标差={sensitivity_main_difference:.3e}；记录数={sensitivity_counts}；4/5/10折均选择同一C")

    parameter_sources = 读CSV(输出根目录 / "02_模型结果/第四问参数来源表.csv")
    parameter_ok = (
        len(parameter_sources) == 18
        and parameter_sources["是否自行设定"].eq("否").all()
        and parameter_sources["参数名称和符号"].is_unique
        and manifest["设置类别权重"] is False
        and manifest["使用SMOTE"] is False
        and manifest["设置漏诊误报代价比"] is False
        and manifest["输出唯一临床阈值"] is False
    )
    验收("参数来源与无自拟代价", parameter_ok, "18项均有来源；类别权重、SMOTE、代价比、唯一临床阈值均为false")

    model_card = (输出根目录 / "02_模型结果/第四问推荐模型卡.md").read_text(encoding="utf-8")
    wording_ok = (
        "AB 空白" in model_card
        and "AE不替代AB" in model_card
        and "逐孕妇外层验证" in model_card
        and "不把任何一个阈值称为临床唯一阈值" in model_card
        and "不是临床确诊模型" in model_card
        and "L1留一系数分布用于稳定性描述" in model_card
    )
    验收("模型卡措辞边界", wording_ok, "标签、层级、阈值、筛查与L1解释边界均已披露")

    image_paths = [path for path in 输出根目录.rglob("*") if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".svg", ".gif", ".bmp", ".webp"}]
    prompts = list((输出根目录 / "05_图表提示词").glob("*.txt"))
    prompt_ok = len(prompts) == 3 and all("MATLAB绘图要求" in path.read_text(encoding="utf-8") and "SVG输出要求" in path.read_text(encoding="utf-8") for path in prompts)
    验收("图形交付约束", not image_paths and prompt_ok, f"图片={len(image_paths)}；合格MATLAB/SVG提示词={len(prompts)}")

    one_click_bytes = 一键脚本路径.read_bytes()
    验收("Windows复现入口", one_click_bytes.startswith(b"\xef\xbb\xbf") and "第四问建模、验证、自审和哈希已完成" in 一键脚本路径.read_text(encoding="utf-8-sig"), "PowerShell脚本UTF-8 BOM且失败退出")

    checklist = pd.DataFrame(checks)
    checklist_path = 独立复核目录 / "第四问独立总控验收清单.csv"
    checklist.to_csv(checklist_path, index=False, encoding="utf-8-sig")
    failed = checklist.loc[~checklist["通过标志"]]
    status = "PASS" if failed.empty else "REJECTED"
    report_lines = [
        "# 第四问独立总控复核报告",
        "",
        f"- 独立审核状态：**{status}**",
        f"- 验收项：{len(checklist)}项；失败：{len(failed)}项。",
        "- 复核方式：不导入建模脚本，从原始Excel、外层预测、完整参数、剪枝路径、整簇抽样序列和文件哈希反向重建。",
        "",
        "## 关键独立结论",
        "",
        "- AB为记录级目标：605条记录、147名孕妇、67条异常；AE全部为是，不能替代AB。",
        "- 主L1路线的147个外层模型、训练内阈值、正则强度和系数均独立重跑并与报告一致。",
        "- BIC剪枝树的147个外层概率模型和树结构独立重跑并与报告一致。",
        f"- 树减逻辑回归的PR曲线下面积差95%区间为[{pr_interval[0]:.6f}, {pr_interval[1]:.6f}]，全为负。",
        "- T13/T18/T21分型标签和指标均复算；T21 Z规则从不报阳性，未定义指标被保留为空值。",
        "- 题面没有漏诊/误报代价比；材料没有设置类别权重、SMOTE或唯一临床阈值。",
        "",
        "## 归档裁决",
        "",
        "全部独立检查通过，允许进入正式目录。" if status == "PASS" else "存在失败项，必须打回，禁止归档。",
    ]
    report_path = 独立复核目录 / "第四问独立总控复核报告.md"
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    pass_record = {
        "审核时间": datetime.now().astimezone().isoformat(),
        "问题": "第四问",
        "状态": status,
        "独立验收项数": int(len(checklist)),
        "失败项数": int(len(failed)),
        "建模运行清单SHA256": 文件哈希(manifest_path),
        "建模结果哈希表SHA256": 文件哈希(hash_table_path),
        "独立复核脚本SHA256": 文件哈希(脚本路径),
        "独立验收清单SHA256": 文件哈希(checklist_path),
        "独立复核报告SHA256": 文件哈希(report_path),
    }
    pass_path = 独立复核目录 / "第四问独立审核PASS记录.json"
    pass_path.write_text(json.dumps(pass_record, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(pass_record, ensure_ascii=False, indent=2), flush=True)
    if status != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
