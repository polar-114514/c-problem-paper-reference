from __future__ import annotations

import hashlib
import json
import math
import platform
import re
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import sklearn
import statsmodels
import statsmodels.api as sm
from scipy.special import expit, logit
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_text


warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")

脚本路径 = Path(__file__).resolve()
项目目录 = 脚本路径.parent
工作区 = 项目目录.parents[2]
工作簿路径 = 工作区 / "00_题目与原始资料/02_原始数据/附件.xlsx"
题目路径 = 工作区 / "00_题目与原始资料/01_题目原文/C题.pdf"
合同路径 = 项目目录 / "00_第四问题意合同与候选设计.md"

输出根目录 = 项目目录 / "正式候选输出"
数据输出目录 = 输出根目录 / "01_数据"
模型输出目录 = 输出根目录 / "02_模型结果"
验证输出目录 = 输出根目录 / "03_验证"
复现输出目录 = 输出根目录 / "04_复现"
图表提示词目录 = 输出根目录 / "05_图表提示词"

内层折数 = 5
正则强度倒数候选 = np.logspace(-4, 4, 10)
整簇自助次数 = 400
区间下分位 = 0.025
区间上分位 = 0.975
标准分数规则阈值 = 3.0
数值概率下限 = np.finfo(float).eps
最大迭代次数 = 5000

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

标签名称 = {
    "任一异常标志": "任一T13_T18_T21异常",
    "T13异常标志": "T13",
    "T18异常标志": "T18",
    "T21异常标志": "T21",
}


def 文件哈希(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def 中文化表头(name: str) -> str:
    translated = str(name)
    translated = translated.replace("SHA256", "安全散列值_SHA256")
    translated = translated.replace("BIC", "贝叶斯信息准则")
    translated = translated.replace("MCC", "马修斯相关系数")
    translated = translated.replace("ROC-AUC", "ROC曲线下面积")
    translated = translated.replace("PR-AUC", "PR曲线下面积")
    translated = translated.replace("Brier", "布里尔")
    return translated


def 写CSV(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    export = frame.rename(columns={column: 中文化表头(column) for column in frame.columns})
    export.to_csv(path, index=False, encoding="utf-8-sig")


def 写JSON(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def 解析孕周(value) -> float:
    text = str(value).strip().lower()
    match = re.fullmatch(r"(\d+)\s*w(?:\s*\+?\s*(\d+)\s*)?", text)
    if not match:
        raise ValueError(f"不能解析孕周：{value!r}")
    week = int(match.group(1))
    day = int(match.group(2) or 0)
    if day < 0 or day > 6:
        raise ValueError(f"孕周天数不在0至6：{value!r}")
    return week + day / 7.0


def 解析Excel日期(value) -> pd.Timestamp:
    if pd.isna(value):
        raise ValueError("检测日期为空")
    if isinstance(value, (int, float, np.integer, np.floating)):
        if not np.isfinite(float(value)):
            raise ValueError(f"检测日期不是有限数值：{value!r}")
        integer_value = int(value)
        if float(value) == integer_value and re.fullmatch(r"\d{8}", str(integer_value)):
            return pd.to_datetime(str(integer_value), format="%Y%m%d", errors="raise")
        return pd.Timestamp("1899-12-30") + pd.to_timedelta(float(value), unit="D")
    return pd.to_datetime(value, errors="raise")


def 解析标签(value) -> frozenset[str]:
    if pd.isna(value) or str(value).strip() == "":
        return frozenset()
    tokens = re.findall(r"T(?:13|18|21)", str(value).upper())
    if not tokens:
        raise ValueError(f"未知AB标签：{value!r}")
    return frozenset(tokens)


def 随机种子() -> int:
    return int(文件哈希(工作簿路径)[:8], 16)


def 构造数据():
    raw = pd.read_excel(工作簿路径, sheet_name="女胎检测数据", engine="openpyxl")
    raw["孕周数"] = raw["检测孕周"].map(解析孕周)
    raw["检测日期标准化"] = raw["检测日期"].map(解析Excel日期).dt.strftime("%Y-%m-%d")
    raw["标签集合"] = raw["染色体的非整倍体"].map(解析标签)
    raw["任一异常标志"] = raw["标签集合"].map(bool).astype(int)
    for chromosome in ["T13", "T18", "T21"]:
        raw[f"{chromosome}异常标志"] = raw["标签集合"].map(lambda labels: int(chromosome in labels))
    raw["AE不健康标志"] = raw["胎儿是否健康"].astype(str).str.strip().eq("否").astype(int)
    raw["抽血事件键"] = (
        raw["孕妇代码"].astype(str)
        + "|"
        + raw["检测日期标准化"]
        + "|"
        + raw["检测抽血次数"].astype(str)
        + "|"
        + raw["检测孕周"].astype(str)
    )
    raw["记录行号"] = np.arange(1, len(raw) + 1)

    data = pd.DataFrame(
        {
            "记录行号": raw["记录行号"].astype(int),
            "原始序号": raw["序号"],
            "孕妇代码": raw["孕妇代码"].astype(str),
            "检测日期": raw["检测日期标准化"],
            "检测孕周原文": raw["检测孕周"].astype(str),
            "抽血事件键": raw["抽血事件键"],
            "AB原始标签": raw["染色体的非整倍体"].fillna("空白=无异常").astype(str),
            "AE出生健康结果": raw["胎儿是否健康"].astype(str),
            "任一异常标志": raw["任一异常标志"].astype(int),
            "T13异常标志": raw["T13异常标志"].astype(int),
            "T18异常标志": raw["T18异常标志"].astype(int),
            "T21异常标志": raw["T21异常标志"].astype(int),
            "孕周数": raw["孕周数"],
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
    data["绝对Z规则连续分数"] = data[["13号染色体Z值", "18号染色体Z值", "21号染色体Z值"]].abs().max(axis=1)

    woman = data.groupby("孕妇代码", sort=True).agg(
        记录数=("记录行号", "size"),
        AB异常记录数=("任一异常标志", "sum"),
        AE不健康记录数=("AE出生健康结果", lambda values: int(pd.Series(values).eq("否").sum())),
        AB原始标签种类数=("AB原始标签", "nunique"),
    )
    woman["AB异常孕妇标志"] = woman["AB异常记录数"].gt(0).astype(int)
    woman["AB阴阳混合标志"] = woman["AB异常记录数"].between(1, woman["记录数"] - 1).astype(int)

    event = data.groupby("抽血事件键", sort=True).agg(
        孕妇代码=("孕妇代码", "first"),
        检测日期=("检测日期", "first"),
        检测孕周原文=("检测孕周原文", "first"),
        记录数=("记录行号", "size"),
        AB状态数=("任一异常标志", "nunique"),
        AB具体标签数=("AB原始标签", "nunique"),
    )

    audit = {
        "工作簿SHA256": 文件哈希(工作簿路径),
        "记录数": int(len(data)),
        "孕妇数": int(data["孕妇代码"].nunique()),
        "抽血事件数": int(data["抽血事件键"].nunique()),
        "多记录事件组数": int(event["记录数"].gt(1).sum()),
        "多记录事件所含记录数": int(event.loc[event["记录数"].gt(1), "记录数"].sum()),
        "同事件AB阴阳不一致组数": int(event.loc[event["记录数"].gt(1), "AB状态数"].gt(1).sum()),
        "AB异常记录数": int(data["任一异常标志"].sum()),
        "AB正常记录数": int(data["任一异常标志"].eq(0).sum()),
        "AB异常孕妇数": int(woman["AB异常孕妇标志"].sum()),
        "AB阴阳混合孕妇数": int(woman["AB阴阳混合标志"].sum()),
        "AE不健康记录数": int(data["AE出生健康结果"].eq("否").sum()),
        "AB异常但AE健康记录数": int((data["任一异常标志"].eq(1) & data["AE出生健康结果"].eq("是")).sum()),
        "AB异常但AE健康孕妇数": int((woman["AB异常孕妇标志"].eq(1) & woman["AE不健康记录数"].eq(0)).sum()),
        "T13阳性记录数": int(data["T13异常标志"].sum()),
        "T18阳性记录数": int(data["T18异常标志"].sum()),
        "T21阳性记录数": int(data["T21异常标志"].sum()),
        "BMI缺失记录数": int(data["孕妇体质指数"].isna().sum()),
        "题面10至25周外记录数": int((~data["孕周数"].between(10, 25, inclusive="both")).sum()),
        "最早检测日期": str(data["检测日期"].min()),
        "最晚检测日期": str(data["检测日期"].max()),
        "检测日期四位年份格式全部成立": bool(data["检测日期"].str.fullmatch(r"\d{4}-\d{2}-\d{2}").all()),
        "孕妇分组与等权标志": True,
    }
    return data, woman.reset_index(), event.reset_index(), audit


def 等孕妇权重(frame: pd.DataFrame) -> np.ndarray:
    counts = frame["孕妇代码"].value_counts()
    return frame["孕妇代码"].map(lambda value: 1.0 / counts[value]).to_numpy(float)


def 构造分层分组器(target_name: str, folds: int):
    target_offset = sum(ord(character) for character in target_name)
    return StratifiedGroupKFold(
        n_splits=folds,
        shuffle=True,
        random_state=(随机种子() + target_offset + folds) % (2**32 - 1),
    )


def 拟合预处理(train: pd.DataFrame):
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    transformed = imputer.fit_transform(train[特征名称])
    transformed = scaler.fit_transform(transformed)
    return imputer, scaler, transformed


def 应用预处理(frame: pd.DataFrame, imputer, scaler):
    return scaler.transform(imputer.transform(frame[特征名称]))


def 新建逻辑回归(c_value: float):
    return LogisticRegression(
        l1_ratio=1.0,
        solver="liblinear",
        C=float(c_value),
        max_iter=最大迭代次数,
        random_state=随机种子(),
    )


def 加权混淆(y_true, y_pred, weight):
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    weight = np.asarray(weight, dtype=float)
    return {
        "真阳性": float(np.sum(weight[(y_true == 1) & (y_pred == 1)])),
        "假阳性": float(np.sum(weight[(y_true == 0) & (y_pred == 1)])),
        "假阴性": float(np.sum(weight[(y_true == 1) & (y_pred == 0)])),
        "真阴性": float(np.sum(weight[(y_true == 0) & (y_pred == 0)])),
    }


def 安全比值(numerator: float, denominator: float):
    return float(numerator / denominator) if denominator > 0 else np.nan


def 混淆指标(y_true, y_pred, weight):
    cells = 加权混淆(y_true, y_pred, weight)
    tp, fp, fn, tn = cells["真阳性"], cells["假阳性"], cells["假阴性"], cells["真阴性"]
    sensitivity = 安全比值(tp, tp + fn)
    specificity = 安全比值(tn, tn + fp)
    precision = 安全比值(tp, tp + fp)
    f1 = 安全比值(2 * precision * sensitivity, precision + sensitivity) if np.isfinite(precision) and np.isfinite(sensitivity) else np.nan
    denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = 安全比值(tp * tn - fp * fn, denominator)
    balanced = np.nanmean([sensitivity, specificity])
    accuracy = 安全比值(tp + tn, tp + fp + fn + tn)
    return {
        **cells,
        "灵敏度": sensitivity,
        "特异度": specificity,
        "精确率": precision,
        "F1分数": f1,
        "马修斯相关系数": mcc,
        "平衡准确率": balanced,
        "准确率": accuracy,
    }


def 选择训练内阈值(y_true, probability, weight):
    probability = np.asarray(probability, dtype=float)
    candidates = np.r_[np.nextafter(np.nanmax(probability), np.inf), np.unique(probability)[::-1]]
    rows = []
    for threshold in candidates:
        metrics = 混淆指标(y_true, probability >= threshold, weight)
        rows.append({"阈值": float(threshold), **metrics})
    curve = pd.DataFrame(rows)
    ranking = curve[["马修斯相关系数", "F1分数", "平衡准确率"]].fillna(-np.inf)
    best_tuple = max(map(tuple, ranking.to_numpy(float)))
    tied = curve.loc[
        ranking.apply(lambda row: tuple(row.to_numpy(float)) == best_tuple, axis=1)
    ]
    threshold = float(np.median(tied["阈值"].to_numpy(float)))
    return threshold, curve


def 计算概率校准(y_true, probability, weight):
    clipped = np.clip(np.asarray(probability, dtype=float), 数值概率下限, 1 - 数值概率下限)
    design = sm.add_constant(logit(clipped), has_constant="add")
    try:
        fit = sm.GLM(np.asarray(y_true, dtype=int), design, family=sm.families.Binomial(), freq_weights=weight).fit()
        intercept = float(fit.params[0])
        slope = float(fit.params[1])
    except Exception:
        intercept, slope = np.nan, np.nan
    observed = float(np.average(y_true, weights=weight))
    predicted = float(np.average(clipped, weights=weight))
    return intercept, slope, observed, predicted


def 计算路线指标(frame: pd.DataFrame, weight: np.ndarray, probability_route: bool):
    y_true = frame["实际异常标志"].to_numpy(int)
    score = frame["连续评分"].to_numpy(float)
    y_pred = frame["预测异常标志"].to_numpy(int)
    metrics = 混淆指标(y_true, y_pred, weight)
    metrics["ROC曲线下面积"] = float(roc_auc_score(y_true, score, sample_weight=weight))
    metrics["PR曲线下面积"] = float(average_precision_score(y_true, score, sample_weight=weight))
    metrics["实际异常比例"] = float(np.average(y_true, weights=weight))
    metrics["预测异常比例"] = float(np.average(y_pred, weights=weight))
    if probability_route:
        probability = np.clip(frame["预测概率"].to_numpy(float), 数值概率下限, 1 - 数值概率下限)
        metrics["布里尔分数"] = float(np.average((probability - y_true) ** 2, weights=weight))
        metrics["对数损失"] = float(log_loss(y_true, probability, sample_weight=weight, labels=[0, 1]))
        intercept, slope, observed, predicted = 计算概率校准(y_true, probability, weight)
        metrics["校准截距"] = intercept
        metrics["校准斜率"] = slope
        metrics["校准实际异常比例"] = observed
        metrics["校准平均预测概率"] = predicted
    else:
        for name in ["布里尔分数", "对数损失", "校准截距", "校准斜率", "校准实际异常比例", "校准平均预测概率"]:
            metrics[name] = np.nan
    return metrics


def 拟合逻辑路线(train: pd.DataFrame, target: str, folds: int = 内层折数):
    splitter = 构造分层分组器(target, folds)
    cross_predictions = np.full((len(正则强度倒数候选), len(train)), np.nan)
    validation_weight = np.zeros(len(train), dtype=float)
    convergence = []
    for fit_index, validation_index in splitter.split(train, train[target], train["孕妇代码"]):
        fit_frame = train.iloc[fit_index]
        validation_frame = train.iloc[validation_index]
        if fit_frame[target].nunique() != 2 or validation_frame[target].nunique() != 2:
            raise RuntimeError(f"{target}的{folds}折内层分组出现单类折")
        imputer, scaler, x_fit = 拟合预处理(fit_frame)
        x_validation = 应用预处理(validation_frame, imputer, scaler)
        fit_weight = 等孕妇权重(fit_frame)
        validation_weight[validation_index] = 等孕妇权重(validation_frame)
        for candidate_index, c_value in enumerate(正则强度倒数候选):
            model = 新建逻辑回归(float(c_value))
            model.fit(x_fit, fit_frame[target].to_numpy(int), sample_weight=fit_weight)
            convergence.append(int(np.max(model.n_iter_) < 最大迭代次数))
            cross_predictions[candidate_index, validation_index] = model.predict_proba(x_validation)[:, 1]
    if not np.isfinite(cross_predictions).all() or not all(convergence):
        raise RuntimeError(f"{target}的L1逻辑回归内层预测不完整或未收敛")
    losses = np.array(
        [
            log_loss(train[target], prediction, sample_weight=validation_weight, labels=[0, 1])
            for prediction in cross_predictions
        ],
        dtype=float,
    )
    selected_index = int(np.argmin(losses))
    selected_c = float(正则强度倒数候选[selected_index])
    selected_cross_prediction = cross_predictions[selected_index]
    threshold, threshold_curve = 选择训练内阈值(
        train[target].to_numpy(int), selected_cross_prediction, validation_weight
    )
    imputer, scaler, x_train = 拟合预处理(train)
    model = 新建逻辑回归(selected_c)
    model.fit(x_train, train[target].to_numpy(int), sample_weight=等孕妇权重(train))
    if int(np.max(model.n_iter_)) >= 最大迭代次数:
        raise RuntimeError(f"{target}的L1逻辑回归全训练拟合未收敛")
    return {
        "路线": "L1正则多因素逻辑回归",
        "模型": model,
        "填补器": imputer,
        "标准化器": scaler,
        "正则强度倒数": selected_c,
        "训练内对数损失": float(losses[selected_index]),
        "训练内阈值": threshold,
        "训练内交叉拟合概率": selected_cross_prediction,
        "训练内阈值曲线": threshold_curve,
        "收敛": True,
    }


def 拟合剪枝树(train: pd.DataFrame, target: str):
    imputer = SimpleImputer(strategy="median")
    x_train = imputer.fit_transform(train[特征名称])
    weight = 等孕妇权重(train)
    base = DecisionTreeClassifier(random_state=随机种子())
    path = base.cost_complexity_pruning_path(x_train, train[target].to_numpy(int), sample_weight=weight)
    candidate_alphas = np.unique(path.ccp_alphas)
    group_count = int(train["孕妇代码"].nunique())
    candidates = []
    for alpha_value in candidate_alphas:
        model = DecisionTreeClassifier(random_state=随机种子(), ccp_alpha=float(alpha_value))
        model.fit(x_train, train[target].to_numpy(int), sample_weight=weight)
        probability = np.clip(model.predict_proba(x_train)[:, 1], 数值概率下限, 1 - 数值概率下限)
        y = train[target].to_numpy(int)
        log_likelihood = float(np.sum(weight * (y * np.log(probability) + (1 - y) * np.log(1 - probability))))
        leaves = int(model.get_n_leaves())
        bic = float(-2 * log_likelihood + leaves * np.log(group_count))
        candidates.append(
            {
                "贝叶斯信息准则": bic,
                "叶节点数": leaves,
                "剪枝复杂度": float(alpha_value),
                "模型": model,
            }
        )
    selected = min(candidates, key=lambda item: (item["贝叶斯信息准则"], item["叶节点数"], item["剪枝复杂度"]))
    selected["路线"] = "贝叶斯信息准则剪枝决策树"
    selected["填补器"] = imputer
    selected["候选树数"] = int(len(candidates))
    return selected


def 剪枝树训练内阈值(train: pd.DataFrame, target: str, folds: int = 内层折数):
    splitter = 构造分层分组器(target + "树", folds)
    prediction = np.full(len(train), np.nan)
    validation_weight = np.zeros(len(train), dtype=float)
    for fit_index, validation_index in splitter.split(train, train[target], train["孕妇代码"]):
        fit_frame = train.iloc[fit_index]
        validation_frame = train.iloc[validation_index]
        if fit_frame[target].nunique() != 2 or validation_frame[target].nunique() != 2:
            raise RuntimeError(f"{target}的树模型{folds}折内层分组出现单类折")
        adapter = 拟合剪枝树(fit_frame, target)
        x_validation = adapter["填补器"].transform(validation_frame[特征名称])
        prediction[validation_index] = adapter["模型"].predict_proba(x_validation)[:, 1]
        validation_weight[validation_index] = 等孕妇权重(validation_frame)
    if not np.isfinite(prediction).all():
        raise RuntimeError(f"{target}的剪枝树内层预测不完整")
    threshold, curve = 选择训练内阈值(train[target].to_numpy(int), prediction, validation_weight)
    return threshold, prediction, curve


def 预测逻辑(adapter, test: pd.DataFrame):
    x_test = 应用预处理(test, adapter["填补器"], adapter["标准化器"])
    return adapter["模型"].predict_proba(x_test)[:, 1]


def 预测树(adapter, test: pd.DataFrame):
    x_test = adapter["填补器"].transform(test[特征名称])
    return adapter["模型"].predict_proba(x_test)[:, 1]


def 标准分数基准(data: pd.DataFrame, target: str):
    if target == "任一异常标志":
        score = data["Z规则连续分数"].to_numpy(float)
    else:
        chromosome = target.replace("T", "").replace("异常标志", "")
        score = data[f"{chromosome}号染色体Z值"].to_numpy(float)
    return score, (score >= 标准分数规则阈值).astype(int)


def 逐孕妇外层候选验证(data: pd.DataFrame, target: str):
    groups = sorted(data["孕妇代码"].unique().tolist())
    prediction_rows = []
    threshold_rows = []
    coefficient_rows = []
    tree_rows = []
    baseline_score, baseline_class = 标准分数基准(data, target)
    for row_index, row in data.iterrows():
        prediction_rows.append(
            {
                "目标": 标签名称[target],
                "路线": "三标准差Z值规则基准",
                "记录行号": int(row["记录行号"]),
                "原始序号": row["原始序号"],
                "孕妇代码": row["孕妇代码"],
                "外层留出孕妇": row["孕妇代码"],
                "实际异常标志": int(row[target]),
                "连续评分": float(baseline_score[row_index]),
                "预测概率": np.nan,
                "训练内阈值": 标准分数规则阈值,
                "预测异常标志": int(baseline_class[row_index]),
                "概率输出标志": 0,
                "多因素综合标志": 0,
                "外层参数": "领域阈值Z=3",
            }
        )
    threshold_rows.append(
        {
            "目标": 标签名称[target],
            "路线": "三标准差Z值规则基准",
            "外层留出孕妇": "固定领域规则",
            "训练内阈值": 标准分数规则阈值,
            "阈值来源": "文献三标准差规则",
        }
    )

    for outer_number, held_out in enumerate(groups, start=1):
        test_mask = data["孕妇代码"].eq(held_out).to_numpy()
        train = data.loc[~test_mask].reset_index(drop=True)
        test = data.loc[test_mask]
        if set(train["孕妇代码"]).intersection(set(test["孕妇代码"])):
            raise RuntimeError("外层逐孕妇留一出现组重叠")

        logistic = 拟合逻辑路线(train, target)
        logistic_probability = 预测逻辑(logistic, test)
        logistic_class = (logistic_probability >= logistic["训练内阈值"]).astype(int)
        threshold_rows.append(
            {
                "目标": 标签名称[target],
                "路线": logistic["路线"],
                "外层留出孕妇": held_out,
                "训练内阈值": logistic["训练内阈值"],
                "阈值来源": "外层训练集内5折分层分组交叉拟合概率的MCC最大点",
                "正则强度倒数": logistic["正则强度倒数"],
                "训练内对数损失": logistic["训练内对数损失"],
            }
        )
        coefficients = logistic["模型"].coef_[0]
        for feature, coefficient in zip(特征名称, coefficients):
            coefficient_rows.append(
                {
                    "目标": 标签名称[target],
                    "外层留出孕妇": held_out,
                    "特征": feature,
                    "标准化系数": float(coefficient),
                    "非零标志": int(coefficient != 0.0),
                    "正则强度倒数": logistic["正则强度倒数"],
                }
            )
        for local_index, (_, row) in enumerate(test.iterrows()):
            prediction_rows.append(
                {
                    "目标": 标签名称[target],
                    "路线": logistic["路线"],
                    "记录行号": int(row["记录行号"]),
                    "原始序号": row["原始序号"],
                    "孕妇代码": row["孕妇代码"],
                    "外层留出孕妇": held_out,
                    "实际异常标志": int(row[target]),
                    "连续评分": float(logistic_probability[local_index]),
                    "预测概率": float(logistic_probability[local_index]),
                    "训练内阈值": logistic["训练内阈值"],
                    "预测异常标志": int(logistic_class[local_index]),
                    "概率输出标志": 1,
                    "多因素综合标志": 1,
                    "外层参数": f"正则强度倒数={logistic['正则强度倒数']:.12g}",
                }
            )

        tree_threshold, _, _ = 剪枝树训练内阈值(train, target)
        tree = 拟合剪枝树(train, target)
        tree_probability = 预测树(tree, test)
        tree_class = (tree_probability >= tree_threshold).astype(int)
        threshold_rows.append(
            {
                "目标": 标签名称[target],
                "路线": tree["路线"],
                "外层留出孕妇": held_out,
                "训练内阈值": tree_threshold,
                "阈值来源": "外层训练集内5折分层分组交叉拟合概率的MCC最大点",
                "剪枝复杂度": tree["剪枝复杂度"],
                "叶节点数": tree["叶节点数"],
            }
        )
        tree_rows.append(
            {
                "目标": 标签名称[target],
                "外层留出孕妇": held_out,
                "剪枝复杂度": tree["剪枝复杂度"],
                "叶节点数": tree["叶节点数"],
                "贝叶斯信息准则": tree["贝叶斯信息准则"],
                "候选树数": tree["候选树数"],
            }
        )
        for local_index, (_, row) in enumerate(test.iterrows()):
            prediction_rows.append(
                {
                    "目标": 标签名称[target],
                    "路线": tree["路线"],
                    "记录行号": int(row["记录行号"]),
                    "原始序号": row["原始序号"],
                    "孕妇代码": row["孕妇代码"],
                    "外层留出孕妇": held_out,
                    "实际异常标志": int(row[target]),
                    "连续评分": float(tree_probability[local_index]),
                    "预测概率": float(tree_probability[local_index]),
                    "训练内阈值": tree_threshold,
                    "预测异常标志": int(tree_class[local_index]),
                    "概率输出标志": 1,
                    "多因素综合标志": 1,
                    "外层参数": f"叶节点数={tree['叶节点数']}",
                }
            )
        if outer_number % 20 == 0 or outer_number == len(groups):
            print(f"{标签名称[target]}逐孕妇外层验证 {outer_number}/{len(groups)}", flush=True)
    return (
        pd.DataFrame(prediction_rows),
        pd.DataFrame(threshold_rows),
        pd.DataFrame(coefficient_rows),
        pd.DataFrame(tree_rows),
    )


def 汇总候选路线(predictions: pd.DataFrame):
    rows = []
    for route, frame in predictions.groupby("路线", sort=False):
        weight = 等孕妇权重(frame)
        probability_route = bool(frame["概率输出标志"].eq(1).all())
        metrics = 计算路线指标(frame, weight, probability_route)
        rows.append(
            {
                "路线": route,
                "外层孕妇数": int(frame["孕妇代码"].nunique()),
                "外层记录数": int(len(frame)),
                "异常记录数": int(frame["实际异常标志"].sum()),
                "概率输出标志": int(probability_route),
                "多因素综合标志": int(frame["多因素综合标志"].eq(1).all()),
                **metrics,
            }
        )
    return pd.DataFrame(rows)


def 孕妇整簇抽样权重(frame: pd.DataFrame, sampled_groups: np.ndarray):
    multiplicity = pd.Series(sampled_groups).value_counts()
    record_counts = frame["孕妇代码"].value_counts()
    return frame["孕妇代码"].map(
        lambda group: float(multiplicity.get(group, 0)) / float(record_counts[group])
    ).to_numpy(float)


def 候选路线整簇自助(predictions: pd.DataFrame):
    routes = predictions["路线"].drop_duplicates().tolist()
    groups = np.array(sorted(predictions["孕妇代码"].unique().tolist()), dtype=object)
    rng = np.random.default_rng(随机种子())
    detail_rows = []
    for repeat in range(1, 整簇自助次数 + 1):
        sampled = rng.choice(groups, size=len(groups), replace=True)
        for route in routes:
            frame = predictions.loc[predictions["路线"].eq(route)].reset_index(drop=True)
            weight = 孕妇整簇抽样权重(frame, sampled)
            if weight[frame["实际异常标志"].eq(1).to_numpy()].sum() == 0 or weight[frame["实际异常标志"].eq(0).to_numpy()].sum() == 0:
                continue
            metrics = 计算路线指标(frame, weight, bool(frame["概率输出标志"].eq(1).all()))
            detail_rows.append({"自助序号": repeat, "路线": route, **metrics})
    detail = pd.DataFrame(detail_rows)
    if detail.groupby("路线")["自助序号"].nunique().min() != 整簇自助次数:
        raise RuntimeError("候选路线孕妇整簇自助存在无效重复")
    summary_rows = []
    metric_names = [
        "灵敏度",
        "特异度",
        "精确率",
        "F1分数",
        "马修斯相关系数",
        "ROC曲线下面积",
        "PR曲线下面积",
        "布里尔分数",
        "对数损失",
        "校准截距",
        "校准斜率",
    ]
    for prefix in [100, 200, 400]:
        for route in routes:
            subset = detail.loc[(detail["路线"].eq(route)) & (detail["自助序号"].le(prefix))]
            for metric in metric_names:
                values = subset[metric].dropna().to_numpy(float)
                if not len(values):
                    continue
                summary_rows.append(
                    {
                        "请求前缀次数": prefix,
                        "有效次数": int(len(values)),
                        "路线": route,
                        "统计量": metric,
                        "中位数": float(np.median(values)),
                        "2.5%分位": float(np.quantile(values, 区间下分位)),
                        "97.5%分位": float(np.quantile(values, 区间上分位)),
                    }
                )
    summary = pd.DataFrame(summary_rows)
    pivot = detail.pivot(index="自助序号", columns="路线", values=["PR曲线下面积", "ROC曲线下面积", "对数损失"])
    difference_rows = []
    for repeat in pivot.index:
        difference_rows.append(
            {
                "自助序号": int(repeat),
                "剪枝树减逻辑回归_PR曲线下面积差": float(
                    pivot.loc[repeat, ("PR曲线下面积", "贝叶斯信息准则剪枝决策树")]
                    - pivot.loc[repeat, ("PR曲线下面积", "L1正则多因素逻辑回归")]
                ),
                "剪枝树减逻辑回归_ROC曲线下面积差": float(
                    pivot.loc[repeat, ("ROC曲线下面积", "贝叶斯信息准则剪枝决策树")]
                    - pivot.loc[repeat, ("ROC曲线下面积", "L1正则多因素逻辑回归")]
                ),
                "剪枝树减逻辑回归_对数损失差": float(
                    pivot.loc[repeat, ("对数损失", "贝叶斯信息准则剪枝决策树")]
                    - pivot.loc[repeat, ("对数损失", "L1正则多因素逻辑回归")]
                ),
            }
        )
    difference = pd.DataFrame(difference_rows)
    return detail, summary, difference


def 选择主路线(candidate_summary: pd.DataFrame, difference: pd.DataFrame):
    eligible = candidate_summary.loc[
        candidate_summary["多因素综合标志"].eq(1) & candidate_summary["概率输出标志"].eq(1)
    ].copy()
    if set(eligible["路线"]) != {"L1正则多因素逻辑回归", "贝叶斯信息准则剪枝决策树"}:
        raise RuntimeError("合格学习路线集合不完整")
    pr_difference = difference["剪枝树减逻辑回归_PR曲线下面积差"].to_numpy(float)
    lower, upper = np.quantile(pr_difference, [区间下分位, 区间上分位])
    if lower > 0:
        selected = "贝叶斯信息准则剪枝决策树"
        reason = "剪枝树相对逻辑回归的孕妇整簇PR曲线下面积差95%区间全为正"
    else:
        selected = "L1正则多因素逻辑回归"
        reason = "剪枝树未以整簇PR曲线下面积差95%区间全正证明优于可解释逻辑回归，按预声明简洁性规则选择逻辑回归"
    return selected, reason, float(lower), float(upper)


def 逐孕妇外层入选路线验证(data: pd.DataFrame, target: str, selected_route: str):
    groups = sorted(data["孕妇代码"].unique().tolist())
    prediction_rows = []
    threshold_rows = []
    baseline_score, baseline_class = 标准分数基准(data, target)
    for row_index, row in data.iterrows():
        prediction_rows.append(
            {
                "目标": 标签名称[target],
                "路线": "对应染色体三标准差Z值规则",
                "记录行号": int(row["记录行号"]),
                "孕妇代码": row["孕妇代码"],
                "实际异常标志": int(row[target]),
                "连续评分": float(baseline_score[row_index]),
                "预测概率": np.nan,
                "训练内阈值": 标准分数规则阈值,
                "预测异常标志": int(baseline_class[row_index]),
                "概率输出标志": 0,
            }
        )
    for outer_number, held_out in enumerate(groups, start=1):
        test_mask = data["孕妇代码"].eq(held_out).to_numpy()
        train = data.loc[~test_mask].reset_index(drop=True)
        test = data.loc[test_mask]
        if selected_route == "L1正则多因素逻辑回归":
            adapter = 拟合逻辑路线(train, target)
            probability = 预测逻辑(adapter, test)
            threshold = adapter["训练内阈值"]
            parameter = f"正则强度倒数={adapter['正则强度倒数']:.12g}"
        else:
            threshold, _, _ = 剪枝树训练内阈值(train, target)
            adapter = 拟合剪枝树(train, target)
            probability = 预测树(adapter, test)
            parameter = f"叶节点数={adapter['叶节点数']}"
        prediction = (probability >= threshold).astype(int)
        threshold_rows.append(
            {
                "目标": 标签名称[target],
                "路线": selected_route,
                "外层留出孕妇": held_out,
                "训练内阈值": threshold,
                "外层参数": parameter,
            }
        )
        for local_index, (_, row) in enumerate(test.iterrows()):
            prediction_rows.append(
                {
                    "目标": 标签名称[target],
                    "路线": selected_route,
                    "记录行号": int(row["记录行号"]),
                    "孕妇代码": row["孕妇代码"],
                    "实际异常标志": int(row[target]),
                    "连续评分": float(probability[local_index]),
                    "预测概率": float(probability[local_index]),
                    "训练内阈值": threshold,
                    "预测异常标志": int(prediction[local_index]),
                    "概率输出标志": 1,
                }
            )
        if outer_number % 30 == 0 or outer_number == len(groups):
            print(f"{标签名称[target]}分型外层验证 {outer_number}/{len(groups)}", flush=True)
    return pd.DataFrame(prediction_rows), pd.DataFrame(threshold_rows)


def 类型指标与整簇区间(type_predictions: pd.DataFrame):
    summary_rows = []
    bootstrap_rows = []
    rng = np.random.default_rng((随机种子() + 2025) % (2**32 - 1))
    groups = np.array(sorted(type_predictions["孕妇代码"].unique().tolist()), dtype=object)
    for (target, route), frame in type_predictions.groupby(["目标", "路线"], sort=False):
        frame = frame.reset_index(drop=True)
        probability_route = bool(frame["概率输出标志"].eq(1).all())
        point = 计算路线指标(frame, 等孕妇权重(frame), probability_route)
        summary_rows.append(
            {
                "异常类型": target,
                "路线": route,
                "阳性记录数": int(frame["实际异常标志"].sum()),
                "阳性孕妇数": int(frame.loc[frame["实际异常标志"].eq(1), "孕妇代码"].nunique()),
                **point,
            }
        )
    sampled_sets = [rng.choice(groups, size=len(groups), replace=True) for _ in range(整簇自助次数)]
    for repeat, sampled in enumerate(sampled_sets, start=1):
        for (target, route), frame in type_predictions.groupby(["目标", "路线"], sort=False):
            frame = frame.reset_index(drop=True)
            weight = 孕妇整簇抽样权重(frame, sampled)
            if weight[frame["实际异常标志"].eq(1).to_numpy()].sum() == 0:
                continue
            metrics = 计算路线指标(frame, weight, bool(frame["概率输出标志"].eq(1).all()))
            bootstrap_rows.append({"自助序号": repeat, "异常类型": target, "路线": route, **metrics})
    bootstrap = pd.DataFrame(bootstrap_rows)
    intervals = []
    for (target, route), frame in bootstrap.groupby(["异常类型", "路线"], sort=False):
        for metric in ["灵敏度", "特异度", "精确率", "F1分数", "马修斯相关系数", "ROC曲线下面积", "PR曲线下面积"]:
            values = frame[metric].dropna().to_numpy(float)
            intervals.append(
                {
                    "异常类型": target,
                    "路线": route,
                    "统计量": metric,
                    "有效次数": int(len(values)),
                    "中位数": float(np.median(values)) if len(values) else np.nan,
                    "2.5%分位": float(np.quantile(values, 区间下分位)) if len(values) else np.nan,
                    "97.5%分位": float(np.quantile(values, 区间上分位)) if len(values) else np.nan,
                }
            )
    return pd.DataFrame(summary_rows), bootstrap, pd.DataFrame(intervals)


def 完整阈值性能曲线(frame: pd.DataFrame):
    probability = frame["预测概率"].to_numpy(float)
    y_true = frame["实际异常标志"].to_numpy(int)
    weight = 等孕妇权重(frame)
    candidates = np.r_[np.nextafter(np.max(probability), np.inf), np.unique(probability)[::-1]]
    rows = []
    for threshold in candidates:
        prediction = (probability >= threshold).astype(int)
        metrics = 混淆指标(y_true, prediction, weight)
        rows.append(
            {
                "概率阈值": float(threshold),
                **metrics,
                "阳性判定比例": float(np.average(prediction, weights=weight)),
            }
        )
    curve = pd.DataFrame(rows)
    maximum = float(curve["马修斯相关系数"].max())
    curve["MCC最大参考点标志"] = curve["马修斯相关系数"].eq(maximum).astype(int)
    return curve


def 校准分箱(frame: pd.DataFrame):
    work = frame.copy().reset_index(drop=True)
    work["校准组"] = pd.qcut(work["预测概率"].rank(method="first"), q=10, labels=False) + 1
    work["孕妇等权"] = 等孕妇权重(work)
    rows = []
    for group, part in work.groupby("校准组", sort=True):
        rows.append(
            {
                "校准十分位组": int(group),
                "记录数": int(len(part)),
                "孕妇数": int(part["孕妇代码"].nunique()),
                "概率下限": float(part["预测概率"].min()),
                "概率上限": float(part["预测概率"].max()),
                "等孕妇权平均预测概率": float(np.average(part["预测概率"], weights=part["孕妇等权"])),
                "等孕妇权实际异常比例": float(np.average(part["实际异常标志"], weights=part["孕妇等权"])),
            }
        )
    return pd.DataFrame(rows)


def 留一系数稳定性(coefficient_detail: pd.DataFrame):
    rows = []
    for feature, frame in coefficient_detail.groupby("特征", sort=False):
        values = frame["标准化系数"].to_numpy(float)
        rows.append(
            {
                "特征": feature,
                "外层拟合次数": int(len(values)),
                "非零比例": float(np.mean(values != 0.0)),
                "正系数比例": float(np.mean(values > 0.0)),
                "负系数比例": float(np.mean(values < 0.0)),
                "标准化系数中位数": float(np.median(values)),
                "标准化系数2.5%分位": float(np.quantile(values, 区间下分位)),
                "标准化系数97.5%分位": float(np.quantile(values, 区间上分位)),
                "正则强度倒数中位数": float(np.median(frame["正则强度倒数"])),
            }
        )
    return pd.DataFrame(rows)


def 全样本入选模型(data: pd.DataFrame, target: str, selected_route: str):
    if selected_route == "L1正则多因素逻辑回归":
        adapter = 拟合逻辑路线(data.reset_index(drop=True), target)
        coefficients = adapter["模型"].coef_[0]
        intercept_standard = float(adapter["模型"].intercept_[0])
        means = adapter["标准化器"].mean_
        scales = adapter["标准化器"].scale_
        medians = adapter["填补器"].statistics_
        raw_coefficients = coefficients / scales
        raw_intercept = float(intercept_standard - np.sum(coefficients * means / scales))
        parameter_rows = [
            {
                "参数": "截距",
                "标准化模型估计值": intercept_standard,
                "原变换尺度估计值": raw_intercept,
                "非零标志": 1,
                "优势尺度倍数": np.nan,
            }
        ]
        reference_rows = []
        for feature, median, mean, scale, coefficient, raw_coefficient in zip(
            特征名称, medians, means, scales, coefficients, raw_coefficients
        ):
            parameter_rows.append(
                {
                    "参数": feature,
                    "标准化模型估计值": float(coefficient),
                    "原变换尺度估计值": float(raw_coefficient),
                    "非零标志": int(coefficient != 0.0),
                    "优势尺度倍数": float(np.exp(coefficient)),
                }
            )
            reference_rows.append(
                {
                    "特征": feature,
                    "训练中位数填补值": float(median),
                    "训练标准化均值": float(mean),
                    "训练标准化尺度": float(scale),
                    "输入变换": "log(1+x)" if feature in {"原始读段数_对数", "唯一比对读段数_对数"} else "原值",
                }
            )
        parameter_rows.extend(
            [
                {
                    "参数": "正则强度倒数",
                    "标准化模型估计值": adapter["正则强度倒数"],
                    "原变换尺度估计值": np.nan,
                    "非零标志": np.nan,
                    "优势尺度倍数": np.nan,
                },
                {
                    "参数": "训练内MCC参考阈值",
                    "标准化模型估计值": adapter["训练内阈值"],
                    "原变换尺度估计值": np.nan,
                    "非零标志": np.nan,
                    "优势尺度倍数": np.nan,
                },
            ]
        )
        return adapter, pd.DataFrame(parameter_rows), pd.DataFrame(reference_rows), ""
    adapter = 拟合剪枝树(data.reset_index(drop=True), target)
    threshold, _, _ = 剪枝树训练内阈值(data.reset_index(drop=True), target)
    parameter_rows = [
        {"参数": "剪枝复杂度", "估计值": adapter["剪枝复杂度"]},
        {"参数": "叶节点数", "估计值": adapter["叶节点数"]},
        {"参数": "贝叶斯信息准则", "估计值": adapter["贝叶斯信息准则"]},
        {"参数": "训练内MCC参考阈值", "估计值": threshold},
    ]
    reference_rows = [
        {"特征": feature, "训练中位数填补值": float(value), "输入变换": "log(1+x)" if feature in {"原始读段数_对数", "唯一比对读段数_对数"} else "原值"}
        for feature, value in zip(特征名称, adapter["填补器"].statistics_)
    ]
    tree_text = export_text(adapter["模型"], feature_names=特征名称)
    return adapter, pd.DataFrame(parameter_rows), pd.DataFrame(reference_rows), tree_text


def 内层折数敏感性(data: pd.DataFrame, target: str):
    rows = []
    for folds in [4, 5, 10]:
        adapter = 拟合逻辑路线(data.reset_index(drop=True), target, folds=folds)
        rows.append(
            {
                "内层分层分组折数": folds,
                "正则强度倒数": adapter["正则强度倒数"],
                "训练内等孕妇权对数损失": adapter["训练内对数损失"],
                "训练内MCC参考阈值": adapter["训练内阈值"],
                "非零特征数": int(np.sum(adapter["模型"].coef_[0] != 0.0)),
            }
        )
    return pd.DataFrame(rows)


def 评价敏感性(data: pd.DataFrame, selected_predictions: pd.DataFrame):
    merged = selected_predictions.merge(
        data[["记录行号", "孕周数", "总GC含量", "抽血事件键", "孕妇体质指数"]],
        on="记录行号",
        how="left",
        validate="one_to_one",
    )
    first_event_rows = set(
        data.sort_values(["抽血事件键", "记录行号"]).groupby("抽血事件键", sort=False)["记录行号"].first().tolist()
    )
    definitions = [
        ("主口径_等孕妇权", np.ones(len(merged), dtype=bool), "等孕妇权"),
        ("记录等权", np.ones(len(merged), dtype=bool), "记录等权"),
        ("仅10至25周", merged["孕周数"].between(10, 25, inclusive="both").to_numpy(), "等孕妇权"),
        ("仅总GC在40%至60%", merged["总GC含量"].between(0.4, 0.6, inclusive="both").to_numpy(), "等孕妇权"),
        ("每个抽血事件仅首条记录", merged["记录行号"].isin(first_event_rows).to_numpy(), "等孕妇权"),
        ("仅BMI完整记录", merged["孕妇体质指数"].notna().to_numpy(), "等孕妇权"),
    ]
    rows = []
    for name, mask, weighting in definitions:
        frame = merged.loc[mask].reset_index(drop=True)
        weight = np.ones(len(frame), dtype=float) if weighting == "记录等权" else 等孕妇权重(frame)
        metrics = 计算路线指标(frame, weight, probability_route=True)
        rows.append(
            {
                "敏感性口径": name,
                "权重口径": weighting,
                "记录数": int(len(frame)),
                "孕妇数": int(frame["孕妇代码"].nunique()),
                "异常记录数": int(frame["实际异常标志"].sum()),
                **metrics,
            }
        )
    z_positive = data["Z规则连续分数"].ge(标准分数规则阈值).astype(int)
    z_absolute = data["绝对Z规则连续分数"].ge(标准分数规则阈值).astype(int)
    for name, prediction, score in [
        ("Z值不小于3", z_positive, data["Z规则连续分数"]),
        ("绝对Z值不小于3", z_absolute, data["绝对Z规则连续分数"]),
    ]:
        frame = pd.DataFrame(
            {
                "孕妇代码": data["孕妇代码"],
                "实际异常标志": data["任一异常标志"],
                "预测异常标志": prediction,
                "连续评分": score,
                "预测概率": np.nan,
            }
        )
        metrics = 计算路线指标(frame, 等孕妇权重(frame), probability_route=False)
        rows.append(
            {
                "敏感性口径": name,
                "权重口径": "等孕妇权",
                "记录数": int(len(frame)),
                "孕妇数": int(frame["孕妇代码"].nunique()),
                "异常记录数": int(frame["实际异常标志"].sum()),
                **metrics,
            }
        )
    return pd.DataFrame(rows)


def 多标签组合结果(type_predictions: pd.DataFrame):
    selected = type_predictions.loc[~type_predictions["路线"].str.contains("Z值规则")].copy()
    pivot_actual = selected.pivot(index="记录行号", columns="目标", values="实际异常标志")
    pivot_predicted = selected.pivot(index="记录行号", columns="目标", values="预测异常标志")
    rows = []
    for record_id in pivot_actual.index:
        actual_set = [label for label in ["T13", "T18", "T21"] if int(pivot_actual.loc[record_id, label]) == 1]
        predicted_set = [label for label in ["T13", "T18", "T21"] if int(pivot_predicted.loc[record_id, label]) == 1]
        rows.append(
            {
                "记录行号": int(record_id),
                "实际标签集合": "+".join(actual_set) if actual_set else "正常",
                "预测标签集合": "+".join(predicted_set) if predicted_set else "正常",
                "标签集合完全一致标志": int(actual_set == predicted_set),
                "逐染色体错误数": int(
                    sum(
                        int(pivot_actual.loc[record_id, label])
                        != int(pivot_predicted.loc[record_id, label])
                        for label in ["T13", "T18", "T21"]
                    )
                ),
            }
        )
    detail = pd.DataFrame(rows)
    summary = pd.DataFrame(
        [
            {
                "记录数": int(len(detail)),
                "标签集合完全一致比例": float(detail["标签集合完全一致标志"].mean()),
                "逐染色体汉明损失": float(detail["逐染色体错误数"].sum() / (3 * len(detail))),
            }
        ]
    )
    return detail, summary


def 构造变量角色表():
    rows = []
    for feature in 特征名称:
        rows.append(
            {
                "变量": feature,
                "角色": "主模型候选特征",
                "处理": "训练折内中位数填补和标准化" if feature != "孕妇体质指数" else "唯一缺失记录在训练折内中位数填补；随后标准化",
                "允许进入模型": "是",
                "理由": "题目明确要求或检测时可得的质量/基线因素",
            }
        )
    rows.extend(
        [
            {"变量": "AB染色体非整倍体", "角色": "目标标签", "处理": "空白为正常；解析T13/T18/T21", "允许进入模型": "否", "理由": "目标不得作特征"},
            {"变量": "AE胎儿是否健康", "角色": "冲突审计", "处理": "只记录与AB不一致", "允许进入模型": "否", "理由": "题目指定AB为判定结果"},
            {"变量": "孕妇代码", "角色": "独立分组", "处理": "逐孕妇留一和整簇自助", "允许进入模型": "否", "理由": "标识符"},
            {"变量": "样本序号", "角色": "溯源", "处理": "只用于对应原始记录", "允许进入模型": "否", "理由": "行号可能携带批次或排序信息"},
            {"变量": "检测日期", "角色": "事件审计", "处理": "只构造抽血事件键", "允许进入模型": "否", "理由": "避免批次和时间泄漏"},
            {"变量": "检测抽血次数", "角色": "事件审计", "处理": "只构造抽血事件键", "允许进入模型": "否", "理由": "可能编码既往复检轨迹"},
            {"变量": "身高与体重", "角色": "不重复入模", "处理": "BMI已进入", "允许进入模型": "否", "理由": "避免与BMI代数冗余"},
        ]
    )
    return pd.DataFrame(rows)


def 构造参数来源表():
    return pd.DataFrame(
        [
            {"参数名称和符号": "主目标Y", "参数值": "AB空白=0；任一T13/T18/T21=1", "来源类型": "题目原文", "来源": "C题.pdf附录1：AB空白即无异常", "是否自行设定": "否"},
            {"参数名称和符号": "分型目标", "参数值": "T13、T18、T21三个一对其余目标", "来源类型": "附件取值", "来源": "附件.xlsx女胎AB列", "是否自行设定": "否"},
            {"参数名称和符号": "独立单位", "参数值": "孕妇", "来源类型": "数据结构", "来源": "同孕妇重复记录；全部记录必须同折", "是否自行设定": "否"},
            {"参数名称和符号": "训练和评价权重", "参数值": "每名孕妇总权重相等", "来源类型": "统计口径", "来源": "重复测量的独立单位为孕妇", "是否自行设定": "否"},
            {"参数名称和符号": "Z值规则阈值", "参数值": "3", "来源类型": "领域文献", "来源": "https://pmc.ncbi.nlm.nih.gov/articles/PMC3019239/", "是否自行设定": "否"},
            {"参数名称和符号": "L1正则强度倒数候选", "参数值": "10^-4至10^4共10个对数等距值", "来源类型": "软件公开默认搜索尺度", "来源": "https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegressionCV.html", "是否自行设定": "否"},
            {"参数名称和符号": "内层分层分组折数", "参数值": str(内层折数), "来源类型": "软件默认与分组适配", "来源": "scikit-learn交叉验证默认5折；本题改为孕妇分组并做4/5/10敏感性", "是否自行设定": "否"},
            {"参数名称和符号": "外层验证", "参数值": "逐孕妇留一147次", "来源类型": "无外层折数参数", "来源": "附件147名女胎孕妇", "是否自行设定": "否"},
            {"参数名称和符号": "树大小候选", "参数值": "训练数据完整最小代价复杂度剪枝路径", "来源类型": "算法生成", "来源": "https://scikit-learn.org/stable/modules/tree.html#minimal-cost-complexity-pruning", "是否自行设定": "否"},
            {"参数名称和符号": "树模型选择", "参数值": "等孕妇权Bernoulli BIC最小", "来源类型": "信息准则", "来源": "叶节点数作参数数；孕妇数作有效样本量", "是否自行设定": "否"},
            {"参数名称和符号": "二分类参考阈值", "参数值": "训练内交叉拟合MCC最大；并列依次F1、平衡准确率", "来源类型": "预声明评价规则", "来源": "类别不平衡且题目要求MCC/F1；不含代价权重", "是否自行设定": "否"},
            {"参数名称和符号": "主模型选择", "参数值": "PR曲线下面积优先；复杂路线必须以整簇区间证明优于解释路线", "来源类型": "预声明模型闸门", "来源": "异常类别不平衡与简洁性要求", "是否自行设定": "否"},
            {"参数名称和符号": "区间水平α", "参数值": "0.05", "来源类型": "统计报告惯例", "来源": "报告95%百分位区间", "是否自行设定": "否"},
            {"参数名称和符号": "整簇自助次数", "参数值": str(整簇自助次数), "来源类型": "项目冻结复核尺度", "来源": "第一至第三问沿用100/200/400前缀收敛", "是否自行设定": "否"},
            {"参数名称和符号": "校准分箱数", "参数值": "10", "来源类型": "十分位展示", "来源": "仅用于可靠性表展示，不参与模型选择", "是否自行设定": "否"},
            {"参数名称和符号": "随机种子", "参数值": str(随机种子()), "来源类型": "源文件派生", "来源": "附件.xlsx SHA256前8位十六进制", "是否自行设定": "否"},
            {"参数名称和符号": "概率数值保护", "参数值": repr(数值概率下限), "来源类型": "机器精度", "来源": "numpy浮点机器精度，仅用于对数计算", "是否自行设定": "否"},
            {"参数名称和符号": "类别权重与代价比", "参数值": "均未设置", "来源类型": "题面缺失", "来源": "题目未给漏诊/误报代价比例", "是否自行设定": "否"},
        ]
    )


def 写图表提示词():
    prompts = {
        "图01_三条候选路线外层验证比较_MATLAB_SVG提示词.txt": """图的目的：比较三标准差Z值基准、L1正则多因素逻辑回归和BIC剪枝决策树的逐孕妇外层验证性能，突出类别不平衡下PR曲线下面积与校准差异。
使用的数据文件：02_模型结果/第四问候选路线统一比较.csv；03_验证/第四问候选路线指标95%区间.csv。
横纵坐标：左图横轴为路线、纵轴为PR曲线下面积和ROC曲线下面积；右图横轴为路线、纵轴为布里尔分数和对数损失，仅绘制概率路线。
分组和颜色：Z值基准用灰色，L1逻辑回归用蓝色，剪枝树用橙色；配色需色盲友好。
需要标注的统计量：点估计、孕妇整簇自助95%区间、异常记录67/605、孕妇级分折147人；Z值基准的概率指标标注“不适用”。
MATLAB绘图要求：读取UTF-8中文CSV；使用tiledlayout；误差棒不得截断；中文字体采用Microsoft YaHei；图例放在图外；不得绘制三维图。
SVG输出要求：使用exportgraphics(gcf,'图01_三条候选路线外层验证比较.svg','ContentType','vector')；保持文字为可编辑矢量对象，白色背景。
""",
        "图02_入选模型阈值性能全曲线_MATLAB_SVG提示词.txt": """图的目的：展示在没有漏诊/误报代价比时，不同概率阈值对应的灵敏度、特异度、精确率、F1、MCC和阳性判定比例，避免虚构唯一临床阈值。
使用的数据文件：02_模型结果/第四问入选模型阈值性能完整曲线.csv。
横纵坐标：横轴为概率阈值；纵轴为0至1的性能指标。阈值按升序绘制。
分组和颜色：灵敏度红色、特异度蓝色、精确率紫色、F1绿色、MCC黑色实线、阳性判定比例灰色虚线。
需要标注的统计量：以空心圆标出MCC最大参考点，但旁注“统计参考，非临床唯一阈值”；不添加任何自行设定的成本权重。
MATLAB绘图要求：使用plot与legend；MCC若为负仍保留真实值；中文字体Microsoft YaHei；网格仅用浅灰主网格。
SVG输出要求：使用exportgraphics输出纯矢量SVG，文件名“图02_入选模型阈值性能全曲线.svg”。
""",
        "图03_入选模型特征稳定性与分型性能_MATLAB_SVG提示词.txt": """图的目的：同时展示L1模型各特征在147次逐孕妇留一中的非零比例和系数方向稳定性，以及T13、T18、T21分型的PR曲线下面积与灵敏度。
使用的数据文件：02_模型结果/第四问入选模型留一系数稳定性.csv；02_模型结果/第四问各异常类型识别指标.csv；03_验证/第四问各异常类型指标95%区间.csv。
横纵坐标：左图纵轴为特征、横轴为非零比例，点颜色表示标准化系数中位数正负；右图横轴为异常类型，纵轴为PR曲线下面积和灵敏度。
分组和颜色：正系数用橙色、负系数用蓝色；分型的入选模型用蓝色、Z值规则用灰色。
需要标注的统计量：T13/T18/T21阳性记录数23/46/13和阳性孕妇数18/30/12；类型指标的孕妇整簇95%区间；系数区间跨0时明确标注“不稳定”。
MATLAB绘图要求：使用tiledlayout；特征名完整中文显示；按非零比例排序；不得把L1系数区间解释为传统Wald置信区间。
SVG输出要求：使用exportgraphics输出“图03_入选模型特征稳定性与分型性能.svg”，ContentType设为vector。
""",
    }
    for filename, content in prompts.items():
        path = 图表提示词目录 / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")


def 生成模型卡(
    audit,
    candidate_summary,
    selected_route,
    selection_reason,
    pr_difference_interval,
    full_parameters,
    threshold_rows,
    type_summary,
    multilabel_summary,
):
    selected = candidate_summary.loc[candidate_summary["路线"].eq(selected_route)].iloc[0]
    baseline = candidate_summary.loc[candidate_summary["路线"].eq("三标准差Z值规则基准")].iloc[0]
    tree = candidate_summary.loc[candidate_summary["路线"].eq("贝叶斯信息准则剪枝决策树")].iloc[0]
    selected_thresholds = threshold_rows.loc[threshold_rows["路线"].eq(selected_route), "训练内阈值"].to_numpy(float)
    type_lines = []
    for _, row in type_summary.loc[type_summary["路线"].eq(selected_route)].iterrows():
        type_lines.append(
            f"- {row['异常类型']}：阳性记录{int(row['阳性记录数'])}、阳性孕妇{int(row['阳性孕妇数'])}；PR曲线下面积{row['PR曲线下面积']:.6f}，灵敏度{row['灵敏度']:.6f}，特异度{row['特异度']:.6f}。"
        )
    c_row = full_parameters.loc[full_parameters["参数"].eq("正则强度倒数")]
    c_text = f"{float(c_row.iloc[0]['标准化模型估计值']):.12g}" if len(c_row) else "不适用"
    nonzero = int(full_parameters.get("非零标志", pd.Series(dtype=float)).fillna(0).sum()) - 1 if "非零标志" in full_parameters else np.nan
    card = rf"""# 第四问推荐模型卡

## 目标和数据口径

- 目标：女胎检测记录 AB 列任一 T13/T18/T21 非整倍体标志；AB 空白按题目附录记为未检出异常。
- 数据：{audit['记录数']}条记录、{audit['孕妇数']}名孕妇、{audit['抽血事件数']}个抽血事件；AB异常{audit['AB异常记录数']}条、涉及{audit['AB异常孕妇数']}名孕妇。
- 标签冲突：AE列{audit['记录数']}条全部为“是”，但AB有{audit['AB异常记录数']}条异常；其中{audit['AB异常但AE健康孕妇数']}名孕妇出现AB异常而AE健康。AE不替代AB，也不进入模型。
- 层级：主预测对象为记录级AB结果；同一孕妇全部记录同折，训练和评价采用等孕妇权重。

## 候选与裁决

- 三标准差Z值规则基准：等孕妇权PR曲线下面积{baseline['PR曲线下面积']:.6f}、ROC曲线下面积{baseline['ROC曲线下面积']:.6f}；它没有概率输出，且未综合其他因素，只作领域基准。
- L1正则多因素逻辑回归：等孕妇权PR曲线下面积{candidate_summary.loc[candidate_summary['路线'].eq('L1正则多因素逻辑回归'),'PR曲线下面积'].iloc[0]:.6f}。
- BIC剪枝决策树：等孕妇权PR曲线下面积{tree['PR曲线下面积']:.6f}。
- 剪枝树减逻辑回归的整簇自助PR曲线下面积差95%区间为[{pr_difference_interval[0]:.6f}, {pr_difference_interval[1]:.6f}]。
- 入选路线：**{selected_route}**。裁决依据：{selection_reason}。

## 模型

对每条记录的17个特征先在训练折内作中位数填补；两个读段计数先作log(1+x)，再对全部特征按训练折均值和标准差标准化。入选模型为

\[
\operatorname{{logit}}\{{P(Y=1\mid x)\}}=\beta_0+\sum_{{k=1}}^{{17}}\beta_k\frac{{\tilde x_k-\mu_k}}{{s_k}},
\]

并以L1惩罚估计系数。全样本内层分组验证选择的正则强度倒数为{c_text}，非零特征数为{nonzero}。完整系数、填补值、均值和尺度见参数表与预处理参照表。

## 逐孕妇外层验证

- PR曲线下面积：{selected['PR曲线下面积']:.6f}；ROC曲线下面积：{selected['ROC曲线下面积']:.6f}。
- 灵敏度：{selected['灵敏度']:.6f}；特异度：{selected['特异度']:.6f}；精确率：{selected['精确率']:.6f}；F1：{selected['F1分数']:.6f}；MCC：{selected['马修斯相关系数']:.6f}。
- 布里尔分数：{selected['布里尔分数']:.6f}；对数损失：{selected['对数损失']:.6f}。
- 校准截距：{selected['校准截距']:.6f}；校准斜率：{selected['校准斜率']:.6f}；等孕妇权实际异常比例{selected['校准实际异常比例']:.6f}，平均预测概率{selected['校准平均预测概率']:.6f}。
- 每个外层测试孕妇的阈值均由其余孕妇的5折分层分组交叉拟合概率按MCC选择；147个训练内阈值范围为[{np.min(selected_thresholds):.6f}, {np.max(selected_thresholds):.6f}]。

## 异常类型识别

{chr(10).join(type_lines)}

- 三个一对其余模型组合后的标签集合完全一致比例为{float(multilabel_summary.iloc[0]['标签集合完全一致比例']):.6f}，逐染色体汉明损失为{float(multilabel_summary.iloc[0]['逐染色体汉明损失']):.6f}。

## 阈值与使用边界

题目没有给出漏诊和误报的数值代价比，因此未设置类别权重、SMOTE或成本函数。MCC阈值只用于产生无泄漏的统计参考混淆矩阵；正式材料同时给出全部样本外概率阈值的完整性能曲线，不把任何一个阈值称为临床唯一阈值。该模型拟合的是附件AB检测标签，不是AE出生健康结局，也不是临床确诊模型。

## 局限性

- T21仅13条阳性记录、12名阳性孕妇，分型指标区间必然较宽。
- 43名孕妇在不同记录间出现AB阴阳变化，同一抽血事件也有2组不一致；模型描述检测记录结果而非恒定胎儿状态。
- L1留一系数分布用于稳定性描述，不是传统独立样本Wald显著性检验。
- 数据仅来自附件样本，外部推广前必须用独立中心数据验证；NIPT属于筛查，不能替代诊断。
"""
    (模型输出目录 / "第四问推荐模型卡.md").write_text(card, encoding="utf-8")


def 生成依赖版本():
    content = "\n".join(
        [
            f"Python={platform.python_version()}",
            f"numpy={np.__version__}",
            f"pandas={pd.__version__}",
            f"scipy={scipy.__version__}",
            f"scikit-learn={sklearn.__version__}",
            f"statsmodels={statsmodels.__version__}",
            f"platform={platform.platform()}",
        ]
    )
    (复现输出目录 / "第四问依赖版本.txt").write_text(content + "\n", encoding="utf-8")


def 执行内部审核(
    data,
    woman,
    event,
    audit,
    candidate_predictions,
    candidate_summary,
    candidate_bootstrap,
    selected_route,
    threshold_rows,
    threshold_curve,
    calibration_bins,
    coefficient_stability,
    type_predictions,
    type_summary,
    type_bootstrap,
    sensitivity,
    fold_sensitivity,
    parameter_sources,
):
    checks = []

    def 验收(name: str, condition, evidence: str):
        passed = bool(condition)
        checks.append({"验收项": name, "通过标志": passed, "状态": "通过" if passed else "失败", "证据": evidence})

    验收("题意与标签", audit["记录数"] == 605 and audit["AB异常记录数"] == 67, "女胎605条；AB异常67条；AB空白作无异常")
    验收("AB与AE冲突", audit["AE不健康记录数"] == 0 and audit["AB异常但AE健康记录数"] == 67 and audit["AB异常但AE健康孕妇数"] == 44, "AE全为是；AB异常67条/44人，AE未替代AB")
    验收("数据层级", audit["孕妇数"] == 147 and audit["抽血事件数"] == 590 and audit["多记录事件组数"] == 15 and audit["同事件AB阴阳不一致组数"] == 2, "147名孕妇、590事件；15个多记录事件、2组AB不一致")
    验收("检测日期解析", audit["检测日期四位年份格式全部成立"] and audit["最早检测日期"] == "2023-03-05" and audit["最晚检测日期"] == "2024-07-08", f"数据实得范围={audit['最早检测日期']}至{audit['最晚检测日期']}；全部为四位年份")
    验收("异常类型样本量", [audit["T13阳性记录数"], audit["T18阳性记录数"], audit["T21阳性记录数"]] == [23, 46, 13], "T13/T18/T21阳性记录为23/46/13")
    验收("主特征与泄漏隔离", len(特征名称) == 17 and not set(特征名称).intersection({"原始序号", "孕妇代码", "检测日期", "抽血次数", "AB原始标签", "AE出生健康结果"}), "17项题面/检测时特征；标识、AB、AE和复检轨迹未入模")
    route_counts = candidate_predictions.groupby("路线").agg(记录数=("记录行号", "size"), 孕妇数=("孕妇代码", "nunique"), 唯一记录数=("记录行号", "nunique"))
    验收("三条不同候选路线", set(route_counts.index) == {"三标准差Z值规则基准", "L1正则多因素逻辑回归", "贝叶斯信息准则剪枝决策树"}, f"路线={route_counts.index.tolist()}")
    验收("逐孕妇外层覆盖", route_counts[["记录数", "唯一记录数"]].eq(605).all().all() and route_counts["孕妇数"].eq(147).all(), "每条路线605条记录恰好一次样本外预测，覆盖147人")
    验收("外层同孕妇同折", candidate_predictions["外层留出孕妇"].eq(candidate_predictions["孕妇代码"]).all(), "每条预测的留出组等于该记录孕妇代码")
    learned = candidate_predictions.loc[candidate_predictions["概率输出标志"].eq(1)]
    验收("概率合法", learned["预测概率"].between(0, 1, inclusive="both").all() and learned["预测概率"].notna().all(), "两条学习路线全部1210个概率位于[0,1]")
    验收("训练内阈值", len(threshold_rows) == 295 and threshold_rows.loc[threshold_rows["路线"].ne("三标准差Z值规则基准"), "外层留出孕妇"].nunique() == 147, "Z规则1条固定阈值；两条学习路线各147个训练内阈值")
    验收("类别不平衡指标完整", {"灵敏度", "特异度", "精确率", "F1分数", "马修斯相关系数", "ROC曲线下面积", "PR曲线下面积"}.issubset(candidate_summary.columns), "未仅报告准确率")
    验收("候选整簇区间", candidate_bootstrap.groupby("路线")["自助序号"].nunique().eq(400).all(), "三条路线各400次孕妇整簇指标复算")
    验收("主路线裁决", selected_route == "L1正则多因素逻辑回归" and int(candidate_summary["主模型标志"].sum()) == 1, f"入选={selected_route}")
    expected_threshold_points = int(candidate_predictions.loc[candidate_predictions["路线"].eq(selected_route), "预测概率"].nunique()) + 1
    验收("阈值完整曲线", len(threshold_curve) == expected_threshold_points and threshold_curve["概率阈值"].is_unique, f"{len(threshold_curve)}个完整经验阈值点")
    验收("校准输出", len(calibration_bins) == 10 and calibration_bins["记录数"].sum() == 605, "10个校准十分位覆盖605条记录")
    验收("系数稳定性", len(coefficient_stability) == 17 and coefficient_stability["外层拟合次数"].eq(147).all(), "17项特征各有147次留一拟合系数")
    type_counts = type_predictions.groupby(["目标", "路线"])["记录行号"].nunique()
    验收("分型外层覆盖", len(type_counts) == 6 and type_counts.eq(605).all(), "T13/T18/T21各含Z规则与入选路线，均覆盖605条")
    验收("分型指标与区间", len(type_summary) == 6 and type_bootstrap.groupby(["异常类型", "路线"])["自助序号"].nunique().min() >= 399, "三类异常均报告混淆、PR/ROC及孕妇整簇区间")
    验收("敏感性", len(sensitivity) == 8 and len(fold_sensitivity) == 3 and set(fold_sensitivity["内层分层分组折数"]) == {4, 5, 10}, "6个评价口径+2个Z规则；内层4/5/10折")
    验收("参数来源", len(parameter_sources) == 18 and parameter_sources["是否自行设定"].eq("否").all() and parameter_sources["参数名称和符号"].is_unique, "18项参数/规则逐项有来源，无自行代价参数")
    csv_paths = list(输出根目录.rglob("*.csv"))
    bad_headers = []
    duplicate_headers = []
    for path in csv_paths:
        columns = pd.read_csv(path, nrows=0).columns.tolist()
        if any(re.search(r"[\u4e00-\u9fff]", str(column)) is None for column in columns):
            bad_headers.append(str(path.relative_to(输出根目录)))
        if len(columns) != len(set(columns)):
            duplicate_headers.append(str(path.relative_to(输出根目录)))
    验收("中文表头", not bad_headers and not duplicate_headers, f"检查{len(csv_paths)}个CSV；无中文表头={bad_headers}；重复表头={duplicate_headers}")
    image_count = len([path for path in 输出根目录.rglob("*") if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".svg", ".gif", ".bmp", ".webp"}])
    prompt_count = len(list(图表提示词目录.glob("*.txt")))
    验收("图形交付约束", image_count == 0 and prompt_count == 3, f"新图像={image_count}；MATLAB/SVG提示词={prompt_count}")
    model_card = (模型输出目录 / "第四问推荐模型卡.md").read_text(encoding="utf-8")
    验收("无虚构代价与临床阈值", "未设置类别权重" in model_card and "不把任何一个阈值称为临床唯一阈值" in model_card and "不是临床确诊模型" in model_card, "模型卡明确无代价比、完整阈值曲线和筛查边界")

    checklist = pd.DataFrame(checks)
    写CSV(checklist, 验证输出目录 / "第四问总控验收清单.csv")
    failed = checklist.loc[~checklist["通过标志"]]
    status = "PASS" if failed.empty else "REJECTED"
    report_lines = [
        "# 第四问内部总控复核报告",
        "",
        f"- 状态：**{status}**",
        f"- 验收项：{len(checklist)}；失败：{len(failed)}。",
        "- 审核范围：题意、标签、层级、泄漏、候选、外层验证、阈值、校准、分型、参数来源、图形约束。",
        "",
    ]
    for _, row in checklist.iterrows():
        report_lines.append(f"- [{'通过' if row['通过标志'] else '失败'}] {row['验收项']}：{row['证据']}")
    if not failed.empty:
        report_lines.extend(["", "存在失败项，必须打回，禁止归档。"])
    else:
        report_lines.extend(["", "全部内部检查通过，允许进入独立复核；独立复核通过前仍不得归档。"])
    (验证输出目录 / "第四问内部总控复核报告.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    return checklist, status


def 生成结果哈希表():
    excluded = {
        "04_复现\\第四问结果文件哈希.csv",
        "04_复现\\第四问自审PASS记录.json",
    }
    rows = []
    for path in sorted(输出根目录.rglob("*")):
        if not path.is_file():
            continue
        relative = str(path.relative_to(输出根目录))
        if relative in excluded:
            continue
        rows.append({"相对路径": relative, "SHA256": 文件哈希(path), "字节数": int(path.stat().st_size)})
    table = pd.DataFrame(rows)
    写CSV(table, 复现输出目录 / "第四问结果文件哈希.csv")
    return table


def main():
    started = datetime.now().astimezone()
    for directory in [数据输出目录, 模型输出目录, 验证输出目录, 复现输出目录, 图表提示词目录]:
        directory.mkdir(parents=True, exist_ok=True)

    print("[1/9] 重建女胎记录、标签、孕妇与抽血事件层", flush=True)
    data, woman, event, audit = 构造数据()
    写CSV(data, 数据输出目录 / "第四问女胎记录级建模表.csv")
    写CSV(woman, 数据输出目录 / "第四问孕妇级标签审计表.csv")
    写CSV(event, 数据输出目录 / "第四问抽血事件重复审计表.csv")
    role_table = 构造变量角色表()
    写CSV(role_table, 数据输出目录 / "第四问变量角色与泄漏禁用表.csv")
    写CSV(pd.DataFrame([audit]), 数据输出目录 / "第四问标签与样本审计摘要.csv")
    写JSON(audit, 数据输出目录 / "第四问数据构造断言.json")

    print("[2/9] 三条路线执行147名孕妇逐人留一与训练内阈值选择", flush=True)
    candidate_predictions, threshold_rows, coefficient_detail, tree_detail = 逐孕妇外层候选验证(data, "任一异常标志")
    candidate_summary = 汇总候选路线(candidate_predictions)

    print("[3/9] 400次孕妇整簇指标区间与主路线裁决", flush=True)
    candidate_bootstrap, candidate_bootstrap_summary, route_difference = 候选路线整簇自助(candidate_predictions)
    selected_route, selection_reason, pr_lower, pr_upper = 选择主路线(candidate_summary, route_difference)
    candidate_summary["主模型标志"] = candidate_summary["路线"].eq(selected_route).astype(int)
    candidate_summary["主审裁决"] = candidate_summary["路线"].map(
        lambda route: selection_reason if route == selected_route else ("领域基准，不满足多因素概率输出门槛" if route == "三标准差Z值规则基准" else "未通过相对可解释路线的PR区间优势闸门")
    )
    selected_predictions = candidate_predictions.loc[candidate_predictions["路线"].eq(selected_route)].reset_index(drop=True)

    print("[4/9] 生成阈值全曲线、校准、全样本参数和稳定性", flush=True)
    threshold_curve = 完整阈值性能曲线(selected_predictions)
    calibration_bins = 校准分箱(selected_predictions)
    coefficient_stability = 留一系数稳定性(coefficient_detail) if selected_route == "L1正则多因素逻辑回归" else pd.DataFrame()
    full_adapter, full_parameters, preprocessing_reference, tree_text = 全样本入选模型(data, "任一异常标志", selected_route)
    fold_sensitivity = 内层折数敏感性(data, "任一异常标志")

    print("[5/9] T13、T18、T21分别执行孕妇级外层验证", flush=True)
    type_prediction_frames = []
    type_threshold_frames = []
    type_full_parameters = []
    for target in ["T13异常标志", "T18异常标志", "T21异常标志"]:
        predictions, thresholds = 逐孕妇外层入选路线验证(data, target, selected_route)
        type_prediction_frames.append(predictions)
        type_threshold_frames.append(thresholds)
        _, parameters, _, _ = 全样本入选模型(data, target, selected_route)
        parameters.insert(0, "异常类型", 标签名称[target])
        type_full_parameters.append(parameters)
    type_predictions = pd.concat(type_prediction_frames, ignore_index=True)
    type_thresholds = pd.concat(type_threshold_frames, ignore_index=True)
    type_parameter_table = pd.concat(type_full_parameters, ignore_index=True)
    type_summary, type_bootstrap, type_intervals = 类型指标与整簇区间(type_predictions)
    multilabel_detail, multilabel_summary = 多标签组合结果(type_predictions)

    print("[6/9] 敏感性、参数来源、模型卡和制图提示词", flush=True)
    sensitivity = 评价敏感性(data, selected_predictions)
    parameter_sources = 构造参数来源表()
    写CSV(parameter_sources, 模型输出目录 / "第四问参数来源表.csv")
    写CSV(candidate_summary, 模型输出目录 / "第四问候选路线统一比较.csv")
    写CSV(full_parameters, 模型输出目录 / "第四问入选模型全样本参数表.csv")
    写CSV(preprocessing_reference, 模型输出目录 / "第四问入选模型预处理参照表.csv")
    写CSV(coefficient_stability, 模型输出目录 / "第四问入选模型留一系数稳定性.csv")
    写CSV(threshold_curve, 模型输出目录 / "第四问入选模型阈值性能完整曲线.csv")
    写CSV(calibration_bins, 模型输出目录 / "第四问入选模型校准十分位表.csv")
    写CSV(type_summary, 模型输出目录 / "第四问各异常类型识别指标.csv")
    写CSV(type_parameter_table, 模型输出目录 / "第四问各异常类型全样本参数表.csv")
    写CSV(multilabel_summary, 模型输出目录 / "第四问多标签组合识别摘要.csv")
    if tree_text:
        (模型输出目录 / "第四问入选决策树文本规则.txt").write_text(tree_text, encoding="utf-8")
    生成模型卡(
        audit,
        candidate_summary,
        selected_route,
        selection_reason,
        (pr_lower, pr_upper),
        full_parameters,
        threshold_rows,
        type_summary,
        multilabel_summary,
    )
    写图表提示词()

    print("[7/9] 写入外层预测、整簇区间、分型和敏感性机器结果", flush=True)
    写CSV(candidate_predictions, 验证输出目录 / "第四问候选路线逐孕妇留一逐记录.csv")
    写CSV(threshold_rows, 验证输出目录 / "第四问候选路线训练内阈值逐孕妇.csv")
    写CSV(coefficient_detail, 验证输出目录 / "第四问逻辑回归留一系数逐次.csv")
    写CSV(tree_detail, 验证输出目录 / "第四问剪枝树留一结构逐次.csv")
    写CSV(candidate_bootstrap, 验证输出目录 / "第四问候选路线指标整簇自助逐次.csv")
    写CSV(candidate_bootstrap_summary, 验证输出目录 / "第四问候选路线指标前缀收敛.csv")
    写CSV(candidate_bootstrap_summary.loc[candidate_bootstrap_summary["请求前缀次数"].eq(400)], 验证输出目录 / "第四问候选路线指标95%区间.csv")
    写CSV(route_difference, 验证输出目录 / "第四问候选路线差异整簇自助逐次.csv")
    写CSV(type_predictions, 验证输出目录 / "第四问各异常类型逐孕妇留一预测.csv")
    写CSV(type_thresholds, 验证输出目录 / "第四问各异常类型训练内阈值逐孕妇.csv")
    写CSV(type_bootstrap, 验证输出目录 / "第四问各异常类型指标整簇自助逐次.csv")
    写CSV(type_intervals, 验证输出目录 / "第四问各异常类型指标95%区间.csv")
    写CSV(multilabel_detail, 验证输出目录 / "第四问多标签组合逐记录结果.csv")
    写CSV(sensitivity, 验证输出目录 / "第四问数据与评价口径敏感性.csv")
    写CSV(fold_sensitivity, 验证输出目录 / "第四问内层折数敏感性.csv")

    print("[8/9] 内部总控审核", flush=True)
    checklist, status = 执行内部审核(
        data,
        woman,
        event,
        audit,
        candidate_predictions,
        candidate_summary,
        candidate_bootstrap,
        selected_route,
        threshold_rows,
        threshold_curve,
        calibration_bins,
        coefficient_stability,
        type_predictions,
        type_summary,
        type_bootstrap,
        sensitivity,
        fold_sensitivity,
        parameter_sources,
    )
    生成依赖版本()

    print("[9/9] 运行清单、结果哈希和自审记录", flush=True)
    one_click = 项目目录 / "一键运行第四问女胎异常判定.ps1"
    manifest = {
        "开始时间": started.isoformat(),
        "结束时间": datetime.now().astimezone().isoformat(),
        "工作簿路径": str(工作簿路径),
        "工作簿SHA256": 文件哈希(工作簿路径),
        "题目SHA256": 文件哈希(题目路径),
        "合同SHA256": 文件哈希(合同路径),
        "建模脚本SHA256": 文件哈希(脚本路径),
        "一键脚本SHA256": 文件哈希(one_click) if one_click.exists() else "运行时不存在",
        "随机种子": 随机种子(),
        "外层验证": "逐孕妇留一",
        "内层分层分组折数": 内层折数,
        "正则强度倒数候选": 正则强度倒数候选.tolist(),
        "整簇自助次数": 整簇自助次数,
        "区间水平": 0.95,
        "Z值规则阈值": 标准分数规则阈值,
        "主路线": selected_route,
        "主路线裁决": selection_reason,
        "设置类别权重": False,
        "使用SMOTE": False,
        "设置漏诊误报代价比": False,
        "输出唯一临床阈值": False,
        "生成图片数": 0,
        "数据断言": audit,
    }
    写JSON(manifest, 复现输出目录 / "第四问运行清单.json")
    hash_table = 生成结果哈希表()
    pass_record = {
        "审核时间": datetime.now().astimezone().isoformat(),
        "问题": "第四问",
        "状态": status,
        "关键检查数": int(len(checklist)),
        "失败检查数": int((~checklist["通过标志"]).sum()),
        "入选模型": selected_route,
        "运行清单SHA256": 文件哈希(复现输出目录 / "第四问运行清单.json"),
        "结果哈希表SHA256": 文件哈希(复现输出目录 / "第四问结果文件哈希.csv"),
        "哈希覆盖文件数": int(len(hash_table)),
    }
    写JSON(pass_record, 复现输出目录 / "第四问自审PASS记录.json")
    print(json.dumps(pass_record, ensure_ascii=False, indent=2), flush=True)
    if status != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
