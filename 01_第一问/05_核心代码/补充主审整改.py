from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
from sklearn.model_selection import GroupKFold


# 当前脚本会动态载入主分析模块；禁止在论文核心代码目录生成__pycache__。
sys.dont_write_bytecode = True


脚本目录 = Path(__file__).resolve().parent
主脚本路径 = 脚本目录 / "优先问题整改分析.py"
输出目录 = 脚本目录.parent / "03_整改后计算输出"


def 载入主脚本() -> Any:
    spec = importlib.util.spec_from_file_location("优先问题整改分析", 主脚本路径)
    if spec is None or spec.loader is None:
        raise RuntimeError("无法载入主整改脚本")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def 写CSV(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def 写JSON(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def 第一问分段GEE(events: pd.DataFrame, module: Any) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    persons = events.sort_values(["孕妇代码", "孕周天数", "抽血次数"]).drop_duplicates("孕妇代码")
    centers = {
        "孕周": float(events["孕周数"].mean()),
        "妇间BMI": float(persons["孕妇平均BMI"].mean()),
        "年龄": float(persons["年龄"].mean()),
        "生产次数": float(persons["生产次数"].mean()),
    }
    data = module.构造第一问变量(events, centers)
    formula = (
        module.第一问公式
        + " + 后段标志 + 后段标志:孕周中心 + 后段标志:I(孕周中心 ** 2)"
        + " + 后段标志:妇间BMI中心 + 后段标志:BMI个体内偏差"
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = smf.gee(
            formula,
            groups="孕妇代码",
            data=data,
            family=sm.families.Gaussian(),
            cov_struct=sm.cov_struct.Exchangeable(),
        )
        fit = model.fit(maxiter=1000)
    if not bool(getattr(fit, "converged", False)):
        raise RuntimeError("第一问分段GEE未收敛")
    warning_text = "；".join(dict.fromkeys(str(item.message) for item in caught))
    params = pd.Series(fit.params)
    ses = pd.Series(fit.bse)
    pvalues = pd.Series(fit.pvalues)
    rows = []
    for term in params.index:
        estimate = float(params[term])
        se = float(ses[term])
        raw_pvalue = float(pvalues[term])
        pvalue_underflow = bool(raw_pvalue == 0.0)
        rows.append(
            {
                "模型角色": "第一问全样本主模型",
                "参数": str(term),
                "估计值": estimate,
                "稳健标准误": se,
                "Wald95%下限": estimate - stats.norm.ppf(0.975) * se,
                "Wald95%上限": estimate + stats.norm.ppf(0.975) * se,
                # statsmodels 在极端显著时可能把双精度尾概率返回为 0；这不是精确零。
                "P值": np.nan if pvalue_underflow else raw_pvalue,
                "P值说明": (
                    "小于双精度浮点最小正数（数值下溢），不报告为精确0"
                    if pvalue_underflow
                    else ""
                ),
                "孕妇数": int(data["孕妇代码"].nunique()),
                "事件数": int(len(data)),
                "相关结构": "可交换相关；孕妇聚类稳健协方差",
                "收敛标志": int(fit.converged),
                "警告": warning_text,
            }
        )

    covariance = pd.DataFrame(fit.cov_params(), index=params.index, columns=params.index)
    interaction_terms = [
        "后段标志:孕周中心",
        "后段标志:I(孕周中心 ** 2)",
        "后段标志:妇间BMI中心",
        "后段标志:BMI个体内偏差",
    ]
    missing = [term for term in interaction_terms if term not in params.index]
    if missing:
        raise RuntimeError(f"第一问分段GEE缺少交互项：{missing}")
    interaction_beta = params.loc[interaction_terms].to_numpy(dtype=float)
    interaction_cov = covariance.loc[interaction_terms, interaction_terms].to_numpy(dtype=float)
    statistic = float(interaction_beta @ np.linalg.pinv(interaction_cov) @ interaction_beta)
    df = len(interaction_terms)
    joint = pd.DataFrame(
        [
            {
                "检验": "序号683前后关键关系相同",
                "零假设": "孕周一次项、孕周二次项、妇间BMI和个体内BMI的四个后段交互同时为0",
                "稳健Wald卡方": statistic,
                "自由度": df,
                "P值": float(stats.chi2.sf(statistic, df)),
                "结论边界": "只检验数据段异质性；不证明683后记录无效，也不解释异质性成因",
            }
        ]
    )

    effect_specs = [
        ("孕周一次项（中心孕周处斜率）", "孕周中心", "后段标志:孕周中心"),
        ("孕周二次项", "I(孕周中心 ** 2)", "后段标志:I(孕周中心 ** 2)"),
        ("妇间BMI效应", "妇间BMI中心", "后段标志:妇间BMI中心"),
        ("个体内BMI效应", "BMI个体内偏差", "后段标志:BMI个体内偏差"),
    ]
    derived_rows = []
    for segment in ("序号683前", "序号683后"):
        for label, base_term, interaction_term in effect_specs:
            contrast = np.zeros(len(params), dtype=float)
            contrast[params.index.get_loc(base_term)] = 1.0
            if segment == "序号683后":
                contrast[params.index.get_loc(interaction_term)] = 1.0
            estimate = float(contrast @ params.to_numpy(dtype=float))
            variance = float(contrast @ covariance.to_numpy(dtype=float) @ contrast)
            se = math.sqrt(max(variance, 0.0))
            z = estimate / se if se > 0 else np.nan
            derived_rows.append(
                {
                    "数据段": segment,
                    "派生效应": label,
                    "估计值": estimate,
                    "稳健标准误": se,
                    "Wald95%下限": estimate - stats.norm.ppf(0.975) * se,
                    "Wald95%上限": estimate + stats.norm.ppf(0.975) * se,
                    "P值": float(2.0 * stats.norm.sf(abs(z))) if np.isfinite(z) else np.nan,
                    "尺度说明": "Y浓度logit尺度；关联而非因果",
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(derived_rows), joint


def 人等权逻辑拟合(
    data: pd.DataFrame,
    week_center: float,
    week_scale: float,
    bmi_center: float,
    bmi_scale: float,
    model_kind: str,
    cutpoint: float | None = None,
) -> dict[str, Any]:
    frame = data.copy()
    week = (frame["孕周数"].to_numpy(dtype=float) - week_center) / week_scale
    y = frame["达标标志"].to_numpy(dtype=float)
    counts = frame.groupby("孕妇代码")["孕妇代码"].transform("size").to_numpy(dtype=float)
    weights = 1.0 / counts
    if model_kind == "不使用BMI的一组模型":
        x = np.column_stack([np.ones(len(frame)), week])
        parameter_count = 2
    elif model_kind == "连续BMI模型":
        bmi = (frame["首次BMI"].to_numpy(dtype=float) - bmi_center) / bmi_scale
        x = np.column_stack([np.ones(len(frame)), week, bmi])
        parameter_count = 3
    elif model_kind == "单切点两组模型":
        if cutpoint is None:
            raise ValueError("单切点模型缺少切点")
        high = (frame["首次BMI"].to_numpy(dtype=float) >= cutpoint).astype(float)
        x = np.column_stack([np.ones(len(frame)), week, high])
        parameter_count = 4  # 三个回归系数，加上一个由数据搜索的切点位置
    else:
        raise ValueError(model_kind)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fit = sm.GLM(y, x, family=sm.families.Binomial(), freq_weights=weights).fit(
            maxiter=300,
            disp=0,
        )
    if not bool(getattr(fit, "converged", False)) or not np.isfinite(fit.llf):
        raise RuntimeError(f"{model_kind}未收敛")
    effective_n = int(frame["孕妇代码"].nunique())
    bic = -2.0 * float(fit.llf) + parameter_count * math.log(effective_n)
    return {
        "模型": model_kind,
        "切点": cutpoint,
        "对数似然": float(fit.llf),
        "BIC": float(bic),
        "参数数（切点计入）": parameter_count,
        "孕妇数": effective_n,
        "事件数": int(len(frame)),
        "警告": "；".join(dict.fromkeys(str(item.message) for item in caught)),
    }


def 候选切点(persons: pd.DataFrame) -> np.ndarray:
    values = np.sort(persons["首次BMI"].dropna().astype(float).unique())
    if len(values) < 2:
        return np.array([], dtype=float)
    return (values[:-1] + values[1:]) / 2.0


def 搜索单切点(
    data: pd.DataFrame,
    week_center: float,
    week_scale: float,
    bmi_center: float,
    bmi_scale: float,
) -> dict[str, Any]:
    persons = data.sort_values(["孕妇代码", "孕周天数"]).drop_duplicates("孕妇代码")
    candidates = 候选切点(persons)
    if len(candidates) == 0:
        raise RuntimeError("没有可识别的BMI切点")
    fits = []
    for cutpoint in candidates:
        try:
            fits.append(
                人等权逻辑拟合(
                    data,
                    week_center,
                    week_scale,
                    bmi_center,
                    bmi_scale,
                    "单切点两组模型",
                    float(cutpoint),
                )
            )
        except Exception:
            continue
    if not fits:
        raise RuntimeError("全部单切点模型均失败")
    best = min(fits, key=lambda row: (row["BIC"], row["切点"]))
    best["可识别候选切点数"] = int(len(candidates))
    best["成功候选切点数"] = int(len(fits))
    return best


def 第二问BMI切点审计(
    events: pd.DataFrame,
    module: Any,
    bootstrap_count: int = 400,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    week_center, week_scale = module.稳健尺度(events["孕周数"])
    bmi_center, bmi_scale = module.稳健尺度(events["首次BMI"])
    base = 人等权逻辑拟合(
        events,
        week_center,
        week_scale,
        bmi_center,
        bmi_scale,
        "不使用BMI的一组模型",
    )
    continuous = 人等权逻辑拟合(
        events,
        week_center,
        week_scale,
        bmi_center,
        bmi_scale,
        "连续BMI模型",
    )
    split = 搜索单切点(events, week_center, week_scale, bmi_center, bmi_scale)
    full_comparison = pd.DataFrame([base, continuous, split])
    full_comparison["相对最小BIC差"] = full_comparison["BIC"] - full_comparison["BIC"].min()
    full_comparison["BIC最优标志"] = (
        full_comparison["BIC"] == full_comparison["BIC"].min()
    ).astype(int)
    full_comparison["说明"] = (
        "孕妇等权二项似然；BIC中把搜索得到的切点位置计作一个参数；不设置最小组人数"
    )

    seed_base = int(module.预期附件哈希[40:48], 16)
    fold_rows = []
    for repeat in range(5):
        splitter = GroupKFold(n_splits=5, shuffle=True, random_state=seed_base + repeat)
        for fold, (train_index, _) in enumerate(
            splitter.split(events, groups=events["孕妇代码"]), start=1
        ):
            train = events.iloc[train_index].copy()
            train_base = 人等权逻辑拟合(
                train,
                week_center,
                week_scale,
                bmi_center,
                bmi_scale,
                "不使用BMI的一组模型",
            )
            train_continuous = 人等权逻辑拟合(
                train,
                week_center,
                week_scale,
                bmi_center,
                bmi_scale,
                "连续BMI模型",
            )
            train_split = 搜索单切点(train, week_center, week_scale, bmi_center, bmi_scale)
            candidates = [train_base, train_continuous, train_split]
            winner = min(candidates, key=lambda row: (row["BIC"], row["模型"]))
            fold_rows.append(
                {
                    "重复": repeat + 1,
                    "折": fold,
                    "训练孕妇数": int(train["孕妇代码"].nunique()),
                    "单切点最优BMI切点": float(train_split["切点"]),
                    "单切点模型BIC": float(train_split["BIC"]),
                    "连续BMI模型BIC": float(train_continuous["BIC"]),
                    "不使用BMI的一组模型BIC": float(train_base["BIC"]),
                    "折内BIC最优模型": str(winner["模型"]),
                    "折内单切点入选标志": int(winner["模型"] == "单切点两组模型"),
                }
            )
    folds = pd.DataFrame(fold_rows)
    cutpoints = folds["单切点最优BMI切点"].to_numpy(dtype=float)
    full_winner = full_comparison.sort_values(["BIC", "模型"]).iloc[0]
    selected_counts = folds["折内BIC最优模型"].value_counts().to_dict()

    bootstrap_rng = np.random.default_rng(int(module.预期附件哈希[56:64], 16))
    bootstrap_rows = []
    for replicate in range(1, bootstrap_count + 1):
        sample = module.整簇重采样(events, bootstrap_rng)
        try:
            sample_base = 人等权逻辑拟合(
                sample, week_center, week_scale, bmi_center, bmi_scale, "不使用BMI的一组模型"
            )
            sample_continuous = 人等权逻辑拟合(
                sample, week_center, week_scale, bmi_center, bmi_scale, "连续BMI模型"
            )
            sample_split = 搜索单切点(
                sample, week_center, week_scale, bmi_center, bmi_scale
            )
            candidates = [sample_base, sample_continuous, sample_split]
            winner = min(candidates, key=lambda row: (row["BIC"], row["模型"]))
            bootstrap_rows.append(
                {
                    "重复序号": replicate,
                    "有效标志": 1,
                    "单切点最优BMI切点": float(sample_split["切点"]),
                    "BIC最优模型": str(winner["模型"]),
                    "单切点模型BIC": float(sample_split["BIC"]),
                    "连续BMI模型BIC": float(sample_continuous["BIC"]),
                    "不使用BMI的一组模型BIC": float(sample_base["BIC"]),
                    "错误": "",
                }
            )
        except Exception as exc:
            bootstrap_rows.append(
                {
                    "重复序号": replicate,
                    "有效标志": 0,
                    "单切点最优BMI切点": np.nan,
                    "BIC最优模型": "",
                    "单切点模型BIC": np.nan,
                    "连续BMI模型BIC": np.nan,
                    "不使用BMI的一组模型BIC": np.nan,
                    "错误": f"{type(exc).__name__}:{exc}",
                }
            )
    bootstraps = pd.DataFrame(bootstrap_rows)
    valid_bootstraps = bootstraps.loc[bootstraps["有效标志"].eq(1)].copy()
    convergence_rows = []
    for prefix in sorted(set(value for value in (100, 200, bootstrap_count) if value <= bootstrap_count)):
        prefix_frame = valid_bootstraps.loc[valid_bootstraps["重复序号"].le(prefix)]
        counts = prefix_frame["BIC最优模型"].value_counts()
        convergence_rows.append(
            {
                "前缀重复数": prefix,
                "有效重复数": int(len(prefix_frame)),
                "一组模型入选比例": float(counts.get("不使用BMI的一组模型", 0) / max(len(prefix_frame), 1)),
                "连续BMI模型入选比例": float(counts.get("连续BMI模型", 0) / max(len(prefix_frame), 1)),
                "单切点模型入选比例": float(counts.get("单切点两组模型", 0) / max(len(prefix_frame), 1)),
                "单切点位置中位数": float(prefix_frame["单切点最优BMI切点"].median()),
                "单切点位置2.5%分位": float(prefix_frame["单切点最优BMI切点"].quantile(0.025)),
                "单切点位置97.5%分位": float(prefix_frame["单切点最优BMI切点"].quantile(0.975)),
            }
        )
    convergence = pd.DataFrame(convergence_rows)
    verdict = {
        "统计分组主裁决": str(full_winner["模型"]),
        "全样本BIC最优值": float(full_winner["BIC"]),
        "全样本最优单切点": float(split["切点"]),
        "25个训练折单切点2.5%分位": float(np.quantile(cutpoints, 0.025)),
        "25个训练折单切点中位数": float(np.median(cutpoints)),
        "25个训练折单切点97.5%分位": float(np.quantile(cutpoints, 0.975)),
        "25个训练折模型入选次数": {str(key): int(value) for key, value in selected_counts.items()},
        "孕妇整簇重采样次数": bootstrap_count,
        "孕妇整簇重采样有效次数": int(len(valid_bootstraps)),
        "孕妇整簇重采样模型入选次数": {
            str(key): int(value)
            for key, value in valid_bootstraps["BIC最优模型"].value_counts().to_dict().items()
        },
        "孕妇整簇重采样单切点2.5%分位": float(
            valid_bootstraps["单切点最优BMI切点"].quantile(0.025)
        ),
        "孕妇整簇重采样单切点中位数": float(
            valid_bootstraps["单切点最优BMI切点"].median()
        ),
        "孕妇整簇重采样单切点97.5%分位": float(
            valid_bootstraps["单切点最优BMI切点"].quantile(0.975)
        ),
        "实施分组": "使用题面示例区间；它们是题面可追溯实施分层，不冒充统计识别切点",
        "解释规则": (
            "若连续BMI或一组模型优于单切点模型，或折间切点大幅漂移，则不声称数据识别出稳定离散BMI阈值；"
            "仍按题面示例区间报告时点，并逐组标注人数与不确定性"
        ),
        "没有自拟科学阈值": True,
    }
    return full_comparison, folds, bootstraps, convergence, verdict


def 统计主分组时点(
    events: pd.DataFrame,
    model: dict[str, Any],
    verdict: dict[str, Any],
    role: str,
    module: Any,
) -> pd.DataFrame:
    persons = events.sort_values(["孕妇代码", "孕周天数"]).drop_duplicates("孕妇代码").copy()
    lower = float(persons["首次BMI"].min())
    upper = float(persons["首次BMI"].max())
    if verdict["统计分组主裁决"] == "单切点两组模型":
        cutpoint = float(verdict["全样本最优单切点"])
        groups = [
            (f"[{lower:.6f},{cutpoint:.6f})", persons.loc[persons["首次BMI"].lt(cutpoint)]),
            (f"[{cutpoint:.6f},{upper:.6f}]", persons.loc[persons["首次BMI"].ge(cutpoint)]),
        ]
    else:
        groups = [(f"[{lower:.6f},{upper:.6f}]", persons)]
    days = np.arange(module.检测起始天, module.检测结束天 + 1, dtype=int)
    rows = []
    for group_name, group in groups:
        planned_parts = []
        for day in days:
            planned = group.copy()
            planned["孕周天数"] = float(day)
            planned["孕周数"] = float(day) / 7.0
            planned_parts.append(planned)
        planned = pd.concat(planned_parts, ignore_index=True)
        probability = module.预测概率(model, planned).reshape(len(days), len(group)).mean(axis=1)
        delay = (days - module.检测起始天) / (module.检测结束天 - module.检测起始天)
        regret = np.maximum(delay, 1.0 - probability)
        index = int(np.flatnonzero(regret == regret.min())[0])
        day = int(days[index])
        rows.append(
            {
                "问题": role,
                "统计BMI组": group_name,
                "孕妇数": int(len(group)),
                "BMI最小值": float(group["首次BMI"].min()),
                "BMI最大值": float(group["首次BMI"].max()),
                "统计折中时点_天": day,
                "统计折中时点_周加天": module.周天文本(day),
                "预计达标比例": float(probability[index]),
                "预计未达标比例": float(1.0 - probability[index]),
                "最大遗憾": float(regret[index]),
                "题面风险等级": module.风险等级(day),
                "分组裁决": verdict["统计分组主裁决"],
                "分组来源": (
                    "第二问BMI分组审计的BIC、折间与整簇稳定性裁决"
                    if role == "第二问"
                    else "为保证第二、三问政策可比，沿用第二问BMI分组裁决；未在第三问多因素条件下独立重识别分组"
                ),
                "说明": (
                    "统计主分组；时点按等待窗口占比与绝对未达标概率的最小最大遗憾确定"
                    if role == "第二问"
                    else "沿用第二问统计分组；年龄、身高、生产次数只调整组内达标概率与时点"
                ),
            }
        )
    return pd.DataFrame(rows)


def 提取统计主时点结果(
    selected: pd.DataFrame,
    replicate: int,
    source: str,
) -> list[dict[str, Any]]:
    rows = []
    for _, row in selected.iterrows():
        day = row.get("统计折中时点_天", np.nan)
        group_label = (
            "统计主组（全体）"
            if row.get("分组裁决") == "不使用BMI的一组模型"
            else row["统计BMI组"]
        )
        rows.append(
            {
                "不确定性来源": source,
                "重复序号": replicate,
                "问题": row["问题"],
                "统计BMI组": group_label,
                "组内孕妇数": row["孕妇数"],
                "折中时点_天": day,
                "折中时点预计达标比例": row.get("预计达标比例", np.nan),
                "有效标志": int(pd.notna(day)),
            }
        )
    return rows


def 汇总统计主时点不确定性(detail: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, group in detail.groupby(
        ["不确定性来源", "问题", "统计BMI组"], sort=False
    ):
        valid = group.loc[group["有效标志"].eq(1)].copy()
        rows.append(
            {
                "不确定性来源": keys[0],
                "问题": keys[1],
                "统计BMI组": keys[2],
                "请求重复数": int(group["重复序号"].nunique()),
                "有效重复数": int(len(valid)),
                "有效率": float(len(valid) / len(group)) if len(group) else np.nan,
                "折中时点2.5%分位_天": (
                    float(valid["折中时点_天"].quantile(0.025)) if len(valid) else np.nan
                ),
                "折中时点中位数_天": (
                    float(valid["折中时点_天"].median()) if len(valid) else np.nan
                ),
                "折中时点97.5%分位_天": (
                    float(valid["折中时点_天"].quantile(0.975)) if len(valid) else np.nan
                ),
                "达标比例2.5%分位": (
                    float(valid["折中时点预计达标比例"].quantile(0.025))
                    if len(valid)
                    else np.nan
                ),
                "达标比例中位数": (
                    float(valid["折中时点预计达标比例"].median()) if len(valid) else np.nan
                ),
                "达标比例97.5%分位": (
                    float(valid["折中时点预计达标比例"].quantile(0.975))
                    if len(valid)
                    else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def 运行统计主时点不确定性(
    events: pd.DataFrame,
    selected_models: dict[str, str],
    measurement_summary: dict[str, float],
    verdict: dict[str, Any],
    module: Any,
    bootstrap_count: int,
    error_count: int,
) -> None:
    bootstrap_rng = np.random.default_rng(int(module.预期附件哈希[16:24], 16))
    error_rng = np.random.default_rng(int(module.预期附件哈希[24:32], 16))
    bootstrap_rows: list[dict[str, Any]] = []
    error_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []

    for replicate in range(1, bootstrap_count + 1):
        sample = module.整簇重采样(events, bootstrap_rng)
        for role in ("第二问", "第三问"):
            try:
                model = module.拟合概率候选(selected_models[role], sample, role)
                selected = 统计主分组时点(sample, model, verdict, role, module)
                bootstrap_rows.extend(
                    提取统计主时点结果(selected, replicate, "孕妇整簇重采样")
                )
            except Exception as exc:
                failure_rows.append(
                    {
                        "不确定性来源": "孕妇整簇重采样",
                        "重复序号": replicate,
                        "问题": role,
                        "错误类型": type(exc).__name__,
                        "错误": str(exc),
                    }
                )

    variance_hat = float(measurement_summary["合并组内方差"])
    df = int(round(measurement_summary["自由度"]))
    for replicate in range(1, error_count + 1):
        sampled_variance = df * variance_hat / float(error_rng.chisquare(df))
        perturbed = events.copy()
        standard_error = np.sqrt(
            sampled_variance / perturbed["记录数"].to_numpy(dtype=float)
        )
        perturbed_y = perturbed["Y浓度"].to_numpy(dtype=float) + error_rng.normal(
            0.0, standard_error, size=len(perturbed)
        )
        perturbed_y = np.clip(
            perturbed_y, np.finfo(float).eps, 1.0 - np.finfo(float).eps
        )
        perturbed["Y浓度"] = perturbed_y
        perturbed["Y浓度logit"] = np.log(perturbed_y / (1.0 - perturbed_y))
        perturbed["达标标志"] = (perturbed_y >= module.达标阈值).astype(int)
        for role in ("第二问", "第三问"):
            try:
                model = module.拟合概率候选(selected_models[role], perturbed, role)
                selected = 统计主分组时点(perturbed, model, verdict, role, module)
                error_rows.extend(
                    提取统计主时点结果(selected, replicate, "检测误差传播")
                )
            except Exception as exc:
                failure_rows.append(
                    {
                        "不确定性来源": "检测误差传播",
                        "重复序号": replicate,
                        "问题": role,
                        "错误类型": type(exc).__name__,
                        "错误": str(exc),
                    }
                )

    bootstrap_detail = pd.DataFrame(bootstrap_rows)
    error_detail = pd.DataFrame(error_rows)
    failures = pd.DataFrame(
        failure_rows,
        columns=["不确定性来源", "重复序号", "问题", "错误类型", "错误"],
    )
    写CSV(bootstrap_detail, 输出目录 / "30_统计主分组时点孕妇整簇重采样逐次.csv")
    写CSV(error_detail, 输出目录 / "30_统计主分组时点检测误差传播逐次.csv")
    写CSV(failures, 输出目录 / "30_统计主分组时点不确定性失败记录.csv")
    combined = pd.concat([bootstrap_detail, error_detail], ignore_index=True)
    写CSV(
        汇总统计主时点不确定性(combined),
        输出目录 / "30_统计主分组时点不确定性汇总.csv",
    )

    convergence_tables = []
    for source, detail, requested in (
        ("孕妇整簇重采样", bootstrap_detail, bootstrap_count),
        ("检测误差传播", error_detail, error_count),
    ):
        for prefix in sorted(
            set(value for value in (100, 200, requested) if value <= requested)
        ):
            summary = 汇总统计主时点不确定性(
                detail.loc[detail["重复序号"].le(prefix)].copy()
            )
            summary["前缀重复数"] = prefix
            summary["不确定性来源"] = source
            convergence_tables.append(summary)
    写CSV(
        pd.concat(convergence_tables, ignore_index=True),
        输出目录 / "30_统计主分组时点不确定性次数收敛.csv",
    )


def 补充覆盖性推断边界() -> None:
    for filename in (
        "14_第二三问主路线三套样本参数表.csv",
        "15_第二三问数据段稀疏与分离诊断.csv",
    ):
        path = 输出目录 / filename
        frame = pd.read_csv(path, encoding="utf-8-sig")
        scope = frame["样本范围"].astype(str)
        expected_flag = np.where(scope.eq("序号683前主分析"), 1, 0)
        expected_boundary = np.select(
            [
                scope.eq("序号683前主分析"),
                scope.eq("序号683后不稳定敏感性"),
            ],
            [
                "仅用于683前主分析样本的关联推断",
                "后段失败仅来自1名孕妇，系数和P值不作有效推断",
            ],
            default="全样本参数只作段基线调整覆盖性描述，不解释为683后条件规律",
        )
        # 重复运行补充审计时不重写已经正确的浮点结果表，避免文本末位漂移。
        if (
            "后段条件推断可用标志" in frame.columns
            and "P值解释边界" in frame.columns
            and np.array_equal(
                pd.to_numeric(frame["后段条件推断可用标志"]).to_numpy(dtype=int),
                expected_flag,
            )
            and np.array_equal(frame["P值解释边界"].astype(str).to_numpy(), expected_boundary)
        ):
            continue
        frame["后段条件推断可用标志"] = expected_flag
        frame["P值解释边界"] = expected_boundary
        写CSV(frame, path)


def 补充题面五组证据标志() -> None:
    uncertainty = pd.read_csv(
        输出目录 / "20_主路线时点不确定性汇总.csv", encoding="utf-8-sig"
    )
    for role in ("第二问", "第三问"):
        path = 输出目录 / f"12_{role}主分析题面分组折中时点.csv"
        frame = pd.read_csv(path, encoding="utf-8-sig")
        frame["输出地位"] = "题面实施分组敏感性；不是统计识别的主分组"
        frame["证据等级"] = np.select(
            [frame["孕妇数"].le(5), frame["孕妇数"].lt(20)],
            ["极稀疏，仅探索性", "稀疏，谨慎解释"],
            default="样本内实施敏感性",
        )
        for source, prefix in (
            ("孕妇整簇重采样", "孕妇整簇"),
            ("检测误差传播", "检测误差"),
        ):
            support = uncertainty.loc[
                uncertainty["问题"].eq(role)
                & uncertainty["不确定性来源"].eq(source),
                ["题面BMI组", "折中时点2.5%分位_天", "折中时点97.5%分位_天"],
            ].rename(
                columns={
                    "折中时点2.5%分位_天": f"{prefix}时点95%下限_天",
                    "折中时点97.5%分位_天": f"{prefix}时点95%上限_天",
                }
            )
            frame = frame.merge(support, on="题面BMI组", how="left", validate="one_to_one")
        写CSV(frame, path)


def 更新统一运行清单(
    module: Any,
    bootstrap_count: int,
    error_count: int,
) -> None:
    manifest_path = 输出目录 / "00_运行清单.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    supplemental_outputs = [
        "22_第一问主模型分段GEE参数.csv",
        "23_第一问主模型前后段派生效应.csv",
        "24_第一问主模型分段异质性Wald检验.csv",
        "25_第二问BMI统计分组支持度.csv",
        "26_第二问BMI切点整簇重采样次数收敛.csv",
        "26_第二问BMI切点整簇重采样逐次.csv",
        "26_第二问单切点折间稳定性.csv",
        "27_第二问BMI统计分组裁决.json",
        "27_第二问统计主分组与折中时点.csv",
        "27_第三问统计主分组与折中时点.csv",
        "29_第二问AFT历史敏感性候选比较.csv",
        "29_第二问AFT历史敏感性引用.json",
        "30_统计主分组时点孕妇整簇重采样逐次.csv",
        "30_统计主分组时点检测误差传播逐次.csv",
        "30_统计主分组时点不确定性失败记录.csv",
        "30_统计主分组时点不确定性汇总.csv",
        "30_统计主分组时点不确定性次数收敛.csv",
    ]
    all_outputs = list(dict.fromkeys([*manifest["核心输出文件"], *supplemental_outputs]))
    missing = [name for name in all_outputs if not (输出目录 / name).is_file()]
    if missing:
        raise RuntimeError(f"统一运行清单缺少输出：{missing}")
    script_path = Path(__file__).resolve()
    manifest["补充生成脚本"] = str(script_path)
    manifest["补充生成脚本SHA256"] = module.文件哈希(script_path)
    manifest["补充运行命令"] = (
        f'python "{script_path}" --输出目录 "{输出目录}" '
        f'--孕妇整簇重采样次数 {bootstrap_count} --检测误差传播次数 {error_count}'
    )
    manifest["补充计算设置"] = {
        "BMI裁决孕妇整簇重采样次数": bootstrap_count,
        "统计主时点孕妇整簇重采样次数": bootstrap_count,
        "统计主时点检测误差传播次数": error_count,
        "说明": "均为不确定性计算的重复次数，不是科学阈值或临床参数",
    }
    manifest["分析生成链"] = [
        {
            "阶段": "主分析",
            "脚本": manifest["生成脚本"],
            "脚本SHA256": manifest["生成脚本SHA256"],
            "运行命令": manifest["运行命令"],
        },
        {
            "阶段": "补充主审",
            "脚本": str(script_path),
            "脚本SHA256": manifest["补充生成脚本SHA256"],
            "运行命令": manifest["补充运行命令"],
        },
    ]
    manifest["核心输出文件"] = all_outputs
    manifest["核心输出SHA256"] = {
        name: module.文件哈希(输出目录 / name) for name in all_outputs
    }
    manifest["分析生成链外的独立复核产物"] = [
        name
        for name in ("90_V2独立复核清单.csv", "91_V2独立复核报告.md", "92_V2自审状态.json")
        if (输出目录 / name).is_file()
    ]
    manifest.pop("目录中其他文件不属于本清单", None)
    写JSON(manifest, manifest_path)


def 导出AFT历史敏感性引用(
    module: Any,
    evidence_directory: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    evidence_directory = evidence_directory.resolve()
    comparison_path = evidence_directory / "第二问AFT候选统一比较.csv"
    assertion_path = evidence_directory / "第二问数据构造断言.json"
    manifest_path = evidence_directory / "第二问运行清单.json"
    script_path = evidence_directory / "第二问无自拟参数重构.py"
    for path in (comparison_path, assertion_path, manifest_path, script_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    comparison = pd.read_csv(comparison_path, encoding="utf-8-sig")
    comparison["当前敏感性地位"] = "历史AFT敏感性；不承担当前主结论"
    comparison["不可与当前路线直接排名原因"] = (
        "AFT估计稀疏观测下的首次达标时间分布；当前主模型估计计划日在场达标概率，目标量不同"
    )
    assertion = json.loads(assertion_path.read_text(encoding="utf-8"))
    old_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    reference = {
        "状态": "RETAINED_AS_REJECTED_SENSITIVITY",
        "目标量": "首次达到4%的区间删失时间分布",
        "当前定位": "回答目标量敏感性；不与计划日在场达标概率主路线直接排名",
        "降级原因": (
            "32名孕妇观测到达标后回落，真实连续轨迹的首次穿越时间在稀疏随访下不可直接识别；"
            "AFT结果依赖单次事件时间和分布假设"
        ),
        "历史包孕妇数": int(assertion["孕妇数"]),
        "历史包事件数": int(assertion["抽血事件数"]),
        "左删失人数": int(assertion["删失结构"]["左删失"]),
        "区间删失人数": int(assertion["删失结构"]["区间删失"]),
        "右删失人数": int(assertion["删失结构"]["右删失"]),
        "达标后回落人数": int(assertion["达标后回落人数"]),
        "原始工作簿SHA256": old_manifest["原始工作簿SHA256"],
        "路径基准": "工作区根目录",
        "历史生成脚本": str(script_path.relative_to(module.工作区)),
        "历史生成脚本SHA256": module.文件哈希(script_path),
        "历史候选比较文件": str(comparison_path.relative_to(module.工作区)),
        "历史候选比较SHA256": module.文件哈希(comparison_path),
        "历史运行清单": str(manifest_path.relative_to(module.工作区)),
        "历史运行清单SHA256": module.文件哈希(manifest_path),
        "冻结证据目录": str(evidence_directory.relative_to(module.工作区)),
        "历史复现入口": "冻结证据仅用于闭合当前引用链；历史AFT完整重跑包保存在后台材料",
    }
    return comparison, reference


def main(
    output_directory: Path,
    bootstrap_count: int,
    error_count: int,
    aft_evidence_directory: Path | None,
) -> None:
    global 输出目录
    输出目录 = output_directory.resolve()
    输出目录.mkdir(parents=True, exist_ok=True)
    module = 载入主脚本()
    records = module.读取男胎记录(module.默认附件)
    events, _ = module.构造事件层(records)

    q1_parameters, q1_effects, q1_joint = 第一问分段GEE(events, module)
    写CSV(q1_parameters, 输出目录 / "22_第一问主模型分段GEE参数.csv")
    写CSV(q1_effects, 输出目录 / "23_第一问主模型前后段派生效应.csv")
    写CSV(q1_joint, 输出目录 / "24_第一问主模型分段异质性Wald检验.csv")

    reference_events = events.loc[events["数据段"].eq("序号683前")].copy()
    bmi_models, bmi_folds, bmi_bootstraps, bmi_convergence, bmi_verdict = 第二问BMI切点审计(
        reference_events, module, bootstrap_count=bootstrap_count
    )
    bmi_models["分析样本"] = "序号683前主分析"
    bmi_folds["分析样本"] = "序号683前主分析"
    bmi_bootstraps["分析样本"] = "序号683前主分析"
    bmi_convergence["分析样本"] = "序号683前主分析"
    bmi_verdict["分析样本"] = "序号683前主分析"
    写CSV(bmi_models, 输出目录 / "25_第二问BMI统计分组支持度.csv")
    写CSV(bmi_folds, 输出目录 / "26_第二问单切点折间稳定性.csv")
    写CSV(bmi_bootstraps, 输出目录 / "26_第二问BMI切点整簇重采样逐次.csv")
    写CSV(bmi_convergence, 输出目录 / "26_第二问BMI切点整簇重采样次数收敛.csv")
    写JSON(bmi_verdict, 输出目录 / "27_第二问BMI统计分组裁决.json")

    route = json.loads((输出目录 / "13_第二三问暂定主路线.json").read_text(encoding="utf-8"))
    selected_models: dict[str, str] = {}
    for role in ("第二问", "第三问"):
        selected_models[role] = route[f"{role}主路线"]
        model = module.拟合概率候选(selected_models[role], reference_events, role)
        写CSV(
            统计主分组时点(reference_events, model, bmi_verdict, role, module),
            输出目录 / f"27_{role}统计主分组与折中时点.csv",
        )

    measurement_summary = json.loads(
        (输出目录 / "04_严格技术复测误差摘要.json").read_text(encoding="utf-8")
    )
    运行统计主时点不确定性(
        reference_events,
        selected_models,
        measurement_summary,
        bmi_verdict,
        module,
        bootstrap_count,
        error_count,
    )

    补充覆盖性推断边界()
    补充题面五组证据标志()

    old_interaction_path = 输出目录 / "07_第一问前后段交互检验.csv"
    old_interaction = pd.read_csv(old_interaction_path, encoding="utf-8-sig")
    old_interaction["模型地位"] = "被第一问分段GEE主模型替代的旧混合效应敏感性"
    old_interaction["推断可用标志"] = 0
    old_interaction["替代证据文件"] = "24_第一问主模型分段异质性Wald检验.csv"
    old_interaction["P值使用边界"] = "不得作为主推断；仅保留旧路线审计痕迹"
    写CSV(old_interaction, old_interaction_path)

    for filename in ("01_全样本抽血事件表.csv", "02_孕周歧义排除事件.csv"):
        path = 输出目录 / filename
        frame = pd.read_csv(path, encoding="utf-8-sig")
        frame = frame.rename(columns={"BMI": "体质指数（BMI）"})
        写CSV(frame, path)
    old_q1_path = 输出目录 / "05_第一问683敏感性参数比较.csv"
    old_q1 = pd.read_csv(old_q1_path, encoding="utf-8-sig").rename(
        columns={"AIC": "赤池信息准则（AIC）", "BIC": "贝叶斯信息准则（BIC）"}
    )
    写CSV(old_q1, old_q1_path)
    bmi_models = bmi_models.rename(columns={"BIC": "贝叶斯信息准则（BIC）"})
    写CSV(bmi_models, 输出目录 / "25_第二问BMI统计分组支持度.csv")

    if aft_evidence_directory is None:
        aft_evidence_directory = (
            module.工作区
            / "05_公共材料"
            / "06_核心代码索引"
            / "复现依赖"
            / "第二问AFT历史敏感性"
        )
    aft_comparison, aft_reference = 导出AFT历史敏感性引用(
        module,
        aft_evidence_directory,
    )
    写CSV(aft_comparison, 输出目录 / "29_第二问AFT历史敏感性候选比较.csv")
    写JSON(aft_reference, 输出目录 / "29_第二问AFT历史敏感性引用.json")

    更新统一运行清单(module, bootstrap_count, error_count)

    print(
        json.dumps(
            {
                "第一问分段GEE": "完成",
                "第一问交互P值": float(q1_joint.loc[0, "P值"]),
                "第二问统计分组裁决": bmi_verdict["统计分组主裁决"],
                "第二问全样本最优单切点": bmi_verdict["全样本最优单切点"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="补充第一问分段GEE和第二问BMI切点稳定性审计")
    parser.add_argument("--输出目录", type=Path, default=输出目录)
    parser.add_argument("--孕妇整簇重采样次数", type=int, default=400)
    parser.add_argument("--检测误差传播次数", type=int, default=400)
    parser.add_argument(
        "--AFT证据目录",
        type=Path,
        default=None,
        help="包含历史AFT脚本、数据断言、候选比较和运行清单的冻结证据目录",
    )
    args = parser.parse_args()
    main(
        args.输出目录,
        args.孕妇整簇重采样次数,
        args.检测误差传播次数,
        args.AFT证据目录,
    )
