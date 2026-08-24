#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""第一问候选 B：受控自然立方样条混合模型。

约束：
1. 只读取共同口径中的 613 个冻结抽血事件；
2. 以孕妇为随机截距和交叉验证分组；
3. 交叉验证测试孕妇只使用固定效应预测；
4. 结构选择阶段使用 ML，结构固定后使用 REML；
5. 不生成任何图像，只输出中文 CSV、Markdown 与 MATLAB-SVG 制图提示词 TXT。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import patsy
import scipy
from patsy import build_design_matrices, dmatrix
from scipy.special import expit
from scipy.stats import chi2, kurtosis, norm, shapiro, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
import sklearn
import statsmodels
import statsmodels.api as sm
from statsmodels.regression.mixed_linear_model import MixedLM
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor


随机种子 = 20250824
交叉验证折数 = 5
聚类自助重复数 = 300
样条保留最小RMSE改善比例 = 0.01
复杂项保留最小RMSE改善比例 = 0.01
响应尺度近似持平比例 = 0.01
np.random.seed(随机种子)


@dataclass
class 拟合结果:
    结果: Any
    设计信息: Any
    设计矩阵: pd.DataFrame
    警告: str
    响应尺度: str
    公式: str
    随机斜率: bool


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest().upper()


def 写CSV(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig", float_format="%.12g")


def 读取并校验(path: Path) -> pd.DataFrame:
    d = pd.read_csv(path, encoding="utf-8-sig")
    必需列 = [
        "孕妇代码", "抽血事件键", "孕周数", "孕妇平均BMI", "BMI个体内偏差",
        "Y染色体浓度均值", "Y浓度对数几率", "年龄", "辅助生殖标志", "生产次数",
        "怀孕次数", "GC含量均值", "原始读段数均值", "比对比例均值",
        "重复读段比例均值", "过滤读段比例均值", "任一记录日期孕周偏差超14天标志",
        "纳入截至25周0天敏感性标志", "纳入主模型标志",
    ]
    缺列 = [c for c in 必需列 if c not in d.columns]
    if 缺列:
        raise ValueError(f"冻结数据缺少列：{缺列}")
    if len(d) != 613 or d["孕妇代码"].nunique() != 167:
        raise ValueError(
            f"冻结样本口径不符：事件={len(d)}，孕妇={d['孕妇代码'].nunique()}，期望 613/167"
        )
    if not (d["纳入主模型标志"] == 1).all():
        raise ValueError("冻结样本中存在未纳入主模型的事件")
    if d[["孕周数", "孕妇平均BMI", "BMI个体内偏差", "Y染色体浓度均值"]].isna().any().any():
        raise ValueError("核心变量存在缺失")
    if not d["Y染色体浓度均值"].between(0, 1, inclusive="neither").all():
        raise ValueError("Y 浓度不全在 (0,1)，不能直接使用既定 logit 变换")
    # 冻结协议修订：怀孕次数缺失 167/613，不得进入主调整块。
    if int(d["怀孕次数"].isna().sum()) != 167:
        raise ValueError("怀孕次数缺失数与冻结协议修订不一致")

    重命名 = {
        "孕妇代码": "woman",
        "抽血事件键": "event",
        "孕周数": "week",
        "孕妇平均BMI": "bmean",
        "BMI个体内偏差": "bdev",
        "Y染色体浓度均值": "y",
        "Y浓度对数几率": "y_logit",
        "年龄": "age",
        "辅助生殖标志": "ivf",
        "生产次数": "parity",
        "怀孕次数": "gravidity",
        "GC含量均值": "gc",
        "原始读段数均值": "reads",
        "比对比例均值": "mapping",
        "重复读段比例均值": "dup",
        "过滤读段比例均值": "filtered",
        "任一记录日期孕周偏差超14天标志": "date_bad",
        "纳入截至25周0天敏感性标志": "within25",
    }
    d = d.rename(columns=重命名).copy()
    d["log_reads"] = np.log10(d["reads"].astype(float))
    return d


def 计算参照值(d: pd.DataFrame) -> dict[str, float]:
    refs: dict[str, float] = {}
    for c in ["week", "bmean", "age", "parity", "gc", "log_reads", "mapping", "dup", "filtered"]:
        refs[f"{c}_mean"] = float(d[c].mean())
        refs[f"{c}_sd"] = float(d[c].std(ddof=1))
    return refs


def 应用参照值(d: pd.DataFrame, refs: dict[str, float]) -> pd.DataFrame:
    x = d.copy()
    x["week_c"] = x["week"] - refs["week_mean"]
    x["week_z"] = x["week_c"] / refs["week_sd"]
    x["bmean_c"] = x["bmean"] - refs["bmean_mean"]
    x["age_c"] = x["age"] - refs["age_mean"]
    x["parity_c"] = x["parity"] - refs["parity_mean"]
    for c in ["gc", "log_reads", "mapping", "dup", "filtered"]:
        x[f"{c}_z"] = (x[c] - refs[f"{c}_mean"]) / refs[f"{c}_sd"]
    return x


def 公式集合() -> dict[str, str]:
    # 辅助生殖仅有11事件/3孕妇，按簇稀疏性不进主调整块；不是按p值筛选。
    临床块 = "age_c + parity_c"
    return {
        "线性": f"1 + week_c + bmean_c + bdev + {临床块}",
        "孕周样条3": (
            "1 + cr(week, df=3, constraints='center') + bmean_c + bdev + " + 临床块
        ),
        "孕周样条4": (
            "1 + cr(week, df=4, constraints='center') + bmean_c + bdev + " + 临床块
        ),
        "孕周样条3+妇间BMI样条3": (
            "1 + cr(week, df=3, constraints='center') "
            "+ cr(bmean, df=3, constraints='center') + bdev + " + 临床块
        ),
        "孕周样条3+孕周妇间BMI交互": (
            "1 + cr(week, df=3, constraints='center') + bmean_c + bdev + "
            + 临床块
            + " + cr(week, df=3, constraints='center'):bmean_c"
        ),
    }


def 响应向量(d: pd.DataFrame, scale: str) -> np.ndarray:
    if scale == "原Y尺度":
        return d["y"].to_numpy(float)
    if scale == "logit尺度":
        return d["y_logit"].to_numpy(float)
    raise ValueError(scale)


def 反变换(eta: np.ndarray | float, scale: str) -> np.ndarray:
    arr = np.asarray(eta, dtype=float)
    return arr if scale == "原Y尺度" else expit(arr)


def 反变换导数(eta: np.ndarray | float, scale: str) -> np.ndarray:
    arr = np.asarray(eta, dtype=float)
    if scale == "原Y尺度":
        return np.ones_like(arr)
    mu = expit(arr)
    return mu * (1 - mu)


def 拟合矩阵(
    y: np.ndarray,
    X: pd.DataFrame,
    groups: pd.Series,
    reml: bool,
    week_z: np.ndarray | None = None,
) -> tuple[Any, str]:
    random_slope = week_z is not None
    exog_re = None
    if random_slope:
        exog_re = pd.DataFrame(
            {"随机截距": np.ones(len(X)), "孕周随机斜率": np.asarray(week_z, float)},
            index=X.index,
        )
    model = MixedLM(endog=y, exog=X, groups=np.asarray(groups), exog_re=exog_re)
    errors: list[str] = []
    all_warnings: list[str] = []
    # statsmodels 在方差接近边界时可能在优化结束后的 Hessian 求逆处失败；
    # 逐个优化器独立重试，避免 method 列表在首次求逆失败时无法继续。
    for method in ["lbfgs", "bfgs", "cg", "powell"]:
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                result = model.fit(
                    reml=reml,
                    method=method,
                    maxiter=1600,
                    disp=False,
                )
            all_warnings.extend(str(w.message) for w in caught)
            warning_text = " | ".join(sorted(set(all_warnings + errors)))
            return result, warning_text
        except Exception as e:
            errors.append(f"{method}失败:{type(e).__name__}:{e}")
    raise RuntimeError("；".join(errors))


def 拟合模型(
    d: pd.DataFrame,
    formula: str,
    scale: str,
    reml: bool,
    design_info: Any | None = None,
    random_slope: bool = False,
) -> 拟合结果:
    if design_info is None:
        X = dmatrix(formula, d, return_type="dataframe")
        design_info = X.design_info
    else:
        X = build_design_matrices([design_info], d, return_type="dataframe")[0]
        X = pd.DataFrame(X, index=d.index, columns=design_info.column_names)
    y = 响应向量(d, scale)
    wz = d["week_z"].to_numpy(float) if random_slope else None
    result, warning_text = 拟合矩阵(y, X, d["woman"], reml, wz)
    return 拟合结果(result, design_info, X, warning_text, scale, formula, random_slope)


def 固定效应预测(fit: 拟合结果, X: pd.DataFrame) -> np.ndarray:
    eta = np.asarray(X) @ np.asarray(fit.结果.fe_params)
    return 反变换(eta, fit.响应尺度)


def 交叉验证(
    d原: pd.DataFrame,
    formula: str,
    scale: str,
    random_slope: bool = False,
) -> tuple[dict[str, float], pd.DataFrame, pd.DataFrame]:
    cv = GroupKFold(n_splits=交叉验证折数, shuffle=True, random_state=随机种子)
    oof = np.full(len(d原), np.nan)
    折行: list[dict[str, Any]] = []
    预测行: list[pd.DataFrame] = []
    for fold, (tr, te) in enumerate(cv.split(d原, groups=d原["woman"]), start=1):
        train0 = d原.iloc[tr].copy()
        test0 = d原.iloc[te].copy()
        if set(train0["woman"]).intersection(set(test0["woman"])):
            raise AssertionError("孕妇分组交叉验证发生泄漏")
        refs = 计算参照值(train0)
        train = 应用参照值(train0, refs)
        test = 应用参照值(test0, refs)
        Xtr = dmatrix(formula, train, return_type="dataframe")
        Xte_m = build_design_matrices([Xtr.design_info], test, return_type="dataframe")[0]
        Xte = pd.DataFrame(Xte_m, index=test.index, columns=Xtr.design_info.column_names)
        ytr = 响应向量(train, scale)
        wz = train["week_z"].to_numpy(float) if random_slope else None
        result, warning_text = 拟合矩阵(ytr, Xtr, train["woman"], False, wz)
        # 对测试折的新孕妇只用固定效应，绝不调用 result.predict 或随机效应。
        eta = np.asarray(Xte) @ np.asarray(result.fe_params)
        pred = 反变换(eta, scale)
        oof[te] = pred
        yte = test0["y"].to_numpy(float)
        折行.append(
            {
                "折号": fold,
                "训练事件数": len(tr),
                "测试事件数": len(te),
                "训练孕妇数": train0["woman"].nunique(),
                "测试孕妇数": test0["woman"].nunique(),
                "孕妇泄漏数": 0,
                "收敛": int(bool(result.converged)),
                "警告": warning_text,
                "RMSE": math.sqrt(mean_squared_error(yte, pred)),
                "MAE": mean_absolute_error(yte, pred),
                "R方": r2_score(yte, pred),
            }
        )
        pr = pd.DataFrame(
            {
                "折号": fold,
                "原行号": te,
                "孕妇代码": test0["woman"].to_numpy(),
                "抽血事件键": test0["event"].to_numpy(),
                "实测Y浓度": yte,
                "固定效应预测Y浓度": pred,
            }
        )
        预测行.append(pr)
    if np.isnan(oof).any():
        raise AssertionError("交叉验证存在未预测事件")
    fold_df = pd.DataFrame(折行)
    pred_df = pd.concat(预测行, ignore_index=True)
    pooled = {
        "组外RMSE": math.sqrt(mean_squared_error(d原["y"], oof)),
        "组外MAE": mean_absolute_error(d原["y"], oof),
        "组外R方": r2_score(d原["y"], oof),
        "折均RMSE": float(fold_df["RMSE"].mean()),
        "折间RMSE标准差": float(fold_df["RMSE"].std(ddof=1)),
        "折均MAE": float(fold_df["MAE"].mean()),
        "折间MAE标准差": float(fold_df["MAE"].std(ddof=1)),
        "全部折收敛": int((fold_df["收敛"] == 1).all()),
        "预测最小值": float(np.min(oof)),
        "预测最大值": float(np.max(oof)),
        "越界预测数": int(np.sum((oof <= 0) | (oof >= 1))),
    }
    return pooled, fold_df, pred_df


def 模型摘要行(name: str, scale: str, fit: 拟合结果, cv: dict[str, float]) -> dict[str, Any]:
    r = fit.结果
    cov_re = np.asarray(r.cov_re, float)
    return {
        "模型": name,
        "响应尺度": scale,
        "固定效应参数数": int(r.k_fe),
        "随机效应协方差参数数": int(r.k_re2),
        "收敛": int(bool(r.converged)),
        "ML对数似然": float(r.llf),
        "AIC": float(r.aic),
        "BIC": float(r.bic),
        "随机截距方差": float(cov_re[0, 0]),
        "残差方差": float(r.scale),
        "拟合警告": fit.警告,
        **cv,
    }


def LRT(reduced: 拟合结果, full: 拟合结果) -> tuple[float, int, float]:
    lr = max(0.0, 2.0 * (float(full.结果.llf) - float(reduced.结果.llf)))
    df = int(full.结果.k_fe - reduced.结果.k_fe)
    if df <= 0:
        raise ValueError("LRT 自由度差必须为正")
    return lr, df, float(chi2.sf(lr, df))


def 选择结构(
    scale: str,
    fits: dict[tuple[str, str], 拟合结果],
    cvs: dict[tuple[str, str], dict[str, float]],
) -> tuple[str, list[dict[str, Any]]]:
    decisions: list[dict[str, Any]] = []
    lr, df, p = LRT(fits[(scale, "线性")], fits[(scale, "孕周样条3")])
    cv_lin, cv_s3 = cvs[(scale, "线性")], cvs[(scale, "孕周样条3")]
    improve = (cv_lin["组外RMSE"] - cv_s3["组外RMSE"]) / cv_lin["组外RMSE"]
    spline_ok = (
        p < 0.05
        and improve >= 样条保留最小RMSE改善比例
        and cv_s3["组外MAE"] <= cv_lin["组外MAE"]
    )
    decisions.append(
        {
            "响应尺度": scale,
            "决策项": "孕周由线性升级为3自由度自然样条",
            "整体检验P值": p,
            "组外RMSE改善比例": improve,
            "组外MAE是否不劣": int(cv_s3["组外MAE"] <= cv_lin["组外MAE"]),
            "是否通过": int(spline_ok),
            "规则": "LRT P<0.05，且组外RMSE至少改善1%，且MAE不劣",
        }
    )
    selected = "孕周样条3" if spline_ok else "线性"
    if selected == "孕周样条3":
        aic3 = fits[(scale, "孕周样条3")].结果.aic
        aic4 = fits[(scale, "孕周样条4")].结果.aic
        cv4 = cvs[(scale, "孕周样条4")]
        improve4 = (cv_s3["组外RMSE"] - cv4["组外RMSE"]) / cv_s3["组外RMSE"]
        df4_ok = (
            (aic3 - aic4) >= 2
            and improve4 >= 复杂项保留最小RMSE改善比例
            and cv4["组外MAE"] <= cv_s3["组外MAE"]
        )
        decisions.append(
            {
                "响应尺度": scale,
                "决策项": "孕周样条由3自由度升级为4自由度",
                "整体检验P值": np.nan,
                "组外RMSE改善比例": improve4,
                "组外MAE是否不劣": int(cv4["组外MAE"] <= cv_s3["组外MAE"]),
                "是否通过": int(df4_ok),
                "规则": "非嵌套比较：AIC至少降低2，且组外RMSE至少改善1%，且MAE不劣",
            }
        )
        if df4_ok:
            selected = "孕周样条4"

    # 妇间 BMI 非线性与交互都只在预先限定的 3-df 孕周样条上检验。
    # 即使其 P 值偶然显著，也必须同时通过组外验证才可保留。
    for extra_name, label in [
        ("孕周样条3+妇间BMI样条3", "妇间BMI由线性升级为3自由度自然样条"),
        ("孕周样条3+孕周妇间BMI交互", "加入孕周与妇间BMI交互"),
    ]:
        lr_e, df_e, p_e = LRT(fits[(scale, "孕周样条3")], fits[(scale, extra_name)])
        cv_e = cvs[(scale, extra_name)]
        improve_e = (cv_s3["组外RMSE"] - cv_e["组外RMSE"]) / cv_s3["组外RMSE"]
        ok_e = (
            selected == "孕周样条3"
            and p_e < 0.05
            and improve_e >= 复杂项保留最小RMSE改善比例
            and cv_e["组外MAE"] <= cv_s3["组外MAE"]
        )
        decisions.append(
            {
                "响应尺度": scale,
                "决策项": label,
                "整体检验P值": p_e,
                "组外RMSE改善比例": improve_e,
                "组外MAE是否不劣": int(cv_e["组外MAE"] <= cv_s3["组外MAE"]),
                "是否通过": int(ok_e),
                "规则": "LRT P<0.05，且组外RMSE至少改善1%，且MAE不劣；仅在3-df主效应结构上保留",
            }
        )
        if ok_e:
            # 两个扩展若都通过，取组外 RMSE 更小者；绝不同时堆叠。
            if selected == "孕周样条3" or cv_e["组外RMSE"] < cvs[(scale, selected)]["组外RMSE"]:
                selected = extra_name
    return selected, decisions


def 构造参照行(refs: dict[str, float], week: float, bmean: float, bdev: float) -> pd.DataFrame:
    raw = pd.DataFrame(
        {
            "week": [week],
            "bmean": [bmean],
            "bdev": [bdev],
            "age": [refs["age_mean"]],
            "ivf": [0.0],
            "parity": [refs["parity_mean"]],
            "gc": [refs["gc_mean"]],
            "log_reads": [refs["log_reads_mean"]],
            "mapping": [refs["mapping_mean"]],
            "dup": [refs["dup_mean"]],
            "filtered": [refs["filtered_mean"]],
        }
    )
    raw["reads"] = 10 ** raw["log_reads"]
    return 应用参照值(raw, refs)


def 设计行(fit: 拟合结果, row: pd.DataFrame) -> np.ndarray:
    mat = build_design_matrices([fit.设计信息], row, return_type="dataframe")[0]
    return np.asarray(mat, float).reshape(-1)


def 固定协方差(fit: 拟合结果) -> np.ndarray:
    k = fit.结果.k_fe
    return np.asarray(fit.结果.cov_params(), float)[:k, :k]


def 单点预测(fit: 拟合结果, x: np.ndarray) -> tuple[float, float, float, float]:
    beta = np.asarray(fit.结果.fe_params, float)
    cov = 固定协方差(fit)
    eta = float(x @ beta)
    mu = float(反变换(eta, fit.响应尺度))
    grad = float(反变换导数(eta, fit.响应尺度)) * x
    se = float(math.sqrt(max(0.0, grad @ cov @ grad)))
    return mu, se, mu - 1.96 * se, mu + 1.96 * se


def 原尺度对比(
    fit: 拟合结果,
    row1: pd.DataFrame,
    row2: pd.DataFrame,
) -> tuple[float, float, float, float, float]:
    x1, x2 = 设计行(fit, row1), 设计行(fit, row2)
    beta = np.asarray(fit.结果.fe_params, float)
    cov = 固定协方差(fit)
    e1, e2 = float(x1 @ beta), float(x2 @ beta)
    m1, m2 = float(反变换(e1, fit.响应尺度)), float(反变换(e2, fit.响应尺度))
    grad = float(反变换导数(e2, fit.响应尺度)) * x2 - float(
        反变换导数(e1, fit.响应尺度)
    ) * x1
    se = float(math.sqrt(max(0.0, grad @ cov @ grad)))
    est = m2 - m1
    z = est / se if se > 0 else np.nan
    p = 2 * norm.sf(abs(z)) if np.isfinite(z) else np.nan
    return est, se, est - 1.96 * se, est + 1.96 * se, float(p)


def 核心效应表(fit: 拟合结果, refs: dict[str, float]) -> pd.DataFrame:
    b0 = refs["bmean_mean"]
    specs = [
        ("孕周由12周增至20周", 构造参照行(refs, 12, b0, 0), 构造参照行(refs, 20, b0, 0)),
        ("孕周由12周增至24周", 构造参照行(refs, 12, b0, 0), 构造参照行(refs, 24, b0, 0)),
        ("妇间平均BMI增加1单位", 构造参照行(refs, 18, b0, 0), 构造参照行(refs, 18, b0 + 1, 0)),
        ("同一孕妇BMI相对本人均值增加1单位", 构造参照行(refs, 18, b0, 0), 构造参照行(refs, 18, b0, 1)),
    ]
    rows = []
    for name, r1, r2 in specs:
        est, se, lo, hi, p = 原尺度对比(fit, r1, r2)
        rows.append(
            {
                "效应名称": name,
                "参照条件": "年龄、生产次数取样本均值；未纳入随机效应",
                "Y浓度变化": est,
                "标准误": se,
                "95%置信区间下限": lo,
                "95%置信区间上限": hi,
                "Y浓度变化百分点": est * 100,
                "百分点下限": lo * 100,
                "百分点上限": hi * 100,
                "对比近似P值": p,
            }
        )
    return pd.DataFrame(rows)


def 曲线预测表(fit: 拟合结果, refs: dict[str, float]) -> pd.DataFrame:
    rows = []
    b0 = refs["bmean_mean"]
    for bdelta, label in [(-2, "平均BMI减2"), (0, "平均BMI"), (2, "平均BMI加2")]:
        for week in np.linspace(11, 25.7142857143, 104):
            row = 构造参照行(refs, float(week), b0 + bdelta, 0)
            mu, se, lo, hi = 单点预测(fit, 设计行(fit, row))
            rows.append(
                {
                    "孕周数": week,
                    "妇间BMI情景": label,
                    "孕妇平均BMI": b0 + bdelta,
                    "BMI个体内偏差": 0,
                    "预测Y浓度": mu,
                    "标准误": se,
                    "95%置信区间下限": lo,
                    "95%置信区间上限": hi,
                    "预测Y浓度百分比": mu * 100,
                    "下限百分比": lo * 100,
                    "上限百分比": hi * 100,
                }
            )
    return pd.DataFrame(rows)


def 整体检验(
    d: pd.DataFrame,
    refs: dict[str, float],
    selected_name: str,
    selected_formula: str,
    scale: str,
    final_ml: 拟合结果,
) -> pd.DataFrame:
    nuisance = "age_c + parity_c"
    if selected_name == "线性":
        no_week = f"1 + bmean_c + bdev + {nuisance}"
    else:
        no_week = f"1 + bmean_c + bdev + {nuisance}"
    no_bmean = selected_formula.replace(" + bmean_c", "")
    no_bdev = selected_formula.replace(" + bdev", "")
    # 扩展项并未与其它扩展堆叠；若被选中，使用更明确的约简式。
    if selected_name == "孕周样条3+妇间BMI样条3":
        no_bmean = f"1 + cr(week, df=3, constraints='center') + bdev + {nuisance}"
    if selected_name == "孕周样条3+孕周妇间BMI交互":
        no_bmean = f"1 + cr(week, df=3, constraints='center') + bdev + {nuisance}"
        no_week = f"1 + bmean_c + bdev + {nuisance}"
    tests = [
        ("孕周总体关联", no_week, "删除整个孕周项后的似然比检验"),
        ("妇间BMI总体关联", no_bmean, "删除整个妇间BMI项及其交互后的似然比检验"),
        ("个体内BMI关联", no_bdev, "删除BMI个体内偏差后的似然比检验"),
    ]
    rows = []
    for name, formula_red, note in tests:
        red = 拟合模型(d, formula_red, scale, False)
        lr, df_lr, p = LRT(red, final_ml)
        rows.append(
            {
                "检验项": name,
                "约简模型公式": formula_red,
                "似然比统计量": lr,
                "自由度": df_lr,
                "P值": p,
                "显著性结论": "显著" if p < 0.05 else "未达0.05显著水平",
                "说明": note,
            }
        )
    # 若最终包含自然样条，另给出相对线性的整体非线性检验。
    if selected_name != "线性":
        linear_formula = 公式集合()["线性"]
        linear_fit = 拟合模型(d, linear_formula, scale, False)
        # 只有 3-df 孕周样条及其扩展与该线性模型按设计空间嵌套。
        if "孕周样条3" in selected_name:
            if selected_name in ["孕周样条3+妇间BMI样条3", "孕周样条3+孕周妇间BMI交互"]:
                s3_fit = 拟合模型(d, 公式集合()["孕周样条3"], scale, False)
                lr_nl, df_nl, p_nl = LRT(linear_fit, s3_fit)
            else:
                lr_nl, df_nl, p_nl = LRT(linear_fit, final_ml)
            rows.append(
                {
                    "检验项": "孕周非线性偏离",
                    "约简模型公式": linear_formula,
                    "似然比统计量": lr_nl,
                    "自由度": df_nl,
                    "P值": p_nl,
                    "显著性结论": "显著" if p_nl < 0.05 else "未达0.05显著水平",
                    "说明": "整组检验自然样条相对线性孕周项是否必要，不解释单个基函数P值",
                }
            )
    return pd.DataFrame(rows)


def 诊断表(fit: 拟合结果, d: pd.DataFrame) -> pd.DataFrame:
    r = fit.结果
    X = fit.设计矩阵
    marginal_eta = np.asarray(X) @ np.asarray(r.fe_params)
    marginal_y = 反变换(marginal_eta, fit.响应尺度)
    resid = np.asarray(r.resid, float)
    cond_fitted = np.asarray(r.fittedvalues, float)
    rho, rho_p = spearmanr(np.abs(resid), cond_fitted)
    try:
        bp_lm, bp_p, bp_f, bp_fp = het_breuschpagan(resid, np.asarray(X, float))
    except Exception:
        bp_lm = bp_p = bp_f = bp_fp = np.nan
    try:
        sw_stat, sw_p = shapiro(resid)
    except Exception:
        sw_stat = sw_p = np.nan
    cov_re = np.asarray(r.cov_re, float)
    ri_var = float(cov_re[0, 0])
    icc = ri_var / (ri_var + float(r.scale))
    x_no_i = np.asarray(X.iloc[:, 1:], float)
    x_std = (x_no_i - x_no_i.mean(axis=0)) / x_no_i.std(axis=0, ddof=1)
    x_std = x_std[:, np.isfinite(x_std).all(axis=0) & (x_std.std(axis=0) > 0)]
    cond = float(np.linalg.cond(x_std)) if x_std.size else np.nan
    vifs = []
    if x_no_i.shape[1] > 0:
        for j in range(x_no_i.shape[1]):
            try:
                vifs.append(float(variance_inflation_factor(np.asarray(X, float), j + 1)))
            except Exception:
                vifs.append(np.nan)
    rows = [
        ("主模型收敛", int(bool(r.converged)), "1为通过"),
        ("随机截距方差", ri_var, "与残差方差共同计算ICC"),
        ("残差方差", float(r.scale), "响应变换尺度"),
        ("组内相关系数ICC", icc, "说明同一孕妇重复测量的相关程度"),
        ("随机截距近奇异标志", int(ri_var < 1e-8), "1为风险"),
        ("绝对残差与条件拟合值Spearman相关", rho, "接近0较好"),
        ("上述相关P值", rho_p, "仅作异方差诊断"),
        ("Breusch-Pagan检验P值", bp_p, "小于0.05提示异方差"),
        ("残差Shapiro-Wilk检验P值", sw_p, "小样本尾部诊断，不作唯一否决依据"),
        ("残差偏度", float(pd.Series(resid).skew()), "接近0较好"),
        ("残差超额峰度", float(kurtosis(resid, fisher=True, bias=False)), "接近0较好"),
        ("固定效应设计矩阵条件数", cond, "过大提示共线性或基函数数值不稳"),
        ("固定效应最大VIF", float(np.nanmax(vifs)) if vifs else np.nan, "样条基函数VIF不单独作变量删除依据"),
        ("边际预测最小Y浓度", float(np.min(marginal_y)), "原Y尺度"),
        ("边际预测最大Y浓度", float(np.max(marginal_y)), "原Y尺度"),
        ("边际预测越界数", int(np.sum((marginal_y <= 0) | (marginal_y >= 1))), "原Y尺度模型需重点检查"),
        ("最大绝对标准化条件残差", float(np.max(np.abs(resid / math.sqrt(r.scale)))), "大值提示离群事件"),
    ]
    return pd.DataFrame(rows, columns=["诊断指标", "数值", "解释"])


def 公式去临床块(formula: str) -> str:
    return formula.replace(" + age_c + parity_c", "")


def 稳健性分析(
    d0: pd.DataFrame,
    refs_main: dict[str, float],
    selected_formula: str,
    scale: str,
    main_effects: pd.DataFrame,
) -> pd.DataFrame:
    scenarios: list[tuple[str, pd.DataFrame, str, str]] = [
        ("主模型", d0, selected_formula, scale),
        ("截至25周0天", d0[d0["within25"] == 1].copy(), selected_formula, scale),
        ("排除日期孕周偏差超14天事件", d0[d0["date_bad"] == 0].copy(), selected_formula, scale),
        ("不加入临床调整块", d0, 公式去临床块(selected_formula), scale),
        ("加入辅助生殖标志敏感性调整", d0, selected_formula + " + ivf", scale),
        ("替代响应尺度", d0, selected_formula, "logit尺度" if scale == "原Y尺度" else "原Y尺度"),
        (
            "加入测序质量调整块",
            d0,
            selected_formula + " + gc_z + log_reads_z + mapping_z + dup_z + filtered_z",
            scale,
        ),
    ]
    main_map = main_effects.set_index("效应名称")["Y浓度变化"].to_dict()
    rows: list[dict[str, Any]] = []
    for scen, subset0, formula, scale_s in scenarios:
        # 中心化保持主样本参照值；样条结点按该敏感性样本重新估计。
        subset = 应用参照值(subset0, refs_main)
        fit = 拟合模型(subset, formula, scale_s, True)
        eff = 核心效应表(fit, refs_main)
        for _, rr in eff.iterrows():
            base = float(main_map[rr["效应名称"]])
            rows.append(
                {
                    "敏感性场景": scen,
                    "事件数": len(subset),
                    "孕妇数": subset["woman"].nunique(),
                    "响应尺度": scale_s,
                    "收敛": int(bool(fit.结果.converged)),
                    "效应名称": rr["效应名称"],
                    "Y浓度变化": rr["Y浓度变化"],
                    "95%置信区间下限": rr["95%置信区间下限"],
                    "95%置信区间上限": rr["95%置信区间上限"],
                    "相对主模型变化比例": (rr["Y浓度变化"] - base) / abs(base) if base != 0 else np.nan,
                    "效应方向与主模型一致": int(np.sign(rr["Y浓度变化"]) == np.sign(base)),
                    "拟合警告": fit.警告,
                }
            )
    return pd.DataFrame(rows)


def 强影响孕妇检查(
    d: pd.DataFrame,
    final_fit: 拟合结果,
    refs: dict[str, float],
) -> pd.DataFrame:
    b0 = refs["bmean_mean"]
    contrast_rows = {
        "孕周12至20周": (构造参照行(refs, 12, b0, 0), 构造参照行(refs, 20, b0, 0)),
        "妇间BMI加1": (构造参照行(refs, 18, b0, 0), 构造参照行(refs, 18, b0 + 1, 0)),
        "个体内BMI加1": (构造参照行(refs, 18, b0, 0), 构造参照行(refs, 18, b0, 1)),
    }
    base: dict[str, tuple[float, float, np.ndarray, np.ndarray]] = {}
    for name, (r1, r2) in contrast_rows.items():
        x1, x2 = 设计行(final_fit, r1), 设计行(final_fit, r2)
        est, se, _, _, _ = 原尺度对比(final_fit, r1, r2)
        base[name] = (est, se, x1, x2)
    full_X = final_fit.设计矩阵
    y = 响应向量(d, final_fit.响应尺度)
    rows: list[dict[str, Any]] = []
    for woman in sorted(d["woman"].unique()):
        keep = d["woman"].to_numpy() != woman
        try:
            result, warning_text = 拟合矩阵(
                y[keep], full_X.loc[keep], d.loc[keep, "woman"], True, None
            )
            beta = np.asarray(result.fe_params, float)
            max_std = 0.0
            rec: dict[str, Any] = {
                "删除孕妇": woman,
                "删除事件数": int((~keep).sum()),
                "收敛": int(bool(result.converged)),
                "拟合警告": warning_text,
            }
            for name, (base_est, base_se, x1, x2) in base.items():
                e1, e2 = float(x1 @ beta), float(x2 @ beta)
                loo_est = float(反变换(e2, final_fit.响应尺度) - 反变换(e1, final_fit.响应尺度))
                std_change = abs(loo_est - base_est) / base_se if base_se > 0 else np.nan
                rec[f"{name}删除后效应"] = loo_est
                rec[f"{name}相对主模型标准误变化"] = std_change
                rec[f"{name}方向翻转"] = int(np.sign(loo_est) != np.sign(base_est))
                if np.isfinite(std_change):
                    max_std = max(max_std, float(std_change))
            rec["最大标准误变化"] = max_std
            rows.append(rec)
        except Exception as e:
            rows.append(
                {
                    "删除孕妇": woman,
                    "删除事件数": int((~keep).sum()),
                    "收敛": 0,
                    "拟合警告": f"失败：{type(e).__name__}: {e}",
                    "最大标准误变化": np.nan,
                }
            )
    out = pd.DataFrame(rows).sort_values("最大标准误变化", ascending=False, na_position="last")
    out.insert(0, "影响排名", np.arange(1, len(out) + 1))
    return out


def 聚类自助法核心效应(
    d: pd.DataFrame,
    final_fit: 拟合结果,
    refs: dict[str, float],
    repeats: int = 聚类自助重复数,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """按孕妇整簇有放回抽样，给异方差下更稳健的效应区间。

    样条结点与中心化参照值固定为主样本，以免每次重抽样改变估计目标；
    同一孕妇被抽到多次时赋予不同的自助簇编号。
    """
    rng = np.random.default_rng(随机种子 + 17)
    women = np.asarray(sorted(d["woman"].unique()))
    positions = {w: np.flatnonzero(d["woman"].to_numpy() == w) for w in women}
    b0 = refs["bmean_mean"]
    contrast_rows = {
        "孕周由12周增至20周": (构造参照行(refs, 12, b0, 0), 构造参照行(refs, 20, b0, 0)),
        "孕周由12周增至24周": (构造参照行(refs, 12, b0, 0), 构造参照行(refs, 24, b0, 0)),
        "妇间平均BMI增加1单位": (构造参照行(refs, 18, b0, 0), 构造参照行(refs, 18, b0 + 1, 0)),
        "同一孕妇BMI相对本人均值增加1单位": (构造参照行(refs, 18, b0, 0), 构造参照行(refs, 18, b0, 1)),
    }
    vectors = {
        name: (设计行(final_fit, r1), 设计行(final_fit, r2))
        for name, (r1, r2) in contrast_rows.items()
    }
    y_all = 响应向量(d, final_fit.响应尺度)
    detail_rows: list[dict[str, Any]] = []
    failures: list[str] = []
    for b in range(1, repeats + 1):
        sampled = rng.choice(women, size=len(women), replace=True)
        row_pos: list[int] = []
        boot_groups: list[str] = []
        for k, w in enumerate(sampled):
            idx = positions[w]
            row_pos.extend(idx.tolist())
            boot_groups.extend([f"自助簇{k:03d}_{w}"] * len(idx))
        Xb = final_fit.设计矩阵.iloc[row_pos].reset_index(drop=True)
        yb = y_all[np.asarray(row_pos, dtype=int)]
        try:
            result, warning_text = 拟合矩阵(
                yb, Xb, pd.Series(boot_groups), True, None
            )
            beta = np.asarray(result.fe_params, float)
            for name, (x1, x2) in vectors.items():
                e1, e2 = float(x1 @ beta), float(x2 @ beta)
                est = float(
                    反变换(e2, final_fit.响应尺度)
                    - 反变换(e1, final_fit.响应尺度)
                )
                detail_rows.append(
                    {
                        "自助重复号": b,
                        "效应名称": name,
                        "Y浓度变化": est,
                        "收敛": int(bool(result.converged)),
                        "拟合警告": warning_text,
                    }
                )
        except Exception as e:
            failures.append(f"第{b}次:{type(e).__name__}:{e}")
    detail = pd.DataFrame(detail_rows)
    successful = int(detail["自助重复号"].nunique()) if len(detail) else 0
    if successful < math.ceil(repeats * 0.95):
        raise RuntimeError(
            f"聚类自助法成功率不足95%：{successful}/{repeats}；{failures[:3]}"
        )
    main = 核心效应表(final_fit, refs).set_index("效应名称")["Y浓度变化"]
    summary_rows: list[dict[str, Any]] = []
    for name, g in detail.groupby("效应名称", sort=False):
        vals = g["Y浓度变化"].to_numpy(float)
        le0 = int(np.sum(vals <= 0))
        ge0 = int(np.sum(vals >= 0))
        p_sign = min(1.0, 2.0 * (min(le0, ge0) + 1) / (len(vals) + 1))
        summary_rows.append(
            {
                "效应名称": name,
                "主模型点估计": float(main[name]),
                "自助法中位数": float(np.median(vals)),
                "自助法标准差": float(np.std(vals, ddof=1)),
                "百分位95%区间下限": float(np.quantile(vals, 0.025)),
                "百分位95%区间上限": float(np.quantile(vals, 0.975)),
                "符号双侧近似P值": p_sign,
                "计划重复数": repeats,
                "成功重复数": successful,
                "与主模型方向一致": int(np.sign(np.median(vals)) == np.sign(main[name])),
            }
        )
    return pd.DataFrame(summary_rows), detail


def 随机斜率比较(
    d0: pd.DataFrame,
    d: pd.DataFrame,
    formula: str,
    scale: str,
    ri_ml: 拟合结果,
    ri_cv: dict[str, float],
) -> pd.DataFrame:
    rows = []
    try:
        rs_ml = 拟合模型(d, formula, scale, False, random_slope=True)
        rs_cv, _, _ = 交叉验证(d0, formula, scale, random_slope=True)
        eig = np.linalg.eigvalsh(np.asarray(rs_ml.结果.cov_re, float))
        eig_ratio = float(np.min(eig) / np.max(eig)) if np.max(eig) > 0 else 0.0
        aic_improve = float(ri_ml.结果.aic - rs_ml.结果.aic)
        bic_improve = float(ri_ml.结果.bic - rs_ml.结果.bic)
        cv_improve = (ri_cv["组外RMSE"] - rs_cv["组外RMSE"]) / ri_cv["组外RMSE"]
        keep = (
            bool(rs_ml.结果.converged)
            and rs_cv["全部折收敛"] == 1
            and aic_improve >= 2
            and cv_improve >= 复杂项保留最小RMSE改善比例
            and eig_ratio >= 1e-6
        )
        revised_keep = (
            bool(rs_ml.结果.converged)
            and rs_cv["全部折收敛"] == 1
            and aic_improve >= 2
            and bic_improve >= 2
            and rs_cv["组外RMSE"] <= ri_cv["组外RMSE"] * 1.01
            and rs_cv["组外MAE"] <= ri_cv["组外MAE"] * 1.01
            and eig_ratio >= 1e-6
        )
        rows.append(
            {
                "随机结构": "随机截距+孕周随机斜率",
                "收敛": int(bool(rs_ml.结果.converged)),
                "全部交叉验证折收敛": rs_cv["全部折收敛"],
                "赤池信息准则相对随机截距改善_AIC": aic_improve,
                "贝叶斯信息准则相对随机截距改善_BIC": bic_improve,
                "组外RMSE": rs_cv["组外RMSE"],
                "组外MAE": rs_cv["组外MAE"],
                "组外RMSE改善比例": cv_improve,
                "随机协方差最小最大特征值比": eig_ratio,
                "按候选原门槛是否保留": int(keep),
                "按主审修正规则是否建议保留": int(revised_keep),
                "拟合警告": rs_ml.警告,
            }
        )
    except Exception as e:
        rows.append(
            {
                "随机结构": "随机截距+孕周随机斜率",
                "收敛": 0,
                "全部交叉验证折收敛": 0,
                "赤池信息准则相对随机截距改善_AIC": np.nan,
                "贝叶斯信息准则相对随机截距改善_BIC": np.nan,
                "组外RMSE": np.nan,
                "组外MAE": np.nan,
                "组外RMSE改善比例": np.nan,
                "随机协方差最小最大特征值比": np.nan,
                "按候选原门槛是否保留": 0,
                "按主审修正规则是否建议保留": 0,
                "拟合警告": f"失败：{type(e).__name__}: {e}",
            }
        )
    rows.insert(
        0,
        {
            "随机结构": "随机截距",
            "收敛": int(bool(ri_ml.结果.converged)),
            "全部交叉验证折收敛": ri_cv["全部折收敛"],
            "赤池信息准则相对随机截距改善_AIC": 0.0,
            "贝叶斯信息准则相对随机截距改善_BIC": 0.0,
            "组外RMSE": ri_cv["组外RMSE"],
            "组外MAE": ri_cv["组外MAE"],
            "组外RMSE改善比例": 0.0,
            "随机协方差最小最大特征值比": 1.0,
            "按候选原门槛是否保留": 1,
            "按主审修正规则是否建议保留": 1,
            "拟合警告": ri_ml.警告,
        },
    )
    return pd.DataFrame(rows)


def 系数表(fit: 拟合结果) -> pd.DataFrame:
    beta = pd.Series(fit.结果.fe_params)
    se = pd.Series(fit.结果.bse_fe, index=beta.index)
    z = beta / se
    rows = []
    for i, name in enumerate(beta.index):
        if name == "Intercept":
            cn = "截距"
        elif name == "bmean_c":
            cn = "妇间平均BMI线性项"
        elif name == "bdev":
            cn = "BMI个体内偏差"
        elif name == "week_c":
            cn = "孕周线性项"
        elif name == "age_c":
            cn = "年龄调整项"
        elif name == "ivf":
            cn = "辅助生殖标志调整项"
        elif name == "parity_c":
            cn = "生产次数调整项"
        elif "cr(week" in name and ":bmean_c" in name:
            cn = "孕周自然样条与妇间BMI交互基函数"
        elif "cr(week" in name:
            cn = "孕周自然样条基函数"
        elif "cr(bmean" in name:
            cn = "妇间BMI自然样条基函数"
        else:
            cn = name
        rows.append(
            {
                "参数序号": i + 1,
                "参数中文说明": cn,
                "设计矩阵列名": name,
                "估计值": beta[name],
                "标准误": se[name],
                "95%置信区间下限": beta[name] - 1.96 * se[name],
                "95%置信区间上限": beta[name] + 1.96 * se[name],
                "Wald_Z": z[name],
                "单参数P值": 2 * norm.sf(abs(z[name])),
                "解释限制": "样条与交互基函数不逐项作科学解释；整体显著性见整组似然比检验",
            }
        )
    return pd.DataFrame(rows)


def 写报告(
    path: Path,
    source: Path,
    selected_scale: str,
    selected_name: str,
    selected_formula: str,
    model_table: pd.DataFrame,
    decision_table: pd.DataFrame,
    test_table: pd.DataFrame,
    effect_table: pd.DataFrame,
    diagnostic_table: pd.DataFrame,
    sensitivity_table: pd.DataFrame,
    influence_table: pd.DataFrame,
    random_table: pd.DataFrame,
    bootstrap_table: pd.DataFrame,
) -> None:
    mt = model_table[(model_table["模型"] == selected_name) & (model_table["响应尺度"] == selected_scale)].iloc[0]
    week_test = test_table[test_table["检验项"] == "孕周总体关联"].iloc[0]
    bbetween_test = test_table[test_table["检验项"] == "妇间BMI总体关联"].iloc[0]
    bwithin_test = test_table[test_table["检验项"] == "个体内BMI关联"].iloc[0]
    e = effect_table.set_index("效应名称")
    influence_top = influence_table.iloc[0]
    direction_ok = sensitivity_table.groupby("效应名称")["效应方向与主模型一致"].min()
    spline_dec = decision_table[decision_table["决策项"] == "孕周由线性升级为3自由度自然样条"]
    spline_pass = int(spline_dec.iloc[0]["是否通过"]) if len(spline_dec) else 0
    interaction_dec = decision_table[decision_table["决策项"] == "加入孕周与妇间BMI交互"]
    interaction_pass = int(interaction_dec.iloc[0]["是否通过"]) if len(interaction_dec) else 0
    random_keep = int(random_table.iloc[-1]["按候选原门槛是否保留"]) if len(random_table) > 1 else 0
    revised_random_keep = int(random_table.iloc[-1]["按主审修正规则是否建议保留"]) if len(random_table) > 1 else 0
    boot = bootstrap_table.set_index("效应名称")
    candidate_pass = spline_pass == 1 and selected_name != "线性"
    verdict = (
        "候选B通过自身路线门槛，可进入主审横向比较；这不等于已被选为最终论文主线。"
        if candidate_pass
        else "候选B未通过‘整体检验+孕妇分组验证’双门槛，应自我否决为主线，只保留为非线性对照。"
    )
    def fmt_p(v: float) -> str:
        return f"{v:.3g}" if v >= 0.001 else f"{v:.2e}"
    lines = [
        "# 第一问候选 B：受控自然立方样条混合模型报告",
        "",
        "## 1. 建模目的与数据口径",
        "",
        "本候选路线用于判断孕周关系是否需要低自由度非线性，并在同一模型中严格区分妇间 BMI 关联与同一孕妇的个体内 BMI 关联。所有结论均为观察性关联，不作因果解释。",
        "",
        f"模型只读取冻结文件 `{source.name}`：613 个抽血事件、167 名孕妇。分析单位是孕妇代码与抽血次数共同定义的抽血事件；A055 第3次抽血的孕周歧义事件已在冻结阶段排除，683 后数据未进入模型。怀孕次数缺失 167/613，按修订协议不进入主调整块，因此没有损失48名孕妇。",
        "",
        "## 2. 模型定义",
        "",
        "令 $Y_{ij}$ 为孕妇 $i$ 第 $j$ 次抽血事件的 Y 染色体浓度，$t_{ij}$ 为孕周，$\\bar B_i$ 为该孕妇在冻结主样本中的平均 BMI，$B_{ij}-\\bar B_i$ 为 BMI 个体内偏差。候选模型写为：",
        "",
        "\\[",
        "g(Y_{ij})=\\beta_0+f(t_{ij})+h(\\bar B_i)+\\beta_w(B_{ij}-\\bar B_i)+\\boldsymbol\\gamma^\\top\\mathbf Z_i+b_i+\\varepsilon_{ij},",
        "\\]",
        "",
        "其中 $f$ 依次比较线性、3自由度自然立方样条与4自由度自然立方样条；$h$ 比较线性与3自由度自然立方样条；$\\mathbf Z_i$ 仅含年龄和生产次数；$b_i\\sim N(0,\\sigma_b^2)$ 为孕妇随机截距。辅助生殖仅11个事件、3名孕妇，因簇层面极稀疏而只作敏感性调整，不是按p值筛除。结构选择用最大似然法，结构固定后用限制性最大似然法重估。",
        "",
        f"最终受控结构为：`{selected_formula}`；响应采用 **{selected_scale}**。",
        "",
        "## 3. 为什么这样控制复杂度",
        "",
        "- 非线性项不解释单个样条基函数的 p 值，而用删除整组项的似然比检验。",
        "- 测试折中的孕妇从未出现在训练折；新孕妇预测只乘固定效应系数，不调用随机效应。",
        "- 样条、BMI非线性和孕周-BMI交互均需同时满足整体检验与组外误差改善，防止只凭拟合优度堆叠项。",
        "- 身高、体重、Y染色体Z值、序号、怀孕次数和辅助生殖标志均未进入主模型；辅助生殖与质量变量只在稳健性分析中加入。",
        "",
        "## 4. 主要结果",
        "",
        f"按孕妇5折分组验证，所选结构的组外 RMSE={mt['组外RMSE']:.6f}，MAE={mt['组外MAE']:.6f}，R²={mt['组外R方']:.4f}。",
        "",
        f"孕周整组检验：LR={week_test['似然比统计量']:.3f}，df={int(week_test['自由度'])}，p={fmt_p(float(week_test['P值']))}。妇间BMI整组检验 p={fmt_p(float(bbetween_test['P值']))}；个体内BMI检验 p={fmt_p(float(bwithin_test['P值']))}。",
        "",
        f"在平均BMI、年龄与生产次数取样本均值的参照条件下，孕周由12周增至20周的预测Y浓度变化为 {e.loc['孕周由12周增至20周','Y浓度变化百分点']:.3f} 个百分点（95%CI {e.loc['孕周由12周增至20周','百分点下限']:.3f} 至 {e.loc['孕周由12周增至20周','百分点上限']:.3f}）。",
        "",
        f"妇间平均BMI每增加1单位，18周时预测Y浓度变化 {e.loc['妇间平均BMI增加1单位','Y浓度变化百分点']:.3f} 个百分点（95%CI {e.loc['妇间平均BMI增加1单位','百分点下限']:.3f} 至 {e.loc['妇间平均BMI增加1单位','百分点上限']:.3f}）；同一孕妇BMI相对本人均值每增加1单位，预测Y浓度变化 {e.loc['同一孕妇BMI相对本人均值增加1单位','Y浓度变化百分点']:.3f} 个百分点（95%CI {e.loc['同一孕妇BMI相对本人均值增加1单位','百分点下限']:.3f} 至 {e.loc['同一孕妇BMI相对本人均值增加1单位','百分点上限']:.3f}）。两者含义不同，不能合并成一个普通BMI回归系数。",
        "",
        "## 5. 交互、随机结构与稳健性",
        "",
        f"孕周与妇间BMI交互的双门槛结果为：{'通过并保留' if interaction_pass else '未同时通过，故不保留'}。孕周随机斜率的保留结果为：{'通过' if random_keep else '未通过，继续使用随机截距'}。",
        f"重要复核边界：上句沿用候选B预先写定的‘新孕妇组外RMSE至少改善1%’门槛；主审随后指出，新孕妇固定效应CV不能充分评价随机结构。按主审修正规则（收敛、非奇异、AIC/BIC显著改善且组外误差不明显恶化），随机斜率为{'建议保留并重估最终模型' if revised_random_keep else '仍不建议保留'}。因此本候选当前随机截距参数表只能作为保守版本，主审不得忽略这项后验规则修正。",
        "",
        f"预设敏感性场景中，各核心效应方向一致性的最小值为 {int(direction_ok.min())}（1代表全部同向）。删除单名孕妇后，最大核心效应变化为主模型标准误的 {float(influence_top['最大标准误变化']):.3f} 倍，最有影响的孕妇为 {influence_top['删除孕妇']}。具体数值见结果表，若发生方向翻转或超过1个标准误，应由主审触发否决。",
        "",
        "## 6. 诊断与边界",
        "",
        f"模型收敛={int(diagnostic_table.loc[diagnostic_table['诊断指标']=='主模型收敛','数值'].iloc[0])}，ICC={float(diagnostic_table.loc[diagnostic_table['诊断指标']=='组内相关系数ICC','数值'].iloc[0]):.4f}，Breusch-Pagan p={fmt_p(float(diagnostic_table.loc[diagnostic_table['诊断指标']=='Breusch-Pagan检验P值','数值'].iloc[0]))}。原尺度残差存在异方差迹象，因此除模型型标准误外，另按孕妇整簇进行{int(bootstrap_table['计划重复数'].iloc[0])}次有放回自助。孕周12至20周的聚类自助95%区间为 {boot.loc['孕周由12周增至20周','百分位95%区间下限']*100:.3f} 至 {boot.loc['孕周由12周增至20周','百分位95%区间上限']*100:.3f} 个百分点；妇间BMI每增加1单位的区间为 {boot.loc['妇间平均BMI增加1单位','百分位95%区间下限']*100:.3f} 至 {boot.loc['妇间平均BMI增加1单位','百分位95%区间上限']*100:.3f} 个百分点；个体内BMI每增加1单位的区间为 {boot.loc['同一孕妇BMI相对本人均值增加1单位','百分位95%区间下限']*100:.3f} 至 {boot.loc['同一孕妇BMI相对本人均值增加1单位','百分位95%区间上限']*100:.3f} 个百分点。",
        "",
        "statsmodels 对原Y小数尺度给出通用的‘方差可能在边界’提示，但随机截距方差相对残差方差对应的ICC并不接近0，且单随机截距协方差为正；因此不把该绝对尺度提示等同于奇异拟合。诊断仍不将残差正态检验机械地作为唯一取舍标准。",
        "",
        "效应只在683前、孕周11至25周6天及本样本BMI支持域内解释。结果不能推出BMI或孕周对Y浓度的因果作用，也不能把后段近确定性BMI-Y关系当作生物学证据。",
        "",
        "## 7. 候选B自我验收结论",
        "",
        verdict,
        "",
        "所有表头为中文；本目录没有生成任何图像，制图需求已改写为 MATLAB 输出 SVG 的提示词 TXT。",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def 写风险文件(path: Path) -> None:
    text = """# 候选 B 风险与自我否决条件

本路线不是因为“用了样条”就自动优于线性模型。出现任一情形，应自我否决为论文主线：

1. 613个抽血事件或167名孕妇口径不符，或混入683后数据；
2. 孕周3自由度自然样条相对线性的整组LRT未达0.05，或按孕妇分组的组外RMSE未改善至少1%，或MAE变差；
3. 交互项仅拟合内显著而分组验证不改善，仍被保留；
4. 测试孕妇的随机截距或随机斜率被用于交叉验证预测；
5. 怀孕次数因167/613缺失进入主模型，造成48名孕妇完整病例损失；或辅助生殖因仅11事件/3孕妇仍被强行作为主调整项；
6. 随机效应协方差奇异、主模型不收敛，或关键结论由单名孕妇删除后发生方向翻转；
6a. 原尺度异方差存在时仍只引用模型型标准误，而不报告按孕妇整簇自助区间；
7. 截至25周0天、排除日期孕周偏差事件或替代响应尺度后，核心方向系统性改变；
8. 把妇间BMI系数误写为同一孕妇增重效应，或把观察性关联写成因果结论；
9. 解释单个样条基函数p值，而没有整组显著性检验；
10. 为本候选新增SVG、PNG或其他图像，而不是只提交制图提示词TXT。

若非线性未过双门槛，应主动退回候选A线性混合模型，不以“更创新”为由保留复杂结构。
"""
    path.write_text(text, encoding="utf-8")


def 写制图提示词(path: Path) -> None:
    text = """图名：第一问孕周-Y染色体浓度受控关系曲线（暂不绘制，仅保留提示词）

数据文件：../02_结果表/05_曲线预测数据.csv

请使用 MATLAB 生成纯矢量 SVG，输出文件名建议为“第一问_孕周Y浓度受控关系曲线.svg”。本阶段不得运行此提示词或生成图像。

绘图要求：
1. 横轴为“孕周（周）”，范围11至25.7；纵轴为“预测Y染色体浓度（%）”。CSV中的百分比列已乘100，不要再次乘100。
2. 主线使用“平均BMI”情景的“预测Y浓度百分比”；其“下限百分比”和“上限百分比”形成95%置信带。置信带用浅蓝色、透明度约0.20，主线用深蓝色、线宽1.8。
3. “平均BMI减2”和“平均BMI加2”仅以灰色虚线和橙色点划线显示预测线，不为两条辅助线重复画置信带；图例写清三个BMI情景对应的具体BMI数值，可从“孕妇平均BMI”列读取。
4. 加一条 y=4 的水平参考虚线，标注“4%可靠性阈值”；该阈值只作题面背景参考，不把本图当作第二问最优时点模型。
5. 不画原始散点，避免同一孕妇重复事件重叠造成视觉伪独立；若后续确需散点，另从冻结样本读取并按孕妇设低透明度，不得把技术复测当独立点。
6. 纵轴从0起，上界按置信带最大值向上留10%余量；不要截断置信带。若下限小于0，显示到0并在图注说明置信区间为固定效应Delta法近似。
7. 中文字体优先“Microsoft YaHei”，找不到时使用“SimHei”；英文字体Arial。画布宽16 cm、高10 cm，白底，坐标轴线宽0.8，网格只保留浅灰色横向主网格。
8. 图题：“孕周与Y染色体浓度的受控关系”；副标题或图注：“孕妇随机截距模型；新孕妇曲线仅含固定效应；年龄和生产次数取样本均值，BMI个体内偏差=0”。
9. 导出必须使用 MATLAB 的 exportgraphics 或 print 的矢量SVG选项；检查SVG文本、曲线和置信带均为矢量对象，不嵌入位图。
10. 图中不使用“因果影响”“导致”等措辞，只写“关联”或“预测关系”。
"""
    path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    here = Path(__file__).resolve()
    candidate_dir = here.parents[1]
    default_input = candidate_dir.parent / "00_共同口径" / "冻结数据" / "第一问主模型冻结样本.csv"
    parser.add_argument("--input", type=Path, default=default_input)
    parser.add_argument("--output", type=Path, default=candidate_dir)
    args = parser.parse_args()
    source = args.input.resolve()
    out = args.output.resolve()
    result_dir = out / "02_结果表"
    report_dir = out / "03_模型报告"
    prompt_dir = out / "04_制图提示词"
    reproduce_dir = out / "05_复现"
    for p in [result_dir, report_dir, prompt_dir, reproduce_dir]:
        p.mkdir(parents=True, exist_ok=True)

    start = time.time()
    d0 = 读取并校验(source)
    refs = 计算参照值(d0)
    d = 应用参照值(d0, refs)
    formulas = 公式集合()
    scales = ["原Y尺度", "logit尺度"]

    fits: dict[tuple[str, str], 拟合结果] = {}
    cvs: dict[tuple[str, str], dict[str, float]] = {}
    model_rows: list[dict[str, Any]] = []
    fold_rows: list[pd.DataFrame] = []
    prediction_rows: list[pd.DataFrame] = []
    for scale in scales:
        for name, formula in formulas.items():
            print(f"[候选比较] {scale} / {name}", flush=True)
            fit = 拟合模型(d, formula, scale, False)
            cv, folds, preds = 交叉验证(d0, formula, scale)
            fits[(scale, name)] = fit
            cvs[(scale, name)] = cv
            model_rows.append(模型摘要行(name, scale, fit, cv))
            folds.insert(0, "模型", name)
            folds.insert(1, "响应尺度", scale)
            fold_rows.append(folds)
            preds.insert(0, "模型", name)
            preds.insert(1, "响应尺度", scale)
            prediction_rows.append(preds)
    model_table = pd.DataFrame(model_rows)
    fold_table = pd.concat(fold_rows, ignore_index=True)
    cv_prediction_table = pd.concat(prediction_rows, ignore_index=True)

    selection_by_scale: dict[str, str] = {}
    decision_rows: list[dict[str, Any]] = []
    for scale in scales:
        selected, decisions = 选择结构(scale, fits, cvs)
        selection_by_scale[scale] = selected
        decision_rows.extend(decisions)
    # 在原Y尺度的组外指标上比较响应变换。若误差相差不到1%，优先logit以保证范围。
    raw_name, log_name = selection_by_scale["原Y尺度"], selection_by_scale["logit尺度"]
    raw_cv, log_cv = cvs[("原Y尺度", raw_name)], cvs[("logit尺度", log_name)]
    if (
        log_cv["组外RMSE"] <= raw_cv["组外RMSE"] * (1 + 响应尺度近似持平比例)
        and log_cv["组外MAE"] <= raw_cv["组外MAE"] * (1 + 响应尺度近似持平比例)
    ):
        selected_scale, selected_name = "logit尺度", log_name
        scale_reason = "logit在原Y尺度的组外RMSE和MAE均不比原尺度模型高1%以上，故优先采用有界响应"
    else:
        selected_scale, selected_name = "原Y尺度", raw_name
        scale_reason = "logit组外RMSE或MAE比原尺度模型高1%以上，故采用原尺度"
    decision_rows.append(
        {
            "响应尺度": "跨尺度",
            "决策项": "最终响应尺度",
            "整体检验P值": np.nan,
            "组外RMSE改善比例": (raw_cv["组外RMSE"] - log_cv["组外RMSE"]) / raw_cv["组外RMSE"],
            "组外MAE是否不劣": int(log_cv["组外MAE"] <= raw_cv["组外MAE"] * 1.01),
            "是否通过": int(selected_scale == "logit尺度"),
            "规则": scale_reason,
        }
    )
    selected_formula = formulas[selected_name]
    decision_table = pd.DataFrame(decision_rows)

    # 结构固定后重新用ML做整组检验，并以REML作为最终参数估计。
    final_ml = fits[(selected_scale, selected_name)]
    final_reml = 拟合模型(d, selected_formula, selected_scale, True)
    test_table = 整体检验(d, refs, selected_name, selected_formula, selected_scale, final_ml)
    effect_table = 核心效应表(final_reml, refs)
    curve_table = 曲线预测表(final_reml, refs)
    coefficient_table = 系数表(final_reml)
    diagnostic_table = 诊断表(final_reml, d)
    sensitivity_table = 稳健性分析(d0, refs, selected_formula, selected_scale, effect_table)
    influence_table = 强影响孕妇检查(d, final_reml, refs)
    random_table = 随机斜率比较(
        d0,
        d,
        selected_formula,
        selected_scale,
        final_ml,
        cvs[(selected_scale, selected_name)],
    )
    print(f"[稳健推断] 按孕妇整簇自助 {聚类自助重复数} 次", flush=True)
    bootstrap_table, bootstrap_detail = 聚类自助法核心效应(
        d, final_reml, refs, 聚类自助重复数
    )
    scale_diag_parts = []
    for scale_check in scales:
        scale_fit = (
            final_reml
            if scale_check == selected_scale
            else 拟合模型(d, selected_formula, scale_check, True)
        )
        part = 诊断表(scale_fit, d)
        part.insert(0, "响应尺度", scale_check)
        scale_diag_parts.append(part)
    scale_diag_table = pd.concat(scale_diag_parts, ignore_index=True)

    selected_summary = pd.DataFrame(
        [
            {
                "冻结事件数": len(d0),
                "冻结孕妇数": d0["woman"].nunique(),
                "怀孕次数缺失事件数": int(d0["gravidity"].isna().sum()),
                "辅助生殖事件数": int(d0["ivf"].sum()),
                "辅助生殖孕妇数": int(d0.loc[d0["ivf"] == 1, "woman"].nunique()),
                "最终响应尺度": selected_scale,
                "最终固定效应结构": selected_name,
                "最终公式": selected_formula,
                "最终估计方法": "REML",
                "随机结构": "孕妇随机截距",
                "交叉验证": "按孕妇5折；测试孕妇仅固定效应预测",
                "输入SHA256": sha256(source),
                "随机种子": 随机种子,
                "聚类自助计划重复数": 聚类自助重复数,
                "聚类自助成功重复数": int(bootstrap_table["成功重复数"].min()),
            }
        ]
    )
    refs_table = pd.DataFrame(
        [{"参照量": k, "数值": v} for k, v in refs.items()]
    )

    写CSV(selected_summary, result_dir / "00_最终模型摘要.csv")
    写CSV(
        model_table.rename(
            columns={
                "ML对数似然": "最大似然对数似然值",
                "AIC": "赤池信息准则_AIC",
                "BIC": "贝叶斯信息准则_BIC",
            }
        ),
        result_dir / "01_候选模型拟合与交叉验证.csv",
    )
    写CSV(decision_table, result_dir / "02_模型选择决策.csv")
    写CSV(test_table, result_dir / "03_整体显著性检验.csv")
    写CSV(
        coefficient_table.rename(columns={"Wald_Z": "沃尔德Z统计量"}),
        result_dir / "04_最终模型固定效应系数.csv",
    )
    写CSV(effect_table, result_dir / "05_核心效应原尺度.csv")
    写CSV(curve_table, result_dir / "06_曲线预测数据.csv")
    写CSV(diagnostic_table, result_dir / "07_诊断指标.csv")
    写CSV(sensitivity_table, result_dir / "08_稳健性分析.csv")
    写CSV(influence_table, result_dir / "09_强影响孕妇逐一删除检查.csv")
    写CSV(random_table, result_dir / "10_随机斜率比较.csv")
    写CSV(
        fold_table.rename(columns={"RMSE": "均方根误差", "MAE": "平均绝对误差"}),
        result_dir / "11_交叉验证折明细.csv",
    )
    写CSV(cv_prediction_table, result_dir / "12_交叉验证逐事件预测.csv")
    写CSV(refs_table, result_dir / "13_中心化与标准化参照值.csv")
    写CSV(scale_diag_table, result_dir / "14_响应尺度诊断对照.csv")
    写CSV(bootstrap_table, result_dir / "15_孕妇整簇自助核心效应.csv")
    写CSV(bootstrap_detail, result_dir / "16_孕妇整簇自助重复明细.csv")

    写报告(
        report_dir / "候选B_受控样条混合模型报告.md",
        source,
        selected_scale,
        selected_name,
        selected_formula,
        model_table,
        decision_table,
        test_table,
        effect_table,
        diagnostic_table,
        sensitivity_table,
        influence_table,
        random_table,
        bootstrap_table,
    )
    写风险文件(out / "00_风险与自我否决条件.md")
    写制图提示词(prompt_dir / "第一问孕周Y浓度受控关系曲线_MATLAB_SVG提示词.txt")

    runtime = {
        "Python": sys.version,
        "操作系统": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "statsmodels": statsmodels.__version__,
        "patsy": patsy.__version__,
        "scikit_learn": sklearn.__version__,
        "脚本SHA256": sha256(here),
        "输入SHA256": sha256(source),
        "运行秒数": time.time() - start,
    }
    (reproduce_dir / "运行环境与哈希.json").write_text(
        json.dumps(runtime, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    output_files = sorted(
        p
        for p in out.rglob("*")
        if p.is_file()
        and p.name != "文件哈希清单.csv"
        and "__pycache__" not in p.parts
        and p.suffix.lower() != ".pyc"
    )
    manifest = pd.DataFrame(
        [
            {
                "相对路径": str(p.relative_to(out)),
                "字节数": p.stat().st_size,
                "哈希值_SHA256": sha256(p),
            }
            for p in output_files
        ]
    )
    写CSV(manifest, reproduce_dir / "文件哈希清单.csv")

    print(
        json.dumps(
            {
                "状态": "完成",
                "事件数": len(d0),
                "孕妇数": d0["woman"].nunique(),
                "响应尺度": selected_scale,
                "结构": selected_name,
                "输出目录": str(out),
                "运行秒数": round(time.time() - start, 3),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
