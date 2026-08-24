from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
from scipy import stats
from scipy.special import expit
import sklearn
from sklearn.model_selection import GroupKFold
import statsmodels
import statsmodels.api as sm
from statsmodels.othermod.betareg import BetaModel
from statsmodels.stats.outliers_influence import variance_inflation_factor


随机种子 = 20250824
孕周中心 = 18.0
自助次数 = 500


@dataclass
class 拟合结果:
    名称: str
    类型: str
    结果: object
    特征: list[str]
    精度模式: str | None = None
    收敛: bool = True
    警告: str = ""


def 文件哈希(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest().upper()


def 保存表(df: pd.DataFrame, path: Path) -> None:
    # 结果表统一采用中文语义表头；括号内仅保留论文通用统计缩写。
    output = df.rename(columns={
        "RMSE": "均方根误差（RMSE）",
        "MAE": "平均绝对误差（MAE）",
    })
    output.to_csv(path, index=False, encoding="utf-8-sig", lineterminator="\n")


def 载入并构造变量(path: Path) -> tuple[pd.DataFrame, dict[str, float]]:
    d = pd.read_csv(path, encoding="utf-8-sig")
    required = [
        "孕妇代码", "抽血事件键", "孕周数", "年龄", "孕妇平均BMI", "BMI个体内偏差",
        "生产次数", "辅助生殖标志", "Y染色体浓度均值", "纳入主模型标志",
        "纳入截至25周0天敏感性标志", "孕周歧义标志",
        "任一记录日期孕周偏差超14天标志", "GC含量均值", "原始读段数均值",
        "唯一比对读段数均值", "比对比例均值", "重复读段比例均值", "过滤读段比例均值",
    ]
    missing = [c for c in required if c not in d.columns]
    if missing:
        raise ValueError(f"冻结数据缺列: {missing}")
    if len(d) != 613 or d["孕妇代码"].nunique() != 167:
        raise AssertionError(f"主样本口径不符: {len(d)}事件/{d['孕妇代码'].nunique()}孕妇")
    if not (d["纳入主模型标志"].eq(1).all() and d["孕周歧义标志"].eq(0).all()):
        raise AssertionError("主模型文件仍含未纳入事件或孕周歧义事件")
    if d["抽血事件键"].duplicated().any():
        raise AssertionError("抽血事件键不唯一")
    if not d["孕周数"].between(10, 26, inclusive="left").all():
        raise AssertionError("孕周越过冻结窗口[10,26)")
    y = d["Y染色体浓度均值"].astype(float)
    if not ((y > 0) & (y < 1)).all():
        raise AssertionError("Beta模型要求响应严格位于(0,1)，冻结样本不满足")
    if d[["孕周数", "年龄", "孕妇平均BMI", "BMI个体内偏差", "生产次数", "Y染色体浓度均值"]].isna().any().any():
        raise AssertionError("主模型核心变量出现缺失")

    person = d.groupby("孕妇代码", sort=True).first(numeric_only=False)
    bmi_center = float(person["孕妇平均BMI"].mean())
    age_center = float(person["年龄"].mean())
    parity_center = float(person["生产次数"].mean())

    d = d.copy()
    d["截距"] = 1.0
    d["孕周中心化"] = d["孕周数"] - 孕周中心
    d["孕周二次项"] = d["孕周中心化"] ** 2
    d["妇间BMI中心化"] = d["孕妇平均BMI"] - bmi_center
    d["年龄中心化"] = d["年龄"] - age_center
    d["生产次数中心化"] = d["生产次数"] - parity_center
    d["辅助生殖标志"] = d["辅助生殖标志"].astype(float)
    d["读段数对数"] = np.log(d["原始读段数均值"].astype(float))

    quality_raw = {
        "GC含量标准化": "GC含量均值",
        "读段数对数标准化": "读段数对数",
        "比对比例标准化": "比对比例均值",
        "重复读段比例标准化": "重复读段比例均值",
        "过滤读段比例标准化": "过滤读段比例均值",
    }
    centers: dict[str, float] = {
        "妇间BMI中心": bmi_center,
        "年龄中心": age_center,
        "生产次数中心": parity_center,
        "孕周中心": 孕周中心,
    }
    for new, old in quality_raw.items():
        m = float(d[old].mean())
        s = float(d[old].std(ddof=0))
        if not np.isfinite(s) or s <= 0:
            raise AssertionError(f"质量变量{old}标准差异常")
        d[new] = (d[old] - m) / s
        centers[f"{old}均值"] = m
        centers[f"{old}标准差"] = s
    return d, centers


主特征线性 = ["截距", "孕周中心化", "妇间BMI中心化", "BMI个体内偏差", "年龄中心化", "生产次数中心化"]
主特征二次 = ["截距", "孕周中心化", "孕周二次项", "妇间BMI中心化", "BMI个体内偏差", "年龄中心化", "生产次数中心化"]
质量特征 = ["GC含量标准化", "读段数对数标准化", "比对比例标准化", "重复读段比例标准化", "过滤读段比例标准化"]


def 精度矩阵(d: pd.DataFrame, mode: str) -> np.ndarray:
    if mode == "常精度":
        return np.ones((len(d), 1), dtype=float)
    if mode == "孕周变精度":
        return np.column_stack([np.ones(len(d)), d["孕周中心化"].to_numpy(float)])
    raise ValueError(mode)


def 精度名称(mode: str) -> list[str]:
    return ["精度截距"] if mode == "常精度" else ["精度截距", "精度方程孕周中心化"]


def 拟合Beta(d: pd.DataFrame, features: list[str], mode: str, cluster: bool = True, group_col: str = "孕妇代码") -> 拟合结果:
    X = d[features].to_numpy(float)
    Z = 精度矩阵(d, mode)
    model = BetaModel(d["Y染色体浓度均值"].to_numpy(float), X, exog_precision=Z)
    caught: list[str] = []
    with warnings.catch_warnings(record=True) as ws:
        warnings.simplefilter("always")
        kwargs = {"maxiter": 2000, "disp": False, "method": "bfgs"}
        if cluster:
            kwargs.update(cov_type="cluster", cov_kwds={"groups": d[group_col].to_numpy(), "use_correction": True})
        res = model.fit(**kwargs)
        caught = [str(w.message) for w in ws]
    converged = bool(bool(getattr(res, "mle_retvals", {}).get("converged", True)) and bool(np.isfinite(res.params).all()))
    return 拟合结果("", "Beta", res, features, mode, converged, " | ".join(sorted(set(caught))))


def 拟合GEE(d: pd.DataFrame, features: list[str], family: str) -> 拟合结果:
    X = d[features].to_numpy(float)
    y = d["Y染色体浓度均值"].to_numpy(float)
    if family == "分数logit":
        fam = sm.families.Binomial(link=sm.families.links.Logit())
    elif family == "高斯恒等":
        fam = sm.families.Gaussian(link=sm.families.links.Identity())
    else:
        raise ValueError(family)
    model = sm.GEE(y, X, groups=d["孕妇代码"].to_numpy(), family=fam, cov_struct=sm.cov_struct.Exchangeable())
    with warnings.catch_warnings(record=True) as ws:
        warnings.simplefilter("always")
        res = model.fit(maxiter=200, ctol=1e-8, cov_type="robust")
    caught = [str(w.message) for w in ws]
    converged = bool(bool(getattr(res, "converged", True)) and bool(np.isfinite(res.params).all()))
    return 拟合结果("", f"GEE-{family}", res, features, None, converged, " | ".join(sorted(set(caught))))


def 预测(fit: 拟合结果, d: pd.DataFrame) -> tuple[np.ndarray, np.ndarray | None]:
    X = d[fit.特征].to_numpy(float)
    if fit.类型 == "Beta":
        Z = 精度矩阵(d, fit.精度模式 or "常精度")
        mu = np.asarray(fit.结果.model.predict(fit.结果.params, exog=X, exog_precision=Z, which="mean"), float)
        phi = np.asarray(fit.结果.model.predict(fit.结果.params, exog=X, exog_precision=Z, which="precision"), float)
        return mu, phi
    return np.asarray(fit.结果.predict(X), float), None


def 模型复杂度(fit: 拟合结果) -> int:
    if fit.类型 == "Beta":
        return len(fit.结果.params)
    return len(fit.结果.params) + 1  # 加1表示交换型工作相关参数


def 公共指标(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    err = y - pred
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae = float(np.mean(np.abs(err)))
    denom = float(np.sum((y - y.mean()) ** 2))
    r2 = float(1 - np.sum(err ** 2) / denom) if denom > 0 else math.nan
    p = np.clip(pred, 1e-10, 1 - 1e-10)
    frac_logloss = float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
    return {"RMSE": rmse, "MAE": mae, "组外R²": r2, "分数对数损失": frac_logloss}


def 建立分组折(d: pd.DataFrame) -> pd.Series:
    cv = GroupKFold(n_splits=5, shuffle=True, random_state=随机种子)
    folds = np.zeros(len(d), dtype=int)
    groups = d["孕妇代码"].to_numpy()
    for k, (_, test) in enumerate(cv.split(np.zeros(len(d)), groups=groups), start=1):
        folds[test] = k
    if set(folds) != {1, 2, 3, 4, 5}:
        raise AssertionError("交叉验证折生成失败")
    check = pd.DataFrame({"孕妇代码": groups, "折": folds}).groupby("孕妇代码")["折"].nunique()
    if not check.eq(1).all():
        raise AssertionError("同一孕妇被拆入多个折")
    return pd.Series(folds, index=d.index, name="交叉验证折")


def 交叉验证(d: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    specs = [
        ("Beta回归-常精度-二次孕周", "Beta", 主特征二次, "常精度"),
        ("Beta回归-孕周变精度-二次孕周", "Beta", 主特征二次, "孕周变精度"),
        ("分数logit GEE-线性孕周", "分数logit", 主特征线性, None),
        ("分数logit GEE-二次孕周", "分数logit", 主特征二次, None),
        ("高斯GEE-二次孕周", "高斯恒等", 主特征二次, None),
    ]
    d = d.copy()
    folds = 建立分组折(d)
    d["交叉验证折"] = folds
    pred_rows: list[dict] = []
    fold_rows: list[dict] = []
    for name, kind, features, mode in specs:
        all_pred = np.full(len(d), np.nan)
        all_ok = True
        for k in range(1, 6):
            train = d.loc[folds.ne(k)].copy()
            test = d.loc[folds.eq(k)].copy()
            try:
                if kind == "Beta":
                    fit = 拟合Beta(train, features, mode or "常精度", cluster=False)
                else:
                    fit = 拟合GEE(train, features, kind)
                fit.名称 = name
                pred, _ = 预测(fit, test)
                if not fit.收敛 or not np.isfinite(pred).all():
                    raise RuntimeError("未收敛或预测非有限")
                all_pred[np.where(folds.eq(k))[0]] = pred
                met = 公共指标(test["Y染色体浓度均值"].to_numpy(float), pred)
                fold_rows.append({
                    "模型": name, "交叉验证折": k, "训练孕妇数": train["孕妇代码"].nunique(),
                    "测试孕妇数": test["孕妇代码"].nunique(), "测试事件数": len(test), "收敛": "是",
                    **met,
                })
            except Exception as e:
                all_ok = False
                fold_rows.append({"模型": name, "交叉验证折": k, "收敛": "否", "错误": str(e)})
        if all_ok and np.isfinite(all_pred).all():
            met = 公共指标(d["Y染色体浓度均值"].to_numpy(float), all_pred)
            pred_outside = int(((all_pred <= 0) | (all_pred >= 1)).sum())
            pred_rows.append({
                "模型": name, "验证方式": "5折按孕妇分组", "事件数": len(d), "孕妇数": d["孕妇代码"].nunique(),
                "模型参数复杂度": len(features) + (1 if mode == "常精度" else 2 if mode else 1),
                "全部折收敛": "是", "越界预测数": pred_outside, **met,
            })
            for i, p in enumerate(all_pred):
                pred_rows[-1]  # 保持上方行引用明确
        else:
            pred_rows.append({"模型": name, "验证方式": "5折按孕妇分组", "全部折收敛": "否"})
        d[f"OOF::{name}"] = all_pred
    return pd.DataFrame(pred_rows), pd.DataFrame(fold_rows), d


def 系数表(fits: list[拟合结果]) -> pd.DataFrame:
    rows: list[dict] = []
    for fit in fits:
        res = fit.结果
        params = np.asarray(res.params, float)
        bse = np.asarray(res.bse, float)
        pval = np.asarray(res.pvalues, float)
        ci = np.asarray(res.conf_int(), float)
        names = list(fit.特征)
        if fit.类型 == "Beta":
            names += 精度名称(fit.精度模式 or "常精度")
        for i, nm in enumerate(names):
            rows.append({
                "模型": fit.名称, "方程部分": "均值方程" if i < len(fit.特征) else "精度方程",
                "参数": nm, "估计值": params[i], "聚类稳健标准误": bse[i],
                "95%置信区间下限": ci[i, 0], "95%置信区间上限": ci[i, 1], "P值": pval[i],
                "重复测量处理": "孕妇聚类稳健协方差" if fit.类型 == "Beta" else "交换型GEE+孕妇聚类稳健协方差",
                "收敛": "是" if fit.收敛 else "否",
            })
    return pd.DataFrame(rows)


def 联合Wald(fit: 拟合结果, terms: list[str]) -> tuple[float, int, float]:
    idx = [fit.特征.index(t) for t in terms]
    b = np.asarray(fit.结果.params, float)[idx]
    cov = np.asarray(fit.结果.cov_params(), float)[np.ix_(idx, idx)]
    stat = float(b.T @ np.linalg.pinv(cov) @ b)
    return stat, len(idx), float(stats.chi2.sf(stat, len(idx)))


def 整体检验表(fits: list[拟合结果]) -> pd.DataFrame:
    groups = [
        ("孕周总体关联", ["孕周中心化", "孕周二次项"]),
        ("孕周非线性", ["孕周二次项"]),
        ("妇间BMI关联", ["妇间BMI中心化"]),
        ("个体内BMI关联", ["BMI个体内偏差"]),
        ("临床调整块", ["年龄中心化", "生产次数中心化"]),
    ]
    rows = []
    for fit in fits:
        for label, terms in groups:
            if not all(t in fit.特征 for t in terms):
                continue
            stat, dfree, p = 联合Wald(fit, terms)
            rows.append({"模型": fit.名称, "检验项": label, "Wald卡方": stat, "自由度": dfree, "P值": p, "显著性水平": 0.05})
    return pd.DataFrame(rows)


def 平均效应(params: np.ndarray, d: pd.DataFrame, features: list[str], target: str, link: str, week: float | None = None) -> float:
    X = d[features].to_numpy(float).copy()
    b = np.asarray(params[: len(features)], float)
    if target.startswith("孕周"):
        if week is not None:
            tc = week - 孕周中心
            X[:, features.index("孕周中心化")] = tc
            if "孕周二次项" in features:
                X[:, features.index("孕周二次项")] = tc ** 2
        else:
            tc = X[:, features.index("孕周中心化")]
        slope_lp = np.full(len(d), b[features.index("孕周中心化")])
        if "孕周二次项" in features:
            slope_lp = slope_lp + 2 * tc * b[features.index("孕周二次项")]
        if link == "恒等":
            return float(np.mean(slope_lp))
        mu = expit(X @ b)
        return float(np.mean(mu * (1 - mu) * slope_lp))
    if target == "妇间BMI":
        j = features.index("妇间BMI中心化")
    elif target == "个体内BMI":
        j = features.index("BMI个体内偏差")
    elif target == "年龄":
        j = features.index("年龄中心化")
    elif target == "生产次数":
        j = features.index("生产次数中心化")
    elif target == "辅助生殖":
        j = features.index("辅助生殖标志")
        X0, X1 = X.copy(), X.copy()
        X0[:, j], X1[:, j] = 0.0, 1.0
        if link == "恒等":
            return float(np.mean(X1 @ b - X0 @ b))
        return float(np.mean(expit(X1 @ b) - expit(X0 @ b)))
    else:
        raise ValueError(target)
    if link == "恒等":
        return float(b[j])
    mu = expit(X @ b)
    return float(np.mean(mu * (1 - mu) * b[j]))


def 边际效应(fit: 拟合结果, d: pd.DataFrame, target: str, week: float | None = None) -> dict[str, float]:
    p = len(fit.特征)
    params = np.asarray(fit.结果.params, float)[:p]
    cov = np.asarray(fit.结果.cov_params(), float)[:p, :p]
    link = "恒等" if fit.类型 == "GEE-高斯恒等" else "logit"
    f = lambda b: 平均效应(b, d, fit.特征, target, link, week)
    est = f(params)
    grad = np.zeros(p)
    for j in range(p):
        h = 1e-5 * max(1.0, abs(params[j]))
        up, dn = params.copy(), params.copy()
        up[j] += h
        dn[j] -= h
        grad[j] = (f(up) - f(dn)) / (2 * h)
    var = float(grad.T @ cov @ grad)
    se = math.sqrt(max(var, 0.0))
    z = est / se if se > 0 else math.nan
    pval = float(2 * stats.norm.sf(abs(z))) if np.isfinite(z) else math.nan
    return {"估计": 100 * est, "标准误": 100 * se, "下限": 100 * (est - 1.96 * se), "上限": 100 * (est + 1.96 * se), "P值": pval}


def 边际效应表(fits: list[拟合结果], d: pd.DataFrame) -> pd.DataFrame:
    rows = []
    targets = [("孕周平均", None, "每增加1周"), ("妇间BMI", None, "孕妇平均BMI每增加1 kg/m²"), ("个体内BMI", None, "同一孕妇BMI相对本人均值每增加1 kg/m²")]
    targets += [(f"孕周局部@{w}周", float(w), "在指定孕周附近每增加1周") for w in (12, 16, 20, 24)]
    for fit in fits:
        for target_label, week, unit in targets:
            target = "孕周平均" if target_label == "孕周平均" else "孕周局部" if target_label.startswith("孕周局部") else target_label
            out = 边际效应(fit, d, target, week)
            rows.append({
                "模型": fit.名称, "效应": target_label, "变化单位": unit,
                "Y浓度变化估计（百分点）": out["估计"], "聚类稳健标准误（百分点）": out["标准误"],
                "95%置信区间下限（百分点）": out["下限"], "95%置信区间上限（百分点）": out["上限"], "P值": out["P值"],
            })
    return pd.DataFrame(rows)


def 边界诊断(d: pd.DataFrame) -> pd.DataFrame:
    y = d["Y染色体浓度均值"].to_numpy(float)
    vals = [
        ("事件数", len(y)), ("孕妇数", d["孕妇代码"].nunique()), ("最小值", y.min()), ("1%分位数", np.quantile(y, .01)),
        ("5%分位数", np.quantile(y, .05)), ("中位数", np.median(y)), ("95%分位数", np.quantile(y, .95)),
        ("99%分位数", np.quantile(y, .99)), ("最大值", y.max()), ("偏度", stats.skew(y, bias=False)),
        ("峰度（超额）", stats.kurtosis(y, fisher=True, bias=False)), ("精确等于0的事件数", int((y == 0).sum())),
        ("精确等于1的事件数", int((y == 1).sum())), ("低于4%的事件比例", float((y < .04).mean())),
        ("低于2%的事件比例", float((y < .02).mean())), ("高于15%的事件比例", float((y > .15).mean())),
        ("高于20%的事件比例", float((y > .20).mean())),
    ]
    return pd.DataFrame(vals, columns=["诊断项目", "数值"])


def 异方差诊断(fits: list[拟合结果], d: pd.DataFrame, beta_naive: 拟合结果 | None = None) -> pd.DataFrame:
    rows = []
    y = d["Y染色体浓度均值"].to_numpy(float)
    for fit in fits:
        mu, phi = 预测(fit, d)
        raw = y - mu
        if fit.类型 == "Beta":
            var = np.clip(mu * (1 - mu) / (1 + phi), 1e-12, None)
            resid = raw / np.sqrt(var)
        elif fit.类型 == "GEE-分数logit":
            resid = raw / np.sqrt(np.clip(mu * (1 - mu), 1e-12, None))
        else:
            resid = raw
        rho, p_rho = stats.spearmanr(np.abs(resid), mu)
        q = pd.qcut(mu, 4, labels=False, duplicates="drop")
        groups = [resid[np.asarray(q == k)] for k in sorted(pd.Series(q).dropna().unique())]
        lev_stat, lev_p = stats.levene(*groups, center="median") if len(groups) >= 2 else (math.nan, math.nan)
        dep = math.nan
        if fit.类型.startswith("GEE"):
            dep_arr = np.asarray(fit.结果.cov_struct.dep_params).reshape(-1)
            dep = float(dep_arr[0]) if len(dep_arr) else math.nan
        rows.append({
            "模型": fit.名称, "收敛": "是" if fit.收敛 else "否", "警告": fit.警告,
            "绝对标准化残差与拟合值Spearman相关": rho, "相关检验P值": p_rho,
            "按拟合值四分位Brown-Forsythe统计量": lev_stat, "Brown-Forsythe检验P值": lev_p,
            "交换型工作相关系数": dep, "原尺度预测越界数": int(((mu <= 0) | (mu >= 1)).sum()),
        })
    if beta_naive is not None:
        beta_cluster = next(f for f in fits if f.名称 == beta_naive.名称)
        p = len(beta_cluster.特征)
        for i, term in enumerate(beta_cluster.特征):
            rows.append({
                "模型": beta_cluster.名称, "诊断项目": f"{term}标准误稳健化比值",
                "聚类稳健标准误/独立样本标准误": float(beta_cluster.结果.bse[i] / beta_naive.结果.bse[i]),
            })
    return pd.DataFrame(rows)


def 共线性诊断(d: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for block_name, features in [("主均值方程", 主特征二次), ("质量敏感性方程", 主特征二次 + 质量特征)]:
        X = d[features].to_numpy(float)
        for j, name in enumerate(features):
            if name == "截距":
                continue
            rows.append({"变量块": block_name, "变量": name, "方差膨胀因子VIF": float(variance_inflation_factor(X, j))})
    return pd.DataFrame(rows)


def 稳健性分析(d: pd.DataFrame) -> tuple[pd.DataFrame, list[拟合结果]]:
    specs = [
        ("主样本", d, 主特征二次),
        ("截至25周0天", d.loc[d["纳入截至25周0天敏感性标志"].eq(1)].copy(), 主特征二次),
        ("排除日期孕周偏差超14天事件", d.loc[d["任一记录日期孕周偏差超14天标志"].eq(0)].copy(), 主特征二次),
        ("加入稀疏辅助生殖标志", d, 主特征二次 + ["辅助生殖标志"]),
        ("加入测序质量调整块", d, 主特征二次 + 质量特征),
        ("线性孕周均值结构", d, 主特征线性),
    ]
    rows = []
    fits = []
    for label, sub, features in specs:
        fit = 拟合GEE(sub, features, "分数logit")
        fit.名称 = f"分数logit GEE::{label}"
        fits.append(fit)
        gest_terms = [x for x in ["孕周中心化", "孕周二次项"] if x in features]
        tests = {
            "孕周总体关联P值": 联合Wald(fit, gest_terms)[2],
            "妇间BMI关联P值": 联合Wald(fit, ["妇间BMI中心化"])[2],
            "个体内BMI关联P值": 联合Wald(fit, ["BMI个体内偏差"])[2],
        }
        ge = 边际效应(fit, sub, "孕周平均")
        be = 边际效应(fit, sub, "妇间BMI")
        we = 边际效应(fit, sub, "个体内BMI")
        rows.append({
            "敏感性方案": label, "事件数": len(sub), "孕妇数": sub["孕妇代码"].nunique(), "收敛": "是" if fit.收敛 else "否",
            "孕周平均边际效应（百分点/周）": ge["估计"], "妇间BMI边际效应（百分点/BMI）": be["估计"],
            "个体内BMI边际效应（百分点/BMI）": we["估计"], **tests,
        })
    return pd.DataFrame(rows), fits


def 簇自助Beta(d: pd.DataFrame, features: list[str], mode: str, n_boot: int = 自助次数) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    rng = np.random.default_rng(随机种子)
    women = np.array(sorted(d["孕妇代码"].unique()))
    coef_rows = []
    effect_rows = []
    success = 0
    for b in range(1, n_boot + 1):
        sampled = rng.choice(women, size=len(women), replace=True)
        parts = []
        for k, w in enumerate(sampled):
            part = d.loc[d["孕妇代码"].eq(w)].copy()
            part["自助簇"] = f"{w}__{k}"
            parts.append(part)
        boot = pd.concat(parts, ignore_index=True)
        try:
            fit = 拟合Beta(boot, features, mode, cluster=False, group_col="自助簇")
            if not fit.收敛:
                continue
            p = len(features)
            pars = np.asarray(fit.结果.params, float)[:p]
            if not np.isfinite(pars).all():
                continue
            success += 1
            for term, value in zip(features, pars):
                coef_rows.append({"自助重复": b, "参数": term, "估计值": value})
            for target in ("孕周平均", "妇间BMI", "个体内BMI"):
                val = 平均效应(pars, d, features, target, "logit") * 100
                effect_rows.append({"自助重复": b, "原尺度效应": target, "Y浓度变化（百分点）": val})
        except Exception:
            continue
    coef = pd.DataFrame(coef_rows)
    eff = pd.DataFrame(effect_rows)
    summaries = []
    for term, g in coef.groupby("参数"):
        vals = g["估计值"].to_numpy(float)
        p_sign = 2 * min((np.sum(vals <= 0) + 1) / (len(vals) + 1), (np.sum(vals >= 0) + 1) / (len(vals) + 1))
        summaries.append({"结果类型": "链接尺度参数", "项目": term, "自助中位数": np.median(vals), "95%分位区间下限": np.quantile(vals, .025), "95%分位区间上限": np.quantile(vals, .975), "符号检验近似P值": min(1.0, p_sign), "成功重复数": success})
    for term, g in eff.groupby("原尺度效应"):
        vals = g["Y浓度变化（百分点）"].to_numpy(float)
        p_sign = 2 * min((np.sum(vals <= 0) + 1) / (len(vals) + 1), (np.sum(vals >= 0) + 1) / (len(vals) + 1))
        summaries.append({"结果类型": "原尺度边际效应（百分点）", "项目": term, "自助中位数": np.median(vals), "95%分位区间下限": np.quantile(vals, .025), "95%分位区间上限": np.quantile(vals, .975), "符号检验近似P值": min(1.0, p_sign), "成功重复数": success})
    coef_long = coef.rename(columns={"参数": "项目", "估计值": "数值"}).copy()
    coef_long.insert(1, "结果类型", "链接尺度参数")
    eff_long = eff.rename(columns={"原尺度效应": "项目", "Y浓度变化（百分点）": "数值"}).copy()
    eff_long.insert(1, "结果类型", "原尺度边际效应（百分点）")
    raw = pd.concat([coef_long, eff_long], ignore_index=True).sort_values(["自助重复", "结果类型", "项目"])
    return pd.DataFrame(summaries), raw, success


def 强影响孕妇(d: pd.DataFrame, full: 拟合结果) -> pd.DataFrame:
    base = np.asarray(full.结果.params, float)
    se = np.asarray(full.结果.bse, float)
    rows = []
    for woman in sorted(d["孕妇代码"].unique()):
        sub = d.loc[d["孕妇代码"].ne(woman)].copy()
        try:
            fit = 拟合GEE(sub, full.特征, "分数logit")
            delta = np.asarray(fit.结果.params, float) - base
            std = np.abs(delta / np.where(se > 0, se, np.nan))
            j = int(np.nanargmax(std))
            rows.append({
                "被删除孕妇": woman, "该孕妇事件数": int(d["孕妇代码"].eq(woman).sum()), "重拟合收敛": "是" if fit.收敛 else "否",
                "最大变化参数": full.特征[j], "最大绝对标准化变化": float(std[j]),
                "孕周线性项变化": float(delta[full.特征.index("孕周中心化")]),
                "孕周二次项变化": float(delta[full.特征.index("孕周二次项")]),
                "妇间BMI项变化": float(delta[full.特征.index("妇间BMI中心化")]),
                "个体内BMI项变化": float(delta[full.特征.index("BMI个体内偏差")]),
            })
        except Exception as e:
            rows.append({"被删除孕妇": woman, "重拟合收敛": "否", "错误": str(e)})
    return pd.DataFrame(rows).sort_values("最大绝对标准化变化", ascending=False, na_position="last")


def 预测网格(fits: list[拟合结果], d: pd.DataFrame, centers: dict[str, float]) -> pd.DataFrame:
    bmi_levels = np.quantile(d.groupby("孕妇代码")["孕妇平均BMI"].first(), [.25, .5, .75])
    rows = []
    for bmi_label, bmi in zip(["妇间BMI第25百分位", "妇间BMI中位数", "妇间BMI第75百分位"], bmi_levels):
        for week in np.linspace(max(11.0, d["孕周数"].min()), d["孕周数"].max(), 120):
            base = d.iloc[[0]].copy()
            base["孕周数"] = week
            base["孕周中心化"] = week - 孕周中心
            base["孕周二次项"] = (week - 孕周中心) ** 2
            base["孕妇平均BMI"] = bmi
            base["妇间BMI中心化"] = bmi - centers["妇间BMI中心"]
            base["BMI个体内偏差"] = 0.0
            base["年龄中心化"] = 0.0
            base["生产次数中心化"] = 0.0
            out = {"BMI水平": bmi_label, "孕妇平均BMI": bmi, "孕周数": week}
            for fit in fits:
                mu, _ = 预测(fit, base)
                out[f"{fit.名称}预测Y浓度"] = float(mu[0])
            rows.append(out)
    return pd.DataFrame(rows)


def 生成报告(outdir: Path, d: pd.DataFrame, cv: pd.DataFrame, overall: pd.DataFrame, effects: pd.DataFrame,
             diagnostics: pd.DataFrame, sensitivity: pd.DataFrame, bootstrap: pd.DataFrame, influence: pd.DataFrame,
             beta_selected: 拟合结果, fits: list[拟合结果], centers: dict[str, float], boot_success: int) -> None:
    cv2 = cv.set_index("模型")
    selected_name = beta_selected.名称
    frac_name = "分数logit GEE-二次孕周"
    def pval(model: str, item: str) -> float:
        return float(overall.loc[(overall["模型"] == model) & (overall["检验项"] == item), "P值"].iloc[0])
    def eff(model: str, item: str) -> pd.Series:
        return effects.loc[(effects["模型"] == model) & (effects["效应"] == item)].iloc[0]
    frac_g = eff(frac_name, "孕周平均")
    frac_b = eff(frac_name, "妇间BMI")
    frac_w = eff(frac_name, "个体内BMI")
    frac_12 = eff(frac_name, "孕周局部@12周")
    frac_16 = eff(frac_name, "孕周局部@16周")
    frac_20 = eff(frac_name, "孕周局部@20周")
    frac_24 = eff(frac_name, "孕周局部@24周")
    frac_fit = next(f for f in fits if f.名称 == frac_name)
    frac_b1 = float(frac_fit.结果.params[frac_fit.特征.index("孕周中心化")])
    frac_b2 = float(frac_fit.结果.params[frac_fit.特征.index("孕周二次项")])
    turning_week = 孕周中心 - frac_b1 / (2 * frac_b2)
    top = influence.iloc[0]
    boot_rate = boot_success / 自助次数
    beta_cv = cv2.loc[selected_name]
    frac_cv = cv2.loc[frac_name]
    gauss_cv = cv2.loc["高斯GEE-二次孕周"]
    beta_choice_reason = "变精度" if beta_selected.精度模式 == "孕周变精度" else "常精度"
    report = fr"""# 候选C：比例边界稳健关系模型报告

## 1. 建模目的与定位

第一问要求解释胎儿 Y 染色体浓度与孕周、BMI 等指标的相关特性并检验显著性。Y 浓度是严格位于 0 与 1 之间的连续比例，直接使用正态同方差模型可能产生越界预测或不恰当的方差结构。因此本候选在统一的 613 个抽血事件、167 名孕妇上比较 Beta 回归、分数 logit GEE 与原尺度高斯 GEE。所有结论均为观察性关联，不解释为因果效应。

本路线遵守冻结口径：以 B+I 为抽血事件、事件等权；A055 第3次抽血因孕周歧义排除；683 后机制段不参与；BMI 拆成妇间均值与个体内偏差；不使用序号、Y-Z 值、身高或体重。辅助生殖仅 11 个事件、3 名孕妇，因簇支持过稀只进入敏感性分析；这不是按 P 值筛变量。怀孕次数的 167 个缺失也不通过完整病例删除来牺牲 48 名孕妇。

## 2. 符号与模型

记第 $i$ 名孕妇第 $j$ 次抽血事件的 Y 染色体浓度为 $Y_{{ij}}\in(0,1)$，孕周为 $t_{{ij}}$，孕妇平均 BMI 为 $\bar B_i$，同一孕妇的 BMI 偏差为 $B_{{ij}}-\bar B_i$。年龄与生产次数构成预先指定的临床调整块。

### 2.1 Beta 边际回归

\[
Y_{{ij}}\sim\operatorname{{Beta}}(\mu_{{ij}}\phi_{{ij}},(1-\mu_{{ij}})\phi_{{ij}}),
\]

\[
\operatorname{{logit}}(\mu_{{ij}})=\beta_0+\beta_1(t_{{ij}}-18)+\beta_2(t_{{ij}}-18)^2
+\beta_3(\bar B_i-{centers['妇间BMI中心']:.4f})+\beta_4(B_{{ij}}-\bar B_i)
+\beta_5(A_i-{centers['年龄中心']:.4f})+\beta_6(P_i-{centers['生产次数中心']:.4f}).
\]

Beta 方差为 $\mu_{{ij}}(1-\mu_{{ij}})/(1+\phi_{{ij}})$。候选内部同时比较常精度与 $\log\phi_{{ij}}=\gamma_0+\gamma_1(t_{{ij}}-18)$ 两种结构，最终 Beta 版本为“{beta_choice_reason}”。点估计采用边际 Beta 似然，标准误按孕妇做簇稳健修正；另以孕妇为抽样单位执行 {自助次数} 次簇自助法。

### 2.2 分数 logit GEE

\[
\operatorname{{logit}}\{{E(Y_{{ij}}\mid X_{{ij}})\}}=X_{{ij}}^T\beta,
\]

孕妇内部采用交换型工作相关矩阵，显著性使用孕妇簇稳健 sandwich 协方差。分数 logit 不要求响应是 0/1 二项结果，只要求条件均值位于 $(0,1)$，因此既保持边界，又允许真实方差偏离二项方差。

## 3. 数据边界与估计可行性

主样本 Y 浓度最小值为 {d['Y染色体浓度均值'].min():.6f}，最大值为 {d['Y染色体浓度均值'].max():.6f}，没有精确 0 或 1，故 Beta 似然无需人为进行边界压缩。簇数为 167，足以使用孕妇簇稳健协方差；{自助次数} 次簇自助成功 {boot_success} 次，成功率 {boot_rate:.1%}。

## 4. 主要结果（以分数 logit GEE 为边界稳健主对照）

- 孕周整组 Wald 检验：$p={pval(frac_name, '孕周总体关联'):.4g}$；二次项检验：$p={pval(frac_name, '孕周非线性'):.4g}$。
- 孕周在样本分布上的平均边际效应为 {frac_g['Y浓度变化估计（百分点）']:.4f} 个百分点/周，95%CI [{frac_g['95%置信区间下限（百分点）']:.4f}, {frac_g['95%置信区间上限（百分点）']:.4f}]。
- 二次项为正，链接尺度曲线的估计最低点约为 {turning_week:.2f} 周。12周局部斜率为 {frac_12['Y浓度变化估计（百分点）']:.4f} 个百分点/周且区间跨0；16、20、24周局部斜率分别为 {frac_16['Y浓度变化估计（百分点）']:.4f}、{frac_20['Y浓度变化估计（百分点）']:.4f}、{frac_24['Y浓度变化估计（百分点）']:.4f} 个百分点/周，说明中后段上升逐渐加快。最低点属于二次均值结构内的描述，不外推到样本窗口之外。
- 妇间 BMI 每增加 1 kg/m²，Y 浓度平均变化 {frac_b['Y浓度变化估计（百分点）']:.4f} 个百分点，95%CI [{frac_b['95%置信区间下限（百分点）']:.4f}, {frac_b['95%置信区间上限（百分点）']:.4f}]，$p={pval(frac_name, '妇间BMI关联'):.4g}$。
- 同一孕妇 BMI 相对本人均值每增加 1 kg/m²，Y 浓度平均变化 {frac_w['Y浓度变化估计（百分点）']:.4f} 个百分点，95%CI [{frac_w['95%置信区间下限（百分点）']:.4f}, {frac_w['95%置信区间上限（百分点）']:.4f}]，$p={pval(frac_name, '个体内BMI关联'):.4g}$。

以上妇间和个体内效应不可互换：前者比较不同孕妇，后者描述同一孕妇随时间的 BMI 偏离。

## 5. 分组验证与稳健性

按孕妇分成5折，测试折孕妇不出现在训练折。{selected_name} 的组外 RMSE={beta_cv['RMSE']:.6f}、MAE={beta_cv['MAE']:.6f}、R²={beta_cv['组外R²']:.4f}；分数 logit GEE 二次孕周的相应指标为 {frac_cv['RMSE']:.6f}、{frac_cv['MAE']:.6f}、{frac_cv['组外R²']:.4f}；高斯 GEE 为 {gauss_cv['RMSE']:.6f}、{gauss_cv['MAE']:.6f}、{gauss_cv['组外R²']:.4f}。这些验证只用于检查新孕妇的固定效应/人口平均组外拟合。候选 C 没有随机斜率，故该 CV 不能评价个体孕周斜率异质性，也不能与含随机斜率模型的条件预测能力直接等同。

敏感性方案覆盖截至25周0天、排除日期孕周偏差事件、稀疏辅助生殖调整、测序质量调整、线性孕周均值结构和原尺度高斯 GEE。具体方向和数值见 `08_稳健性分析.csv`，不得只凭某一个 P 值下结论。

逐一删除孕妇的重拟合中，最大标准化参数变化为 {top['最大绝对标准化变化']:.3f}（删除 {top['被删除孕妇']}，影响参数为 {top['最大变化参数']}）。该诊断用于识别结论是否被单一孕妇主导。

## 6. BetaModel 聚类稳健推断的可行边界

1. `BetaModel(cov_type='cluster')` 能在 167 个相互独立的孕妇簇下给出 sandwich 标准误；孕妇簇自助法还能独立核验区间与符号稳定性。
2. 但该模型的似然本身仍把边际观测按 Beta 密度相乘，簇稳健协方差只修正推断，不会生成孕妇随机截距，也不会估计个体特异效应。
3. 当均值模型或 Beta 边际分布严重错设、簇数很少、或少数孕妇强影响时，稳健标准误不能挽救点估计。AIC 只能在两个 Beta 版本之间比较，不能与 GEE 的准似然直接横向比较。
4. 分数 logit GEE 估计人口平均关系并允许稳健的簇内相关推断，但同样没有孕妇随机截距或随机斜率。因此，Beta 回归适合作为“比例边界和异方差”的稳健性对照；若论文主线强调层级随机效应或个体斜率差异，应由混合效应候选承担。分数 logit GEE 可作为人口平均关系的合法主候选，但不能替代个体特异随机效应解释。

## 7. 候选C的主审建议与否决条件

候选 C 不建议以纯 BetaModel 单独成为最终主线；建议把分数 logit GEE 作为边界稳健主对照，把 Beta 结果和簇自助区间放入稳健性部分。若最终混合/样条主模型与本候选在孕周、妇间 BMI、个体内 BMI 的方向或区间上明显冲突，应暂停定稿并回查均值函数、强影响孕妇和响应尺度。

触发以下任一条件时，候选 C 不得作为主线：Beta 簇自助成功率低于90%；GEE任一分组验证折不收敛；组外预测明显劣于简单线性基线；核心结论由单一孕妇删除后翻转；或边界链接与原尺度模型的方向无法解释地冲突。
"""
    (outdir / "候选C_边界稳健模型报告.md").write_text(report, encoding="utf-8")


def 生成制图提示词(outdir: Path) -> None:
    prompt = """候选C后续统一制图提示词（当前阶段禁止实际绘图）

统一要求：
1. 仅使用 MATLAB；最终输出纯矢量 SVG，设置 `set(groot,'defaultAxesFontName','Microsoft YaHei')` 和 `set(groot,'defaultTextFontName','Microsoft YaHei')`。
2. 白色背景，宽约16 cm、高约10 cm，300 dpi仅用于预览，最终 `exportgraphics(...,'ContentType','vector')` 输出 SVG。
3. 所有标题、坐标轴、图例、注释使用中文；Y浓度统一显示为百分比；不得把观测关联写成因果效应。

图C1：比例模型预测关系曲线
- 数据文件：`14_原尺度预测网格.csv`，可叠加 `12_主样本固定效应预测.csv` 中的真实事件点。
- 横轴：孕周数；纵轴：Y染色体浓度（%）。
- 按“妇间BMI第25百分位/中位数/第75百分位”分三种线型或颜色。
- 主曲线使用“分数logit GEE-二次孕周预测Y浓度”；用较细虚线叠加最终选定Beta模型预测。真实点用浅灰、小尺寸、透明散点，仅作分布背景。
- 不得连同一孕妇的事件点；图注写明固定年龄、生产次数为样本中心，BMI个体内偏差为0。
- 输出建议文件名：`候选C_孕周BMI边界模型预测.svg`。

图C2：按孕妇分组验证的残差诊断
- 数据文件：`12_主样本固定效应预测.csv`。
- 横轴：分数logit GEE二次孕周的OOF预测Y浓度（%）；纵轴：OOF残差（观测-预测，百分点）。
- 画 y=0 水平虚线；叠加局部线性/LOWESS趋势（只作诊断，不给因果解释）。
- 颜色按交叉验证折1-5，但透明度低；右上角标注5折按孕妇划分、事件数613、孕妇数167。
- 输出建议文件名：`候选C_组外残差诊断.svg`。

图C3：原尺度边际效应区间图
- 数据文件：`05_原尺度边际效应.csv`。
- 只筛选“孕周平均、妇间BMI、个体内BMI”三个效应与三个模型（最终Beta、分数logit GEE、高斯GEE）。
- 横轴：Y浓度变化（百分点/相应单位）；纵轴：效应名称；点为估计，横线为95%CI，竖线x=0。
- 用颜色区分模型，不以星号代替置信区间；单位在副标题中明确。
- 输出建议文件名：`候选C_边际效应稳健性对照.svg`。

图C4：强影响孕妇诊断（可选）
- 数据文件：`09_强影响孕妇.csv`。
- 仅显示“最大绝对标准化变化”最大的前15名；横轴为标准化变化，纵轴为孕妇代码；颜色表示“最大变化参数”。
- 添加参考线x=1，标题明确这是逐一删除孕妇的敏感性诊断。
- 输出建议文件名：`候选C_强影响孕妇诊断.svg`。
"""
    (outdir / "候选C_制图提示词.txt").write_text(prompt, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="第一问候选C：Beta/GEE边界稳健模型")
    here = Path(__file__).resolve().parent
    default_input = here.parent / "00_共同口径" / "冻结数据" / "第一问主模型冻结样本.csv"
    parser.add_argument("--input", type=Path, default=default_input)
    parser.add_argument("--bootstrap", type=int, default=自助次数)
    args = parser.parse_args()
    input_path = args.input.resolve()
    outdir = here
    d, centers = 载入并构造变量(input_path)

    cv_summary, cv_folds, d = 交叉验证(d)
    beta_const = 拟合Beta(d, 主特征二次, "常精度", cluster=True)
    beta_const.名称 = "Beta回归-常精度-二次孕周"
    beta_var = 拟合Beta(d, 主特征二次, "孕周变精度", cluster=True)
    beta_var.名称 = "Beta回归-孕周变精度-二次孕周"
    frac_linear = 拟合GEE(d, 主特征线性, "分数logit")
    frac_linear.名称 = "分数logit GEE-线性孕周"
    frac_quad = 拟合GEE(d, 主特征二次, "分数logit")
    frac_quad.名称 = "分数logit GEE-二次孕周"
    gauss_quad = 拟合GEE(d, 主特征二次, "高斯恒等")
    gauss_quad.名称 = "高斯GEE-二次孕周"
    fits = [beta_const, beta_var, frac_linear, frac_quad, gauss_quad]
    if not all(f.收敛 for f in fits):
        failed = [f.名称 for f in fits if not f.收敛]
        raise RuntimeError(f"全样本模型未收敛: {failed}")

    # 预先锁定的简约选择：变精度Beta须AIC至少降低2，且组外RMSE不比常精度差0.5%以上。
    cv_idx = cv_summary.set_index("模型")
    aic_gain = float(beta_const.结果.aic - beta_var.结果.aic)
    rmse_const = float(cv_idx.loc[beta_const.名称, "RMSE"])
    rmse_var = float(cv_idx.loc[beta_var.名称, "RMSE"])
    beta_selected = beta_var if (aic_gain >= 2 and rmse_var <= rmse_const * 1.005) else beta_const

    beta_naive = 拟合Beta(d, 主特征二次, beta_selected.精度模式 or "常精度", cluster=False)
    beta_naive.名称 = beta_selected.名称
    selected_fits = [beta_selected, frac_quad, gauss_quad]

    boundary = 边界诊断(d)
    coefs = 系数表(fits)
    overall = 整体检验表(fits)
    effects = 边际效应表(selected_fits, d)
    diagnostics = 异方差诊断(selected_fits, d, beta_naive)
    collin = 共线性诊断(d)
    sensitivity, sensitivity_fits = 稳健性分析(d)
    influence = 强影响孕妇(d, frac_quad)
    boot_summary, boot_raw, boot_success = 簇自助Beta(d, beta_selected.特征, beta_selected.精度模式 or "常精度", args.bootstrap)
    if boot_success < math.ceil(args.bootstrap * 0.9):
        raise RuntimeError(f"Beta簇自助成功率低于90%: {boot_success}/{args.bootstrap}")

    pred_table = d[["孕妇代码", "抽血事件键", "孕周数", "孕妇平均BMI", "BMI个体内偏差", "年龄", "生产次数", "Y染色体浓度均值", "交叉验证折"]].copy()
    for fit in selected_fits:
        pred, _ = 预测(fit, d)
        pred_table[f"{fit.名称}固定效应预测"] = pred
        pred_table[f"{fit.名称}拟合残差"] = d["Y染色体浓度均值"].to_numpy(float) - pred
        oof_col = f"OOF::{fit.名称}"
        if oof_col in d.columns:
            pred_table[f"{fit.名称}OOF预测"] = d[oof_col]
            pred_table[f"{fit.名称}OOF残差"] = d["Y染色体浓度均值"] - d[oof_col]
    grid = 预测网格(selected_fits, d, centers)

    model_rows = []
    for fit in fits:
        pred, _ = 预测(fit, d)
        model_rows.append({
            "模型": fit.名称, "模型类型": fit.类型, "均值方程参数数": len(fit.特征), "总参数复杂度": 模型复杂度(fit),
            "收敛": "是" if fit.收敛 else "否", "AIC（仅Beta内部可比）": float(fit.结果.aic) if fit.类型 == "Beta" else math.nan,
            "BIC（仅Beta内部可比）": float(fit.结果.bic) if fit.类型 == "Beta" else math.nan,
            "全样本RMSE（仅描述）": 公共指标(d["Y染色体浓度均值"].to_numpy(float), pred)["RMSE"],
            "是否Beta内部选定版本": "是" if fit is beta_selected else "否",
        })
    model_compare = pd.DataFrame(model_rows).merge(cv_summary, on="模型", how="left", suffixes=("", "_组外"))

    保存表(boundary, outdir / "01_样本边界与分布诊断.csv")
    保存表(model_compare, outdir / "02_模型比较与分组验证.csv")
    保存表(coefs, outdir / "03_模型系数与聚类稳健推断.csv")
    保存表(overall, outdir / "04_整体显著性检验.csv")
    保存表(effects, outdir / "05_原尺度边际效应.csv")
    保存表(cv_folds, outdir / "06_交叉验证折明细.csv")
    保存表(diagnostics, outdir / "07_异方差与相关结构诊断.csv")
    保存表(sensitivity, outdir / "08_稳健性分析.csv")
    保存表(influence, outdir / "09_强影响孕妇.csv")
    保存表(collin, outdir / "10_共线性诊断.csv")
    保存表(boot_summary, outdir / "11_Beta簇自助法摘要.csv")
    保存表(boot_raw, outdir / "11b_Beta簇自助法原始结果.csv")
    保存表(pred_table, outdir / "12_主样本固定效应预测.csv")
    保存表(pd.DataFrame({"孕妇代码": d["孕妇代码"], "交叉验证折": d["交叉验证折"]}).drop_duplicates().sort_values(["交叉验证折", "孕妇代码"]), outdir / "13_孕妇交叉验证折分配.csv")
    保存表(grid, outdir / "14_原尺度预测网格.csv")

    生成报告(outdir, d, cv_summary, overall, effects, diagnostics, sensitivity, boot_summary, influence,
             beta_selected, fits, centers, boot_success)
    生成制图提示词(outdir)

    summary = {
        "冻结样本": {"事件数": len(d), "孕妇数": int(d["孕妇代码"].nunique()), "输入SHA256": 文件哈希(input_path)},
        "随机种子": 随机种子,
        "Beta内部选定版本": beta_selected.名称,
        "Beta常精度相对变精度AIC差": aic_gain,
        "Beta簇自助": {"计划次数": args.bootstrap, "成功次数": boot_success, "成功率": boot_success / args.bootstrap},
        "模型收敛": {f.名称: bool(f.收敛) for f in fits},
        "软件": {"Python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__, "statsmodels": statsmodels.__version__, "scikit-learn": sklearn.__version__},
        "方法边界": "BetaModel使用孕妇簇稳健协方差与孕妇簇自助，但没有随机截距；纯Beta结果只作边界稳健性对照。",
    }
    (outdir / "候选C_运行摘要.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    files = []
    for p in sorted(outdir.iterdir()):
        if p.is_file() and p.name != "候选C_复现清单.json":
            files.append({"文件名": p.name, "字节数": p.stat().st_size, "SHA256": 文件哈希(p)})
    manifest = {"输入文件": str(input_path), "输入SHA256": 文件哈希(input_path), "输出文件": files}
    (outdir / "候选C_复现清单.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
