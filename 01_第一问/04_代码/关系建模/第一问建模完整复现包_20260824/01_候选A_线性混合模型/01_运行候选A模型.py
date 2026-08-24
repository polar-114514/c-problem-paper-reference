from __future__ import annotations

import hashlib
import json
import math
import platform
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import patsy
import scipy
import sklearn
import statsmodels
import statsmodels.formula.api as smf
from scipy.special import expit
from scipy.stats import chi2, kurtosis, skew
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor


随机种子 = 20250824
交叉验证重复数 = 5
交叉验证折数 = 5
整簇自助次数 = 200

本文件 = Path(__file__).resolve()
候选目录 = 本文件.parent
共同口径目录 = 候选目录.parent / "00_共同口径"
输入文件 = 共同口径目录 / "冻结数据" / "第一问主模型冻结样本.csv"
源工作簿 = (
    Path(__file__).resolve().parents[5]
    / "00_题目与原始资料"
    / "02_原始数据"
    / "附件.xlsx"
)


def 文件哈希(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def 写CSV(df: pd.DataFrame, name: str) -> Path:
    path = 候选目录 / name
    中文表头映射 = {
        "AIC": "赤池信息准则_AIC",
        "BIC": "贝叶斯信息准则_BIC",
        "边际R2": "边际决定系数_R2",
        "条件R2": "条件决定系数_R2",
        "RMSE": "均方根误差_RMSE",
        "MAE": "平均绝对误差_MAE",
        "折内R2": "折内决定系数_R2",
        "组外R2": "组外决定系数_R2",
        "RMSE均值": "均方根误差均值_RMSE",
        "RMSE标准差": "均方根误差标准差_RMSE",
        "MAE均值": "平均绝对误差均值_MAE",
        "MAE标准差": "平均绝对误差标准差_MAE",
        "组外R2均值": "组外决定系数均值_R2",
        "组外R2标准差": "组外决定系数标准差_R2",
        "Wald_Z": "沃尔德Z统计量",
        "Wald_P值_参考点斜率": "参考点斜率沃尔德P值",
        "logit尺度参考点斜率系数": "对数几率尺度参考点斜率系数",
        "logit尺度95CI下限": "对数几率尺度95%置信区间下限",
        "logit尺度95CI上限": "对数几率尺度95%置信区间上限",
        "估计值_logit尺度": "对数几率尺度估计值",
        "95CI下限": "95%置信区间下限",
        "95CI上限": "95%置信区间上限",
        "原尺度95CI下限_百分点": "原尺度95%置信区间下限_百分点",
        "原尺度95CI上限_百分点": "原尺度95%置信区间上限_百分点",
        "自助95CI下限_百分点": "自助95%置信区间下限_百分点",
        "自助95CI上限_百分点": "自助95%置信区间上限_百分点",
        "固定效应均值95CI下限": "固定效应均值95%置信区间下限",
        "固定效应均值95CI上限": "固定效应均值95%置信区间上限",
    }
    out = df.rename(columns=中文表头映射)
    df_columns = list(out.columns)
    if len(df_columns) != len(set(df_columns)):
        raise ValueError(f"中文表头映射后出现重名：{name}")
    out.to_csv(path, index=False, encoding="utf-8-sig", float_format="%.12g")
    return path


def 写文本(text: str, name: str) -> Path:
    path = 候选目录 / name
    path.write_text(text, encoding="utf-8")
    return path


@dataclass(frozen=True)
class 模型规格:
    代号: str
    名称: str
    公式右侧: str
    随机公式: str = "1"


模型规格表 = {
    "空模型_随机截距": 模型规格("空模型_随机截距", "空模型（随机截距）", "1"),
    "核心线性_随机截距": 模型规格(
        "核心线性_随机截距",
        "核心线性模型（随机截距）",
        "孕周中心 + 妇间BMI中心 + BMI个体内偏差",
    ),
    "调整线性_随机截距": 模型规格(
        "调整线性_随机截距",
        "临床调整线性模型（随机截距）",
        "孕周中心 + 妇间BMI中心 + BMI个体内偏差 + 年龄中心 + 生产次数中心",
    ),
    "孕周二次_随机截距": 模型规格(
        "孕周二次_随机截距",
        "孕周二次模型（随机截距）",
        "孕周中心 + I(孕周中心 ** 2) + 妇间BMI中心 + BMI个体内偏差 + 年龄中心 + 生产次数中心",
    ),
    "二次交互_随机截距": 模型规格(
        "二次交互_随机截距",
        "低阶曲率交互模型（随机截距）",
        "孕周中心 + I(孕周中心 ** 2) + 妇间BMI中心 + I(妇间BMI中心 ** 2) + BMI个体内偏差 + 孕周中心:妇间BMI中心 + 年龄中心 + 生产次数中心",
    ),
    "调整线性_随机斜率": 模型规格(
        "调整线性_随机斜率",
        "临床调整线性模型（孕周随机斜率）",
        "孕周中心 + 妇间BMI中心 + BMI个体内偏差 + 年龄中心 + 生产次数中心",
        "1 + 孕周中心",
    ),
    "孕周二次_随机斜率": 模型规格(
        "孕周二次_随机斜率",
        "孕周二次模型（孕周随机斜率）",
        "孕周中心 + I(孕周中心 ** 2) + 妇间BMI中心 + BMI个体内偏差 + 年龄中心 + 生产次数中心",
        "1 + 孕周中心",
    ),
}


def 读取并核验数据() -> pd.DataFrame:
    df = pd.read_csv(输入文件, encoding="utf-8-sig")
    必需列 = [
        "孕妇代码",
        "抽血事件键",
        "孕周数",
        "孕妇体质指数_BMI",
        "年龄",
        "辅助生殖标志",
        "生产次数",
        "怀孕次数",
        "Y染色体浓度均值",
        "GC含量均值",
        "原始读段数均值",
        "比对比例均值",
        "重复读段比例均值",
        "过滤读段比例均值",
        "任一记录日期孕周偏差超14天标志",
        "纳入截至25周0天敏感性标志",
    ]
    缺列 = [c for c in 必需列 if c not in df.columns]
    if 缺列:
        raise ValueError(f"冻结数据缺少必需列：{缺列}")
    if len(df) != 613 or df["孕妇代码"].nunique() != 167:
        raise ValueError(f"冻结样本口径错误：{len(df)}个事件、{df['孕妇代码'].nunique()}名孕妇")
    if df["抽血事件键"].duplicated().any():
        raise ValueError("抽血事件键不唯一")
    if not ((df["Y染色体浓度均值"] > 0) & (df["Y染色体浓度均值"] < 1)).all():
        raise ValueError("Y浓度不全在(0,1)，不能直接作logit变换")
    核心列 = ["孕周数", "孕妇体质指数_BMI", "年龄", "辅助生殖标志", "生产次数", "Y染色体浓度均值"]
    if df[核心列].isna().any().any():
        raise ValueError("主模型核心或预设临床调整变量存在缺失")
    if int(df["怀孕次数"].isna().sum()) != 167:
        raise ValueError("怀孕次数缺失数与冻结协议不一致")
    return df.copy()


def 构造分析变量(
    raw: pd.DataFrame,
    中心值: dict[str, float] | None = None,
    质量尺度: dict[str, tuple[float, float]] | None = None,
    重算BMI分解: bool = True,
) -> tuple[pd.DataFrame, dict[str, float], dict[str, tuple[float, float]]]:
    d = raw.copy()
    d["孕妇代码"] = d["孕妇代码"].astype(str)
    d["Y浓度"] = d["Y染色体浓度均值"].astype(float)
    d["Y浓度logit"] = np.log(d["Y浓度"] / (1.0 - d["Y浓度"]))
    if 重算BMI分解:
        d["妇间BMI"] = d.groupby("孕妇代码", observed=True)["孕妇体质指数_BMI"].transform("mean")
        d["BMI个体内偏差"] = d["孕妇体质指数_BMI"] - d["妇间BMI"]
    else:
        d["妇间BMI"] = d["孕妇平均BMI"]
        d["BMI个体内偏差"] = d["BMI个体内偏差"]

    if 中心值 is None:
        person = d.sort_values("孕妇代码").drop_duplicates("孕妇代码")
        中心值 = {
            "孕周": float(d["孕周数"].mean()),
            "妇间BMI": float(person["妇间BMI"].mean()),
            "年龄": float(person["年龄"].mean()),
            "生产次数": float(person["生产次数"].mean()),
        }
    d["孕周中心"] = d["孕周数"] - 中心值["孕周"]
    d["妇间BMI中心"] = d["妇间BMI"] - 中心值["妇间BMI"]
    d["年龄中心"] = d["年龄"] - 中心值["年龄"]
    d["生产次数中心"] = d["生产次数"] - 中心值["生产次数"]

    d["对数原始读段数"] = np.log(d["原始读段数均值"].astype(float))
    质量源列 = {
        "GC含量标准化": "GC含量均值",
        "对数读段数标准化": "对数原始读段数",
        "比对比例标准化": "比对比例均值",
        "重复读段比例标准化": "重复读段比例均值",
        "过滤读段比例标准化": "过滤读段比例均值",
    }
    if 质量尺度 is None:
        质量尺度 = {}
        for 新列, 源列 in 质量源列.items():
            均值 = float(d[源列].mean())
            标准差 = float(d[源列].std(ddof=0))
            if not np.isfinite(标准差) or 标准差 <= 0:
                raise ValueError(f"质量变量{源列}没有可用变异")
            质量尺度[新列] = (均值, 标准差)
    for 新列, 源列 in 质量源列.items():
        均值, 标准差 = 质量尺度[新列]
        d[新列] = (d[源列] - 均值) / 标准差
    return d, 中心值, 质量尺度


def 公式(spec: 模型规格, response: str = "Y浓度logit") -> str:
    return f"{response} ~ {spec.公式右侧}"


def 拟合混合模型(
    spec: 模型规格,
    data: pd.DataFrame,
    response: str = "Y浓度logit",
    reml: bool = False,
):
    model = smf.mixedlm(
        公式(spec, response),
        data=data,
        groups=data["孕妇代码"],
        re_formula=spec.随机公式,
    )
    全部警告: list[str] = []
    最后异常: Exception | None = None
    for method in ("lbfgs", "powell"):
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                result = model.fit(reml=reml, method=method, maxiter=2000, disp=False)
            警告 = [str(w.message) for w in caught]
            全部警告.extend(警告)
            if bool(getattr(result, "converged", False)) and np.isfinite(result.llf):
                result._候选A优化器 = method
                result._候选A警告 = "；".join(dict.fromkeys(全部警告))
                return result
        except Exception as exc:  # 保留异常供审计，不静默失败
            最后异常 = exc
            全部警告.append(f"{method}:{type(exc).__name__}:{exc}")
    raise RuntimeError(f"混合模型未收敛：{spec.名称}；{最后异常}；{'；'.join(全部警告)}")


def 随机方差摘要(result) -> tuple[float, float, float, bool]:
    cov = np.asarray(result.cov_re, dtype=float)
    随机截距方差 = float(cov[0, 0]) if cov.size else 0.0
    随机斜率方差 = float(cov[1, 1]) if cov.shape[0] > 1 else np.nan
    残差方差 = float(result.scale)
    最小特征值 = float(np.linalg.eigvalsh(cov).min()) if cov.size else 0.0
    奇异 = bool(随机截距方差 < 1e-8 or 最小特征值 < 1e-8)
    return 随机截距方差, 随机斜率方差, 残差方差, 奇异


def 混合模型R2(result) -> tuple[float, float]:
    固定预测 = np.asarray(result.model.exog @ np.asarray(result.fe_params), dtype=float)
    固定方差 = float(np.var(固定预测, ddof=1))
    cov = np.asarray(result.cov_re, dtype=float)
    if cov.shape == (1, 1):
        随机方差 = float(cov[0, 0])
    else:
        Z = np.asarray(result.model.exog_re, dtype=float)
        随机方差 = float(np.mean(np.einsum("ij,jk,ik->i", Z, cov, Z)))
    残差方差 = float(result.scale)
    总方差 = 固定方差 + 随机方差 + 残差方差
    return 固定方差 / 总方差, (固定方差 + 随机方差) / 总方差


def 模型复杂度(result) -> int:
    q = int(np.asarray(result.cov_re).shape[0])
    return int(len(result.fe_params) + q * (q + 1) / 2 + 1)


def LRT(完整模型, 简化模型) -> tuple[float, int, float]:
    统计量 = max(0.0, 2.0 * (float(完整模型.llf) - float(简化模型.llf)))
    自由度 = max(1, 模型复杂度(完整模型) - 模型复杂度(简化模型))
    return 统计量, 自由度, float(chi2.sf(统计量, 自由度))


def 构造固定设计(result, data: pd.DataFrame) -> np.ndarray:
    info = result.model.data.design_info
    return np.asarray(patsy.build_design_matrices([info], data, return_type="dataframe")[0], dtype=float)


def 仅固定效应预测(result, data: pd.DataFrame, response: str = "Y浓度logit") -> np.ndarray:
    X = 构造固定设计(result, data)
    eta = X @ np.asarray(result.fe_params, dtype=float)
    return expit(eta) if response == "Y浓度logit" else eta


def 原尺度平均边际效应(
    result,
    data: pd.DataFrame,
    variable: str,
    delta: float = 1.0,
    draws: int = 2000,
    rng: np.random.Generator | None = None,
) -> tuple[float, float, float]:
    d0 = data.copy()
    d1 = data.copy()
    d1[variable] = d1[variable] + delta
    X0 = 构造固定设计(result, d0)
    X1 = 构造固定设计(result, d1)
    beta = np.asarray(result.fe_params, dtype=float)
    point = float(np.mean(expit(X1 @ beta) - expit(X0 @ beta)))
    if draws <= 0:
        return point, np.nan, np.nan
    names = list(result.fe_params.index)
    cov = np.asarray(result.cov_params().loc[names, names], dtype=float)
    cov = (cov + cov.T) / 2.0
    eigval, eigvec = np.linalg.eigh(cov)
    cov = eigvec @ np.diag(np.clip(eigval, 1e-12, None)) @ eigvec.T
    rng = np.random.default_rng(随机种子 + 911) if rng is None else rng
    beta_draw = rng.multivariate_normal(beta, cov, size=draws)
    # 分批计算，避免创建过大的事件×抽样三维数组。
    effects = []
    for start in range(0, draws, 200):
        b = beta_draw[start : start + 200].T
        effects.extend(np.mean(expit(X1 @ b) - expit(X0 @ b), axis=0).tolist())
    lo, hi = np.quantile(np.asarray(effects), [0.025, 0.975])
    return point, float(lo), float(hi)


def 分配孕妇折(data: pd.DataFrame, seed: int, k: int) -> dict[str, int]:
    counts = data.groupby("孕妇代码", observed=True).size().rename("事件数").reset_index()
    rng = np.random.default_rng(seed)
    counts["随机序"] = rng.random(len(counts))
    counts = counts.sort_values(["事件数", "随机序"], ascending=[False, True])
    fold_sizes = [0] * k
    mapping: dict[str, int] = {}
    for row in counts.itertuples(index=False):
        choices = np.flatnonzero(np.asarray(fold_sizes) == min(fold_sizes))
        fold = int(rng.choice(choices))
        mapping[str(row.孕妇代码)] = fold + 1
        fold_sizes[fold] += int(row.事件数)
    return mapping


def 交叉验证(raw: pd.DataFrame):
    逐折行: list[dict] = []
    折分行: list[dict] = []
    重复汇总行: list[dict] = []
    specs = [
        模型规格表["空模型_随机截距"],
        模型规格表["核心线性_随机截距"],
        模型规格表["调整线性_随机截距"],
        模型规格表["孕周二次_随机截距"],
        模型规格表["二次交互_随机截距"],
        模型规格表["调整线性_随机斜率"],
        模型规格表["孕周二次_随机斜率"],
    ]
    for repeat in range(1, 交叉验证重复数 + 1):
        mapping = 分配孕妇折(raw, 随机种子 + 1009 * repeat, 交叉验证折数)
        for person, fold in sorted(mapping.items()):
            折分行.append(
                {
                    "重复轮次": repeat,
                    "折号": fold,
                    "孕妇代码": person,
                    "事件数": int((raw["孕妇代码"].astype(str) == person).sum()),
                }
            )
        for spec in specs:
            全部真实: list[float] = []
            全部预测: list[float] = []
            失败折 = 0
            for fold in range(1, 交叉验证折数 + 1):
                test_mask = raw["孕妇代码"].astype(str).map(mapping).eq(fold)
                train0, test0 = raw.loc[~test_mask].copy(), raw.loc[test_mask].copy()
                train, centers, quality_scales = 构造分析变量(train0, 重算BMI分解=True)
                test, _, _ = 构造分析变量(
                    test0,
                    中心值=centers,
                    质量尺度=quality_scales,
                    重算BMI分解=True,
                )
                if set(train["孕妇代码"]) & set(test["孕妇代码"]):
                    raise AssertionError("交叉验证发生孕妇泄漏")
                try:
                    fit = 拟合混合模型(spec, train, reml=False)
                    pred = 仅固定效应预测(fit, test)
                    true = test["Y浓度"].to_numpy(dtype=float)
                    rmse = float(np.sqrt(np.mean((true - pred) ** 2)))
                    mae = float(np.mean(np.abs(true - pred)))
                    r2 = float(1.0 - np.sum((true - pred) ** 2) / np.sum((true - true.mean()) ** 2))
                    conv = 1
                    warn = getattr(fit, "_候选A警告", "")
                    optimizer = getattr(fit, "_候选A优化器", "")
                    全部真实.extend(true.tolist())
                    全部预测.extend(pred.tolist())
                except Exception as exc:
                    rmse = mae = r2 = np.nan
                    conv = 0
                    warn = f"{type(exc).__name__}:{exc}"
                    optimizer = ""
                    失败折 += 1
                逐折行.append(
                    {
                        "模型代号": spec.代号,
                        "模型名称": spec.名称,
                        "重复轮次": repeat,
                        "折号": fold,
                        "训练孕妇数": int(train0["孕妇代码"].nunique()),
                        "测试孕妇数": int(test0["孕妇代码"].nunique()),
                        "训练事件数": len(train0),
                        "测试事件数": len(test0),
                        "RMSE": rmse,
                        "MAE": mae,
                        "折内R2": r2,
                        "收敛标志": conv,
                        "优化器": optimizer,
                        "警告或异常": warn,
                    }
                )
            if 全部真实:
                y = np.asarray(全部真实)
                p = np.asarray(全部预测)
                重复汇总行.append(
                    {
                        "模型代号": spec.代号,
                        "模型名称": spec.名称,
                        "重复轮次": repeat,
                        "RMSE": float(np.sqrt(np.mean((y - p) ** 2))),
                        "MAE": float(np.mean(np.abs(y - p))),
                        "组外R2": float(1.0 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2)),
                        "失败折数": 失败折,
                    }
                )
    repeat_df = pd.DataFrame(重复汇总行)
    汇总 = (
        repeat_df.groupby(["模型代号", "模型名称"], as_index=False)
        .agg(
            RMSE均值=("RMSE", "mean"),
            RMSE标准差=("RMSE", "std"),
            MAE均值=("MAE", "mean"),
            MAE标准差=("MAE", "std"),
            组外R2均值=("组外R2", "mean"),
            组外R2标准差=("组外R2", "std"),
            失败折数=("失败折数", "sum"),
        )
        .sort_values("RMSE均值")
        .reset_index(drop=True)
    )
    return pd.DataFrame(逐折行), repeat_df, 汇总, pd.DataFrame(折分行)


def 模型模型比较(data: pd.DataFrame):
    fits = {}
    rows = []
    for key, spec in 模型规格表.items():
        fit = 拟合混合模型(spec, data, reml=False)
        fits[key] = fit
        v0, v1, ve, singular = 随机方差摘要(fit)
        r2m, r2c = 混合模型R2(fit)
        rows.append(
            {
                "模型代号": key,
                "模型名称": spec.名称,
                "固定效应参数数": len(fit.fe_params),
                "总参数数": 模型复杂度(fit),
                "对数似然": float(fit.llf),
                "AIC": float(fit.aic),
                "BIC": float(fit.bic),
                "边际R2": r2m,
                "条件R2": r2c,
                "随机截距方差": v0,
                "孕周随机斜率方差": v1,
                "残差方差": ve,
                "奇异随机效应标志": int(singular),
                "收敛标志": int(bool(fit.converged)),
                "优化器": getattr(fit, "_候选A优化器", ""),
                "拟合警告": getattr(fit, "_候选A警告", ""),
            }
        )
    return fits, pd.DataFrame(rows)


def 整体检验(data: pd.DataFrame, fits: dict):
    linear = fits["调整线性_随机截距"]
    week_quad = fits["孕周二次_随机截距"]
    full_quad = fits["二次交互_随机截距"]
    rows = []

    # 线性候选的整块检验，供结构比较和线性退回方案审计。
    linear_specs = [
        ("线性模型_孕周块", "妇间BMI中心 + BMI个体内偏差 + 年龄中心 + 生产次数中心", 1),
        ("线性模型_BMI总体块", "孕周中心 + 年龄中心 + 生产次数中心", 2),
        ("线性模型_妇间BMI项", "孕周中心 + BMI个体内偏差 + 年龄中心 + 生产次数中心", 1),
        ("线性模型_BMI个体内项", "孕周中心 + 妇间BMI中心 + 年龄中心 + 生产次数中心", 1),
        ("线性模型_临床调整块", "孕周中心 + 妇间BMI中心 + BMI个体内偏差", 2),
    ]
    for name, rhs, expected_df in linear_specs:
        spec = 模型规格(f"删去{name}", f"删去{name}", rhs)
        reduced = 拟合混合模型(spec, data, reml=False)
        stat, df, p = LRT(linear, reduced)
        rows.append(
            {
                "检验块": name,
                "完整模型": 模型规格表["调整线性_随机截距"].名称,
                "简化模型": spec.名称,
                "似然比统计量": stat,
                "自由度": df,
                "预期自由度": expected_df,
                "P值": p,
                "显著性结论_0.05": "显著" if p < 0.05 else "不显著",
            }
        )

    # 孕周二次模型是候选A的最简非线性结构；所有核心变量均作整块检验。
    week_specs = [
        ("孕周二次模型_孕周总体块", "妇间BMI中心 + BMI个体内偏差 + 年龄中心 + 生产次数中心", 2),
        ("孕周二次模型_BMI总体块", "孕周中心 + I(孕周中心 ** 2) + 年龄中心 + 生产次数中心", 2),
        ("孕周二次模型_妇间BMI项", "孕周中心 + I(孕周中心 ** 2) + BMI个体内偏差 + 年龄中心 + 生产次数中心", 1),
        ("孕周二次模型_BMI个体内项", "孕周中心 + I(孕周中心 ** 2) + 妇间BMI中心 + 年龄中心 + 生产次数中心", 1),
        ("孕周二次模型_临床调整块", "孕周中心 + I(孕周中心 ** 2) + 妇间BMI中心 + BMI个体内偏差", 2),
    ]
    for name, rhs, expected_df in week_specs:
        spec = 模型规格(f"删去{name}", f"删去{name}", rhs)
        reduced = 拟合混合模型(spec, data, reml=False)
        stat, df, p = LRT(week_quad, reduced)
        rows.append(
            {
                "检验块": name,
                "完整模型": 模型规格表["孕周二次_随机截距"].名称,
                "简化模型": spec.名称,
                "似然比统计量": stat,
                "自由度": df,
                "预期自由度": expected_df,
                "P值": p,
                "显著性结论_0.05": "显著" if p < 0.05 else "不显著",
            }
        )

    full_specs = [
        ("完整低阶模型_孕周总体块", "妇间BMI中心 + I(妇间BMI中心 ** 2) + BMI个体内偏差 + 年龄中心 + 生产次数中心", 3),
        ("完整低阶模型_BMI总体块", "孕周中心 + I(孕周中心 ** 2) + 年龄中心 + 生产次数中心", 4),
        ("完整低阶模型_妇间BMI块", "孕周中心 + I(孕周中心 ** 2) + BMI个体内偏差 + 年龄中心 + 生产次数中心", 3),
        ("完整低阶模型_BMI个体内项", "孕周中心 + I(孕周中心 ** 2) + 妇间BMI中心 + I(妇间BMI中心 ** 2) + 孕周中心:妇间BMI中心 + 年龄中心 + 生产次数中心", 1),
        ("完整低阶模型_临床调整块", "孕周中心 + I(孕周中心 ** 2) + 妇间BMI中心 + I(妇间BMI中心 ** 2) + BMI个体内偏差 + 孕周中心:妇间BMI中心", 2),
    ]
    for name, rhs, expected_df in full_specs:
        spec = 模型规格(f"删去{name}", f"删去{name}", rhs)
        reduced = 拟合混合模型(spec, data, reml=False)
        stat, df, p = LRT(full_quad, reduced)
        rows.append(
            {
                "检验块": name,
                "完整模型": 模型规格表["二次交互_随机截距"].名称,
                "简化模型": spec.名称,
                "似然比统计量": stat,
                "自由度": df,
                "预期自由度": expected_df,
                "P值": p,
                "显著性结论_0.05": "显著" if p < 0.05 else "不显著",
            }
        )

    stat, df, p = LRT(week_quad, linear)
    rows.append(
        {
            "检验块": "孕周二次项增量块",
            "完整模型": 模型规格表["孕周二次_随机截距"].名称,
            "简化模型": 模型规格表["调整线性_随机截距"].名称,
            "似然比统计量": stat,
            "自由度": df,
            "预期自由度": 1,
            "P值": p,
            "显著性结论_0.05": "显著" if p < 0.05 else "不显著",
        }
    )
    stat, df, p = LRT(full_quad, week_quad)
    rows.append(
        {
            "检验块": "妇间BMI二次项与孕周-BMI交互增量块",
            "完整模型": 模型规格表["二次交互_随机截距"].名称,
            "简化模型": 模型规格表["孕周二次_随机截距"].名称,
            "似然比统计量": stat,
            "自由度": df,
            "预期自由度": 2,
            "P值": p,
            "显著性结论_0.05": "显著" if p < 0.05 else "不显著",
        }
    )
    return pd.DataFrame(rows)


def 选择候选A结构(
    模型比较: pd.DataFrame,
    检验表: pd.DataFrame,
    cv: pd.DataFrame,
    cv_repeat: pd.DataFrame,
) -> tuple[str, list[str]]:
    reasons: list[str] = []
    cvmap = cv.set_index("模型代号")["RMSE均值"].to_dict()
    p_week2 = float(检验表.loc[检验表["检验块"] == "孕周二次项增量块", "P值"].iloc[0])
    p_extra = float(检验表.loc[检验表["检验块"] == "妇间BMI二次项与孕周-BMI交互增量块", "P值"].iloc[0])
    linear_rmse = float(cvmap["调整线性_随机截距"])
    week_rmse = float(cvmap["孕周二次_随机截距"])
    full_rmse = float(cvmap["二次交互_随机截距"])
    week_gain = (linear_rmse - week_rmse) / linear_rmse
    repeat_pivot = cv_repeat.pivot(index="重复轮次", columns="模型代号", values="RMSE")
    week_improve_rounds = int((repeat_pivot["孕周二次_随机截距"] < repeat_pivot["调整线性_随机截距"]).sum())
    if p_week2 < 0.05 and week_gain >= 0.005 and week_improve_rounds == 交叉验证重复数:
        chosen = "孕周二次_随机截距"
        reasons.append(
            f"孕周二次项增量检验P={p_week2:.4g}，分组CV的RMSE相对改善{week_gain:.2%}，且{week_improve_rounds}/{交叉验证重复数}轮均改善，保留孕周二次项。"
        )
    else:
        chosen = "调整线性_随机截距"
        reasons.append(
            f"孕周二次项增量检验P={p_week2:.4g}，分组CV的RMSE相对改善{week_gain:.2%}，改善轮次{week_improve_rounds}/{交叉验证重复数}；未同时越过预设门槛，退回线性结构。"
        )

    extra_gain = (week_rmse - full_rmse) / week_rmse
    if chosen == "孕周二次_随机截距" and p_extra < 0.05 and extra_gain >= 0.005:
        chosen = "二次交互_随机截距"
        reasons.append(f"在孕周二次项基础上，妇间BMI二次项与交互增量检验P={p_extra:.4g}且CV再改善{extra_gain:.2%}，保留完整低阶块。")
    else:
        reasons.append(f"在孕周二次项基础上，妇间BMI二次项与交互增量检验P={p_extra:.4g}、CV再改善{extra_gain:.2%}；不保留无充分证据的附加项。")

    compare = 模型比较.set_index("模型代号")
    if chosen == "孕周二次_随机截距":
        base_key, rs_key = "孕周二次_随机截距", "孕周二次_随机斜率"
    elif chosen == "调整线性_随机截距":
        base_key, rs_key = "调整线性_随机截距", "调整线性_随机斜率"
    else:
        base_key = rs_key = ""
    if base_key:
        rs_row = compare.loc[rs_key]
        base_row = compare.loc[base_key]
        base_rmse = float(cvmap[base_key])
        rs_rmse = float(cvmap[rs_key])
        rs_gain = (base_rmse - rs_rmse) / base_rmse
    else:
        rs_row = base_row = None
        rs_gain = -np.inf
    rs_ok = (
        bool(base_key)
        and
        int(rs_row["收敛标志"]) == 1
        and int(rs_row["奇异随机效应标志"]) == 0
        and float(rs_row["AIC"]) <= float(base_row["AIC"]) - 2.0
        and float(rs_row["BIC"]) <= float(base_row["BIC"]) - 2.0
        and rs_rmse <= base_rmse * 1.01
    )
    if rs_ok:
        chosen = rs_key
        reasons.append(
            f"随机斜率收敛且非奇异，AIC/BIC分别下降{float(base_row['AIC'])-float(rs_row['AIC']):.3f}/{float(base_row['BIC'])-float(rs_row['BIC']):.3f}，新孕妇固定效应CV的RMSE变化{(-rs_gain):+.2%}（未明显恶化），保留孕周随机斜率。"
        )
    elif base_key:
        reasons.append(
            f"随机斜率相对同一固定效应随机截距模型的AIC/BIC变化为{float(rs_row['AIC'])-float(base_row['AIC']):.3f}/{float(rs_row['BIC'])-float(base_row['BIC']):.3f}，新孕妇固定效应CV的RMSE变化{(-rs_gain):+.2%}，且奇异标志={int(rs_row['奇异随机效应标志'])}；不满足全部保留条件。"
        )
    return chosen, reasons


def 关键效应表(result, data: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(随机种子 + 444)
    variable_info = [
        ("孕周中心", "孕周", "每增加1周"),
        ("妇间BMI中心", "妇间BMI", "孕妇平均BMI每增加1 kg/m^2"),
        ("BMI个体内偏差", "BMI个体内", "同一孕妇BMI相对本人均值每增加1 kg/m^2"),
    ]
    rows = []
    for variable, label, unit in variable_info:
        ame, lo, hi = 原尺度平均边际效应(result, data, variable, 1.0, 2000, rng)
        coefficient = float(result.fe_params.get(variable, np.nan))
        if variable in result.fe_params.index:
            se = float(result.bse_fe.iloc[result.fe_params.index.get_loc(variable)])
            z = coefficient / se
            p = float(2 * scipy.stats.norm.sf(abs(z)))
            clo, chi = coefficient - 1.96 * se, coefficient + 1.96 * se
        else:
            se = z = p = clo = chi = np.nan
        rows.append(
            {
                "效应": label,
                "变化单位": unit,
                "logit尺度参考点斜率系数": coefficient,
                "标准误": se,
                "logit尺度95CI下限": clo,
                "logit尺度95CI上限": chi,
                "Wald_Z": z,
                "Wald_P值_参考点斜率": p,
                "原尺度平均边际效应_百分点": ame * 100.0,
                "原尺度95CI下限_百分点": lo * 100.0,
                "原尺度95CI上限_百分点": hi * 100.0,
            }
        )
    return pd.DataFrame(rows)


def 所有参数表(result) -> pd.DataFrame:
    rows = []
    for idx, name in enumerate(result.fe_params.index):
        coef = float(result.fe_params.iloc[idx])
        se = float(result.bse_fe.iloc[idx])
        z = coef / se
        rows.append(
            {
                "参数": name,
                "估计值_logit尺度": coef,
                "标准误": se,
                "95CI下限": coef - 1.96 * se,
                "95CI上限": coef + 1.96 * se,
                "Wald_Z": z,
                "P值": float(2 * scipy.stats.norm.sf(abs(z))),
            }
        )
    return pd.DataFrame(rows)


def 诊断(result, data: pd.DataFrame):
    conditional_fitted = np.asarray(result.fittedvalues, dtype=float)
    conditional_resid = np.asarray(result.resid, dtype=float)
    fixed_y = 仅固定效应预测(result, data)
    X = np.asarray(result.model.exog, dtype=float)
    try:
        lm, lm_p, fval, f_p = het_breuschpagan(conditional_resid, X)
    except Exception:
        lm = lm_p = fval = f_p = np.nan
    names = list(result.model.exog_names)
    vif_rows = []
    for i, name in enumerate(names):
        if name == "Intercept":
            continue
        try:
            vif = float(variance_inflation_factor(X, i))
        except Exception:
            vif = np.nan
        vif_rows.append({"变量": name, "方差膨胀因子VIF": vif})
    v0, v1, ve, singular = 随机方差摘要(result)
    icc = v0 / (v0 + ve) if np.isfinite(v0 + ve) and v0 + ve > 0 else np.nan
    cov_re = np.asarray(result.cov_re, dtype=float)
    if cov_re.shape[0] > 1 and v0 > 0 and v1 > 0:
        random_corr = float(cov_re[0, 1] / math.sqrt(v0 * v1))
        random_min_eig = float(np.linalg.eigvalsh(cov_re).min())
    else:
        random_corr = np.nan
        random_min_eig = float(np.linalg.eigvalsh(cov_re).min()) if cov_re.size else np.nan
    diag_rows = [
        {"诊断项目": "收敛", "统计量": int(bool(result.converged)), "P值": np.nan, "判读": "通过" if result.converged else "失败"},
        {"诊断项目": "随机效应奇异", "统计量": int(singular), "P值": np.nan, "判读": "通过" if not singular else "警报"},
        {"诊断项目": "随机截距方差", "统计量": v0, "P值": np.nan, "判读": "logit尺度"},
        {"诊断项目": "孕周随机斜率方差", "统计量": v1, "P值": np.nan, "判读": "logit尺度；随机截距模型为空"},
        {"诊断项目": "残差方差", "统计量": ve, "P值": np.nan, "判读": "logit尺度"},
        {"诊断项目": "中心孕周处组内相关系数ICC", "统计量": icc, "P值": np.nan, "判读": "随机斜率模型中ICC随孕周变化，此值仅对应中心孕周"},
        {"诊断项目": "随机截距与随机斜率相关系数", "统计量": random_corr, "P值": np.nan, "判读": "随机截距模型为空"},
        {"诊断项目": "随机效应协方差最小特征值", "统计量": random_min_eig, "P值": np.nan, "判读": "接近0提示奇异"},
        {"诊断项目": "Breusch-Pagan异方差LM", "统计量": lm, "P值": lm_p, "判读": "P<0.05提示异方差"},
        {"诊断项目": "Breusch-Pagan异方差F", "统计量": fval, "P值": f_p, "判读": "P<0.05提示异方差"},
        {"诊断项目": "条件残差偏度", "统计量": float(skew(conditional_resid, bias=False)), "P值": np.nan, "判读": "绝对值越小越对称"},
        {"诊断项目": "条件残差超额峰度", "统计量": float(kurtosis(conditional_resid, fisher=True, bias=False)), "P值": np.nan, "判读": "0附近接近正态"},
        {"诊断项目": "原尺度固定效应预测低于0的比例", "统计量": float(np.mean(fixed_y < 0)), "P值": np.nan, "判读": "logit反变换理论上应为0"},
        {"诊断项目": "原尺度固定效应预测高于1的比例", "统计量": float(np.mean(fixed_y > 1)), "P值": np.nan, "判读": "logit反变换理论上应为0"},
    ]
    event = data[["孕妇代码", "抽血事件键", "孕周数", "孕妇体质指数_BMI", "妇间BMI", "BMI个体内偏差", "Y浓度"]].copy()
    event = event.rename(columns={"孕妇体质指数_BMI": "孕妇BMI"})
    event["固定效应预测Y浓度"] = fixed_y
    event["条件拟合logit"] = conditional_fitted
    event["条件残差logit"] = conditional_resid
    event["标准化条件残差"] = conditional_resid / math.sqrt(max(float(result.scale), 1e-15))
    return pd.DataFrame(diag_rows), pd.DataFrame(vif_rows), event


def 获取关键效应点估计(result, data: pd.DataFrame) -> dict[str, float]:
    out = {}
    for variable, label in [("孕周中心", "孕周"), ("妇间BMI中心", "妇间BMI"), ("BMI个体内偏差", "BMI个体内")]:
        out[label] = 原尺度平均边际效应(result, data, variable, draws=0)[0] * 100.0
    return out


def 整簇自助(raw: pd.DataFrame, spec: 模型规格, 主结果, 主分析数据: pd.DataFrame):
    rng = np.random.default_rng(随机种子 + 777)
    persons = np.asarray(sorted(raw["孕妇代码"].astype(str).unique()))
    main_effect = 获取关键效应点估计(主结果, 主分析数据)
    rows = []
    for b in range(1, 整簇自助次数 + 1):
        sampled = rng.choice(persons, size=len(persons), replace=True)
        chunks = []
        for copy_idx, person in enumerate(sampled):
            chunk = raw.loc[raw["孕妇代码"].astype(str).eq(person)].copy()
            chunk["孕妇代码"] = [f"{person}__自助{copy_idx:03d}"] * len(chunk)
            chunks.append(chunk)
        boot0 = pd.concat(chunks, ignore_index=True)
        boot, _, _ = 构造分析变量(boot0, 重算BMI分解=True)
        try:
            fit = 拟合混合模型(spec, boot, reml=False)
            effects = 获取关键效应点估计(fit, boot)
            rows.append(
                {
                    "自助序号": b,
                    "收敛标志": 1,
                    "孕周效应_百分点": effects["孕周"],
                    "妇间BMI效应_百分点": effects["妇间BMI"],
                    "BMI个体内效应_百分点": effects["BMI个体内"],
                    "警告或异常": getattr(fit, "_候选A警告", ""),
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "自助序号": b,
                    "收敛标志": 0,
                    "孕周效应_百分点": np.nan,
                    "妇间BMI效应_百分点": np.nan,
                    "BMI个体内效应_百分点": np.nan,
                    "警告或异常": f"{type(exc).__name__}:{exc}",
                }
            )
    detail = pd.DataFrame(rows)
    ok = detail.loc[detail["收敛标志"].eq(1)].copy()
    summary_rows = []
    for label, col in [
        ("孕周", "孕周效应_百分点"),
        ("妇间BMI", "妇间BMI效应_百分点"),
        ("BMI个体内", "BMI个体内效应_百分点"),
    ]:
        values = ok[col].dropna().to_numpy(dtype=float)
        direction = np.sign(main_effect[label])
        summary_rows.append(
            {
                "效应": label,
                "主模型点估计_百分点": main_effect[label],
                "有效自助次数": len(values),
                "自助中位数_百分点": float(np.median(values)),
                "自助95CI下限_百分点": float(np.quantile(values, 0.025)),
                "自助95CI上限_百分点": float(np.quantile(values, 0.975)),
                "与主模型方向一致比例": float(np.mean(np.sign(values) == direction)),
            }
        )
    return detail, pd.DataFrame(summary_rows)


def 留一孕妇影响(raw: pd.DataFrame, spec: 模型规格, main_result, main_data: pd.DataFrame):
    main_effect = 获取关键效应点估计(main_result, main_data)
    coefficient_names = {"孕周": "孕周中心", "妇间BMI": "妇间BMI中心", "BMI个体内": "BMI个体内偏差"}
    main_coef = {k: float(main_result.fe_params[v]) for k, v in coefficient_names.items()}
    se_map = {
        "孕周": float(main_result.bse_fe.iloc[main_result.fe_params.index.get_loc("孕周中心")]) if "孕周中心" in main_result.fe_params.index else np.nan,
        "妇间BMI": float(main_result.bse_fe.iloc[main_result.fe_params.index.get_loc("妇间BMI中心")]) if "妇间BMI中心" in main_result.fe_params.index else np.nan,
        "BMI个体内": float(main_result.bse_fe.iloc[main_result.fe_params.index.get_loc("BMI个体内偏差")]) if "BMI个体内偏差" in main_result.fe_params.index else np.nan,
    }
    rows = []
    for person in sorted(raw["孕妇代码"].astype(str).unique()):
        sub0 = raw.loc[~raw["孕妇代码"].astype(str).eq(person)].copy()
        sub, _, _ = 构造分析变量(sub0, 重算BMI分解=True)
        try:
            fit = 拟合混合模型(spec, sub, reml=False)
            effects = 获取关键效应点估计(fit, sub)
            shifts = {k: effects[k] - main_effect[k] for k in main_effect}
            coef_shift_std = {
                k: (float(fit.fe_params[coefficient_names[k]]) - main_coef[k]) / se_map[k]
                for k in coefficient_names
            }
            denom = {k: max(abs(main_effect[k]), 1e-6) for k in main_effect}
            rows.append(
                {
                    "删去孕妇": person,
                    "被删事件数": int((raw["孕妇代码"].astype(str) == person).sum()),
                    "收敛标志": 1,
                    "孕周效应_百分点": effects["孕周"],
                    "妇间BMI效应_百分点": effects["妇间BMI"],
                    "BMI个体内效应_百分点": effects["BMI个体内"],
                    "孕周效应变化_百分点": shifts["孕周"],
                    "妇间BMI效应变化_百分点": shifts["妇间BMI"],
                    "BMI个体内效应变化_百分点": shifts["BMI个体内"],
                    "孕周系数变化_主模型标准误": coef_shift_std["孕周"],
                    "妇间BMI系数变化_主模型标准误": coef_shift_std["妇间BMI"],
                    "BMI个体内系数变化_主模型标准误": coef_shift_std["BMI个体内"],
                    "最大标准化系数变化": max(abs(v) for v in coef_shift_std.values()),
                    "最大相对变化": max(abs(shifts[k]) / denom[k] for k in shifts),
                    "任一核心效应方向翻转标志": int(any(np.sign(effects[k]) != np.sign(main_effect[k]) for k in main_effect)),
                    "警告或异常": getattr(fit, "_候选A警告", ""),
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "删去孕妇": person,
                    "被删事件数": int((raw["孕妇代码"].astype(str) == person).sum()),
                    "收敛标志": 0,
                    "最大标准化系数变化": np.nan,
                    "最大相对变化": np.nan,
                    "任一核心效应方向翻转标志": np.nan,
                    "警告或异常": f"{type(exc).__name__}:{exc}",
                }
            )
    detail = pd.DataFrame(rows).sort_values(["收敛标志", "最大标准化系数变化"], ascending=[False, False])
    return detail


def 稳健性分析(raw: pd.DataFrame, selected_spec: 模型规格, main_result, main_data: pd.DataFrame):
    main_effect = 获取关键效应点估计(main_result, main_data)
    rows = []

    def add_result(
        name: str,
        subset: pd.DataFrame,
        response: str = "Y浓度logit",
        quality: bool = False,
        ivf_sensitivity: bool = False,
    ):
        d, _, _ = 构造分析变量(subset, 重算BMI分解=True)
        if quality:
            rhs = selected_spec.公式右侧 + " + GC含量标准化 + 对数读段数标准化 + 比对比例标准化 + 重复读段比例标准化 + 过滤读段比例标准化"
            spec = 模型规格(name, name, rhs, selected_spec.随机公式)
        elif ivf_sensitivity:
            rhs = selected_spec.公式右侧 + " + 辅助生殖标志"
            spec = 模型规格(name, name, rhs, selected_spec.随机公式)
        else:
            spec = 模型规格(name, name, selected_spec.公式右侧, selected_spec.随机公式)
        try:
            fit = 拟合混合模型(spec, d, response=response, reml=False)
            if response == "Y浓度logit":
                effects = 获取关键效应点估计(fit, d)
            else:
                effects = {}
                for variable, label in [("孕周中心", "孕周"), ("妇间BMI中心", "妇间BMI"), ("BMI个体内偏差", "BMI个体内")]:
                    d1 = d.copy()
                    d1[variable] = d1[variable] + 1.0
                    delta_pred = (构造固定设计(fit, d1) - 构造固定设计(fit, d)) @ np.asarray(fit.fe_params, dtype=float)
                    effects[label] = float(np.mean(delta_pred)) * 100.0
            rows.append(
                {
                    "分析口径": name,
                    "事件数": len(d),
                    "孕妇数": d["孕妇代码"].nunique(),
                    "响应尺度": "logit后反变换" if response == "Y浓度logit" else "原始比例",
                    "孕周效应_百分点": effects["孕周"],
                    "妇间BMI效应_百分点": effects["妇间BMI"],
                    "BMI个体内效应_百分点": effects["BMI个体内"],
                    "孕周方向与主模型一致": int(np.sign(effects["孕周"]) == np.sign(main_effect["孕周"])),
                    "妇间BMI方向与主模型一致": int(np.sign(effects["妇间BMI"]) == np.sign(main_effect["妇间BMI"])),
                    "BMI个体内方向与主模型一致": int(np.sign(effects["BMI个体内"]) == np.sign(main_effect["BMI个体内"])),
                    "收敛标志": int(bool(fit.converged)),
                    "警告或异常": getattr(fit, "_候选A警告", ""),
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "分析口径": name,
                    "事件数": len(d),
                    "孕妇数": d["孕妇代码"].nunique(),
                    "响应尺度": response,
                    "收敛标志": 0,
                    "警告或异常": f"{type(exc).__name__}:{exc}",
                }
            )

    rows.append(
        {
            "分析口径": "主模型",
            "事件数": len(main_data),
            "孕妇数": main_data["孕妇代码"].nunique(),
            "响应尺度": "logit后反变换",
            "孕周效应_百分点": main_effect["孕周"],
            "妇间BMI效应_百分点": main_effect["妇间BMI"],
            "BMI个体内效应_百分点": main_effect["BMI个体内"],
            "孕周方向与主模型一致": 1,
            "妇间BMI方向与主模型一致": 1,
            "BMI个体内方向与主模型一致": 1,
            "收敛标志": int(bool(main_result.converged)),
            "警告或异常": getattr(main_result, "_候选A警告", ""),
        }
    )
    add_result("截至25周0天", raw.loc[raw["纳入截至25周0天敏感性标志"].eq(1)].copy())
    add_result("删除日期孕周偏差超14天事件", raw.loc[raw["任一记录日期孕周偏差超14天标志"].eq(0)].copy())
    add_result("原始浓度尺度线性混合模型", raw.copy(), response="Y浓度")
    add_result("加入辅助生殖标志敏感性", raw.copy(), ivf_sensitivity=True)
    add_result("加入五项测序质量调整", raw.copy(), quality=True)
    return pd.DataFrame(rows)


def 预测关系表(result, data: pd.DataFrame, centers: dict[str, float], selected_spec: 模型规格):
    rng = np.random.default_rng(随机种子 + 1919)
    names = list(result.fe_params.index)
    beta = np.asarray(result.fe_params, dtype=float)
    cov = np.asarray(result.cov_params().loc[names, names], dtype=float)
    cov = (cov + cov.T) / 2.0
    eigval, eigvec = np.linalg.eigh(cov)
    cov = eigvec @ np.diag(np.clip(eigval, 1e-12, None)) @ eigvec.T
    draws = rng.multivariate_normal(beta, cov, size=3000)
    bmi_levels = [28.0, 32.0, 36.0, 40.0]
    weeks = np.arange(10.0, 25.0001, 0.25)
    rows = []
    for bmi in bmi_levels:
        grid = pd.DataFrame(
            {
                "孕妇代码": ["参考孕妇"] * len(weeks),
                "孕周数": weeks,
                "孕妇体质指数_BMI": [bmi] * len(weeks),
                "妇间BMI": [bmi] * len(weeks),
                "BMI个体内偏差": [0.0] * len(weeks),
                "孕周中心": weeks - centers["孕周"],
                "妇间BMI中心": bmi - centers["妇间BMI"],
                "年龄": [centers["年龄"]] * len(weeks),
                "年龄中心": [0.0] * len(weeks),
                "辅助生殖标志": [0] * len(weeks),
                "生产次数": [centers["生产次数"]] * len(weeks),
                "生产次数中心": [0.0] * len(weeks),
            }
        )
        X = 构造固定设计(result, grid)
        point = expit(X @ beta)
        pred_draw = expit(X @ draws.T)
        lo = np.quantile(pred_draw, 0.025, axis=1)
        hi = np.quantile(pred_draw, 0.975, axis=1)
        for w, p, l, h in zip(weeks, point, lo, hi):
            rows.append(
                {
                    "孕周": w,
                    "妇间BMI参考值": bmi,
                    "BMI个体内偏差": 0.0,
                    "年龄参考值": centers["年龄"],
                    "受孕方式": "自然受孕",
                    "生产次数参考值": centers["生产次数"],
                    "预测Y浓度": float(p),
                    "固定效应均值95CI下限": float(l),
                    "固定效应均值95CI上限": float(h),
                    "模型代号": selected_spec.代号,
                }
            )
    return pd.DataFrame(rows)


def 生成报告(
    selected_key: str,
    reasons: list[str],
    model_compare: pd.DataFrame,
    tests: pd.DataFrame,
    cv: pd.DataFrame,
    effects: pd.DataFrame,
    diagnostic: pd.DataFrame,
    robustness: pd.DataFrame,
    bootstrap_summary: pd.DataFrame,
    influence: pd.DataFrame,
    elapsed: float,
) -> str:
    selected_name = 模型规格表[selected_key].名称
    mrow = model_compare.set_index("模型代号").loc[selected_key]
    cvrow = cv.set_index("模型代号").loc[selected_key]
    effect_map = effects.set_index("效应")
    test_map = tests.set_index("检验块")
    if selected_key.startswith("孕周二次_"):
        test_keys = {
            "孕周": "孕周二次模型_孕周总体块",
            "BMI总体": "孕周二次模型_BMI总体块",
            "妇间BMI": "孕周二次模型_妇间BMI项",
            "BMI个体内": "孕周二次模型_BMI个体内项",
        }
        selected_formula_note = "最终式在上述线性式中增加 $\\beta_{T2}T_{ij}^2$，但不增加无证据支持的妇间 BMI 二次项或孕周-BMI 交互。"
        if selected_key.endswith("随机斜率"):
            selected_formula_note += " 随机部分改为 $u_{0i}+u_{1i}T_{ij}$，允许不同孕妇具有不同孕周变化斜率；$u_{0i}$ 与 $u_{1i}$ 允许相关。"
    elif selected_key == "二次交互_随机截距":
        test_keys = {
            "孕周": "完整低阶模型_孕周总体块",
            "BMI总体": "完整低阶模型_BMI总体块",
            "妇间BMI": "完整低阶模型_妇间BMI块",
            "BMI个体内": "完整低阶模型_BMI个体内项",
        }
        selected_formula_note = "最终式在上述线性式中增加孕周二次项、妇间 BMI 二次项与孕周-BMI 交互。"
    else:
        test_keys = {
            "孕周": "线性模型_孕周块",
            "BMI总体": "线性模型_BMI总体块",
            "妇间BMI": "线性模型_妇间BMI项",
            "BMI个体内": "线性模型_BMI个体内项",
        }
        selected_formula_note = "最终式采用上述线性固定效应结构。"
    influence_ok = influence.loc[influence["收敛标志"].eq(1)]
    direction_flip_count = int(influence_ok["任一核心效应方向翻转标志"].sum())
    max_standardized = float(influence_ok["最大标准化系数变化"].max())
    unstable_boot = bootstrap_summary.loc[bootstrap_summary["与主模型方向一致比例"] < 0.90, "效应"].astype(str).tolist()
    if unstable_boot:
        self_denial = "触发预设自我否决条件：" + "、".join(unstable_boot) + " 的整簇自助方向保持率低于90%。因此候选A不能直接成为最终主线，只可作为可解释基线或对照，除非主审修改预设门槛并给出新的统计依据。"
    else:
        self_denial = "未触发整簇自助方向保持率门槛；仍需与其他候选比较组外误差和非线性形态。"
    lines = [
        "# 第一问候选 A：线性/低阶曲率混合效应模型报告",
        "",
        "> 候选阶段材料，仅供主审比较；未写入第一问正式模型目录。所有关系均为观察性条件关联，不作因果解释。",
        "",
        "## 1. 建模目的与数据口径",
        "",
        "题目要求分析胎儿 Y 染色体浓度与孕周、BMI 等指标的相关特性，建立关系模型并检验显著性。候选 A 用混合效应模型显式处理同一孕妇的重复抽血，并把 BMI 拆成妇间水平与个体内偏差，避免把两类含义不同的关联压成一个系数。",
        "",
        "主拟合使用冻结的 613 个抽血事件、167 名孕妇。每个 `孕妇代码+抽血次数` 事件等权；同一事件的多次检测先取 Y 浓度算术均值。A055 第 3 次抽血孕周元数据冲突，整次事件不进入主拟合。序号、Y 染色体 Z 值、身高和体重均未进入模型；683 后机制段未混入。怀孕次数缺失 167/613，按冻结协议不进入主调整块，避免完整病例分析额外丢失 48 名孕妇。",
        "",
        "## 2. 模型定义",
        "",
        "记孕妇为 $i$、抽血事件为 $j$。令 $Y_{ij}$ 为事件级 Y 染色体浓度，采用 $Z_{ij}=\\log[Y_{ij}/(1-Y_{ij})]$ 保证回到原尺度后的预测位于 $(0,1)$。令 $T_{ij}$ 为中心化孕周，$\\bar B_i$ 为孕妇事件级平均 BMI，$B_{ij}-\\bar B_i$ 为 BMI 个体内偏差。预设主临床调整只包括年龄和生产次数。辅助生殖仅见于 3 名孕妇（11 个事件），因簇层样本过稀而只作敏感性，不是按 P 值筛选。",
        "",
        "线性随机截距模型为：",
        "",
        "\\[",
        "Z_{ij}=\\beta_0+\\beta_TT_{ij}+\\beta_B(\\bar B_i-\\bar B)+\\beta_W(B_{ij}-\\bar B_i)+\\boldsymbol{\\gamma}^{\\mathsf T}\\mathbf X_i+u_i+\\varepsilon_{ij},",
        "\\]",
        "",
        "其中 $u_i\\sim N(0,\\sigma_u^2)$ 控制孕妇层未观测基线异质性，$\\varepsilon_{ij}\\sim N(0,\\sigma_e^2)$ 为事件级残差。低阶扩展仅增加 $T^2$、妇间 BMI 二次项与 $T\\times\\bar B_i$，并与线性式进行整块似然比检验；随机斜率也只有在非奇异、AIC 与分组验证同时支持时才保留。",
        "",
        selected_formula_note,
        "",
        "## 3. 结构选择",
        "",
    ]
    lines.extend([f"- {r}" for r in reasons])
    lines.extend(
        [
            "",
            f"候选 A 内部推荐结构为 **{selected_name}**。其 ML AIC={mrow['AIC']:.3f}、BIC={mrow['BIC']:.3f}，边际 $R^2$={mrow['边际R2']:.3f}、条件 $R^2$={mrow['条件R2']:.3f}；5×5 按孕妇分组交叉验证得到 RMSE={cvrow['RMSE均值']:.5f}±{cvrow['RMSE标准差']:.5f}、MAE={cvrow['MAE均值']:.5f}、组外 $R^2$={cvrow['组外R2均值']:.3f}。测试孕妇从未进入训练折，预测只使用固定效应。",
            "",
            "## 4. 核心效应及显著性",
            "",
        ]
    )
    for label in ["孕周", "妇间BMI", "BMI个体内"]:
        row = effect_map.loc[label]
        lines.append(
            f"- **{label}**：{row['变化单位']}，平均预测 Y 浓度变化 {row['原尺度平均边际效应_百分点']:.3f} 个百分点（95% CI {row['原尺度95CI下限_百分点']:.3f} 至 {row['原尺度95CI上限_百分点']:.3f}）。"
        )
    lines.extend(
        [
            "",
            f"整块似然比检验：孕周总体块 $P={test_map.loc[test_keys['孕周'],'P值']:.4g}$；BMI 总体块 $P={test_map.loc[test_keys['BMI总体'],'P值']:.4g}$；妇间 BMI 块/项 $P={test_map.loc[test_keys['妇间BMI'],'P值']:.4g}$；BMI 个体内项 $P={test_map.loc[test_keys['BMI个体内'],'P值']:.4g}$。因此论文中应把妇间 BMI 与个体内 BMI 分开叙述，不能只给一个总体 BMI 相关系数。",
            "",
            "上述置信区间是固定效应参数协方差的模拟区间，已经反变换到 Y 浓度原尺度；最终推断还应结合整簇自助法区间，避免只依赖渐近正态近似。",
            "",
            "## 5. 稳健性与诊断",
            "",
            f"所选模型收敛标志为 {int(mrow['收敛标志'])}，随机效应奇异标志为 {int(mrow['奇异随机效应标志'])}。留一孕妇分析中，核心效应方向翻转的删除次数为 {direction_flip_count}/{len(influence_ok)}，最大标准化系数变化为 {max_standardized:.3f} 个主模型标准误。整簇 bootstrap 的有效重复数与方向保持率见 `13_整簇自助法汇总.csv`。",
            "",
            "稳健性分析预先覆盖截至 25w+0、删除日期孕周偏差超 14 天事件、原始浓度尺度、加入辅助生殖标志、加入五项测序质量变量五个方向。辅助生殖项和质量变量都只用于稳健性检查，不作为主临床关系的核心解释。具体效应及方向一致性见 `11_稳健性结果表.csv`。",
            "",
            "异方差、残差偏度/峰度、VIF 与随机方差见 `09_模型诊断汇总.csv` 和 `10_共线性诊断.csv`。即使某项 P 值显著，也只能称为在当前样本、协变量与重复测量结构下的条件关联。",
            "",
            "## 6. 候选 A 的自我否决判断",
            "",
            self_denial,
            "",
            "候选 A 的优势是公式短、显著性检验清楚、原尺度效应可直接解释；主要风险是低阶函数可能不能表达复杂非线性，而且 logit-正态残差假设仍需与样条/GEE/Beta 等候选比较。主审只有在候选 A 不触发以下条件时才可选它：随机效应非奇异且收敛；关键效应方向在预设敏感性分析和整簇 bootstrap 中稳定；按孕妇分组的预测不明显差于更灵活候选；残差不存在无法解释的系统曲线。",
            "",
            f"本脚本运行耗时 {elapsed:.1f} 秒。所有数值均由 `01_运行候选A模型.py` 从冻结 CSV 一键重算；本目录未生成任何图像，建议图只写成 MATLAB-SVG 提示词 TXT。",
        ]
    )
    return "\n".join(lines) + "\n"


def 主程序():
    start = time.time()
    np.random.seed(随机种子)
    raw = 读取并核验数据()
    data, centers, quality_scales = 构造分析变量(raw, 重算BMI分解=True)

    fits, compare_df = 模型模型比较(data)
    tests_df = 整体检验(data, fits)
    cv_fold, cv_repeat, cv_summary, fold_map = 交叉验证(raw)
    selected_key, reasons = 选择候选A结构(compare_df, tests_df, cv_summary, cv_repeat)
    selected_spec = 模型规格表[selected_key]

    # 固定效应结构选定后用 REML 复拟合，参数估计和区间均来自该结果。
    final_result = 拟合混合模型(selected_spec, data, reml=True)
    effects_df = 关键效应表(final_result, data)
    params_df = 所有参数表(final_result)
    diagnostics_df, vif_df, event_diag_df = 诊断(final_result, data)
    robustness_df = 稳健性分析(raw, selected_spec, final_result, data)
    bootstrap_detail, bootstrap_summary = 整簇自助(raw, selected_spec, final_result, data)
    influence_df = 留一孕妇影响(raw, selected_spec, final_result, data)
    prediction_df = 预测关系表(final_result, data, centers, selected_spec)

    写CSV(compare_df, "03_模型比较表.csv")
    写CSV(tests_df, "04_整体显著性检验表.csv")
    写CSV(effects_df, "05_核心效应估计表.csv")
    写CSV(params_df, "06_全部固定效应参数表.csv")
    写CSV(cv_summary, "07_分组交叉验证汇总.csv")
    写CSV(cv_repeat, "07a_分组交叉验证逐轮.csv")
    写CSV(cv_fold, "07b_分组交叉验证逐折.csv")
    写CSV(fold_map, "08_分组交叉验证折分名单.csv")
    写CSV(diagnostics_df, "09_模型诊断汇总.csv")
    写CSV(vif_df, "10_共线性诊断.csv")
    写CSV(robustness_df, "11_稳健性结果表.csv")
    写CSV(bootstrap_detail, "12_整簇自助法逐次.csv")
    写CSV(bootstrap_summary, "13_整簇自助法汇总.csv")
    写CSV(influence_df, "14_留一孕妇影响分析.csv")
    写CSV(event_diag_df, "15_模型诊断逐事件.csv")
    写CSV(prediction_df, "16_预测关系数据.csv")

    elapsed = time.time() - start
    report = 生成报告(
        selected_key,
        reasons,
        compare_df,
        tests_df,
        cv_summary,
        effects_df,
        diagnostics_df,
        robustness_df,
        bootstrap_summary,
        influence_df,
        elapsed,
    )
    写文本(report, "02_候选A建模报告.md")

    prompt1 = f"""图名：第一问候选A——孕周与Y染色体浓度的调整后关系（不同妇间BMI）
数据文件：{候选目录 / '16_预测关系数据.csv'}
仅在后续统一制图阶段执行；当前阶段禁止生成图像。

请用 MATLAB 读取 UTF-8 CSV。横轴为“孕周（周）”，纵轴为“预测Y染色体浓度（%）”；把预测Y浓度、固定效应均值95CI上下限乘100。按妇间BMI参考值28、32、36、40分别画4条连续实线，每条线配同色半透明95%置信带。图中加入4%水平虚线，但不要把该阈值解释成问题一的模型因变量。图例用“BMI=28”等中文标签；不绘制原始散点以免遮挡，可在图注写“613个抽血事件，167名孕妇；模型仅用固定效应预测”。配色使用色盲友好的深蓝、青绿、橙、紫；白底；中文字体优先Microsoft YaHei，英数Times New Roman；宽16 cm、高10 cm、300 dpi预览。坐标范围由数据决定但Y轴下限不得小于0。使用 exportgraphics(...,'ContentType','vector') 输出纯矢量 SVG，文件名：第一问_候选A_调整后孕周BMI关系.svg。不得把线外区域外推到孕周10—25周之外。
"""
    写文本(prompt1, "17_制图提示词_调整后关系曲线.txt")

    prompt2 = f"""图名：第一问候选A——混合模型残差与影响诊断
数据文件：{候选目录 / '15_模型诊断逐事件.csv'}
仅在后续统一制图阶段执行；当前阶段禁止生成图像。

请用 MATLAB 生成2×2诊断图：(a) 固定效应预测Y浓度(%)与实测Y浓度(%)，加45度参考线；(b) 条件拟合logit与标准化条件残差，加y=0虚线和局部平滑趋势；(c) 标准化条件残差QQ图及正态参考线；(d) 按孕周分箱后残差箱线图，分箱仅用于诊断展示，不改变模型。点按孕妇代码着色会造成颜色过多，因此统一使用半透明深蓝点；绝对标准化残差>3的事件以红色空心圆标记并标注抽血事件键。中文字体Microsoft YaHei，白底，宽18 cm、高14 cm。使用 exportgraphics(...,'ContentType','vector') 输出纯矢量SVG，文件名：第一问_候选A_残差诊断.svg。图注必须写“条件残差含估计随机效应；诊断不用于重新筛选样本”。
"""
    写文本(prompt2, "18_制图提示词_残差诊断.txt")

    # 最后写清单；清单不包含自身哈希，避免自引用不稳定。
    outputs = sorted(p for p in 候选目录.iterdir() if p.is_file() and p.name != "19_复现信息与文件哈希.json")
    manifest = {
        "随机种子": 随机种子,
        "交叉验证": f"{交叉验证重复数}次重复×{交叉验证折数}折，按孕妇整组划分",
        "整簇自助次数": 整簇自助次数,
        "候选A推荐模型代号": selected_key,
        "主样本事件数": len(raw),
        "主样本孕妇数": int(raw["孕妇代码"].nunique()),
        "变量中心值": centers,
        "质量变量标准化参数": {k: {"均值": v[0], "标准差": v[1]} for k, v in quality_scales.items()},
        "输入冻结数据": str(输入文件),
        "输入冻结数据SHA256": 文件哈希(输入文件),
        "原始工作簿": str(源工作簿),
        "原始工作簿SHA256": 文件哈希(源工作簿),
        "Python": sys.version,
        "平台": platform.platform(),
        "依赖版本": {
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "statsmodels": statsmodels.__version__,
            "patsy": patsy.__version__,
            "scikit-learn": sklearn.__version__,
        },
        "输出文件SHA256": {p.name: 文件哈希(p) for p in outputs},
        "未生成图像": True,
    }
    (候选目录 / "19_复现信息与文件哈希.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"状态": "完成", "推荐模型": selected_key, "耗时秒": round(time.time() - start, 3)}, ensure_ascii=False))


if __name__ == "__main__":
    主程序()
