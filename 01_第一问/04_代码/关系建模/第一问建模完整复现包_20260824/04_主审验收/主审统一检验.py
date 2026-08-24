from __future__ import annotations

import argparse
import hashlib
import json
import math
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
from scipy.special import expit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor


默认数据 = (
    Path(__file__).resolve().parents[1]
    / "00_共同口径"
    / "冻结数据"
    / "第一问主模型冻结样本.csv"
)
默认输出目录 = Path(__file__).resolve().parent / "统一复核结果"


def 文件哈希(path: Path) -> str:
    摘要 = hashlib.sha256()
    with path.open("rb") as 文件:
        for 数据块 in iter(lambda: 文件.read(1024 * 1024), b""):
            摘要.update(数据块)
    return 摘要.hexdigest()


def 读取并准备(path: Path) -> pd.DataFrame:
    source = pd.read_csv(path)
    rename = {
        "孕妇代码": "woman",
        "孕周数": "week",
        "孕妇体质指数_BMI": "bmi",
        "年龄": "age",
        "生产次数": "parity",
        "辅助生殖标志": "art",
        "Y染色体浓度均值": "y_mean",
        "Y染色体浓度中位数": "y_median",
        "纳入截至25周0天敏感性标志": "within25",
        "任一记录日期孕周偏差超14天标志": "date_bad",
        "GC含量均值": "gc",
        "原始读段数均值": "reads",
        "比对比例均值": "map_ratio",
        "重复读段比例均值": "dup_ratio",
        "过滤读段比例均值": "filter_ratio",
    }
    missing = [列 for 列 in rename if 列 not in source.columns]
    if missing:
        raise RuntimeError(f"冻结数据缺列：{missing}")
    data = source.rename(columns=rename)
    if len(data) != 613 or data["woman"].nunique() != 167:
        raise RuntimeError("主模型冻结样本必须为613个事件、167名孕妇。")
    return 准备子样本(data)


def 准备子样本(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy().reset_index(drop=True)
    data["bmi_between"] = data.groupby("woman")["bmi"].transform("mean")
    data["bmi_within"] = data["bmi"] - data["bmi_between"]
    data["week_c"] = data["week"] - 18.0
    data["bmi_between_c"] = data["bmi_between"] - 32.0
    data["age_c"] = data["age"] - 29.0
    data["logit_y"] = np.log(data["y_mean"] / (1.0 - data["y_mean"]))
    data["logit_y_median"] = np.log(data["y_median"] / (1.0 - data["y_median"]))
    for column in ["gc", "reads", "map_ratio", "dup_ratio", "filter_ratio"]:
        sd = float(data[column].std(ddof=1))
        data[column + "_z"] = (data[column] - float(data[column].mean())) / sd
    return data


def 拟合混合模型(formula: str, data: pd.DataFrame, random_formula: str, reml: bool = False):
    model = smf.mixedlm(
        formula,
        data,
        groups=data["woman"],
        re_formula=random_formula,
        missing="raise",
    )
    errors = []
    for method, maxiter in [("lbfgs", 3000), ("powell", 5000)]:
        try:
            result = model.fit(reml=reml, method=method, maxiter=maxiter, disp=False)
            if result.converged:
                return result, method, ";".join(errors)
            errors.append(f"{method}:未收敛")
        except Exception as exc:
            errors.append(f"{method}:{type(exc).__name__}:{exc}")
    raise RuntimeError("；".join(errors))


def 原尺度预测(result, data: pd.DataFrame, response_scale: str) -> np.ndarray:
    eta = np.asarray(result.predict(data), dtype=float)
    return eta if response_scale == "原Y尺度" else expit(eta)


def 诊断(result, data: pd.DataFrame, response_scale: str) -> dict[str, float]:
    resid = np.asarray(result.resid, dtype=float)
    fitted = np.asarray(result.fittedvalues, dtype=float)
    bp_x = np.column_stack(
        [np.ones(len(data)), fitted, data["week_c"], data["bmi_between_c"], data["bmi_within"]]
    )
    bp = het_breuschpagan(resid, bp_x)
    pred_y = 原尺度预测(result, data, response_scale)
    fixed_link = np.asarray(result.model.exog @ result.fe_params, dtype=float)
    fixed_var = float(np.var(fixed_link, ddof=1))
    if result.cov_re.shape == (2, 2):
        z = np.column_stack([np.ones(len(data)), data["week_c"]])
    else:
        z = np.ones((len(data), 1))
    random_var = float(
        np.mean(np.einsum("ij,jk,ik->i", z, result.cov_re.to_numpy(), z))
    )
    residual_var = float(result.scale)
    total = fixed_var + random_var + residual_var
    return {
        "条件残差标准差": float(np.std(resid, ddof=1)),
        "条件残差偏度": float(stats.skew(resid)),
        "条件残差超额峰度": float(stats.kurtosis(resid)),
        "Shapiro检验P值": float(stats.shapiro(resid).pvalue),
        "Breusch-Pagan检验P值": float(bp[1]),
        "绝对残差与拟合值Spearman相关": float(stats.spearmanr(np.abs(resid), fitted).statistic),
        "大于3倍残差标准差事件数": int(np.sum(np.abs(resid) > 3 * np.std(resid, ddof=1))),
        "边际R2": fixed_var / total,
        "条件R2": (fixed_var + random_var) / total,
        "固定效应预测最小Y浓度": float(np.min(pred_y)),
        "固定效应预测最大Y浓度": float(np.max(pred_y)),
        "预测越出0到1范围数": int(np.sum((pred_y <= 0) | (pred_y >= 1))),
    }


def 随机结构指标(result) -> dict[str, float]:
    cov = result.cov_re.to_numpy()
    eig = np.linalg.eigvalsh(cov)
    output = {
        "随机效应协方差最小最大特征值比": float(eig.min() / eig.max()),
        "随机截距标准差": float(math.sqrt(cov[0, 0])),
        "残差标准差": float(math.sqrt(result.scale)),
    }
    if cov.shape == (2, 2):
        output["随机孕周斜率标准差"] = float(math.sqrt(cov[1, 1]))
        output["随机截距斜率相关"] = float(cov[0, 1] / math.sqrt(cov[0, 0] * cov[1, 1]))
        output["18周处ICC"] = float(cov[0, 0] / (cov[0, 0] + result.scale))
    else:
        output["随机孕周斜率标准差"] = np.nan
        output["随机截距斜率相关"] = np.nan
        output["18周处ICC"] = float(cov[0, 0] / (cov[0, 0] + result.scale))
    return output


def 固定效应VIF(result) -> float:
    x = np.asarray(result.model.exog, dtype=float)
    names = list(result.model.exog_names)
    values = []
    for index, name in enumerate(names):
        if name == "Intercept":
            continue
        values.append(variance_inflation_factor(x, index))
    return float(max(values))


def 参照预测(result, response_scale: str, week: float, bmi_between: float, bmi_within: float = 0.0) -> float:
    row = pd.DataFrame(
        {
            "week": [week],
            "week_c": [week - 18.0],
            "bmi_between": [bmi_between],
            "bmi_between_c": [bmi_between - 32.0],
            "bmi_within": [bmi_within],
            "age": [29.0],
            "age_c": [0.0],
            "parity": [0.0],
            "art": [0.0],
            "gc_z": [0.0],
            "reads_z": [0.0],
            "map_ratio_z": [0.0],
            "dup_ratio_z": [0.0],
            "filter_ratio_z": [0.0],
        }
    )
    eta = float(result.predict(row).iloc[0])
    return eta if response_scale == "原Y尺度" else float(expit(eta))


def 场景效应(result, response_scale: str) -> dict[str, float]:
    week12 = 参照预测(result, response_scale, 12.0, 32.0)
    week20 = 参照预测(result, response_scale, 20.0, 32.0)
    week24 = 参照预测(result, response_scale, 24.0, 32.0)
    bmi0 = 参照预测(result, response_scale, 18.0, 32.0)
    bmi_between1 = 参照预测(result, response_scale, 18.0, 33.0)
    bmi_within1 = 参照预测(result, response_scale, 18.0, 32.0, 1.0)
    quadratic_name = next(
        (name for name in result.fe_params.index if "week_c ** 2" in name), None
    )
    if quadratic_name is None:
        turning = np.nan
    else:
        turning = 18.0 - float(result.fe_params["week_c"]) / (
            2 * float(result.fe_params[quadratic_name])
        )
    return {
        "12周预测Y浓度": week12,
        "20周预测Y浓度": week20,
        "24周预测Y浓度": week24,
        "12至20周预测变化": week20 - week12,
        "12至24周预测变化": week24 - week12,
        "妇间BMI增加1的18周预测变化": bmi_between1 - bmi0,
        "个体内BMI增加1的18周预测变化": bmi_within1 - bmi0,
        "曲线最低点孕周": turning,
    }


def 分组交叉验证(data: pd.DataFrame, candidates: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
    fold_rows = []
    seeds = [20250824, 20250825, 20250826, 20250827, 20250828]
    for seed in seeds:
        splitter = GroupKFold(n_splits=5, shuffle=True, random_state=seed)
        splits = list(splitter.split(data, groups=data["woman"]))
        for candidate in candidates:
            for fold, (train_index, test_index) in enumerate(splits, 1):
                train = data.iloc[train_index].copy()
                test = data.iloc[test_index].copy()
                try:
                    result, method, warning = 拟合混合模型(
                        candidate["公式"], train, candidate["随机公式"], reml=False
                    )
                    prediction = 原尺度预测(result, test, candidate["响应尺度"])
                    rmse = math.sqrt(mean_squared_error(test["y_mean"], prediction))
                    mae = mean_absolute_error(test["y_mean"], prediction)
                    r2 = r2_score(test["y_mean"], prediction)
                    converged = 1
                    error = warning
                except Exception as exc:
                    rmse = mae = r2 = np.nan
                    converged = 0
                    method = ""
                    error = f"{type(exc).__name__}:{exc}"
                fold_rows.append(
                    {
                        "随机种子": seed,
                        "折号": fold,
                        "候选模型": candidate["名称"],
                        "测试事件数": len(test),
                        "测试孕妇数": test["woman"].nunique(),
                        "收敛": converged,
                        "优化器": method,
                        "RMSE": rmse,
                        "MAE": mae,
                        "R2": r2,
                        "异常": error,
                    }
                )
    folds = pd.DataFrame(fold_rows)
    summary = (
        folds.groupby("候选模型", as_index=False)
        .agg(
            交叉验证折数=("折号", "size"),
            收敛折数=("收敛", "sum"),
            RMSE均值=("RMSE", "mean"),
            RMSE标准差=("RMSE", "std"),
            MAE均值=("MAE", "mean"),
            MAE标准差=("MAE", "std"),
            R2均值=("R2", "mean"),
            R2标准差=("R2", "std"),
        )
        .sort_values("RMSE均值")
    )
    return folds, summary


def 主程序(data_path: Path, output_dir: Path) -> None:
    warnings.filterwarnings("ignore")
    output_dir.mkdir(parents=True, exist_ok=True)
    data = 读取并准备(data_path)

    base = "bmi_between_c + bmi_within + age_c + parity"
    candidates = [
        {"名称": "原尺度线性随机斜率", "公式": f"y_mean ~ week_c + {base}", "响应尺度": "原Y尺度", "随机公式": "1+week_c"},
        {"名称": "原尺度二次随机斜率", "公式": f"y_mean ~ week_c + I(week_c**2) + {base}", "响应尺度": "原Y尺度", "随机公式": "1+week_c"},
        {"名称": "原尺度样条随机斜率", "公式": f"y_mean ~ cr(week, df=3, constraints='center') + {base}", "响应尺度": "原Y尺度", "随机公式": "1+week_c"},
        {"名称": "logit线性随机斜率", "公式": f"logit_y ~ week_c + {base}", "响应尺度": "logit尺度", "随机公式": "1+week_c"},
        {"名称": "logit二次随机截距", "公式": f"logit_y ~ week_c + I(week_c**2) + {base}", "响应尺度": "logit尺度", "随机公式": "1"},
        {"名称": "logit二次随机斜率", "公式": f"logit_y ~ week_c + I(week_c**2) + {base}", "响应尺度": "logit尺度", "随机公式": "1+week_c"},
        {"名称": "logit样条随机斜率", "公式": f"logit_y ~ cr(week, df=3, constraints='center') + {base}", "响应尺度": "logit尺度", "随机公式": "1+week_c"},
    ]

    fit_rows = []
    fitted = {}
    for candidate in candidates:
        result, method, warning = 拟合混合模型(
            candidate["公式"], data, candidate["随机公式"], reml=False
        )
        fitted[candidate["名称"]] = result
        row = {
            "候选模型": candidate["名称"],
            "响应尺度": candidate["响应尺度"],
            "随机结构": "随机截距+孕周随机斜率" if "week_c" in candidate["随机公式"] else "随机截距",
            "固定效应参数数": len(result.fe_params),
            "总参数数": len(result.params) + 1,
            "收敛": int(result.converged),
            "优化器": method,
            "对数似然": result.llf,
            "AIC": result.aic,
            "BIC": result.bic,
            "最大VIF": 固定效应VIF(result),
            "拟合异常": warning,
        }
        row.update(随机结构指标(result))
        row.update(诊断(result, data, candidate["响应尺度"]))
        fit_rows.append(row)
    fit_table = pd.DataFrame(fit_rows)

    folds, cv_summary = 分组交叉验证(data, candidates)
    fit_table = fit_table.merge(cv_summary, on="候选模型", how="left")
    fit_table.rename(
        columns={"AIC": "赤池信息准则_AIC", "BIC": "贝叶斯信息准则_BIC"}
    ).to_csv(output_dir / "01_候选模型统一比较.csv", index=False, encoding="utf-8-sig")
    folds.rename(
        columns={
            "RMSE": "均方根误差_RMSE",
            "MAE": "平均绝对误差_MAE",
            "R2": "决定系数_R2",
        }
    ).to_csv(output_dir / "02_按孕妇分组交叉验证逐折.csv", index=False, encoding="utf-8-sig")
    cv_summary.to_csv(output_dir / "03_按孕妇分组交叉验证汇总.csv", index=False, encoding="utf-8-sig")

    main_name = "logit二次随机斜率"
    main = fitted[main_name]
    conf = main.conf_int()
    parameter_labels = {
        "Intercept": "截距",
        "week_c": "孕周一次项",
        "I(week_c ** 2)": "孕周二次项",
        "bmi_between_c": "妇间BMI效应",
        "bmi_within": "个体内BMI效应",
        "age_c": "年龄效应",
        "parity": "生产次数效应",
    }
    coef_rows = []
    for name, value in main.fe_params.items():
        coef_rows.append(
            {
                "参数": parameter_labels.get(name, name),
                "内部参数名": name,
                "估计值_logit尺度": value,
                "标准误": main.bse_fe[name],
                "Wald_P值": main.pvalues[name],
                "95%置信区间下限": conf.loc[name, 0],
                "95%置信区间上限": conf.loc[name, 1],
                "比值比": math.exp(value),
            }
        )
    pd.DataFrame(coef_rows).to_csv(output_dir / "04_推荐主线固定效应.csv", index=False, encoding="utf-8-sig")

    full_formula = next(c["公式"] for c in candidates if c["名称"] == main_name)
    reduced = {
        "孕周总体（二次项整体）": f"logit_y ~ {base}",
        "孕周非线性（二次项）": f"logit_y ~ week_c + {base}",
        "妇间BMI": "logit_y ~ week_c + I(week_c**2) + bmi_within + age_c + parity",
        "个体内BMI": "logit_y ~ week_c + I(week_c**2) + bmi_between_c + age_c + parity",
        "临床调整块": "logit_y ~ week_c + I(week_c**2) + bmi_between_c + bmi_within",
        "年龄": "logit_y ~ week_c + I(week_c**2) + bmi_between_c + bmi_within + parity",
        "生产次数": "logit_y ~ week_c + I(week_c**2) + bmi_between_c + bmi_within + age_c",
    }
    lrt_rows = []
    for label, formula in reduced.items():
        red, _, warning = 拟合混合模型(formula, data, "1+week_c", reml=False)
        degrees = len(main.fe_params) - len(red.fe_params)
        lr = 2 * (main.llf - red.llf)
        lrt_rows.append(
            {
                "检验项": label,
                "似然比统计量": lr,
                "自由度": degrees,
                "P值": stats.chi2.sf(lr, degrees),
                "0.05显著性结论": "显著" if stats.chi2.sf(lr, degrees) < 0.05 else "不显著",
                "简化模型拟合异常": warning,
            }
        )
    ri = fitted["logit二次随机截距"]
    lrt_rows.append(
        {
            "检验项": "随机斜率结构（方差参数在边界，P值不作主依据）",
            "似然比统计量": 2 * (main.llf - ri.llf),
            "自由度": 2,
            "P值": np.nan,
            "0.05显著性结论": "按AIC/BIC、非奇异和不恶化CV保留",
            "简化模型拟合异常": "随机效应方差检验存在边界问题",
        }
    )
    pd.DataFrame(lrt_rows).to_csv(output_dir / "05_推荐主线整体显著性检验.csv", index=False, encoding="utf-8-sig")

    sensitivity_specs = []
    sensitivity_specs.append(("主模型", data, full_formula, "logit尺度", "1+week_c"))
    sensitivity_specs.append(("截至25周0天", 准备子样本(data.loc[data["within25"] == 1]), full_formula, "logit尺度", "1+week_c"))
    sensitivity_specs.append(("排除日期孕周偏差超14天事件", 准备子样本(data.loc[data["date_bad"] == 0]), full_formula, "logit尺度", "1+week_c"))
    median_formula = full_formula.replace("logit_y ~", "logit_y_median ~")
    sensitivity_specs.append(("抽血事件内Y浓度改用中位数", data, median_formula, "logit尺度", "1+week_c"))
    raw_formula = full_formula.replace("logit_y ~", "y_mean ~")
    sensitivity_specs.append(("原Y尺度替代", data, raw_formula, "原Y尺度", "1+week_c"))
    sensitivity_specs.append(("加入稀疏辅助生殖标志", data, full_formula + " + art", "logit尺度", "1+week_c"))
    quality_formula = full_formula + " + gc_z + reads_z + map_ratio_z + dup_ratio_z + filter_ratio_z"
    sensitivity_specs.append(("加入测序质量调整块", data, quality_formula, "logit尺度", "1+week_c"))
    no_clinical = "logit_y ~ week_c + I(week_c**2) + bmi_between_c + bmi_within"
    sensitivity_specs.append(("不加入年龄与生产次数", data, no_clinical, "logit尺度", "1+week_c"))
    sensitivity_specs.append(("仅随机截距", data, full_formula, "logit尺度", "1"))

    sensitivity_rows = []
    for label, subset, formula, scale, random_formula in sensitivity_specs:
        result, method, warning = 拟合混合模型(formula, subset, random_formula, reml=False)
        effect = 场景效应(result, scale)
        row = {
            "敏感性场景": label,
            "事件数": len(subset),
            "孕妇数": subset["woman"].nunique(),
            "响应尺度": scale,
            "随机结构": "随机截距+孕周随机斜率" if "week_c" in random_formula else "随机截距",
            "收敛": int(result.converged),
            "优化器": method,
            "AIC": result.aic,
            "BIC": result.bic,
            "孕周一次项估计": result.fe_params.get("week_c", np.nan),
            "孕周二次项估计": result.fe_params.get("I(week_c ** 2)", np.nan),
            "妇间BMI估计": result.fe_params.get("bmi_between_c", np.nan),
            "妇间BMI_P值": result.pvalues.get("bmi_between_c", np.nan),
            "个体内BMI估计": result.fe_params.get("bmi_within", np.nan),
            "个体内BMI_P值": result.pvalues.get("bmi_within", np.nan),
            "年龄估计": result.fe_params.get("age_c", np.nan),
            "生产次数估计": result.fe_params.get("parity", np.nan),
            "拟合异常": warning,
        }
        row.update(effect)
        row.update(诊断(result, subset, scale))
        sensitivity_rows.append(row)
    pd.DataFrame(sensitivity_rows).rename(
        columns={"AIC": "赤池信息准则_AIC", "BIC": "贝叶斯信息准则_BIC"}
    ).to_csv(output_dir / "06_推荐主线稳健性分析.csv", index=False, encoding="utf-8-sig")

    diagnostic_rows = []
    main_diag = 诊断(main, data, "logit尺度")
    main_diag.update(随机结构指标(main))
    main_diag["最大VIF"] = 固定效应VIF(main)
    for key, value in main_diag.items():
        diagnostic_rows.append({"诊断指标": key, "数值": value})
    pd.DataFrame(diagnostic_rows).to_csv(output_dir / "07_推荐主线诊断汇总.csv", index=False, encoding="utf-8-sig")

    event_diagnostic = pd.DataFrame(
        {
            "抽血事件键": data["抽血事件键"],
            "孕妇代码": data["woman"],
            "孕周数": data["week"],
            "孕妇平均BMI": data["bmi_between"],
            "BMI个体内偏差": data["bmi_within"],
            "观测Y浓度": data["y_mean"],
            "固定效应预测Y浓度": expit(np.asarray(main.predict(data), dtype=float)),
            "条件预测Y浓度": expit(np.asarray(main.fittedvalues, dtype=float)),
            "条件残差_logit尺度": np.asarray(main.resid, dtype=float),
            "标准化条件残差": np.asarray(main.resid, dtype=float) / math.sqrt(main.scale),
        }
    )
    event_diagnostic.to_csv(
        output_dir / "09_推荐主线逐事件诊断.csv", index=False, encoding="utf-8-sig"
    )

    decision = pd.DataFrame(
        [
            ("孕周函数", "二次项", "线性模型整体拟合和组外误差较差；3自由度样条与二次项组外误差近似，但二次项参数更少、BIC更优"),
            ("BMI函数", "妇间线性+个体内线性", "妇间BMI二次与孕周交互未获整体检验和分组验证支持"),
            ("响应尺度", "logit(Y)", "原尺度RMSE略低，但存在明显异方差；logit保持预测边界、MAE更低且残差异方差显著减弱"),
            ("随机结构", "孕妇随机截距+孕周随机斜率", "ML AIC/BIC明显改善、协方差非奇异，新孕妇固定效应CV未发生有意义恶化"),
            ("主临床调整", "年龄+生产次数", "二者无缺失；怀孕次数缺失167/613，辅助生殖仅3名孕妇，后两者不进主调整"),
            ("测序质量变量", "仅敏感性调整", "用于检验核心效应稳定，不解释为生物学关系"),
        ],
        columns=["决策维度", "主审选择", "依据"],
    )
    decision.to_csv(output_dir / "08_主审模型选择决策.csv", index=False, encoding="utf-8-sig")

    files = sorted(path for path in output_dir.iterdir() if path.is_file() and path.name != "主审统一检验运行清单.json")
    manifest = {
        "运行时间": datetime.now().astimezone().isoformat(timespec="seconds"),
        "输入文件": str(data_path),
        "输入文件SHA256": 文件哈希(data_path),
        "事件数": len(data),
        "孕妇数": data["woman"].nunique(),
        "推荐主线": main_name,
        "推荐主线公式": full_formula,
        "交叉验证": "5个随机种子×5折；按孕妇分组；测试孕妇预测只用固定效应",
        "输出文件SHA256": {path.name: 文件哈希(path) for path in files},
    }
    (output_dir / "主审统一检验运行清单.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="第一问候选模型统一复核，不绘图。")
    parser.add_argument("--数据", type=Path, default=默认数据)
    parser.add_argument("--输出目录", type=Path, default=默认输出目录)
    args = parser.parse_args()
    主程序(args.数据, args.输出目录)
