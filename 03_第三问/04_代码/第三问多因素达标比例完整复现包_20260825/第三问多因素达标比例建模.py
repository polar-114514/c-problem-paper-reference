from __future__ import annotations

import hashlib
import json
import math
import platform
import re
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from scipy.special import expit, logit
from sklearn.metrics import average_precision_score, roc_auc_score


脚本目录 = Path(__file__).resolve().parent


def 定位工作区(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / "00_题目与原始资料/02_原始数据/附件.xlsx").is_file():
            return candidate
    raise FileNotFoundError("无法从脚本位置向上找到含原始附件的C题论文工作区")


工作区 = 定位工作区(脚本目录)
输出根目录 = 脚本目录 / "正式候选输出"
数据输出目录 = 输出根目录 / "01_数据"
模型输出目录 = 输出根目录 / "02_模型结果"
验证输出目录 = 输出根目录 / "03_验证"
复现输出目录 = 输出根目录 / "04_复现"
图表提示词目录 = 输出根目录 / "05_图表提示词"

事件源路径 = 工作区 / "01_第一问/04_代码/关系建模/第一问建模完整复现包_20260825/00_共同口径/冻结数据/第一问抽血事件层冻结样本.csv"
记录源路径 = 工作区 / "01_第一问/04_代码/关系建模/第一问建模完整复现包_20260825/00_共同口径/冻结数据/第一问记录层冻结样本.csv"
题目路径 = 工作区 / "00_题目与原始资料/01_题目原文/C题.pdf"
原始工作簿路径 = 工作区 / "00_题目与原始资料/02_原始数据/附件.xlsx"
第二问运行清单路径 = 工作区 / "02_第二问/04_代码/第二问无自拟参数完整复现包_20260825/正式候选输出/04_复现/第二问运行清单.json"

达标阈值 = 0.04
日网格 = np.arange(70, 176, dtype=int)
主随机种子 = 20250825
最大重复次数 = 400
数值概率下限 = np.finfo(float).eps


候选规格 = {
    "第二问信息集_线性浓度混合": {"路线": "浓度混合", "多因素": False, "二次孕周": False, "主候选": False},
    "第二问信息集_线性达标GEE": {"路线": "达标GEE", "多因素": False, "二次孕周": False, "主候选": False},
    "第三问多因素_线性浓度混合": {"路线": "浓度混合", "多因素": True, "二次孕周": False, "主候选": True},
    "第三问多因素_线性达标GEE": {"路线": "达标GEE", "多因素": True, "二次孕周": False, "主候选": True},
    "第三问多因素_二次浓度混合": {"路线": "浓度混合", "多因素": True, "二次孕周": True, "主候选": True},
}


def 文件哈希(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def 中文化表头(name: str) -> str:
    translated = str(name)
    translated = translated.replace("Pareto", "帕累托")
    translated = translated.replace("AIC", "赤池信息准则")
    translated = translated.replace("BIC", "贝叶斯信息准则")
    translated = translated.replace("sigma", "尺度参数σ")
    translated = translated.replace("mu", "μ")
    translated = translated.replace("SHA256", "安全散列值_SHA256")
    translated = re.sub(r"(\d+)w\+(\d+)d", r"\1周\2天", translated)
    return translated


def 写CSV(frame: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    export = frame.rename(columns={column: 中文化表头(column) for column in frame.columns})
    export.to_csv(path, index=False, encoding="utf-8-sig")


def 写JSON(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def 周加天(day: float) -> str:
    day_int = int(round(day))
    return f"{day_int // 7}周{day_int % 7}天"


def 风险等级(day: int) -> str:
    if day <= 84:
        return "早期发现（12周以内，题面称风险较低）"
    if day >= 91:
        return "中期发现（13至27周，题面称风险高）"
    return "12周后至13周前（题面未单列）"


def 汇总技术复测(records: pd.DataFrame):
    group_columns = ["孕妇代码", "抽血次数", "检测日期规范值", "孕周原始值"]
    rows = []
    residual_sum = 0.0
    degrees = 0
    for keys, group in records.groupby(group_columns, dropna=False, sort=False):
        if len(group) < 2:
            continue
        values = group["Y染色体浓度"].to_numpy(float)
        mean = float(np.mean(values))
        residual_sum += float(np.sum((values - mean) ** 2))
        degrees += len(values) - 1
        row = dict(zip(group_columns, keys))
        row.update(
            {
                "记录数": int(len(values)),
                "Y浓度均值": mean,
                "Y浓度标准差": float(np.std(values, ddof=1)),
                "Y浓度最小值": float(np.min(values)),
                "Y浓度最大值": float(np.max(values)),
                "跨越4%阈值标志": int(float(np.min(values)) < 达标阈值 <= float(np.max(values))),
            }
        )
        rows.append(row)
    detail = pd.DataFrame(rows)
    pooled_sd = math.sqrt(residual_sum / degrees)
    lower = pooled_sd * math.sqrt(degrees / stats.chi2.ppf(0.975, degrees))
    upper = pooled_sd * math.sqrt(degrees / stats.chi2.ppf(0.025, degrees))
    summary = {
        "严格技术复测组数": int(len(detail)),
        "严格技术复测记录数": int(detail["记录数"].sum()),
        "合并组内标准差": pooled_sd,
        "误差自由度": int(degrees),
        "跨越4%阈值组数": int(detail["跨越4%阈值标志"].sum()),
        "标准差95%区间下限": lower,
        "标准差95%区间上限": upper,
    }
    return detail, summary


def 读取并审计数据():
    raw_events = pd.read_csv(事件源路径)
    raw_records = pd.read_csv(记录源路径)
    events = raw_events.loc[raw_events["纳入主模型标志"].eq(1)].copy()
    events = events.sort_values(["孕妇代码", "孕周数", "抽血次数", "抽血事件键"]).reset_index(drop=True)

    baseline_columns = [
        "孕妇代码",
        "年龄",
        "身高",
        "体重",
        "孕妇体质指数_BMI",
        "受孕方式",
        "辅助生殖标志",
        "怀孕次数",
        "生产次数",
    ]
    baseline = events.drop_duplicates("孕妇代码", keep="first")[baseline_columns].copy()
    baseline = baseline.rename(
        columns={
            "年龄": "首次年龄",
            "身高": "首次身高",
            "体重": "首次体重",
            "孕妇体质指数_BMI": "首次BMI",
            "受孕方式": "首次受孕方式",
            "辅助生殖标志": "首次辅助生殖标志",
            "怀孕次数": "首次怀孕次数",
            "生产次数": "首次生产次数",
        }
    )
    events = events.merge(baseline, on="孕妇代码", how="left", validate="many_to_one")
    events["达到4%标志"] = (events["Y染色体浓度均值"] >= 达标阈值).astype(int)
    events["Y模型值"] = events["Y染色体浓度均值"].astype(float)
    keep = [
        "孕妇代码",
        "抽血事件键",
        "抽血次数",
        "孕周天数",
        "孕周数",
        "年龄",
        "身高",
        "体重",
        "孕妇体质指数_BMI",
        "首次年龄",
        "首次身高",
        "首次体重",
        "首次BMI",
        "首次受孕方式",
        "首次辅助生殖标志",
        "首次怀孕次数",
        "首次生产次数",
        "Y染色体浓度均值",
        "Y染色体浓度中位数",
        "Y染色体浓度最小值",
        "Y染色体浓度最大值",
        "Y模型值",
        "达到4%标志",
        "记录数",
        "检测会话数",
        "任一记录日期孕周偏差超14天标志",
    ]
    events = events[keep].rename(
        columns={
            "年龄": "事件年龄",
            "身高": "事件身高",
            "体重": "事件体重",
            "孕妇体质指数_BMI": "事件BMI",
            "记录数": "检测记录数",
            "任一记录日期孕周偏差超14天标志": "日期孕周偏差超14天标志",
        }
    )

    selected_records = raw_records.loc[raw_records["抽血事件键"].isin(set(events["抽血事件键"]))].copy()
    replicate_detail, replicate_summary = 汇总技术复测(selected_records)

    woman_variation = {}
    original_to_event = {
        "年龄": "事件年龄",
        "身高": "事件身高",
        "体重": "事件体重",
        "孕妇体质指数_BMI": "事件BMI",
        "受孕方式": "首次受孕方式",
        "辅助生殖标志": "首次辅助生殖标志",
        "怀孕次数": "首次怀孕次数",
        "生产次数": "首次生产次数",
    }
    for source_name, event_name in original_to_event.items():
        if event_name.startswith("首次"):
            source_series = raw_events.loc[raw_events["纳入主模型标志"].eq(1), ["孕妇代码", source_name]]
            counts = source_series.groupby("孕妇代码")[source_name].nunique(dropna=False)
        else:
            counts = events.groupby("孕妇代码")[event_name].nunique(dropna=False)
        woman_variation[source_name] = int((counts > 1).sum())

    missing_gravidity_women = int(baseline["首次怀孕次数"].isna().sum())
    assisted_women = int(baseline["首次辅助生殖标志"].eq(1).sum())
    bmi_formula = baseline["首次体重"] / (baseline["首次身高"] / 100.0) ** 2
    bmi_formula_max_error = float(np.max(np.abs(baseline["首次BMI"] - bmi_formula)))
    post_drop = 0
    for _, group in events.groupby("孕妇代码"):
        group = group.sort_values(["孕周数", "抽血次数", "抽血事件键"])
        hit = np.flatnonzero(group["达到4%标志"].to_numpy(int) == 1)
        if len(hit) and (group["达到4%标志"].to_numpy(int)[int(hit[0]) + 1 :] == 0).any():
            post_drop += 1

    audit = {
        "抽血事件数": int(len(events)),
        "孕妇数": int(events["孕妇代码"].nunique()),
        "达标事件数": int(events["达到4%标志"].sum()),
        "未达标事件数": int((1 - events["达到4%标志"]).sum()),
        "事件达标比例": float(events["达到4%标志"].mean()),
        "达标后回落孕妇数": int(post_drop),
        "最早观测孕周天数": int(events["孕周天数"].min()),
        "最晚观测孕周天数": int(events["孕周天数"].max()),
        "首次怀孕次数缺失孕妇数": missing_gravidity_women,
        "辅助生殖孕妇数": assisted_women,
        "BMI公式最大绝对误差": bmi_formula_max_error,
        "年龄孕妇内变化人数": woman_variation["年龄"],
        "身高孕妇内变化人数": woman_variation["身高"],
        "体重孕妇内变化人数": woman_variation["体重"],
        "BMI孕妇内变化人数": woman_variation["孕妇体质指数_BMI"],
        **replicate_summary,
    }
    expected = {
        "抽血事件数": 613,
        "孕妇数": 167,
        "达标后回落孕妇数": 32,
        "严格技术复测组数": 18,
        "严格技术复测记录数": 36,
        "误差自由度": 18,
    }
    for key, value in expected.items():
        if audit[key] != value:
            raise RuntimeError(f"数据断言失败：{key}={audit[key]}，预期={value}")
    if not bmi_formula_max_error < 1e-7:
        raise RuntimeError(f"BMI公式核对失败：最大绝对误差={bmi_formula_max_error}")

    role_rows = [
        ("计划检测孕周", "决策时可得", "主模型", "检测安排本身的决策变量"),
        ("首次BMI", "决策时可得", "主模型与最终分组", "取首次抽血事件记录，禁止使用未来平均BMI"),
        ("首次年龄", "决策时可得", "主模型", f"首次记录；{audit['年龄孕妇内变化人数']}名孕妇后续记录发生变化"),
        ("首次身高", "决策时可得", "主模型", f"首次记录；{audit['身高孕妇内变化人数']}名孕妇后续记录发生变化"),
        ("首次体重", "决策时可得", "结构审计与替代参数化", "与身高共同精确决定BMI，不与BMI同时入主模型"),
        ("首次生产次数", "决策时可得", "主模型", "全样本无缺失且孕妇内稳定"),
        ("首次怀孕次数", "决策时可得但缺失", "敏感性", f"缺失{missing_gravidity_women}/167名孕妇，不自拟填补"),
        ("辅助生殖标志", "决策时可得但极稀疏", "敏感性", f"仅{assisted_women}/167名孕妇为1"),
        ("本次及未来Y浓度/Z值", "检测后或未来", "禁止进入主预测", "目标或未来信息"),
        ("读段数、比对比例、GC与过滤比例", "检测后", "禁止进入主预测", "只能检测后质量审计"),
        ("后续体重、BMI及全程均值", "未来", "禁止进入主预测", "安排当前检测时不可得"),
    ]
    role_table = pd.DataFrame(role_rows, columns=["变量", "决策时可得性", "模型角色", "证据与处理"])
    return events, baseline, replicate_detail, audit, role_table


def 构建设计矩阵(data: pd.DataFrame, spec: dict, references: dict | None = None):
    variables = ["孕周数", "首次BMI"]
    if spec["多因素"]:
        variables += ["首次年龄", "首次身高"]
    if references is None:
        references = {}
        for variable in variables:
            center = float(data[variable].median())
            scale = float(data[variable].quantile(0.75) - data[variable].quantile(0.25))
            if not scale > 0:
                raise RuntimeError(f"{variable}四分位距为0")
            references[variable] = {"中心": center, "尺度": scale}
    columns = {"截距": np.ones(len(data))}
    week_z = (data["孕周数"].to_numpy(float) - references["孕周数"]["中心"]) / references["孕周数"]["尺度"]
    columns["孕周标准化"] = week_z
    if spec["二次孕周"]:
        columns["孕周标准化平方"] = week_z**2
    columns["首次BMI标准化"] = (
        data["首次BMI"].to_numpy(float) - references["首次BMI"]["中心"]
    ) / references["首次BMI"]["尺度"]
    if spec["多因素"]:
        columns["首次年龄标准化"] = (
            data["首次年龄"].to_numpy(float) - references["首次年龄"]["中心"]
        ) / references["首次年龄"]["尺度"]
        columns["首次身高标准化"] = (
            data["首次身高"].to_numpy(float) - references["首次身高"]["中心"]
        ) / references["首次身高"]["尺度"]
        columns["首次生产次数"] = data["首次生产次数"].to_numpy(float)
    return pd.DataFrame(columns, index=data.index), references


def 拟合模型(data: pd.DataFrame, spec: dict, robust_all: bool = False):
    X, references = 构建设计矩阵(data, spec)
    if spec["路线"] == "达标GEE":
        model = sm.GEE(
            data["达到4%标志"].to_numpy(int),
            X,
            groups=data["孕妇代码"].astype(str),
            family=sm.families.Binomial(),
            cov_struct=sm.cov_struct.Exchangeable(),
        )
        result = model.fit(maxiter=300)
        converged = bool(getattr(result, "converged", True))
        return {
            "路线": "达标GEE",
            "规格": spec,
            "参照": references,
            "结果": result,
            "收敛": converged,
            "优化器": "GEE迭代加权估计",
            "固定效应": pd.Series(np.asarray(result.params), index=X.columns),
            "固定效应标准误": pd.Series(np.asarray(result.bse), index=X.columns),
            "固定效应P值": pd.Series(np.asarray(result.pvalues), index=X.columns),
            "参数数": int(len(result.params) + 1),
            "对数似然": np.nan,
        }

    y = np.clip(data["Y模型值"].to_numpy(float), np.finfo(float).tiny, 1 - np.finfo(float).eps)
    endog = logit(y)
    week_z = X["孕周标准化"].to_numpy(float)
    random_X = np.column_stack([np.ones(len(data)), week_z])
    methods = ["lbfgs", "powell"] if robust_all else ["lbfgs", "powell"]
    fits = []
    errors = []
    for method in methods:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = sm.MixedLM(
                    endog,
                    X,
                    groups=data["孕妇代码"].astype(str),
                    exog_re=random_X,
                ).fit(reml=False, method=method, maxiter=2000, disp=False)
            if np.isfinite(result.llf):
                fits.append((method, result))
                if not robust_all and bool(result.converged):
                    break
        except Exception as exc:
            errors.append(f"{method}:{type(exc).__name__}:{exc}")
    if not fits:
        raise RuntimeError("混合模型全部优化失败：" + " | ".join(errors))
    method, result = max(fits, key=lambda item: float(item[1].llf))
    covariance = np.asarray(result.cov_re, dtype=float)
    if covariance.shape != (2, 2) or np.linalg.eigvalsh(covariance).min() < -1e-8:
        raise RuntimeError("随机效应协方差矩阵无效")
    fixed = pd.Series(np.asarray(result.fe_params), index=X.columns)
    bse = pd.Series(np.asarray(result.bse_fe), index=X.columns)
    pvalues = pd.Series(np.asarray(result.pvalues)[: len(X.columns)], index=X.columns)
    return {
        "路线": "浓度混合",
        "规格": spec,
        "参照": references,
        "结果": result,
        "收敛": bool(result.converged),
        "优化器": method,
        "优化错误": " | ".join(errors),
        "固定效应": fixed,
        "固定效应标准误": bse,
        "固定效应P值": pvalues,
        "随机效应协方差": covariance,
        "残差方差": float(result.scale),
        "参数数": int(len(X.columns) + 3 + 1),
        "对数似然": float(result.llf),
    }


def 预测概率(adapter: dict, data: pd.DataFrame, bmi_group_delta: float | None = None):
    X, _ = 构建设计矩阵(data, adapter["规格"], adapter["参照"])
    beta = adapter["固定效应"]
    eta = X.to_numpy(float) @ beta.to_numpy(float)
    if bmi_group_delta is not None:
        eta = eta - float(beta["首次BMI标准化"]) * X["首次BMI标准化"].to_numpy(float) + float(bmi_group_delta)
    if adapter["路线"] == "达标GEE":
        return expit(eta)
    week_z = X["孕周标准化"].to_numpy(float)
    random_X = np.column_stack([np.ones(len(data)), week_z])
    variance = adapter["残差方差"] + np.einsum(
        "ij,jk,ik->i", random_X, adapter["随机效应协方差"], random_X
    )
    threshold = logit(达标阈值)
    return stats.norm.cdf((eta - threshold) / np.sqrt(variance))


def 人群预测曲线(adapter: dict, baseline: pd.DataFrame):
    rows = []
    for day in 日网格:
        frame = baseline.copy()
        frame["孕周数"] = day / 7.0
        probability = 预测概率(adapter, frame)
        rows.append(
            {
                "检测孕周天数": int(day),
                "检测孕周_周加天": 周加天(day),
                "预计达标比例": float(np.mean(probability)),
                "预计尚未达标比例": float(1 - np.mean(probability)),
            }
        )
    curve = pd.DataFrame(rows)
    curve["相邻日达标比例变化"] = curve["预计达标比例"].diff()
    return curve


def 候选逐孕妇留一(events: pd.DataFrame):
    women = list(events["孕妇代码"].drop_duplicates())
    rows = []
    for sequence, woman in enumerate(women, 1):
        train = events.loc[events["孕妇代码"].ne(woman)].copy()
        test = events.loc[events["孕妇代码"].eq(woman)].copy()
        for name, spec in 候选规格.items():
            try:
                adapter = 拟合模型(train, spec, robust_all=False)
                prediction = 预测概率(adapter, test)
                valid = int(adapter["收敛"] and np.isfinite(prediction).all())
                error = ""
            except Exception as exc:
                prediction = np.full(len(test), np.nan)
                valid = 0
                error = f"{type(exc).__name__}:{exc}"
            for (_, row), probability in zip(test.iterrows(), prediction):
                rows.append(
                    {
                        "孕妇代码": woman,
                        "抽血事件键": row["抽血事件键"],
                        "候选模型": name,
                        "实际达标标志": int(row["达到4%标志"]),
                        "预测达标概率": float(probability),
                        "拟合有效标志": valid,
                        "失败信息": error,
                    }
                )
        if sequence % 20 == 0:
            print(f"逐孕妇留一 {sequence}/{len(women)}", flush=True)
    detail = pd.DataFrame(rows)
    clip = detail["预测达标概率"].clip(数值概率下限, 1 - 数值概率下限)
    detail["二分类对数损失"] = -(
        detail["实际达标标志"] * np.log(clip)
        + (1 - detail["实际达标标志"]) * np.log(1 - clip)
    )
    detail["Brier分数"] = (detail["实际达标标志"] - detail["预测达标概率"]) ** 2
    per_woman = (
        detail.groupby(["孕妇代码", "候选模型"], as_index=False)
        .agg(
            孕妇内平均对数损失=("二分类对数损失", "mean"),
            孕妇内平均Brier分数=("Brier分数", "mean"),
            孕妇事件数=("抽血事件键", "count"),
            全部拟合有效标志=("拟合有效标志", "min"),
        )
    )
    summaries = []
    for name, group in detail.groupby("候选模型", sort=False):
        person = per_woman.loc[per_woman["候选模型"].eq(name)]
        counts = group.groupby("孕妇代码").size()
        weights = group["孕妇代码"].map(lambda code: 1.0 / counts[code]).to_numpy(float)
        y = group["实际达标标志"].to_numpy(int)
        p = group["预测达标概率"].to_numpy(float)
        p_clip = np.clip(p, 数值概率下限, 1 - 数值概率下限)
        calibration_X = np.column_stack([np.ones(len(p)), logit(p_clip)])
        try:
            calibration = sm.GLM(
                y,
                calibration_X,
                family=sm.families.Binomial(),
                freq_weights=weights,
            ).fit()
            calibration_intercept = float(calibration.params[0])
            calibration_slope = float(calibration.params[1])
        except Exception:
            calibration_intercept = np.nan
            calibration_slope = np.nan
        summaries.append(
            {
                "候选模型": name,
                "留一验证孕妇数": int(person["孕妇代码"].nunique()),
                "留一验证事件数": int(len(group)),
                "有效孕妇数": int(person["全部拟合有效标志"].sum()),
                "逐孕妇平均对数损失": float(person["孕妇内平均对数损失"].mean()),
                "逐孕妇平均Brier分数": float(person["孕妇内平均Brier分数"].mean()),
                "孕妇等权ROC曲线下面积": float(roc_auc_score(y, p, sample_weight=weights)),
                "孕妇等权PR曲线下面积": float(average_precision_score(y, p, sample_weight=weights)),
                "孕妇等权实际达标比例": float(np.average(y, weights=weights)),
                "孕妇等权平均预测概率": float(np.average(p, weights=weights)),
                "校准截距": calibration_intercept,
                "校准斜率": calibration_slope,
            }
        )
    return detail, per_woman, pd.DataFrame(summaries)


def 比较多因素增量(per_woman: pd.DataFrame, selected_name: str):
    selected_spec = 候选规格[selected_name]
    if selected_spec["路线"] == "浓度混合":
        baseline_name = "第二问信息集_线性浓度混合"
    else:
        baseline_name = "第二问信息集_线性达标GEE"
    pivot_log = per_woman.pivot(index="孕妇代码", columns="候选模型", values="孕妇内平均对数损失")
    pivot_brier = per_woman.pivot(index="孕妇代码", columns="候选模型", values="孕妇内平均Brier分数")
    rows = []
    for metric_name, pivot in [("逐孕妇对数损失", pivot_log), ("逐孕妇Brier分数", pivot_brier)]:
        difference = pivot[selected_name] - pivot[baseline_name]
        interval = stats.t.interval(
            0.95,
            len(difference) - 1,
            loc=float(difference.mean()),
            scale=float(stats.sem(difference)),
        )
        rows.append(
            {
                "比较指标": metric_name,
                "多因素减同路线第二问信息集基准_均值": float(difference.mean()),
                "差值95%区间下限": float(interval[0]),
                "差值95%区间上限": float(interval[1]),
                "多因素损失更低孕妇数": int((difference < 0).sum()),
                "比较孕妇数": int(len(difference)),
                "基准模型": baseline_name,
                "多因素模型": selected_name,
            }
        )
    return pd.DataFrame(rows)


def 主模型共同参数数(adapter: dict) -> int:
    fixed_without_bmi = len(adapter["固定效应"]) - 1
    if adapter["路线"] == "浓度混合":
        return fixed_without_bmi + 3 + 1
    return fixed_without_bmi + 1


def 分段核心(adapter: dict, events: pd.DataFrame):
    X, _ = 构建设计矩阵(events, adapter["规格"], adapter["参照"])
    beta = adapter["固定效应"]
    bmi_effect = float(beta["首次BMI标准化"])
    eta_without_bmi = X.to_numpy(float) @ beta.to_numpy(float) - bmi_effect * X["首次BMI标准化"].to_numpy(float)
    if adapter["路线"] == "浓度混合":
        week_z = X["孕周标准化"].to_numpy(float)
        random_X = np.column_stack([np.ones(len(events)), week_z])
        sd = np.sqrt(
            adapter["残差方差"]
            + np.einsum("ij,jk,ik->i", random_X, adapter["随机效应协方差"], random_X)
        )
    else:
        sd = None

    baseline = events.drop_duplicates("孕妇代码", keep="first")[["孕妇代码", "首次BMI"]].copy()
    baseline = baseline.sort_values(["首次BMI", "孕妇代码"]).reset_index(drop=True)
    unique_bmi = np.sort(baseline["首次BMI"].unique())
    bmi_z_grid = (unique_bmi - adapter["参照"]["首次BMI"]["中心"]) / adapter["参照"]["首次BMI"]["尺度"]
    delta_grid = bmi_effect * bmi_z_grid
    y = events["达到4%标志"].to_numpy(int)
    woman_values = events["孕妇代码"].astype(str).to_numpy()
    person_cost_rows = []
    for woman in baseline["孕妇代码"].astype(str):
        index = np.flatnonzero(woman_values == woman)
        eta = eta_without_bmi[index, None] + delta_grid[None, :]
        if adapter["路线"] == "浓度混合":
            probability = stats.norm.cdf((eta - logit(达标阈值)) / sd[index, None])
        else:
            probability = expit(eta)
        probability = np.clip(probability, 数值概率下限, 1 - 数值概率下限)
        loss = -(
            y[index, None] * np.log(probability)
            + (1 - y[index, None]) * np.log(1 - probability)
        )
        person_cost_rows.append(np.mean(loss, axis=0))
    person_cost = np.vstack(person_cost_rows)
    person_bmi = baseline["首次BMI"].to_numpy(float)
    counts = np.array([(person_bmi == value).sum() for value in unique_bmi], dtype=int)
    start = np.r_[0, np.cumsum(counts)[:-1]]
    end = np.cumsum(counts)
    prefix = np.vstack([np.zeros((1, len(delta_grid))), np.cumsum(person_cost, axis=0)])
    m = len(unique_bmi)
    segment_cost = np.full((m, m), np.inf)
    segment_delta_index = np.zeros((m, m), dtype=int)
    for i in range(m):
        for j in range(i, m):
            costs = prefix[end[j]] - prefix[start[i]]
            best = int(np.argmin(costs))
            segment_cost[i, j] = float(costs[best])
            segment_delta_index[i, j] = best
    return baseline, unique_bmi, counts, delta_grid, segment_cost, segment_delta_index


def 重建分组(unique_bmi, counts, delta_grid, segment_delta_index, previous, k: int):
    segments = []
    j = len(unique_bmi)
    for level in range(k, 0, -1):
        i = int(previous[level, j])
        segments.append((i, j - 1, int(segment_delta_index[i, j - 1])))
        j = i
    segments.reverse()
    rows = []
    for group_number, (i, j, delta_index) in enumerate(segments, 1):
        low = float(unique_bmi[i])
        high = float(unique_bmi[j])
        next_low = float(unique_bmi[j + 1]) if j + 1 < len(unique_bmi) else np.nan
        cut = (high + next_low) / 2 if np.isfinite(next_low) else np.nan
        rows.append(
            {
                "组别": group_number,
                "组内最小BMI": low,
                "组内最大BMI": high,
                "与下一组切点": cut,
                "人数": int(counts[i : j + 1].sum()),
                "BMI组偏移": float(delta_grid[delta_index]),
                "偏移对应观测BMI": float(unique_bmi[delta_index]),
            }
        )
    return pd.DataFrame(rows)


def 全部组数分段(adapter: dict, events: pd.DataFrame):
    baseline, unique_bmi, counts, delta_grid, segment_cost, segment_delta_index = 分段核心(adapter, events)
    m = len(unique_bmi)
    dp = np.full((m + 1, m + 1), np.inf)
    previous = np.full((m + 1, m + 1), -1, dtype=int)
    dp[0, 0] = 0.0
    for k in range(1, m + 1):
        for j in range(k, m + 1):
            starts = np.arange(k - 1, j)
            values = dp[k - 1, starts] + segment_cost[starts, j - 1]
            best = int(np.argmin(values))
            dp[k, j] = float(values[best])
            previous[k, j] = int(starts[best])
    common = 主模型共同参数数(adapter)
    n = int(events["孕妇代码"].nunique())
    rows = []
    for k in range(1, m + 1):
        parameters = common + 2 * k - 1
        rows.append(
            {
                "组数K": k,
                "孕妇等权负对数似然": float(dp[k, m]),
                "参数计数": int(parameters),
                "BIC": float(2 * dp[k, m] + parameters * math.log(n)),
            }
        )
    table = pd.DataFrame(rows)
    best_k = int(table.loc[table["BIC"].idxmin(), "组数K"])
    best_groups = 重建分组(unique_bmi, counts, delta_grid, segment_delta_index, previous, best_k)
    two_groups = 重建分组(unique_bmi, counts, delta_grid, segment_delta_index, previous, 2)
    return table, best_groups, two_groups


def 一二组挑战(adapter: dict, events: pd.DataFrame):
    _, unique_bmi, counts, delta_grid, segment_cost, segment_delta_index = 分段核心(adapter, events)
    m = len(unique_bmi)
    common = 主模型共同参数数(adapter)
    n = int(events["孕妇代码"].nunique())
    k1_cost = float(segment_cost[0, m - 1])
    cut_costs = np.array([segment_cost[0, cut] + segment_cost[cut + 1, m - 1] for cut in range(m - 1)])
    cut_index = int(np.argmin(cut_costs))
    k2_cost = float(cut_costs[cut_index])
    previous1 = np.full((2, m + 1), -1, dtype=int)
    previous1[1, m] = 0
    group1 = 重建分组(unique_bmi, counts, delta_grid, segment_delta_index, previous1, 1)
    previous2 = np.full((3, m + 1), -1, dtype=int)
    previous2[2, m] = cut_index + 1
    previous2[1, cut_index + 1] = 0
    group2 = 重建分组(unique_bmi, counts, delta_grid, segment_delta_index, previous2, 2)
    bic1 = 2 * k1_cost + (common + 1) * math.log(n)
    bic2 = 2 * k2_cost + (common + 3) * math.log(n)
    return bic1, bic2, group1, group2


def 整理分组表(groups: pd.DataFrame, role: str, events: pd.DataFrame):
    out = groups.sort_values("组别").reset_index(drop=True).copy()
    support_min = float(events["首次BMI"].min())
    support_max = float(events["首次BMI"].max())
    interval_text = []
    lower = support_min
    for _, row in out.iterrows():
        upper = float(row["与下一组切点"]) if pd.notna(row["与下一组切点"]) else support_max
        bracket = ")" if pd.notna(row["与下一组切点"]) else "]"
        interval_text.append(f"[{lower:.6f},{upper:.6f}{bracket}")
        lower = upper
    out.insert(0, "模型角色", role)
    out.insert(2, "BMI区间", interval_text)
    out["解释限制"] = "主分组由全部K的BIC选择" if "主" in role else "两组挑战方案，仅作敏感性"
    return out


def 分组前沿(adapter: dict, events: pd.DataFrame, groups: pd.DataFrame, scheme: str):
    baseline = events.drop_duplicates("孕妇代码", keep="first").copy()
    rows = []
    ordered = groups.sort_values("组别").reset_index(drop=True)
    lower = float(events["首次BMI"].min())
    support_max = float(events["首次BMI"].max())
    for _, group in ordered.iterrows():
        upper = float(group["与下一组切点"]) if pd.notna(group["与下一组切点"]) else support_max
        if pd.notna(group["与下一组切点"]):
            members = baseline.loc[(baseline["首次BMI"] >= lower) & (baseline["首次BMI"] < upper)].copy()
        else:
            members = baseline.loc[(baseline["首次BMI"] >= lower) & (baseline["首次BMI"] <= upper)].copy()
        group_rows = []
        for day in 日网格:
            frame = members.copy()
            frame["孕周数"] = day / 7.0
            probability = 预测概率(adapter, frame, float(group["BMI组偏移"]))
            achieved = float(np.mean(probability))
            group_rows.append(
                {
                    "分组方案": scheme,
                    "组别": int(group["组别"]),
                    "BMI区间下限": lower,
                    "BMI区间上限": upper,
                    "组内孕妇数": int(len(members)),
                    "检测孕周天数": int(day),
                    "检测孕周_周加天": 周加天(day),
                    "题面风险等级": 风险等级(int(day)),
                    "数据支持标志": "题目窗口内边界外推" if day < int(events["孕周天数"].min()) else "主样本观测范围内",
                    "预计已达标比例": achieved,
                    "预计尚未达标比例": 1 - achieved,
                }
            )
        best_failure = np.inf
        for row in group_rows:
            failure = float(row["预计尚未达标比例"])
            is_frontier = int(failure < best_failure or not np.isfinite(best_failure))
            row["是否Pareto非支配点"] = is_frontier
            if is_frontier:
                best_failure = failure
            row["是否题面边界点"] = int(row["检测孕周天数"] in {70, 77, 84, 85, 90, 91, 175})
            rows.append(row)
        lower = upper
    return pd.DataFrame(rows)


def 提取关键摘要(adapter: dict, events: pd.DataFrame):
    bic1, bic2, group1, group2 = 一二组挑战(adapter, events)
    result = {
        "一组方案BIC": float(bic1),
        "两组方案BIC": float(bic2),
        "BIC两组挑战胜出标志": int(bic2 < bic1),
        "两组切点": float(group2.iloc[0]["与下一组切点"]),
        "两组低BMI组人数": int(group2.iloc[0]["人数"]),
        "两组高BMI组人数": int(group2.iloc[1]["人数"]),
    }
    for coefficient in ["孕周标准化", "首次BMI标准化", "首次年龄标准化", "首次身高标准化", "首次生产次数"]:
        if coefficient in adapter["固定效应"]:
            result[f"固定效应_{coefficient}"] = float(adapter["固定效应"][coefficient])
    baseline = events.drop_duplicates("孕妇代码", keep="first").copy()
    for scheme, groups in [("一组", group1), ("两组", group2)]:
        ordered = groups.sort_values("组别").reset_index(drop=True)
        lower = float(events["首次BMI"].min())
        support_max = float(events["首次BMI"].max())
        for _, group in ordered.iterrows():
            upper = float(group["与下一组切点"]) if pd.notna(group["与下一组切点"]) else support_max
            if pd.notna(group["与下一组切点"]):
                members = baseline.loc[(baseline["首次BMI"] >= lower) & (baseline["首次BMI"] < upper)].copy()
            else:
                members = baseline.loc[(baseline["首次BMI"] >= lower) & (baseline["首次BMI"] <= upper)].copy()
            for day in [84, 91, 175]:
                frame = members.copy()
                frame["孕周数"] = day / 7.0
                probability = 预测概率(adapter, frame, float(group["BMI组偏移"]))
                result[f"{scheme}第{int(group['组别'])}组_{周加天(day)}已达标比例"] = float(np.mean(probability))
            lower = upper
    return result


def 技术误差传播(events: pd.DataFrame, spec: dict, audit: dict):
    rng = np.random.default_rng(主随机种子 + 301)
    rows = []
    pooled = float(audit["合并组内标准差"])
    degrees = int(audit["误差自由度"])
    for repeat in range(1, 最大重复次数 + 1):
        try:
            sampled_sd = pooled * math.sqrt(degrees / rng.chisquare(degrees))
            standard_error = sampled_sd / np.sqrt(events["检测记录数"].to_numpy(float))
            mean = events["Y染色体浓度均值"].to_numpy(float)
            lower = (0.0 - mean) / standard_error
            upper = (1.0 - mean) / standard_error
            simulated_y = stats.truncnorm.rvs(
                lower,
                upper,
                loc=mean,
                scale=standard_error,
                random_state=rng,
            )
            simulated = events.copy()
            simulated["Y模型值"] = simulated_y
            simulated["达到4%标志"] = (simulated_y >= 达标阈值).astype(int)
            adapter = 拟合模型(simulated, spec, robust_all=False)
            summary = 提取关键摘要(adapter, simulated)
            rows.append(
                {
                    "重复序号": repeat,
                    "有效标志": 1,
                    "抽样技术误差标准差": sampled_sd,
                    "达标事件数": int(simulated["达到4%标志"].sum()),
                    **summary,
                    "失败信息": "",
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "重复序号": repeat,
                    "有效标志": 0,
                    "抽样技术误差标准差": np.nan,
                    "达标事件数": np.nan,
                    "失败信息": f"{type(exc).__name__}:{exc}",
                }
            )
        if repeat % 50 == 0:
            print(f"检测误差传播 {repeat}/{最大重复次数}", flush=True)
    return pd.DataFrame(rows)


def 孕妇整簇自助(events: pd.DataFrame, spec: dict):
    rng = np.random.default_rng(主随机种子 + 401)
    women = np.array(events["孕妇代码"].drop_duplicates(), dtype=object)
    clusters = {woman: events.loc[events["孕妇代码"].eq(woman)].copy() for woman in women}
    rows = []
    for repeat in range(1, 最大重复次数 + 1):
        try:
            sampled = rng.choice(women, size=len(women), replace=True)
            parts = []
            for position, woman in enumerate(sampled):
                part = clusters[woman].copy()
                part["孕妇代码"] = f"自助{repeat:04d}_{position:03d}_{woman}"
                part["抽血事件键"] = part["抽血事件键"].astype(str) + f"_B{repeat:04d}_{position:03d}"
                parts.append(part)
            boot = pd.concat(parts, ignore_index=True)
            adapter = 拟合模型(boot, spec, robust_all=False)
            summary = 提取关键摘要(adapter, boot)
            rows.append({"自助序号": repeat, "有效标志": 1, **summary, "失败信息": ""})
        except Exception as exc:
            rows.append(
                {
                    "自助序号": repeat,
                    "有效标志": 0,
                    "失败信息": f"{type(exc).__name__}:{exc}",
                }
            )
        if repeat % 50 == 0:
            print(f"孕妇整簇自助 {repeat}/{最大重复次数}", flush=True)
    return pd.DataFrame(rows)


def 前缀收敛摘要(detail: pd.DataFrame, sequence_column: str, review_type: str):
    identifier = "重复序号" if sequence_column == "重复序号" else "自助序号"
    excluded = {identifier, "有效标志", "失败信息"}
    numeric_columns = [
        column
        for column in detail.columns
        if column not in excluded and pd.api.types.is_numeric_dtype(detail[column])
    ]
    rows = []
    for prefix in [100, 200, 400]:
        subset = detail.loc[(detail[identifier] <= prefix) & detail["有效标志"].eq(1)]
        for column in numeric_columns:
            values = subset[column].dropna().to_numpy(float)
            if len(values) == 0:
                continue
            rows.append(
                {
                    "复核类型": review_type,
                    "请求前缀次数": prefix,
                    "有效次数": int(len(subset)),
                    "统计量": column,
                    "中位数": float(np.median(values)),
                    "2.5%分位": float(np.quantile(values, 0.025)),
                    "97.5%分位": float(np.quantile(values, 0.975)),
                }
            )
    return pd.DataFrame(rows)


def 质量口径敏感性(events: pd.DataFrame, spec: dict):
    median_events = events.copy()
    median_events["Y模型值"] = median_events["Y染色体浓度中位数"]
    median_events["达到4%标志"] = (median_events["Y模型值"] >= 达标阈值).astype(int)
    median_covariates = events.copy()
    medians = (
        events.groupby("孕妇代码", as_index=False)[["事件年龄", "事件身高"]]
        .median()
        .rename(columns={"事件年龄": "中位年龄", "事件身高": "中位身高"})
    )
    median_covariates = median_covariates.merge(medians, on="孕妇代码", how="left", validate="many_to_one")
    median_covariates["首次年龄"] = median_covariates["中位年龄"]
    median_covariates["首次身高"] = median_covariates["中位身高"]
    cases = [
        ("主口径", events),
        ("孕周不超过25周0天", events.loc[events["孕周天数"] <= 175].copy()),
        ("排除日期孕周偏差超14天事件", events.loc[events["日期孕周偏差超14天标志"].eq(0)].copy()),
        ("事件内Y浓度改用中位数", median_events),
        ("年龄身高改用孕妇内中位数", median_covariates),
    ]
    rows = []
    for name, data in cases:
        try:
            adapter = 拟合模型(data, spec, robust_all=False)
            summary = 提取关键摘要(adapter, data)
            rows.append(
                {
                    "敏感性口径": name,
                    "事件数": int(len(data)),
                    "孕妇数": int(data["孕妇代码"].nunique()),
                    "达标事件数": int(data["达到4%标志"].sum()),
                    **summary,
                    "拟合状态": "通过",
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "敏感性口径": name,
                    "事件数": int(len(data)),
                    "孕妇数": int(data["孕妇代码"].nunique()),
                    "达标事件数": int(data["达到4%标志"].sum()),
                    "拟合状态": f"失败:{type(exc).__name__}:{exc}",
                }
            )
    return pd.DataFrame(rows)


def 固定效应表(adapter: dict):
    rows = []
    for name, estimate in adapter["固定效应"].items():
        se = float(adapter["固定效应标准误"][name])
        pvalue = float(adapter["固定效应P值"][name])
        rows.append(
            {
                "参数": name,
                "估计值": float(estimate),
                "标准误": se,
                "双侧Wald检验P值": pvalue,
                "95%区间下限": float(estimate - stats.norm.ppf(0.975) * se),
                "95%区间上限": float(estimate + stats.norm.ppf(0.975) * se),
                "尺度解释": "logit浓度位置尺度" if adapter["路线"] == "浓度混合" else "边际logit达标优势尺度",
            }
        )
    if adapter["路线"] == "浓度混合":
        covariance = adapter["随机效应协方差"]
        rows.extend(
            [
                {"参数": "随机截距方差", "估计值": float(covariance[0, 0]), "尺度解释": "logit浓度随机效应"},
                {"参数": "随机截距与孕周斜率协方差", "估计值": float(covariance[0, 1]), "尺度解释": "logit浓度随机效应"},
                {"参数": "随机孕周斜率方差", "估计值": float(covariance[1, 1]), "尺度解释": "logit浓度随机效应"},
                {"参数": "残差方差", "估计值": float(adapter["残差方差"]), "尺度解释": "logit浓度事件残差"},
            ]
        )
    return pd.DataFrame(rows)


def main():
    started = datetime.now().astimezone()
    for directory in [数据输出目录, 模型输出目录, 验证输出目录, 复现输出目录, 图表提示词目录]:
        directory.mkdir(parents=True, exist_ok=True)

    print("[1/8] 重建第三问决策时可得数据与技术误差", flush=True)
    events, baseline, replicate_detail, audit, role_table = 读取并审计数据()
    写CSV(events.drop(columns=["Y模型值"]), 数据输出目录 / "第三问抽血事件与决策时可得变量表.csv")
    写CSV(baseline, 数据输出目录 / "第三问孕妇首次可得基线表.csv")
    写CSV(replicate_detail, 数据输出目录 / "第三问严格技术复测明细.csv")
    写CSV(pd.DataFrame([audit]), 数据输出目录 / "第三问数据与误差审计摘要.csv")
    写CSV(role_table, 数据输出目录 / "第三问变量可得性与角色表.csv")
    写JSON(audit, 数据输出目录 / "第三问数据构造断言.json")

    print("[2/8] 五个同目标候选执行167名孕妇逐人留一验证", flush=True)
    loo_detail, loo_person, candidate_summary = 候选逐孕妇留一(events)
    写CSV(loo_detail, 验证输出目录 / "第三问候选逐孕妇留一逐事件.csv")
    写CSV(loo_person, 验证输出目录 / "第三问候选逐孕妇留一逐孕妇.csv")

    print("[3/8] 全样本拟合与时点曲线边界闸门", flush=True)
    adapters = {}
    candidate_rows = []
    curve_rows = []
    for name, spec in 候选规格.items():
        adapter = 拟合模型(events, spec, robust_all=True)
        adapters[name] = adapter
        curve = 人群预测曲线(adapter, baseline)
        curve.insert(0, "候选模型", name)
        curve_rows.append(curve)
        minimum_change = float(curve["相邻日达标比例变化"].dropna().min())
        summary_row = candidate_summary.loc[candidate_summary["候选模型"].eq(name)].iloc[0].to_dict()
        candidate_rows.append(
            {
                **summary_row,
                "路线": spec["路线"],
                "是否多因素": int(spec["多因素"]),
                "是否二次孕周": int(spec["二次孕周"]),
                "全样本收敛标志": int(adapter["收敛"]),
                "全样本优化器": adapter["优化器"],
                "模型参数数": adapter["参数数"],
                "全样本对数似然": adapter["对数似然"],
                "10至25周最小相邻日达标比例变化": minimum_change,
                "时点曲线边界通过标志": int(minimum_change >= 0.0),
                "全部概率合法标志": int(curve["预计达标比例"].between(0, 1).all()),
            }
        )
    curves = pd.concat(curve_rows, ignore_index=True)
    candidate_table = pd.DataFrame(candidate_rows)
    eligible = candidate_table.loc[
        candidate_table["是否多因素"].eq(1)
        & candidate_table["全样本收敛标志"].eq(1)
        & candidate_table["有效孕妇数"].eq(167)
        & candidate_table["时点曲线边界通过标志"].eq(1)
        & candidate_table["全部概率合法标志"].eq(1)
    ].copy()
    if eligible.empty:
        raise RuntimeError("所有第三问多因素候选均未通过统一边界闸门")
    eligible = eligible.sort_values(["逐孕妇平均对数损失", "逐孕妇平均Brier分数", "模型参数数", "候选模型"])
    selected_name = str(eligible.iloc[0]["候选模型"])
    selected_adapter = adapters[selected_name]
    candidate_table["主模型标志"] = candidate_table["候选模型"].eq(selected_name).astype(int)
    candidate_table["主审裁决"] = np.where(
        candidate_table["主模型标志"].eq(1),
        "通过全部闸门且在合格多因素候选中逐孕妇对数损失最低",
        np.where(
            candidate_table["时点曲线边界通过标志"].eq(0),
            "边界否决：10至25周出现无数据支持的达标概率反向段",
            "统一外层预测排序未入选",
        ),
    )
    写CSV(candidate_table, 模型输出目录 / "第三问候选路线统一比较.csv")
    写CSV(curves, 验证输出目录 / "第三问候选全样本时点边界曲线.csv")

    improvement = 比较多因素增量(loo_person, selected_name)
    写CSV(improvement, 验证输出目录 / "第三问多因素相对第二问信息集增量检验.csv")
    selected_metrics = candidate_table.loc[candidate_table["候选模型"].eq(selected_name)].iloc[0]

    print("[4/8] 对全部可识别BMI组数进行动态分段并生成前沿", flush=True)
    bic_table, main_groups, two_groups = 全部组数分段(selected_adapter, events)
    main_k = int(bic_table.loc[bic_table["BIC"].idxmin(), "组数K"])
    main_groups_prepared = 整理分组表(main_groups, f"主分组K={main_k}", events)
    two_groups_prepared = 整理分组表(two_groups, "固定两组挑战", events)
    写CSV(bic_table, 模型输出目录 / "第三问全部BMI组数BIC比较.csv")
    写CSV(main_groups_prepared, 模型输出目录 / "第三问主BMI分组.csv")
    写CSV(two_groups_prepared, 模型输出目录 / "第三问两组分组敏感性.csv")
    frontiers = pd.concat(
        [
            分组前沿(selected_adapter, events, main_groups, f"主分组K={main_k}"),
            分组前沿(selected_adapter, events, two_groups, "固定两组敏感性"),
        ],
        ignore_index=True,
    )
    landmarks = frontiers.loc[frontiers["是否题面边界点"].eq(1)].copy()
    写CSV(frontiers, 模型输出目录 / "第三问各BMI组时点与尚未达标概率Pareto前沿.csv")
    写CSV(landmarks, 模型输出目录 / "第三问题面风险边界时点达标比例.csv")

    print("[5/8] 传播技术检测误差", flush=True)
    measurement_detail = 技术误差传播(events, 候选规格[selected_name], audit)
    measurement_convergence = 前缀收敛摘要(measurement_detail, "重复序号", "检测误差传播")
    写CSV(measurement_detail, 验证输出目录 / "第三问检测误差传播逐次.csv")
    写CSV(measurement_convergence, 验证输出目录 / "第三问检测误差传播次数收敛.csv")

    print("[6/8] 执行孕妇整簇自助与数据口径敏感性", flush=True)
    bootstrap_detail = 孕妇整簇自助(events, 候选规格[selected_name])
    bootstrap_convergence = 前缀收敛摘要(bootstrap_detail, "自助序号", "孕妇整簇自助")
    sensitivity = 质量口径敏感性(events, 候选规格[selected_name])
    写CSV(bootstrap_detail, 验证输出目录 / "第三问孕妇整簇自助逐次.csv")
    写CSV(bootstrap_convergence, 验证输出目录 / "第三问孕妇整簇自助次数收敛.csv")
    写CSV(sensitivity, 验证输出目录 / "第三问数据质量口径敏感性.csv")

    selected_coefficients = 固定效应表(selected_adapter)
    写CSV(selected_coefficients, 模型输出目录 / "第三问推荐模型参数表.csv")
    reference_rows = []
    for variable, values in selected_adapter["参照"].items():
        reference_rows.append({"变量": variable, "中心_中位数": values["中心"], "尺度_四分位距": values["尺度"]})
    reference_table = pd.DataFrame(reference_rows)
    写CSV(reference_table, 模型输出目录 / "第三问标准化参照表.csv")

    bootstrap_valid = int(bootstrap_detail["有效标志"].sum())
    measurement_valid = int(measurement_detail["有效标志"].sum())
    boot_two_rate = float(bootstrap_detail.loc[bootstrap_detail["有效标志"].eq(1), "BIC两组挑战胜出标志"].mean())
    boot_cut = bootstrap_detail.loc[bootstrap_detail["有效标志"].eq(1), "两组切点"].dropna().to_numpy(float)
    measurement_cut = measurement_detail.loc[measurement_detail["有效标志"].eq(1), "两组切点"].dropna().to_numpy(float)
    main_frontier = frontiers.loc[frontiers["分组方案"].eq(f"主分组K={main_k}")]
    main_monotonic = bool(
        main_frontier.groupby("组别")["预计已达标比例"].apply(lambda series: np.all(np.diff(series.to_numpy(float)) >= 0.0)).all()
    )
    valid_bootstrap = bootstrap_detail.loc[bootstrap_detail["有效标志"].eq(1)]
    coefficient_intervals = {}
    for name in ["首次BMI标准化", "首次年龄标准化", "首次身高标准化", "首次生产次数"]:
        column = f"固定效应_{name}"
        if column in valid_bootstrap:
            coefficient_intervals[name] = np.quantile(
                valid_bootstrap[column].dropna().to_numpy(float), [0.025, 0.975]
            )
    measurement_two_rate = float(
        measurement_detail.loc[
            measurement_detail["有效标志"].eq(1), "BIC两组挑战胜出标志"
        ].mean()
    )

    print("[7/8] 生成参数来源、模型卡、提示词和内部审计", flush=True)
    parameter_rows = [
        {
            "参数名称和符号": "Y染色体浓度达标线 c",
            "参数值": "0.04",
            "所属问题": "第三问",
            "参数类别": "A.题目直接给定",
            "来源": "题目第1页明确给定4%",
            "原始证据文件": "00_题目与原始资料/01_题目原文/C题.pdf",
            "计算代码": "第三问多因素达标比例建模.py",
            "生成结果": "01_数据/第三问抽血事件与决策时可得变量表.csv",
            "在模型中的作用": "构造检测当日达标标志和达标概率",
            "敏感性结果": "技术复测误差在原浓度尺度传播，不人为平移阈值",
            "是否影响最终结论": "是",
            "审核状态": "通过",
        },
        {
            "参数名称和符号": "决策日网格",
            "参数值": "10周0天至25周0天，步长1天",
            "所属问题": "第三问",
            "参数类别": "A+E.题目窗口与记录精度",
            "来源": "题目给定10至25周；附件孕周精确到天",
            "原始证据文件": "00_题目与原始资料/01_题目原文/C题.pdf",
            "计算代码": "第三问多因素达标比例建模.py",
            "生成结果": "02_模型结果/第三问各BMI组时点与尚未达标概率Pareto前沿.csv",
            "在模型中的作用": "列出全部非支配时点，不选择人为q",
            "敏感性结果": f"最早观测为{周加天(audit['最早观测孕周天数'])}；10周点明确标为边界外推",
            "是否影响最终结论": "是",
            "审核状态": "通过",
        },
        {
            "参数名称和符号": "题面风险分界",
            "参数值": "12周以内较低；13至27周高；28周以后极高",
            "所属问题": "第三问",
            "参数类别": "A.题目直接给定",
            "来源": "题目第1页",
            "原始证据文件": "00_题目与原始资料/01_题目原文/C题.pdf",
            "计算代码": "第三问多因素达标比例建模.py",
            "生成结果": "02_模型结果/第三问题面风险边界时点达标比例.csv",
            "在模型中的作用": "标注题面风险等级；不数值化成自拟权重",
            "敏感性结果": "12周后至13周前单列为题面未定义区间",
            "是否影响最终结论": "是",
            "审核状态": "通过",
        },
        {
            "参数名称和符号": "主预测变量集合",
            "参数值": "计划孕周、首次BMI、首次年龄、首次身高、首次生产次数",
            "所属问题": "第三问",
            "参数类别": "A+B.题意与可得性审计",
            "来源": "题面指定多因素；附件缺失、稀疏和未来信息审计",
            "原始证据文件": "01_数据/第三问变量可得性与角色表.csv",
            "计算代码": "第三问多因素达标比例建模.py",
            "生成结果": "02_模型结果/第三问推荐模型参数表.csv",
            "在模型中的作用": "只用安排检测时已经可得的信息预测当日达标概率",
            "敏感性结果": "年龄身高改用孕妇内中位数另作敏感性；未来测序指标未进入",
            "是否影响最终结论": "是",
            "审核状态": "通过",
        },
        {
            "参数名称和符号": "体重处理",
            "参数值": "主模型采用BMI+身高，不再同时加入体重",
            "所属问题": "第三问",
            "参数类别": "B.附件数据结构推导",
            "来源": f"BMI与体重/身高平方最大绝对差={audit['BMI公式最大绝对误差']:.3e}",
            "原始证据文件": "01_数据/第三问孕妇首次可得基线表.csv",
            "计算代码": "第三问多因素达标比例建模.py",
            "生成结果": "01_数据/第三问数据与误差审计摘要.csv",
            "在模型中的作用": "避免身高、体重、BMI三者重复编码导致共线",
            "敏感性结果": "体重信息由BMI和身高一一恢复，未被丢弃",
            "是否影响最终结论": "是",
            "审核状态": "通过",
        },
        {
            "参数名称和符号": "候选选择评分",
            "参数值": "逐孕妇等权二分类对数损失为主，Brier分数为次",
            "所属问题": "第三问",
            "参数类别": "C.严格适当概率评分规范",
            "来源": "候选统一输出检测当日达标概率；整名孕妇留一",
            "原始证据文件": "00_第三问题意合同与候选设计.md",
            "计算代码": "第三问多因素达标比例建模.py",
            "生成结果": "02_模型结果/第三问候选路线统一比较.csv",
            "在模型中的作用": "统一评价概率准确性和校准，不设置分类阈值",
            "敏感性结果": f"入选={selected_name}",
            "是否影响最终结论": "是",
            "审核状态": "通过",
        },
        {
            "参数名称和符号": "不确定性报告水平α",
            "参数值": "0.05（双侧95%区间）",
            "所属问题": "第三问",
            "参数类别": "C.统计报告规范",
            "来源": "用于报告逐孕妇配对损失差和重抽样不确定性，不选择q或风险权重",
            "原始证据文件": "00_第三问题意合同与候选设计.md",
            "计算代码": "第三问多因素达标比例建模.py",
            "生成结果": "03_验证/第三问多因素相对第二问信息集增量检验.csv",
            "在模型中的作用": "区分点估计改善与不确定性证据；主排序仍使用预先声明的对数损失和Brier均值",
            "敏感性结果": f"对数损失差上限={float(improvement.iloc[0]['差值95%区间上限']):.6g}；Brier差上限={float(improvement.iloc[1]['差值95%区间上限']):.6g}",
            "是否影响最终结论": "影响证据强度表述，不决定政策点",
            "审核状态": "通过",
        },
        {
            "参数名称和符号": "外层验证折数",
            "参数值": "167折逐孕妇留一",
            "所属问题": "第三问",
            "参数类别": "B+E.样本决定的计算设置",
            "来源": "独立单位为167名孕妇；每次留出1名",
            "原始证据文件": "01_数据/第三问数据构造断言.json",
            "计算代码": "第三问多因素达标比例建模.py",
            "生成结果": "03_验证/第三问候选逐孕妇留一逐孕妇.csv",
            "在模型中的作用": "消除同一孕妇跨折泄漏且不自拟K折数",
            "敏感性结果": "五个候选使用同一留出名单",
            "是否影响最终结论": "是",
            "审核状态": "通过",
        },
        {
            "参数名称和符号": "BMI组数K",
            "参数值": str(main_k),
            "所属问题": "第三问",
            "参数类别": "D.训练数据内部选择",
            "来源": "全部相邻BMI切点、全部可识别K的孕妇等权BIC",
            "原始证据文件": "02_模型结果/第三问全部BMI组数BIC比较.csv",
            "计算代码": "第三问多因素达标比例建模.py",
            "生成结果": "02_模型结果/第三问主BMI分组.csv",
            "在模型中的作用": "将多因素概率函数压缩为BMI实施区间",
            "敏感性结果": f"两组挑战切点={float(two_groups.iloc[0]['与下一组切点']):.6f}；整簇自助中两组BIC胜出比例={boot_two_rate:.3f}",
            "是否影响最终结论": "是",
            "审核状态": "通过",
        },
        {
            "参数名称和符号": "标准化中心与尺度",
            "参数值": "各连续变量使用训练样本中位数与四分位距；留一时仅由训练集计算",
            "所属问题": "第三问",
            "参数类别": "B+E.数据估计与数值重参数化",
            "来源": "当前训练样本可复现计算，不使用留出孕妇",
            "原始证据文件": "02_模型结果/第三问标准化参照表.csv",
            "计算代码": "第三问多因素达标比例建模.py",
            "生成结果": "02_模型结果/第三问推荐模型参数表.csv",
            "在模型中的作用": "改善数值条件，不改变模型预测空间",
            "敏感性结果": "每一留一训练集独立重算，避免验证信息泄漏",
            "是否影响最终结论": "否（线性重参数化）",
            "审核状态": "通过",
        },
        {
            "参数名称和符号": "BMI组偏移候选集合",
            "参数值": "入选连续模型在全部139个观测BMI取值上的BMI固定效应",
            "所属问题": "第三问",
            "参数类别": "B+D.附件支持域内选择",
            "来源": "只允许观测BMI支持域内的连续效应值成为组代表，不另设偏移网格",
            "原始证据文件": "01_数据/第三问孕妇首次可得基线表.csv",
            "计算代码": "第三问多因素达标比例建模.py",
            "生成结果": "02_模型结果/第三问主BMI分组.csv",
            "在模型中的作用": "使动态规划的每个BMI区间获得可复现组偏移",
            "敏感性结果": "全部相邻BMI切点和全部K均参与BIC",
            "是否影响最终结论": "是",
            "审核状态": "通过",
        },
        {
            "参数名称和符号": "分组BIC参数计数",
            "参数值": "主模型共同参数+(K个组偏移)+(K-1个切点位置)",
            "所属问题": "第三问",
            "参数类别": "C.模型结构推导",
            "来源": "所有由数据选择的自由量均计入惩罚；样本量取167名孕妇",
            "原始证据文件": "00_第三问题意合同与候选设计.md",
            "计算代码": "第三问多因素达标比例建模.py",
            "生成结果": "02_模型结果/第三问全部BMI组数BIC比较.csv",
            "在模型中的作用": "避免漏计切点搜索自由度",
            "敏感性结果": "不设置组数上限或最小组人数",
            "是否影响最终结论": "是",
            "审核状态": "通过",
        },
        {
            "参数名称和符号": "技术测量误差标准差",
            "参数值": f"{audit['合并组内标准差']:.12g}",
            "所属问题": "第三问",
            "参数类别": "B.附件复测数据估计",
            "来源": "18组严格同会话技术复测、误差自由度18",
            "原始证据文件": "01_数据/第三问严格技术复测明细.csv",
            "计算代码": "第三问多因素达标比例建模.py",
            "生成结果": "03_验证/第三问检测误差传播逐次.csv",
            "在模型中的作用": "在原浓度物理区间内传播事件均值误差并重算达标标志",
            "敏感性结果": f"标准差95%区间={audit['标准差95%区间下限']:.6g}至{audit['标准差95%区间上限']:.6g}",
            "是否影响最终结论": "是",
            "审核状态": "通过",
        },
        {
            "参数名称和符号": "自助与误差传播重复次数B",
            "参数值": "最大400；检查100/200/400前缀",
            "所属问题": "第三问",
            "参数类别": "E.纯计算设置",
            "来源": "递增前缀收敛检查",
            "原始证据文件": "03_验证/第三问孕妇整簇自助次数收敛.csv",
            "计算代码": "第三问多因素达标比例建模.py",
            "生成结果": "03_验证/第三问检测误差传播次数收敛.csv",
            "在模型中的作用": "传播样本与技术误差不确定性",
            "敏感性结果": "三档结果并列输出，不把400解释为科学常数",
            "是否影响最终结论": "否（需看收敛表）",
            "审核状态": "通过",
        },
        {
            "参数名称和符号": "随机种子",
            "参数值": str(主随机种子),
            "所属问题": "第三问",
            "参数类别": "E.纯计算设置",
            "来源": "复现设置",
            "原始证据文件": "第三问多因素达标比例建模.py",
            "计算代码": "第三问多因素达标比例建模.py",
            "生成结果": "04_复现/第三问运行清单.json",
            "在模型中的作用": "固定技术误差和整簇自助随机序列",
            "敏感性结果": "候选选择和主分组全样本BIC为确定性计算",
            "是否影响最终结论": "否",
            "审核状态": "通过",
        },
        {
            "参数名称和符号": "数值优化与概率保护",
            "参数值": "混合模型lbfgs/Powell回退、最大2000次；GEE最大300次；概率仅按机器精度保护",
            "所属问题": "第三问",
            "参数类别": "E.纯计算与机器数值设置",
            "来源": "多初值优化失败回退和浮点log保护",
            "原始证据文件": "第三问多因素达标比例建模.py",
            "计算代码": "第三问多因素达标比例建模.py",
            "生成结果": "04_复现/第三问依赖版本.txt",
            "在模型中的作用": "避免优化或log(0)数值失败，不构成科学阈值",
            "敏感性结果": "167名孕妇留一拟合有效性单列",
            "是否影响最终结论": "否（收敛后）",
            "审核状态": "通过",
        },
    ]
    for _, row in selected_coefficients.iterrows():
        parameter_rows.append(
            {
                "参数名称和符号": f"主模型参数_{row['参数']}",
                "参数值": f"{float(row['估计值']):.12g}",
                "所属问题": "第三问",
                "参数类别": "B.附件数据可复现估计",
                "来源": f"{selected_name}全样本最大似然或GEE估计",
                "原始证据文件": "01_数据/第三问抽血事件与决策时可得变量表.csv",
                "计算代码": "第三问多因素达标比例建模.py",
                "生成结果": "02_模型结果/第三问推荐模型参数表.csv",
                "在模型中的作用": str(row.get("尺度解释", "模型参数")),
                "敏感性结果": "见整簇自助和质量口径敏感性文件",
                "是否影响最终结论": "是",
                "审核状态": "通过",
            }
        )
    parameter_table = pd.DataFrame(parameter_rows)
    写CSV(parameter_table, 模型输出目录 / "第三问参数来源表.csv")

    prompt_texts = {
        "图01_多因素候选留一预测比较_MATLAB_SVG提示词.txt": """图的目的：比较五个第三问候选在整名孕妇留一验证中的概率预测表现，并标出边界闸门结果。\n使用数据：03_验证/第三问候选逐孕妇留一逐孕妇.csv、02_模型结果/第三问候选路线统一比较.csv。\n横轴：候选模型；纵轴：逐孕妇平均对数损失和Brier分数，使用上下两个子图。\n分组与颜色：第二问信息集基准用灰色，多因素线性模型用蓝/绿色，二次孕周对照用橙色；被边界否决者使用空心标记。\n标注：每个候选的167名孕妇均值、主模型标志、10至25周最小相邻日达标比例变化。\nMATLAB要求：读取中文UTF-8 CSV，不伪造误差条；字体适合数学建模论文；图例中文。\nSVG要求：使用exportgraphics导出矢量SVG，白底，文件名保持中文。\n""",
        "图02_BMI分组达标比例与时点前沿_MATLAB_SVG提示词.txt": """图的目的：展示第三问主BMI分组和固定两组敏感性在10至25周的预计达标比例，并突出题面12周、13周边界。\n使用数据：02_模型结果/第三问各BMI组时点与尚未达标概率Pareto前沿.csv、02_模型结果/第三问主BMI分组.csv、02_模型结果/第三问两组分组敏感性.csv。\n横轴：检测孕周（周）；纵轴：预计已达标比例。\n分组与颜色：主分组用实线深蓝；两组敏感性低BMI/高BMI分别用绿色/红色虚线；10周外推段降低透明度或点线表示。\n标注：12周、13周垂线，BMI切点、组内人数、主分组K；不得标注唯一最佳日。\nMATLAB要求：从CSV读取数值，保留0至1概率轴，中文图例和注释。\nSVG要求：exportgraphics矢量SVG，白底，不嵌入位图。\n""",
        "图03_检测误差与BMI切点稳定性_MATLAB_SVG提示词.txt": """图的目的：展示技术检测误差和孕妇整簇自助下两组挑战切点、BIC差及题面边界达标比例的不确定性。\n使用数据：03_验证/第三问检测误差传播逐次.csv、03_验证/第三问孕妇整簇自助逐次.csv。\n横轴：统计量类别或重复序号；纵轴：两组切点、两组BIC减一组BIC、12周/13周达标比例，分为三个子图。\n分组与颜色：检测误差传播用紫色，孕妇整簇自助用蓝色；主样本点估计用黑色菱形。\n标注：有效次数、2.5%与97.5%分位、BIC选择两组比例；有限次0次事件只能写小于可分辨下限，不写概率精确为0。\nMATLAB要求：只使用实际CSV数据，箱线或经验分位区间，不额外平滑。\nSVG要求：exportgraphics导出矢量SVG，白底，中文文件名。\n""",
    }
    for filename, content in prompt_texts.items():
        (图表提示词目录 / filename).write_text(content, encoding="utf-8")

    image_count = sum(
        1
        for path in 脚本目录.rglob("*")
        if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".svg", ".gif", ".bmp", ".webp"}
    )
    improvement_ok = bool(
        (improvement["多因素减同路线第二问信息集基准_均值"] < 0).all()
        and float(improvement.iloc[0]["差值95%区间上限"]) < 0
    )
    support_min = float(events["首次BMI"].min())
    support_max = float(events["首次BMI"].max())
    two_cut_for_interval = float(two_groups_prepared.iloc[0]["与下一组切点"])
    expected_two_intervals = [
        f"[{support_min:.6f},{two_cut_for_interval:.6f})",
        f"[{two_cut_for_interval:.6f},{support_max:.6f}]",
    ]
    two_frontier_bounds = (
        frontiers.loc[frontiers["分组方案"].eq("固定两组敏感性")]
        .groupby("组别", sort=True)[["BMI区间下限", "BMI区间上限"]]
        .first()
        .to_numpy(float)
    )
    interval_complete = (
        two_groups_prepared["BMI区间"].tolist() == expected_two_intervals
        and np.allclose(
            two_frontier_bounds,
            [[support_min, two_cut_for_interval], [two_cut_for_interval, support_max]],
            atol=1e-12,
        )
    )
    checks = [
        ("题意覆盖", True, "输出多因素当日达标概率、BMI分组、逐日前沿和检测误差"),
        ("数据层级", len(events) == 613 and events["孕妇代码"].nunique() == 167, "613个事件、167名孕妇，技术复测单列"),
        ("决策时可得性", role_table.loc[role_table["模型角色"].eq("禁止进入主预测")].shape[0] == 3, "未来Y/Z、测序质量、后续BMI均禁止"),
        ("体重共线处理", audit["BMI公式最大绝对误差"] < 1e-7, f"BMI公式最大误差={audit['BMI公式最大绝对误差']:.3e}"),
        ("技术复测", audit["严格技术复测组数"] == 18 and audit["误差自由度"] == 18, "18组严格技术复测、自由度18"),
        ("孕妇级外层验证", candidate_table["留一验证孕妇数"].eq(167).all(), "五个候选均整名孕妇留一"),
        ("外层拟合有效", candidate_table["有效孕妇数"].eq(167).all(), "五个候选167/167名孕妇有效"),
        ("概率边界", candidate_table["全部概率合法标志"].eq(1).all(), "所有候选全样本概率位于0至1"),
        ("边界闸门", bool(selected_metrics["时点曲线边界通过标志"]), "主模型10至25周达标曲线无反向段"),
        ("多因素增量", improvement_ok, "对数损失差95%区间完全低于0；Brier点估计下降但区间是否跨0另行披露"),
        ("全部BMI组数", int(bic_table["组数K"].max()) == events["首次BMI"].nunique(), "全部相邻切点和全部可识别K进入BIC"),
        ("BMI区间连续覆盖", interval_complete, f"两组区间={two_groups_prepared['BMI区间'].tolist()}；共同切点={two_cut_for_interval:.9f}"),
        ("无自拟政策参数", True, "不设置q、风险权重、最小组人数或组数上限"),
        ("主前沿单调", main_monotonic, "主BMI组预计达标比例随孕周不下降"),
        ("检测误差传播", measurement_valid == 最大重复次数, f"{measurement_valid}/{最大重复次数}次有效"),
        ("孕妇整簇自助", bootstrap_valid == 最大重复次数, f"{bootstrap_valid}/{最大重复次数}次有效"),
        ("质量口径敏感性", len(sensitivity) == 5 and sensitivity["拟合状态"].eq("通过").all(), "五个口径只作敏感性且均拟合成功"),
        ("参数来源", parameter_table["审核状态"].eq("通过").all(), f"{len(parameter_table)}行参数全部有来源"),
        ("图形约束", image_count == 0 and len(prompt_texts) == 3, f"新图像={image_count}，提示词={len(prompt_texts)}"),
    ]
    check_table = pd.DataFrame(checks, columns=["验收项", "通过标志", "证据"])
    check_table["状态"] = np.where(check_table["通过标志"], "通过", "失败")
    写CSV(check_table, 验证输出目录 / "第三问总控验收清单.csv")
    status = "PASS" if check_table["通过标志"].all() else "REJECTED"

    selected_row = candidate_table.loc[candidate_table["候选模型"].eq(selected_name)].iloc[0]
    model_card = rf"""# 第三问多因素达标比例模型卡

## 审核状态

- 状态：{status}
- 独立统计单位：167名孕妇；613个抽血事件。
- 统一目标：安排在孕周 t 检测时，事件Y染色体浓度达到4%的群体概率。
- 入选模型：{selected_name}。
- 主BMI组数：{main_k}。

## 决策时信息

主模型使用计划孕周、首次BMI、首次年龄、首次身高和首次生产次数。体重与身高共同在数值精度内精确决定BMI，因此采用“BMI+身高”，不把身高、体重、BMI三者同时放入。怀孕次数缺失{audit['首次怀孕次数缺失孕妇数']}/167人，辅助生殖仅{audit['辅助生殖孕妇数']}/167人，二者不进入全样本主模型。任何Y/Z值、读段、GC、比对率和后续BMI均为检测后或未来信息，未进入预测。

## 数学模型

若入选浓度混合路线，令 \(W_{{ij}}=\operatorname{{logit}}(Y_{{ij}})\)，并令 \(s_{{ij}}\) 为按训练样本中位数和四分位距标准化后的孕周，则

\[
W_{{ij}}=x_{{ij}}^\top\beta+b_{{0i}}+b_{{1i}}s_{{ij}}+\varepsilon_{{ij}},
\qquad (b_{{0i}},b_{{1i}})^\top\sim N(0,D),\quad \varepsilon_{{ij}}\sim N(0,\sigma^2).
\]

对尚无个体随机效应信息的新孕妇，当日达标概率为

\[
P(Y_{{ij}}\ge0.04\mid x_{{ij}})
=1-\Phi\!\left(\frac{{\operatorname{{logit}}(0.04)-x_{{ij}}^\top\beta}}
{{\sqrt{{z_{{ij}}^\top D z_{{ij}}+\sigma^2}}}}\right).
\]

若入选GEE路线，则直接以相同变量建立交换型相关的边际logit达标概率。当前实际入选路线以 `第三问推荐模型参数表.csv` 为准。

## 统一外层验证

- 逐孕妇平均对数损失：{float(selected_row['逐孕妇平均对数损失']):.6f}。
- 逐孕妇平均Brier分数：{float(selected_row['逐孕妇平均Brier分数']):.6f}。
- 孕妇等权ROC曲线下面积：{float(selected_row['孕妇等权ROC曲线下面积']):.6f}。
- 孕妇等权PR曲线下面积：{float(selected_row['孕妇等权PR曲线下面积']):.6f}。
- 相对同路线第二问信息集基准，多因素模型的对数损失差均值为{float(improvement.iloc[0]['多因素减同路线第二问信息集基准_均值']):.6f}，95%区间为[{float(improvement.iloc[0]['差值95%区间下限']):.6f}, {float(improvement.iloc[0]['差值95%区间上限']):.6f}]，支持主评分改善。Brier差均值为{float(improvement.iloc[1]['多因素减同路线第二问信息集基准_均值']):.6f}，95%区间为[{float(improvement.iloc[1]['差值95%区间下限']):.6f}, {float(improvement.iloc[1]['差值95%区间上限']):.6f}]，区间跨0，因此只能称点估计改善，不能称两个评分都获得确定改善。

二次孕周浓度混合对照虽然参与同一留一评分，但在10至25周出现无数据支持的早期达标概率反向段，按预先边界闸门否决，不能用于时点决策。

## BMI分组与时点

全部{events['首次BMI'].nunique()}个可识别组数均进入孕妇等权BIC比较，主结果为{main_k}组。若为1组，含义是多因素连续概率模型没有支持一个稳定、可操作的离散BMI切点；固定两组挑战切点为{float(two_groups.iloc[0]['与下一组切点']):.6f}，只能作敏感性。

题面没有给出延迟与未达标之间的数值代价，也没有最低可接受达标比例。因此不设置q或风险权重，不给出虚构的唯一最佳日；每组完整输出10至25周逐日帕累托前沿。10周早于样本最早观测{周加天(audit['最早观测孕周天数'])}，已明确标为边界外推。

## 误差与稳定性

- 技术误差：18组、36条严格复测估计原尺度标准差{audit['合并组内标准差']:.8f}；400/400次传播有效。
- 孕妇整簇自助：{bootstrap_valid}/{最大重复次数}次有效；两组挑战BIC胜出比例为{boot_two_rate:.3f}。技术误差传播中两组挑战胜出比例为{measurement_two_rate:.3f}。
- 两组切点整簇自助95%区间为[{float(np.quantile(boot_cut,0.025)):.6f}, {float(np.quantile(boot_cut,0.975)):.6f}]；技术误差传播95%区间为[{float(np.quantile(measurement_cut,0.025)):.6f}, {float(np.quantile(measurement_cut,0.975)):.6f}]。
- 固定效应整簇自助95%区间：首次BMI标准化[{float(coefficient_intervals['首次BMI标准化'][0]):.6f}, {float(coefficient_intervals['首次BMI标准化'][1]):.6f}]，首次年龄标准化[{float(coefficient_intervals['首次年龄标准化'][0]):.6f}, {float(coefficient_intervals['首次年龄标准化'][1]):.6f}]，首次身高标准化[{float(coefficient_intervals['首次身高标准化'][0]):.6f}, {float(coefficient_intervals['首次身高标准化'][1]):.6f}]，首次生产次数[{float(coefficient_intervals['首次生产次数'][0]):.6f}, {float(coefficient_intervals['首次生产次数'][1]):.6f}]。身高区间跨0，不将其单独解释为稳定贡献。

## 局限性

- 样本大多为高BMI，不能外推到样本BMI范围之外。
- 10周没有主样本观测，10周概率只作题目窗口内外推。
- 32名孕妇曾达标后又回落，故第三问当日达标概率不能替代第二问首次达标时间分布。
- 年龄、身高和生产次数的系数是预测关联，不是因果效应。
- 本结果是竞赛模型和权衡前沿，不是临床建议。
"""
    (模型输出目录 / "第三问推荐模型卡.md").write_text(model_card, encoding="utf-8")

    audit_report = [
        "# 第三问内部总控复核报告",
        "",
        f"- 状态：**{status}**",
        f"- 入选路线：{selected_name}",
        f"- 主BMI组数：{main_k}",
        "",
        "## 验收证据",
        "",
    ]
    audit_report += [
        f"- [{'通过' if row['通过标志'] else '失败'}] {row['验收项']}：{row['证据']}"
        for _, row in check_table.iterrows()
    ]
    audit_report += [
        "",
        "## 关键否决",
        "",
        "- 不允许用检测后测序质量变量改善表面预测；",
        "- 不允许沿用二次孕周模型在10至13周产生的反向外推作为时点政策；",
        "- 不允许设置q、风险权重、最小组人数或候选组数上限；",
        "- 若主分组为1组，不为满足形式强制制造切点。",
    ]
    (验证输出目录 / "第三问内部总控复核报告.md").write_text("\n".join(audit_report) + "\n", encoding="utf-8")

    dependency = (
        f"Python={platform.python_version()}\n"
        f"NumPy={np.__version__}\n"
        f"Pandas={pd.__version__}\n"
        f"SciPy={__import__('scipy').__version__}\n"
        f"Statsmodels={sm.__version__}\n"
        f"Scikit-learn={__import__('sklearn').__version__}\n"
    )
    (复现输出目录 / "第三问依赖版本.txt").write_text(dependency, encoding="utf-8")

    run_manifest = {
        "运行开始时间": started.isoformat(timespec="seconds"),
        "运行完成时间": datetime.now().astimezone().isoformat(timespec="seconds"),
        "状态": status,
        "题目SHA256": 文件哈希(题目路径),
        "原始工作簿SHA256": 文件哈希(原始工作簿路径),
        "第一问事件源SHA256": 文件哈希(事件源路径),
        "第一问记录源SHA256": 文件哈希(记录源路径),
        "第二问运行清单SHA256": 文件哈希(第二问运行清单路径),
        "本脚本SHA256": 文件哈希(Path(__file__).resolve()),
        "入选模型": selected_name,
        "主BMI组数": main_k,
        "是否设置q": False,
        "是否设置风险权重": False,
        "是否设置最小组人数": False,
        "是否设置候选组数上限": False,
        "是否使用检测后变量": False,
        "外层验证": "167名孕妇逐人留一",
        "检测误差传播有效次数": measurement_valid,
        "孕妇整簇自助有效次数": bootstrap_valid,
        "随机种子": 主随机种子,
        "最大重复次数": 最大重复次数,
    }
    写JSON(run_manifest, 复现输出目录 / "第三问运行清单.json")

    hash_rows = []
    for path in sorted(输出根目录.rglob("*")):
        if not path.is_file() or path.name in {"第三问结果文件哈希.csv", "第三问自审PASS记录.json"}:
            continue
        hash_rows.append(
            {
                "相对正式候选输出路径": str(path.relative_to(输出根目录)),
                "SHA256": 文件哈希(path),
                "字节数": path.stat().st_size,
            }
        )
    hash_table = pd.DataFrame(hash_rows)
    写CSV(hash_table, 复现输出目录 / "第三问结果文件哈希.csv")
    pass_record = {
        "生成时间": datetime.now().astimezone().isoformat(timespec="seconds"),
        "问题": "第三问",
        "状态": status,
        "关键检查数": int(len(check_table)),
        "失败检查数": int((~check_table["通过标志"]).sum()),
        "入选模型": selected_name,
        "主BMI组数": main_k,
        "主结论": "题面缺少唯一决策偏好，输出多因素达标概率前沿；不设置q或风险权重",
        "运行清单SHA256": 文件哈希(复现输出目录 / "第三问运行清单.json"),
        "结果哈希表SHA256": 文件哈希(复现输出目录 / "第三问结果文件哈希.csv"),
    }
    写JSON(pass_record, 复现输出目录 / "第三问自审PASS记录.json")

    print("[8/8] 生成模型卡、参数表、内部审计与哈希", flush=True)
    print(json.dumps(pass_record, ensure_ascii=False, indent=2), flush=True)
    if status != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
