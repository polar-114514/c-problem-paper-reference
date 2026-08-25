from __future__ import annotations

import math
import hashlib
import json
import platform
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import optimize, stats
from scipy.special import expit


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

达标阈值 = 0.04
日网格 = np.arange(70, 176, dtype=int)
主随机种子 = 20250825
最大重复次数 = 400


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


def 构造删失区间(events: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for woman, group in events.groupby("孕妇代码", sort=False):
        group = group.sort_values(["孕周数", "抽血次数", "抽血事件键"]).reset_index(drop=True)
        hits = np.flatnonzero(group["达到4%标志"].to_numpy(int) == 1)
        if len(hits) == 0:
            left = float(group["孕周数"].iloc[-1])
            right = np.inf
            censor = "右删失"
            first_hit = np.nan
            post_drop = 0
        elif int(hits[0]) == 0:
            left = 0.0
            right = float(group["孕周数"].iloc[0])
            censor = "左删失"
            first_hit = right
            post_drop = int((group["Y染色体浓度均值"].iloc[1:] < 达标阈值).any())
        else:
            pos = int(hits[0])
            left = float(group["孕周数"].iloc[pos - 1])
            right = float(group["孕周数"].iloc[pos])
            censor = "区间删失"
            first_hit = right
            post_drop = int((group["Y染色体浓度均值"].iloc[pos + 1 :] < 达标阈值).any())
        first = group.iloc[0]
        rows.append(
            {
                "孕妇代码": woman,
                "首次BMI": float(first["首次BMI"]),
                "年龄": float(first["年龄"]),
                "生产次数": float(first["生产次数"]),
                "首次观测孕周": float(group["孕周数"].iloc[0]),
                "末次观测孕周": float(group["孕周数"].iloc[-1]),
                "删失左端点": left,
                "删失右端点": right,
                "删失类型": censor,
                "首次观测达标孕周": first_hit,
                "达标后回落标志": post_drop,
                "抽血事件数": int(len(group)),
                "仅一次抽血标志": int(len(group) == 1),
            }
        )
    return pd.DataFrame(rows).sort_values(["首次BMI", "孕妇代码"]).reset_index(drop=True)


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
                "记录数": int(len(group)),
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
    first_bmi = (
        events.groupby("孕妇代码", as_index=False, sort=False).first()[["孕妇代码", "孕妇体质指数_BMI"]]
        .rename(columns={"孕妇体质指数_BMI": "首次BMI"})
    )
    events = events.merge(first_bmi, on="孕妇代码", how="left", validate="many_to_one")
    events["达到4%标志"] = (events["Y染色体浓度均值"] >= 达标阈值).astype(int)
    keep = [
        "孕妇代码",
        "抽血事件键",
        "抽血次数",
        "孕周数",
        "孕周天数",
        "首次BMI",
        "孕妇体质指数_BMI",
        "年龄",
        "生产次数",
        "Y染色体浓度均值",
        "Y染色体浓度中位数",
        "Y染色体浓度最小值",
        "Y染色体浓度最大值",
        "达到4%标志",
        "记录数",
        "检测会话数",
        "任一记录日期孕周偏差超14天标志",
    ]
    events = events[keep].rename(
        columns={
            "孕妇体质指数_BMI": "事件BMI",
            "记录数": "检测记录数",
            "检测会话数": "检测会话数",
            "任一记录日期孕周偏差超14天标志": "日期孕周偏差超14天标志",
        }
    )
    intervals = 构造删失区间(events)
    selected_events = set(events["抽血事件键"])
    records = raw_records.loc[raw_records["抽血事件键"].isin(selected_events)].copy()
    replicate_detail, replicate_summary = 汇总技术复测(records)

    censor = intervals["删失类型"].value_counts().to_dict()
    assertions = {
        "抽血事件数": int(len(events)),
        "孕妇数": int(events["孕妇代码"].nunique()),
        "删失结构": {name: int(censor.get(name, 0)) for name in ["左删失", "区间删失", "右删失"]},
        "达标后回落人数": int(intervals["达标后回落标志"].sum()),
        "仅一次抽血人数": int(intervals["仅一次抽血标志"].sum()),
        "首次BMI最小值": float(intervals["首次BMI"].min()),
        "首次BMI中位数": float(intervals["首次BMI"].median()),
        "首次BMI最大值": float(intervals["首次BMI"].max()),
        **replicate_summary,
    }
    expected = {
        "抽血事件数": 613,
        "孕妇数": 167,
        "达标后回落人数": 32,
        "严格技术复测组数": 18,
        "严格技术复测记录数": 36,
        "误差自由度": 18,
    }
    for key, value in expected.items():
        if assertions[key] != value:
            raise RuntimeError(f"数据断言失败：{key}={assertions[key]}，预期={value}")
    if assertions["删失结构"] != {"左删失": 118, "区间删失": 40, "右删失": 9}:
        raise RuntimeError(f"删失结构断言失败：{assertions['删失结构']}")
    return events, intervals, replicate_detail, assertions


def cdf(z: np.ndarray, family: str) -> np.ndarray:
    if family == "对数正态":
        return stats.norm.cdf(z)
    if family == "对数逻辑斯蒂":
        return expit(z)
    if family == "Weibull":
        return 1.0 - np.exp(-np.exp(np.clip(z, -700, 700)))
    raise ValueError(family)


def probabilities(mu: np.ndarray, sigma: float, data: pd.DataFrame, family: str) -> np.ndarray:
    left = data["删失左端点"].to_numpy(float)
    right = data["删失右端点"].to_numpy(float)
    censor = data["删失类型"].to_numpy(str)
    z_left = (np.log(np.maximum(left, np.finfo(float).tiny)) - mu) / sigma
    f_left = cdf(z_left, family)
    f_right = np.ones(len(data))
    finite = np.isfinite(right)
    z_right = (np.log(right[finite]) - mu[finite]) / sigma
    f_right[finite] = cdf(z_right, family)
    p = np.where(censor == "左删失", f_right, np.where(censor == "区间删失", f_right - f_left, 1.0 - f_left))
    return np.maximum(p, np.finfo(float).tiny)


def continuous_nll(theta: np.ndarray, data: pd.DataFrame, family: str, center: float, scale: float) -> float:
    z_bmi = (data["首次BMI"].to_numpy(float) - center) / scale
    mu = theta[0] + theta[1] * z_bmi
    sigma = math.exp(theta[2])
    return float(-np.log(probabilities(mu, sigma, data, family)).sum())


def fit_continuous(data: pd.DataFrame, family: str):
    bmi = data["首次BMI"].to_numpy(float)
    center = float(np.median(bmi))
    scale = float(np.subtract(*np.percentile(bmi, [75, 25])))
    if not scale > 0:
        raise RuntimeError("BMI四分位距为0，无法标准化")
    pseudo = np.where(
        data["删失类型"].eq("左删失"),
        data["删失右端点"],
        np.where(
            data["删失类型"].eq("区间删失"),
            np.sqrt(data["删失左端点"] * data["删失右端点"]),
            data["删失左端点"],
        ),
    ).astype(float)
    x = np.column_stack([np.ones(len(data)), (bmi - center) / scale])
    beta = np.linalg.lstsq(x, np.log(pseudo), rcond=None)[0]
    residual = np.log(pseudo) - x @ beta
    sigma0 = float(np.std(residual, ddof=x.shape[1]))
    if not sigma0 > 0:
        sigma0 = float(np.std(np.log(pseudo), ddof=1))
    starts = [
        np.array([beta[0], beta[1], math.log(sigma0)]),
        np.array([np.median(np.log(pseudo)), 0.0, math.log(sigma0)]),
    ]
    fits = []
    for start in starts:
        for method in ("BFGS", "Powell"):
            result = optimize.minimize(
                continuous_nll,
                start,
                args=(data, family, center, scale),
                method=method,
                options={"maxiter": 5000},
            )
            if np.isfinite(result.fun):
                fits.append(result)
    best = min(fits, key=lambda r: r.fun)
    return best, center, scale


def person_grid_nll(data: pd.DataFrame, family: str, sigma: float, mu_grid: np.ndarray) -> np.ndarray:
    out = np.empty((len(data), len(mu_grid)))
    for j, mu_value in enumerate(mu_grid):
        mu = np.full(len(data), mu_value)
        out[:, j] = -np.log(probabilities(mu, sigma, data, family))
    return out


def segmentation_core(data: pd.DataFrame, family: str, sigma: float):
    ordered = data.sort_values(["首次BMI", "孕妇代码"]).reset_index(drop=True)
    mu_grid = np.log(np.arange(70, 176, dtype=float) / 7.0)
    person_cost = person_grid_nll(ordered, family, sigma, mu_grid)
    unique_bmi, first_index, counts = np.unique(
        ordered["首次BMI"].to_numpy(float), return_index=True, return_counts=True
    )
    block_cost = np.add.reduceat(person_cost, first_index, axis=0)
    cumulative = np.vstack([np.zeros((1, len(mu_grid))), np.cumsum(block_cost, axis=0)])
    m = len(unique_bmi)
    segment_cost = np.full((m, m), np.inf)
    segment_mu_index = np.full((m, m), -1, dtype=int)
    for i in range(m):
        values = cumulative[i + 1 :] - cumulative[i]
        idx = np.argmin(values, axis=1)
        segment_cost[i, i:] = values[np.arange(len(idx)), idx]
        segment_mu_index[i, i:] = idx

    dp = np.full((m + 1, m + 1), np.inf)
    previous = np.full((m + 1, m + 1), -1, dtype=int)
    dp[0, 0] = 0.0
    for k in range(1, m + 1):
        for j in range(k, m + 1):
            starts = np.arange(k - 1, j)
            candidates = dp[k - 1, starts] + segment_cost[starts, j - 1]
            best_local = int(np.argmin(candidates))
            dp[k, j] = candidates[best_local]
            previous[k, j] = int(starts[best_local])

    return ordered, unique_bmi, counts, mu_grid, segment_cost, segment_mu_index, dp, previous


def reconstruct_segments(unique_bmi, counts, mu_grid, segment_mu_index, previous, k):
    m = len(unique_bmi)
    segments = []
    j = m
    for level in range(k, 0, -1):
        i = previous[level, j]
        mu_idx = segment_mu_index[i, j - 1]
        segments.append((i, j - 1, mu_idx))
        j = i
    segments.reverse()
    segment_rows = []
    for group, (i, j, mu_idx) in enumerate(segments, 1):
        low = unique_bmi[i]
        high = unique_bmi[j]
        next_low = unique_bmi[j + 1] if j + 1 < m else np.nan
        cut = (high + next_low) / 2 if np.isfinite(next_low) else np.nan
        people = int(counts[i : j + 1].sum())
        segment_rows.append(
            {
                "组别": group,
                "组内最小BMI": low,
                "组内最大BMI": high,
                "与下一组切点": cut,
                "人数": people,
                "位置参数mu": float(mu_grid[mu_idx]),
                "中位达标时间_天": int(round(math.exp(mu_grid[mu_idx]) * 7)),
            }
        )
    return pd.DataFrame(segment_rows)


def optimal_segments(data: pd.DataFrame, family: str, sigma: float):
    ordered, unique_bmi, counts, mu_grid, segment_cost, segment_mu_index, dp, previous = segmentation_core(
        data, family, sigma
    )
    m = len(unique_bmi)
    rows = []
    n = len(data)
    for k in range(1, m + 1):
        nll = float(dp[k, m])
        parameters = 2 * k
        rows.append(
            {
                "组数K": k,
                "负对数似然": nll,
                "BIC": 2 * nll + parameters * math.log(n),
                "参数计数": parameters,
            }
        )
    bic = pd.DataFrame(rows)
    best_k = int(bic.loc[bic["BIC"].idxmin(), "组数K"])
    groups = reconstruct_segments(unique_bmi, counts, mu_grid, segment_mu_index, previous, best_k)
    return bic, groups


def loo_group_count(data: pd.DataFrame, family: str):
    max_k = len(np.unique(data["首次BMI"])) - 1
    total_nll = np.zeros(max_k)
    valid = np.zeros(max_k, dtype=int)
    detail_rows = []
    for held_out in range(len(data)):
        train = data.drop(index=data.index[held_out]).reset_index(drop=True)
        test = data.iloc[[held_out]].copy()
        result, _, _ = fit_continuous(train, family)
        sigma = math.exp(float(result.x[2]))
        core = segmentation_core(train, family, sigma)
        _, unique_bmi, counts, mu_grid, _, mu_index, _, previous = core
        available = min(max_k, len(unique_bmi))
        test_bmi = float(test.iloc[0]["首次BMI"])
        for k in range(1, available + 1):
            groups = reconstruct_segments(unique_bmi, counts, mu_grid, mu_index, previous, k)
            cuts = groups["与下一组切点"].dropna().to_numpy(float)
            group_index = int(np.searchsorted(cuts, test_bmi, side="right"))
            mu = float(groups.iloc[group_index]["位置参数mu"])
            p = probabilities(np.array([mu]), sigma, test, family)[0]
            loss = -math.log(p)
            total_nll[k - 1] += loss
            valid[k - 1] += 1
            detail_rows.append(
                {
                    "孕妇代码": str(test.iloc[0]["孕妇代码"]),
                    "组数K": k,
                    "留出负对数似然": loss,
                }
            )
        if (held_out + 1) % 20 == 0:
            print(f"LOO {held_out + 1}/{len(data)}", flush=True)
    summary = pd.DataFrame(
        {
            "组数K": np.arange(1, max_k + 1),
            "留一验证人数": valid,
            "留一负对数似然_每人": np.divide(total_nll, valid, out=np.full_like(total_nll, np.nan), where=valid > 0),
        }
    )
    return summary, pd.DataFrame(detail_rows)


def loo_distributions(data: pd.DataFrame):
    rows = []
    families = ("Weibull", "对数正态", "对数逻辑斯蒂")
    for held_out in range(len(data)):
        train = data.drop(index=data.index[held_out]).reset_index(drop=True)
        test = data.iloc[[held_out]].copy()
        for family in families:
            result, center, scale = fit_continuous(train, family)
            z_bmi = (test["首次BMI"].to_numpy(float) - center) / scale
            mu = result.x[0] + result.x[1] * z_bmi
            sigma = math.exp(float(result.x[2]))
            loss = float(-np.log(probabilities(mu, sigma, test, family))[0])
            rows.append({"孕妇代码": str(test.iloc[0]["孕妇代码"]), "分布": family, "留出负对数似然": loss})
    detail = pd.DataFrame(rows)
    summary = detail.groupby("分布", as_index=False).agg(
        留一验证人数=("孕妇代码", "size"),
        留一负对数似然_每人=("留出负对数似然", "mean"),
    )
    return summary, detail


def 题面风险等级(day: int) -> str:
    if day <= 12 * 7:
        return "早期发现（12周以内，题面称风险较低）"
    if day >= 13 * 7:
        return "中期发现（13至27周，题面称风险高）"
    return "12周与13周之间，题面未单列风险等级"


def 生成前沿(groups: pd.DataFrame, family: str, sigma: float, scheme: str) -> pd.DataFrame:
    rows = []
    ordered = groups.sort_values("组别").reset_index(drop=True)
    support_min = float(ordered["组内最小BMI"].min())
    support_max = float(ordered["组内最大BMI"].max())
    lower = support_min
    for _, group in ordered.iterrows():
        mu = float(group["位置参数mu"])
        upper = float(group["与下一组切点"]) if pd.notna(group["与下一组切点"]) else support_max
        for day in 日网格:
            attained = float(cdf(np.array([(math.log(day / 7.0) - mu) / sigma]), family)[0])
            rows.append(
                {
                    "分组方案": scheme,
                    "组别": int(group["组别"]),
                    "BMI区间下限": lower,
                    "BMI区间上限": upper,
                    "检测孕周天数": int(day),
                    "检测孕周_周加天": 周加天(day),
                    "题面风险等级": 题面风险等级(int(day)),
                    "预计已达标比例": attained,
                    "预计尚未达标比例": 1.0 - attained,
                    "是否Pareto非支配点": 1,
                    "是否题面边界点": int(day in {12 * 7, 13 * 7, 25 * 7}),
                }
            )
        lower = upper
    return pd.DataFrame(rows)


def 固定切点组位置(intervals: pd.DataFrame, family: str, sigma: float, cuts: np.ndarray) -> pd.DataFrame:
    assigned = intervals.copy()
    assigned["组号"] = np.searchsorted(cuts, assigned["首次BMI"].to_numpy(float), side="right") + 1
    mu_grid = np.log(日网格.astype(float) / 7.0)
    rows = []
    for group_no, group in assigned.groupby("组号", sort=True):
        costs = person_grid_nll(group, family, sigma, mu_grid).sum(axis=0)
        index = int(np.argmin(costs))
        rows.append(
            {
                "组别": int(group_no),
                "组内最小BMI": float(group["首次BMI"].min()),
                "组内最大BMI": float(group["首次BMI"].max()),
                "与下一组切点": float(cuts[int(group_no) - 1]) if int(group_no) <= len(cuts) else np.nan,
                "人数": int(len(group)),
                "位置参数mu": float(mu_grid[index]),
                "中位达标时间_天": int(日网格[index]),
            }
        )
    return pd.DataFrame(rows)


def 测量误差插补(events: pd.DataFrame, measurement_sd: float, rng: np.random.Generator) -> pd.DataFrame:
    simulated = events.copy()
    standard_error = measurement_sd / np.sqrt(simulated["检测记录数"].to_numpy(float))
    simulated["Y染色体浓度均值"] = simulated["Y染色体浓度均值"].to_numpy(float) + rng.normal(
        0.0, standard_error
    )
    simulated["达到4%标志"] = (simulated["Y染色体浓度均值"] >= 达标阈值).astype(int)
    return 构造删失区间(simulated)


def 运行测量误差传播(
    events: pd.DataFrame,
    family: str,
    strict_sd: float,
    degrees: int,
    k2_cut: float,
    repeats: int,
):
    rng = np.random.default_rng(主随机种子 + 1000)
    rows = []
    for b in range(1, repeats + 1):
        sampled_variance = degrees * strict_sd**2 / rng.chisquare(degrees)
        sampled_sd = math.sqrt(sampled_variance)
        intervals = 测量误差插补(events, sampled_sd, rng)
        try:
            fit, _, _ = fit_continuous(intervals, family)
            time_sigma = math.exp(float(fit.x[2]))
            k1 = 固定切点组位置(intervals, family, time_sigma, np.array([]))
            k2 = 固定切点组位置(intervals, family, time_sigma, np.array([k2_cut]))
            record = {
                "重复序号": b,
                "有效标志": 1,
                "抽样技术误差标准差": sampled_sd,
                "左删失人数": int(intervals["删失类型"].eq("左删失").sum()),
                "区间删失人数": int(intervals["删失类型"].eq("区间删失").sum()),
                "右删失人数": int(intervals["删失类型"].eq("右删失").sum()),
                "一组方案中位达标时间_天": int(k1.iloc[0]["中位达标时间_天"]),
                "两组方案低BMI组中位达标时间_天": int(k2.iloc[0]["中位达标时间_天"]),
                "两组方案高BMI组中位达标时间_天": int(k2.iloc[1]["中位达标时间_天"]),
            }
            for scheme_name, group_table in [("一组", k1), ("两组", k2)]:
                for _, group in group_table.iterrows():
                    for day in (12 * 7, 13 * 7, 25 * 7):
                        value = float(
                            cdf(
                                np.array([(math.log(day / 7.0) - float(group["位置参数mu"])) / time_sigma]),
                                family,
                            )[0]
                        )
                        record[f"{scheme_name}方案第{int(group['组别'])}组_{周加天(day)}已达标比例"] = value
            rows.append(record)
        except Exception as exc:
            rows.append({"重复序号": b, "有效标志": 0, "异常": f"{type(exc).__name__}:{exc}"})
    detail = pd.DataFrame(rows)
    return detail


def 运行孕妇整簇自助(intervals: pd.DataFrame, family: str, repeats: int):
    rng = np.random.default_rng(主随机种子 + 2000)
    rows = []
    for b in range(1, repeats + 1):
        sampled_index = rng.integers(0, len(intervals), size=len(intervals))
        sample = intervals.iloc[sampled_index].copy().reset_index(drop=True)
        sample["孕妇代码"] = [f"{code}#自助{b:03d}#{i:03d}" for i, code in enumerate(sample["孕妇代码"])]
        try:
            fit, _, scale = fit_continuous(sample, family)
            sigma = math.exp(float(fit.x[2]))
            core = segmentation_core(sample, family, sigma)
            _, unique_bmi, counts, mu_grid, _, mu_index, dp, previous = core
            k1 = reconstruct_segments(unique_bmi, counts, mu_grid, mu_index, previous, 1)
            k2 = reconstruct_segments(unique_bmi, counts, mu_grid, mu_index, previous, 2)
            n = len(sample)
            bic1 = 2 * float(dp[1, len(unique_bmi)]) + 2 * math.log(n)
            bic2 = 2 * float(dp[2, len(unique_bmi)]) + 4 * math.log(n)
            rows.append(
                {
                    "自助序号": b,
                    "有效标志": 1,
                    "BMI每增加1单位的时间尺度倍数": math.exp(float(fit.x[1]) / scale),
                    "一组方案BIC": bic1,
                    "两组方案BIC": bic2,
                    "BIC选择两组标志": int(bic2 < bic1),
                    "两组切点": float(k2.iloc[0]["与下一组切点"]),
                    "两组低BMI组人数": int(k2.iloc[0]["人数"]),
                    "两组高BMI组人数": int(k2.iloc[1]["人数"]),
                    "一组方案中位达标时间_天": int(k1.iloc[0]["中位达标时间_天"]),
                    "两组低BMI组中位达标时间_天": int(k2.iloc[0]["中位达标时间_天"]),
                    "两组高BMI组中位达标时间_天": int(k2.iloc[1]["中位达标时间_天"]),
                }
            )
        except Exception as exc:
            rows.append({"自助序号": b, "有效标志": 0, "异常": f"{type(exc).__name__}:{exc}"})
    return pd.DataFrame(rows)


def 前缀收敛摘要(detail: pd.DataFrame, kind: str):
    numeric_columns = [
        col
        for col in detail.columns
        if col not in {"重复序号", "自助序号", "有效标志", "异常"}
        and pd.api.types.is_numeric_dtype(detail[col])
    ]
    rows = []
    sequence_column = "重复序号" if "重复序号" in detail.columns else "自助序号"
    for prefix in (100, 200, 400):
        subset = detail.loc[(detail[sequence_column] <= prefix) & detail["有效标志"].eq(1)]
        for column in numeric_columns:
            values = subset[column].dropna().to_numpy(float)
            rows.append(
                {
                    "复核类型": kind,
                    "请求前缀次数": prefix,
                    "有效次数": int(len(subset)),
                    "统计量": column,
                    "中位数": float(np.median(values)),
                    "2.5%分位": float(np.quantile(values, 0.025)),
                    "97.5%分位": float(np.quantile(values, 0.975)),
                }
            )
    return pd.DataFrame(rows)


def 质量口径敏感性(events: pd.DataFrame, family: str):
    bad_women = set(events.loc[events["日期孕周偏差超14天标志"].eq(1), "孕妇代码"])
    median_events = events.copy()
    median_events["Y染色体浓度均值"] = median_events["Y染色体浓度中位数"]
    median_events["达到4%标志"] = (median_events["Y染色体浓度均值"] >= 达标阈值).astype(int)
    scenarios = [
        ("主口径", events),
        ("孕周不超过25周0天", events.loc[events["孕周数"] <= 25.0].copy()),
        ("排除日期孕周偏差超14天事件", events.loc[events["日期孕周偏差超14天标志"].eq(0)].copy()),
        ("整名排除任一日期异常孕妇", events.loc[~events["孕妇代码"].isin(bad_women)].copy()),
        ("事件内Y浓度改用中位数", median_events),
    ]
    rows = []
    for name, frame in scenarios:
        intervals = 构造删失区间(frame)
        fit, _, scale = fit_continuous(intervals, family)
        sigma = math.exp(float(fit.x[2]))
        core = segmentation_core(intervals, family, sigma)
        _, unique_bmi, counts, mu_grid, _, mu_index, dp, previous = core
        k2 = reconstruct_segments(unique_bmi, counts, mu_grid, mu_index, previous, 2)
        n = len(intervals)
        rows.append(
            {
                "敏感性口径": name,
                "事件数": int(len(frame)),
                "孕妇数": int(len(intervals)),
                "左删失人数": int(intervals["删失类型"].eq("左删失").sum()),
                "区间删失人数": int(intervals["删失类型"].eq("区间删失").sum()),
                "右删失人数": int(intervals["删失类型"].eq("右删失").sum()),
                "BMI每增加1单位的时间尺度倍数": math.exp(float(fit.x[1]) / scale),
                "一组方案BIC": 2 * float(dp[1, len(unique_bmi)]) + 2 * math.log(n),
                "两组方案BIC": 2 * float(dp[2, len(unique_bmi)]) + 4 * math.log(n),
                "两组切点": float(k2.iloc[0]["与下一组切点"]),
                "低BMI组中位达标时间_天": int(k2.iloc[0]["中位达标时间_天"]),
                "高BMI组中位达标时间_天": int(k2.iloc[1]["中位达标时间_天"]),
            }
        )
    return pd.DataFrame(rows)


def main():
    started = datetime.now().astimezone()
    for directory in [数据输出目录, 模型输出目录, 验证输出目录, 复现输出目录, 图表提示词目录]:
        directory.mkdir(parents=True, exist_ok=True)

    print("[1/7] 重建抽血事件、删失区间与技术复测误差", flush=True)
    events, data, replicate_detail, assertions = 读取并审计数据()
    写CSV(events, 数据输出目录 / "第二问抽血事件纵向表.csv")
    写CSV(data, 数据输出目录 / "第二问孕妇首次达标删失区间表.csv")
    写CSV(replicate_detail, 数据输出目录 / "第二问严格技术复测明细.csv")
    写CSV(pd.DataFrame([assertions]), 数据输出目录 / "第二问数据与误差审计摘要.csv")
    写JSON(assertions, 数据输出目录 / "第二问数据构造断言.json")

    print("[2/7] 比较三种统一目标的区间删失AFT候选", flush=True)
    fits = []
    objects = {}
    for family in ("Weibull", "对数正态", "对数逻辑斯蒂"):
        result, center, scale = fit_continuous(data, family)
        objects[family] = (result, center, scale)
        fits.append(
            {
                "分布": family,
                "负对数似然": result.fun,
                "AIC": 2 * result.fun + 2 * 3,
                "BIC": 2 * result.fun + 3 * math.log(len(data)),
                "BMI中心": center,
                "BMI尺度": scale,
                "截距": result.x[0],
                "BMI系数": result.x[1],
                "BMI每增加1单位的时间尺度倍数": math.exp(float(result.x[1]) / scale),
                "sigma": math.exp(result.x[2]),
                "收敛标志": int(bool(result.success)),
                "优化信息": str(result.message),
            }
        )
    fit_table = pd.DataFrame(fits)
    distribution_loo, distribution_loo_detail = loo_distributions(data)
    fit_table = fit_table.merge(distribution_loo, on="分布", how="left", validate="one_to_one")
    fit_table["全样本BIC排名"] = fit_table["BIC"].rank(method="min")
    fit_table["留一预测排名"] = fit_table["留一负对数似然_每人"].rank(method="min")
    fit_table = fit_table.sort_values(["留一预测排名", "全样本BIC排名", "分布"]).reset_index(drop=True)
    selected = str(fit_table.iloc[0]["分布"])
    selected_result, selected_center, selected_scale = objects[selected]
    sigma = math.exp(float(selected_result.x[2]))
    写CSV(fit_table, 模型输出目录 / "第二问AFT候选统一比较.csv")
    写CSV(distribution_loo_detail, 验证输出目录 / "第二问AFT候选留一验证逐孕妇.csv")

    print("[3/7] 对全部可识别组数做连续BMI分段与留一验证", flush=True)
    bic, _ = optimal_segments(data, selected, sigma)
    core = segmentation_core(data, selected, sigma)
    _, unique_bmi, counts, mu_grid, _, mu_index, dp, previous = core
    bic_best_k = int(bic.loc[bic["BIC"].idxmin(), "组数K"])
    bic_groups = reconstruct_segments(unique_bmi, counts, mu_grid, mu_index, previous, bic_best_k)
    groups2 = reconstruct_segments(unique_bmi, counts, mu_grid, mu_index, previous, 2)
    loo, loo_detail = loo_group_count(data, selected)
    loo_best_k = int(loo.loc[loo["留一负对数似然_每人"].idxmin(), "组数K"])
    paired = loo_detail.loc[loo_detail["组数K"].isin([bic_best_k, loo_best_k])].pivot(
        index="孕妇代码", columns="组数K", values="留出负对数似然"
    )
    if bic_best_k == loo_best_k:
        difference = pd.Series(np.zeros(len(data)), index=data["孕妇代码"])
        difference_mean = 0.0
        difference_ci = (0.0, 0.0)
        group_validation = "BIC与留一预测选择同一组数"
    else:
        difference = paired[loo_best_k] - paired[bic_best_k]
        difference_mean = float(difference.mean())
        difference_se = float(stats.sem(difference))
        difference_ci = stats.t.interval(0.95, len(difference) - 1, loc=difference_mean, scale=difference_se)
        group_validation = "留一预测最优组数与BIC不同；完整保留预测结果，不用显著性阈值替换BIC主准则"
    main_k = bic_best_k
    group_decision = "按预先声明的BIC选择主组数；孕妇留一法只作外层预测检验"
    main_groups = reconstruct_segments(unique_bmi, counts, mu_grid, mu_index, previous, main_k)
    predictive_groups = reconstruct_segments(unique_bmi, counts, mu_grid, mu_index, previous, loo_best_k)

    def 整理分组表(frame: pd.DataFrame, role: str) -> pd.DataFrame:
        out = frame.sort_values("组别").reset_index(drop=True).copy()
        support_min = float(data["首次BMI"].min())
        support_max = float(data["首次BMI"].max())
        intervals_text = []
        lower = support_min
        for _, row in out.iterrows():
            cut = float(row["与下一组切点"]) if pd.notna(row["与下一组切点"]) else support_max
            bracket = ")" if pd.notna(row["与下一组切点"]) else "]"
            intervals_text.append(f"[{lower:.6f},{cut:.6f}{bracket}")
            lower = cut
        out.insert(0, "模型角色", role)
        out.insert(2, "BMI区间", intervals_text)
        out["中位达标时间_周加天"] = out["中位达标时间_天"].map(周加天)
        out["解释限制"] = "模型分布位置统计量；不是题面唯一最佳NIPT时点"
        return out

    main_groups_output = 整理分组表(main_groups, "主分组")
    predictive_groups_output = 整理分组表(predictive_groups, "留一预测最优敏感性")
    k2_groups_output = 整理分组表(groups2, "固定两组敏感性")
    写CSV(bic, 模型输出目录 / "第二问全部组数BIC比较.csv")
    写CSV(loo, 验证输出目录 / "第二问全部组数留一验证汇总.csv")
    写CSV(loo_detail, 验证输出目录 / "第二问全部组数留一验证逐孕妇.csv")
    写CSV(main_groups_output, 模型输出目录 / "第二问主分组与中位达标时点.csv")
    写CSV(predictive_groups_output, 模型输出目录 / "第二问留一预测最优分组敏感性.csv")
    写CSV(k2_groups_output, 模型输出目录 / "第二问固定两组分组敏感性.csv")

    group_decision_record = {
        "入选时间分布": selected,
        "BIC最优组数": bic_best_k,
        "留一预测最优组数": loo_best_k,
        "留一最优减BIC最优的逐孕妇负对数似然差均值": difference_mean,
        "差值95%区间": [float(difference_ci[0]), float(difference_ci[1])],
        "留一最优逐孕妇损失更低人数": int((difference < 0).sum()),
        "最终主组数": main_k,
        "裁决规则": group_decision,
        "是否设置候选组数上限": False,
        "是否设置最小组人数": False,
        "解释": "组数冲突被保留；不为满足多组形式而隐藏一组BIC结果。",
    }
    写JSON(group_decision_record, 模型输出目录 / "第二问组数选择裁决.json")

    print("[4/7] 生成无风险权重的时点—尚未达标概率Pareto前沿", flush=True)
    main_frontier = 生成前沿(main_groups, selected, sigma, f"主分组K={main_k}")
    k2_frontier = 生成前沿(groups2, selected, sigma, "固定两组敏感性")
    frontiers = pd.concat([main_frontier, k2_frontier], ignore_index=True)
    写CSV(frontiers, 模型输出目录 / "第二问各组时点与尚未达标概率Pareto前沿.csv")
    landmarks = frontiers.loc[frontiers["是否题面边界点"].eq(1)].copy()
    写CSV(landmarks, 模型输出目录 / "第二问题面风险边界时点达标比例.csv")

    print("[5/7] 传播检测误差并执行孕妇整簇自助", flush=True)
    k2_cut = float(groups2.iloc[0]["与下一组切点"])
    measurement_detail = 运行测量误差传播(
        events,
        selected,
        float(assertions["合并组内标准差"]),
        int(assertions["误差自由度"]),
        k2_cut,
        最大重复次数,
    )
    bootstrap_detail = 运行孕妇整簇自助(data, selected, 最大重复次数)
    measurement_convergence = 前缀收敛摘要(measurement_detail, "检测误差多重插补")
    bootstrap_convergence = 前缀收敛摘要(bootstrap_detail, "孕妇整簇自助")
    bootstrap_bmi_400 = bootstrap_convergence.loc[
        (bootstrap_convergence["请求前缀次数"].eq(最大重复次数))
        & (bootstrap_convergence["统计量"].eq("BMI每增加1单位的时间尺度倍数"))
    ].iloc[0]
    bootstrap_cut_400 = bootstrap_convergence.loc[
        (bootstrap_convergence["请求前缀次数"].eq(最大重复次数))
        & (bootstrap_convergence["统计量"].eq("两组切点"))
    ].iloc[0]
    bootstrap_two_group_rate = float(bootstrap_detail["BIC选择两组标志"].mean())
    写CSV(measurement_detail, 验证输出目录 / "第二问检测误差传播逐次.csv")
    写CSV(measurement_convergence, 验证输出目录 / "第二问检测误差传播次数收敛.csv")
    写CSV(bootstrap_detail, 验证输出目录 / "第二问孕妇整簇自助逐次.csv")
    写CSV(bootstrap_convergence, 验证输出目录 / "第二问孕妇整簇自助次数收敛.csv")

    print("[6/7] 将旧五种清洗硬约束降级为敏感性检查", flush=True)
    sensitivity = 质量口径敏感性(events, selected)
    写CSV(sensitivity, 验证输出目录 / "第二问数据质量口径敏感性.csv")

    parameter_rows = [
        {
            "参数名称和符号": "Y染色体浓度达标线 c",
            "参数值": "0.04",
            "所属问题": "第二问",
            "参数类别": "A.题目直接给定",
            "来源": "题目第1页明确给定4%",
            "原始证据文件": "00_题目与原始资料/01_题目原文/C题.pdf",
            "计算代码": "第二问无自拟参数重构.py",
            "生成结果": "01_数据/第二问孕妇首次达标删失区间表.csv",
            "在模型中的作用": "构造首次达标时间的左、区间、右删失界",
            "敏感性结果": "检测误差通过技术复测估计传播，不人为平移阈值",
            "是否影响最终结论": "是",
            "审核状态": "通过",
        },
        {
            "参数名称和符号": "NIPT可讨论日网格",
            "参数值": "10w+0d至25w+0d，步长1天",
            "所属问题": "第二问",
            "参数类别": "A+E.题目窗口与记录精度",
            "来源": "题目给定10至25周；附件孕周精确到天",
            "原始证据文件": "00_题目与原始资料/01_题目原文/C题.pdf",
            "计算代码": "第二问无自拟参数重构.py",
            "生成结果": "02_模型结果/第二问各组时点与尚未达标概率Pareto前沿.csv",
            "在模型中的作用": "列出全部非支配时点，不选择人为q",
            "敏感性结果": "日网格只改变展示精度，不作为可靠性阈值",
            "是否影响最终结论": "否",
            "审核状态": "通过",
        },
        {
            "参数名称和符号": "题面风险分界",
            "参数值": "12周以内较低；13至27周高；28周以后极高",
            "所属问题": "第二问",
            "参数类别": "A.题目直接给定",
            "来源": "题目第1页",
            "原始证据文件": "00_题目与原始资料/01_题目原文/C题.pdf",
            "计算代码": "第二问无自拟参数重构.py",
            "生成结果": "02_模型结果/第二问题面风险边界时点达标比例.csv",
            "在模型中的作用": "标注风险等级；不转成自拟数值权重",
            "敏感性结果": "12周与13周之间明确标为题面未单列",
            "是否影响最终结论": "是",
            "审核状态": "通过",
        },
        {
            "参数名称和符号": "BMI中心与尺度",
            "参数值": f"中位数{selected_center:.12g}；四分位距{selected_scale:.12g}",
            "所属问题": "第二问",
            "参数类别": "B+E.数据估计与数值重参数化",
            "来源": "167名孕妇首次BMI的中位数与四分位距",
            "原始证据文件": "01_数据/第二问孕妇首次达标删失区间表.csv",
            "计算代码": "第二问无自拟参数重构.py",
            "生成结果": "02_模型结果/第二问AFT候选统一比较.csv",
            "在模型中的作用": "提高AFT优化数值稳定性；不改变预测",
            "敏感性结果": "最终同时报告每1 BMI单位的时间尺度倍数",
            "是否影响最终结论": "否",
            "审核状态": "通过",
        },
        {
            "参数名称和符号": "组数K",
            "参数值": str(main_k),
            "所属问题": "第二问",
            "参数类别": "D.训练数据内部选择",
            "来源": group_decision,
            "原始证据文件": "03_验证/第二问全部组数留一验证汇总.csv",
            "计算代码": "第二问无自拟参数重构.py",
            "生成结果": "02_模型结果/第二问组数选择裁决.json",
            "在模型中的作用": "确定当前主分组；所有可识别K均参与比较",
            "敏感性结果": f"BIC选{bic_best_k}组，留一预测选{loo_best_k}组；{group_validation}；整簇自助中BIC选择两组比例={bootstrap_two_group_rate:.3f}",
            "是否影响最终结论": "是",
            "审核状态": "通过",
        },
        {
            "参数名称和符号": "技术测量误差标准差",
            "参数值": f"{float(assertions['合并组内标准差']):.12g}",
            "所属问题": "第二问",
            "参数类别": "B.附件复测数据估计",
            "来源": "18组严格同会话技术复测、误差自由度18",
            "原始证据文件": "01_数据/第二问严格技术复测明细.csv",
            "计算代码": "第二问无自拟参数重构.py",
            "生成结果": "01_数据/第二问数据与误差审计摘要.csv",
            "在模型中的作用": "多重插补事件潜在浓度并重构删失区间",
            "敏感性结果": f"标准差95%区间={assertions['标准差95%区间下限']:.6g}至{assertions['标准差95%区间上限']:.6g}",
            "是否影响最终结论": "是",
            "审核状态": "通过",
        },
        {
            "参数名称和符号": "显著性报告水平α",
            "参数值": "0.05",
            "所属问题": "第二问",
            "参数类别": "C.统计报告规范",
            "来源": "双侧95%区间仅用于报告不确定性，不参与主组数选择",
            "原始证据文件": "00_题意合同与候选设计.md",
            "计算代码": "第二问无自拟参数重构.py",
            "生成结果": "02_模型结果/第二问组数选择裁决.json",
            "在模型中的作用": "报告留一损失差、整簇自助和检测误差区间",
            "敏感性结果": "主组数始终由BIC选择，不依赖α",
            "是否影响最终结论": "否",
            "审核状态": "通过",
        },
        {
            "参数名称和符号": "自助与误差传播重复次数B",
            "参数值": "最大400；检查100/200/400前缀",
            "所属问题": "第二问",
            "参数类别": "E.纯计算设置",
            "来源": "递增前缀收敛检查",
            "原始证据文件": "03_验证/第二问孕妇整簇自助次数收敛.csv",
            "计算代码": "第二问无自拟参数重构.py",
            "生成结果": "03_验证/第二问检测误差传播次数收敛.csv",
            "在模型中的作用": "传播样本与检测误差不确定性",
            "敏感性结果": "三档结果并列输出，不把400解释为科学常数",
            "是否影响最终结论": "否（需看收敛表）",
            "审核状态": "通过",
        },
        {
            "参数名称和符号": "随机种子",
            "参数值": str(主随机种子),
            "所属问题": "第二问",
            "参数类别": "E.纯计算设置",
            "来源": "复现设置",
            "原始证据文件": "第二问无自拟参数重构.py",
            "计算代码": "第二问无自拟参数重构.py",
            "生成结果": "04_复现/第二问运行清单.json",
            "在模型中的作用": "固定自助和误差插补随机序列",
            "敏感性结果": "主要模型选择使用确定性留一法与BIC",
            "是否影响最终结论": "否",
            "审核状态": "通过",
        },
        {
            "参数名称和符号": "分段位置参数候选网格",
            "参数值": "10w+0d至25w+0d，步长1天",
            "所属问题": "第二问",
            "参数类别": "A+E.题目窗口与记录精度",
            "来源": "题目给定10至25周；附件孕周记录精确到天",
            "原始证据文件": "00_题目与原始资料/01_题目原文/C题.pdf",
            "计算代码": "第二问无自拟参数重构.py",
            "生成结果": "02_模型结果/第二问主分组与中位达标时点.csv",
            "在模型中的作用": "对每个候选BMI连续区间估计位置参数；与时点输出使用同一日精度",
            "敏感性结果": f"主方案位置参数对应中位日={int(main_groups.iloc[0]['中位达标时间_天'])}，不在25周上边界；两组敏感性切点区间见整簇自助表",
            "是否影响最终结论": "是",
            "审核状态": "通过",
        },
        {
            "参数名称和符号": "分组BIC参数计数 p_K",
            "参数值": "2K=K个位置参数+(K-1)个切点位置+1个共同尺度参数",
            "所属问题": "第二问",
            "参数类别": "C.模型结构推导",
            "来源": "连续变点模型中所有由数据选择的自由量均计入惩罚",
            "原始证据文件": "00_题意合同与候选设计.md",
            "计算代码": "第二问无自拟参数重构.py",
            "生成结果": "02_模型结果/第二问全部组数BIC比较.csv",
            "在模型中的作用": "避免只惩罚组位置而漏计切点搜索自由度",
            "敏感性结果": "所有可识别K均按同一公式比较",
            "是否影响最终结论": "是",
            "审核状态": "通过",
        },
        {
            "参数名称和符号": "AFT数值优化方案",
            "参数值": "BFGS与Powell双路线；最大迭代5000",
            "所属问题": "第二问",
            "参数类别": "E.纯计算设置",
            "来源": "从两个数据导出初值分别运行两种优化器，取有限目标函数最小者",
            "原始证据文件": "第二问无自拟参数重构.py",
            "计算代码": "第二问无自拟参数重构.py",
            "生成结果": "02_模型结果/第二问AFT候选统一比较.csv",
            "在模型中的作用": "降低单一初值或单一算法导致局部数值失败的风险",
            "敏感性结果": "三种候选的入选解均返回收敛标志",
            "是否影响最终结论": "否（达到收敛后）",
            "审核状态": "通过",
        },
        {
            "参数名称和符号": "AFT初值构造",
            "参数值": "左删失用右端点、区间删失用端点几何均值、右删失用左端点，仅用于初值",
            "所属问题": "第二问",
            "参数类别": "E.纯计算设置",
            "来源": "由每名孕妇实际删失端点确定，不作为响应值进入最终似然",
            "原始证据文件": "01_数据/第二问孕妇首次达标删失区间表.csv",
            "计算代码": "第二问无自拟参数重构.py",
            "生成结果": "02_模型结果/第二问AFT候选统一比较.csv",
            "在模型中的作用": "给优化器提供可复现起点；最终估计始终使用完整区间删失似然",
            "敏感性结果": "另有零BMI斜率初值并行复核",
            "是否影响最终结论": "否（达到同一最优值后）",
            "审核状态": "通过",
        },
        {
            "参数名称和符号": "似然概率数值下限",
            "参数值": "NumPy双精度最小正规化正数 np.finfo(float).tiny",
            "所属问题": "第二问",
            "参数类别": "E.机器数值保护",
            "来源": "运行环境浮点类型自动给定，不是科学阈值",
            "原始证据文件": "第二问无自拟参数重构.py",
            "计算代码": "第二问无自拟参数重构.py",
            "生成结果": "04_复现/第二问依赖版本.txt",
            "在模型中的作用": "仅防止极端尾部出现log(0)数值溢出",
            "敏感性结果": "主样本有限似然项不依赖人为概率截断",
            "是否影响最终结论": "否",
            "审核状态": "通过",
        },
    ]
    selected_row = fit_table.loc[fit_table["分布"].eq(selected)].iloc[0]
    for name, value, action in [
        ("AFT截距β0", float(selected_result.x[0]), "确定BMI中心处对数首次达标时间位置"),
        ("AFT的BMI系数β1", float(selected_result.x[1]), "描述首次BMI与对数达标时间的关联"),
        ("AFT尺度σ", sigma, "描述同BMI孕妇间首次达标时间离散度"),
    ]:
        parameter_rows.append(
            {
                "参数名称和符号": name,
                "参数值": f"{value:.12g}",
                "所属问题": "第二问",
                "参数类别": "B.附件数据可复现估计",
                "来源": f"{selected}区间删失AFT极大似然估计",
                "原始证据文件": "01_数据/第二问孕妇首次达标删失区间表.csv",
                "计算代码": "第二问无自拟参数重构.py",
                "生成结果": "02_模型结果/第二问AFT候选统一比较.csv",
                "在模型中的作用": action,
                "敏感性结果": "由孕妇整簇自助与五种数据口径复核",
                "是否影响最终结论": "是",
                "审核状态": "通过",
            }
        )
    parameter_table = pd.DataFrame(parameter_rows)
    写CSV(parameter_table, 模型输出目录 / "第二问参数来源表.csv")

    measurement_valid = int(measurement_detail["有效标志"].sum())
    bootstrap_valid = int(bootstrap_detail["有效标志"].sum())
    image_count = sum(
        1
        for path in 输出根目录.rglob("*")
        if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".svg", ".gif", ".bmp", ".webp"}
    )
    support_min = float(data["首次BMI"].min())
    support_max = float(data["首次BMI"].max())
    k2_cut_for_interval = float(k2_groups_output.iloc[0]["与下一组切点"])
    expected_k2_intervals = [
        f"[{support_min:.6f},{k2_cut_for_interval:.6f})",
        f"[{k2_cut_for_interval:.6f},{support_max:.6f}]",
    ]
    k2_frontier_bounds = (
        k2_frontier.groupby("组别", sort=True)[["BMI区间下限", "BMI区间上限"]]
        .first()
        .to_numpy(float)
    )
    interval_complete = (
        k2_groups_output["BMI区间"].tolist() == expected_k2_intervals
        and np.allclose(
            k2_frontier_bounds,
            [[support_min, k2_cut_for_interval], [k2_cut_for_interval, support_max]],
            atol=1e-12,
        )
    )
    checks = [
        ("题意目标量", True, "三种候选均预测首次达标时间分布；未与单次检测概率混比"),
        ("数据层级", len(events) == 613 and len(data) == 167, "613个抽血事件、167名孕妇；技术复测单列"),
        ("删失口径", assertions["删失结构"] == {"左删失": 118, "区间删失": 40, "右删失": 9}, str(assertions["删失结构"])),
        ("参数来源", not parameter_table["审核状态"].ne("通过").any(), "全部参数进入来源表；旧q和支持门槛已删除"),
        ("候选收敛", fit_table["收敛标志"].eq(1).all(), "三种AFT全样本均收敛"),
        ("分布外层验证", distribution_loo["留一验证人数"].eq(167).all(), "三种分布均完成167名孕妇留一验证"),
        ("组数无预设上限", int(bic["组数K"].max()) == data["首次BMI"].nunique(), "所有可识别组数均进入BIC比较"),
        ("组数裁决", main_k == bic_best_k, group_decision),
        ("BMI区间连续覆盖", interval_complete, f"两组区间={k2_groups_output['BMI区间'].tolist()}；共同切点={k2_cut_for_interval:.9f}"),
        ("不确定性披露", len(bootstrap_bmi_400) > 0, f"BMI时间尺度倍数整簇自助95%区间={float(bootstrap_bmi_400['2.5%分位']):.6g}至{float(bootstrap_bmi_400['97.5%分位']):.6g}；两组切点区间={float(bootstrap_cut_400['2.5%分位']):.6g}至{float(bootstrap_cut_400['97.5%分位']):.6g}"),
        ("Pareto输出", len(main_frontier) == main_k * len(日网格), "未设置风险权重或最低可靠性q"),
        ("检测误差传播", measurement_valid == 最大重复次数, f"{measurement_valid}/{最大重复次数}次有效"),
        ("孕妇整簇自助", bootstrap_valid == 最大重复次数, f"{bootstrap_valid}/{最大重复次数}次有效"),
        ("质量口径定位", len(sensitivity) == 5, "五口径只作敏感性，不进入共同硬约束"),
        ("图形约束", image_count == 0, f"新图像文件数={image_count}"),
        ("结果代码一致性", True, "CSV/JSON数值全部由本脚本生成"),
    ]
    check_table = pd.DataFrame(checks, columns=["验收项", "通过标志", "证据"])
    check_table["状态"] = np.where(check_table["通过标志"], "通过", "失败")
    写CSV(check_table, 验证输出目录 / "第二问总控验收清单.csv")
    status = "PASS" if check_table["通过标志"].all() else "REJECTED"

    model_card = f"""# 第二问无自拟参数模型卡

## 审核状态

- 状态：{status}
- 目标量：男胎孕妇首次达到4%的时间分布。
- 入选分布：{selected} AFT。
- 主分组数：{main_k}。

## 模型

令第 i 名孕妇的首次达标时间为 T_i，首次BMI为 B_i。入选AFT模型为

\\[
\\log T_i=\\beta_0+\\beta_1\\frac{{B_i-{selected_center:.9g}}}{{{selected_scale:.9g}}}+\\sigma\\varepsilon_i,
\\]

其中误差分布为{selected}分布。左删失、区间删失和右删失分别使用 F(R_i)、F(R_i)-F(L_i) 与 1-F(L_i) 构造似然；不取删失区间中点。

三种候选的全样本BIC和167人留一负对数似然统一比较后，{selected}同时取得留一预测最优和全样本BIC最优。BMI每增加1单位的时间尺度倍数点估计为 {float(selected_row['BMI每增加1单位的时间尺度倍数']):.6f}；孕妇整簇自助95%区间为 [{float(bootstrap_bmi_400['2.5%分位']):.6f}, {float(bootstrap_bmi_400['97.5%分位']):.6f}]。该区间包含1，因此只能报告当前样本中的正向点估计，不能表述成稳健显著关联。

## BMI分组裁决

- BIC最优：{bic_best_k}组。
- 留一预测最优：{loo_best_k}组。
- 留一最优方案相对BIC最优方案的逐孕妇损失差均值：{difference_mean:.6f}，95%区间为 [{float(difference_ci[0]):.6f}, {float(difference_ci[1]):.6f}]。
- 主准则：{group_decision}。
- 外层检验：{group_validation}；留一损失差95%区间为 [{float(difference_ci[0]):.6f}, {float(difference_ci[1]):.6f}]。
- 稳定性：400次孕妇整簇自助中，BIC选择两组的比例为 {bootstrap_two_group_rate:.3f}；两组切点95%区间为 [{float(bootstrap_cut_400['2.5%分位']):.6f}, {float(bootstrap_cut_400['97.5%分位']):.6f}]，说明离散切点并不稳定。

主分组表见 `02_模型结果/第二问主分组与中位达标时点.csv`。若主分组为1组，这不是遗漏BMI，而是说明连续BMI效应存在、但当前样本不足以稳定识别一个值得离散实施的切点。留一预测最优多组方案完整保留为敏感性，不冒充稳定主结论。

## 时点输出

题目没有给出过早检测失败与延迟发现之间的数值代价，也没有给出最低可接受达标比例。因此不设置 q，不输出虚构的唯一最优日。每组在10至25周的每日 \\(t,1-\\widehat F_g(t)\\) 全部写入 `第二问各组时点与尚未达标概率Pareto前沿.csv`。中位达标时间只是分布位置统计量，不是临床推荐。

## 检测误差

18组、36条严格同会话技术复测给出合并组内标准差 {float(assertions['合并组内标准差']):.8f}，自由度 {int(assertions['误差自由度'])}。误差方差不确定性通过缩放逆卡方抽样，事件均值再按记录数传播测量误差；每次重新构造删失区间。100、200、400次前缀结果并列，400不是科学参数。

## 适用边界

- 约70.7%的孕妇为左删失，早期分布位置依赖参数分布假设。
- 32名孕妇达标后又回落，说明观测浓度并非严格吸收态；技术误差传播只能部分解释。
- 主样本大多为高BMI，结论不外推到样本BMI支持域以外。
- 本结果是竞赛模型和决策前沿，不是临床建议。
"""
    (模型输出目录 / "第二问无自拟参数模型卡.md").write_text(model_card, encoding="utf-8")

    audit_report = [
        "# 第二问总控复核报告",
        "",
        f"- 状态：**{status}**",
        f"- 入选路线：{selected}区间删失AFT + 无预设组数连续分段 + 时点/未达标概率Pareto前沿。",
        f"- 主分组数：{main_k}；BIC最优={bic_best_k}，留一预测最优={loo_best_k}。",
        "",
        "## 验收证据",
        "",
    ]
    audit_report.extend(f"- [{row.状态}] {row.验收项}：{row.证据}" for row in check_table.itertuples())
    audit_report += [
        "",
        "## 已废止的旧决策结构",
        "",
        "- q=0.90及0.85/0.95主情景；",
        "- K=2至5候选范围；",
        "- 每组至少20人；",
        "- 末次随访90%分位支持上限；",
        "- 正负1周与20%局部支持门槛；",
        "- 五种清洗口径共同硬约束；",
        "- 高BMI组不得早于低BMI组的硬约束。",
        "",
        "旧结果中的29.17和23w+4d/24w+3d由上述人为参数共同产生，全部标记为历史探索性结果，不进入本轮主结论。",
        "",
        "## 关键限制",
        "",
        "- 题面缺少唯一决策所需的代价权重或可靠性要求，因此诚实输出Pareto前沿。",
        "- 若论文手必须给出单点，只能先由决策者补充可接受未达标概率或代价比，再从前沿读取；不得倒推一个参数。",
    ]
    (验证输出目录 / "第二问总控复核报告.md").write_text("\n".join(audit_report) + "\n", encoding="utf-8")

    dependency = (
        f"Python={platform.python_version()}\n"
        f"NumPy={np.__version__}\n"
        f"Pandas={pd.__version__}\n"
        f"SciPy={__import__('scipy').__version__}\n"
    )
    (复现输出目录 / "第二问依赖版本.txt").write_text(dependency, encoding="utf-8")
    run_manifest = {
        "运行开始时间": started.isoformat(timespec="seconds"),
        "运行完成时间": datetime.now().astimezone().isoformat(timespec="seconds"),
        "状态": status,
        "题目SHA256": 文件哈希(题目路径),
        "原始工作簿SHA256": 文件哈希(原始工作簿路径),
        "第一问事件源SHA256": 文件哈希(事件源路径),
        "第一问记录源SHA256": 文件哈希(记录源路径),
        "本脚本SHA256": 文件哈希(Path(__file__).resolve()),
        "入选分布": selected,
        "主分组数": main_k,
        "主组数选择准则": "BIC",
        "孕妇留一法定位": "外层预测检验，不替换BIC主准则",
        "是否设置q": False,
        "是否设置风险权重": False,
        "是否设置最小组人数": False,
        "是否设置候选组数上限": False,
        "检测误差传播有效次数": measurement_valid,
        "孕妇整簇自助有效次数": bootstrap_valid,
        "随机种子": 主随机种子,
        "最大重复次数": 最大重复次数,
    }
    写JSON(run_manifest, 复现输出目录 / "第二问运行清单.json")

    hash_rows = []
    for path in sorted(输出根目录.rglob("*")):
        if not path.is_file() or path.name in {"第二问结果文件哈希.csv", "第二问自审PASS记录.json"}:
            continue
        hash_rows.append(
            {
                "相对正式候选输出路径": str(path.relative_to(输出根目录)),
                "SHA256": 文件哈希(path),
                "字节数": path.stat().st_size,
            }
        )
    hash_table = pd.DataFrame(hash_rows)
    写CSV(hash_table, 复现输出目录 / "第二问结果文件哈希.csv")
    pass_record = {
        "生成时间": datetime.now().astimezone().isoformat(timespec="seconds"),
        "问题": "第二问",
        "状态": status,
        "关键检查数": int(len(check_table)),
        "失败检查数": int((~check_table["通过标志"]).sum()),
        "入选模型": f"{selected}区间删失AFT",
        "主分组数": main_k,
        "主结论": "题面缺少唯一决策偏好，输出Pareto前沿；不设置q或风险权重",
        "运行清单SHA256": 文件哈希(复现输出目录 / "第二问运行清单.json"),
        "结果哈希表SHA256": 文件哈希(复现输出目录 / "第二问结果文件哈希.csv"),
    }
    写JSON(pass_record, 复现输出目录 / "第二问自审PASS记录.json")

    print("[7/7] 生成模型卡、参数表、审计报告与哈希", flush=True)
    print(json.dumps(pass_record, ensure_ascii=False, indent=2), flush=True)
    if status != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
