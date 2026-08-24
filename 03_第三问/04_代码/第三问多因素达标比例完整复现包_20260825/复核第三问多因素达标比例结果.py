from __future__ import annotations

import hashlib
import json
import math
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from scipy.special import expit, logit
from sklearn.metrics import average_precision_score, roc_auc_score


脚本路径 = Path(__file__).resolve()
项目目录 = 脚本路径.parent


def 定位工作区(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / "00_题目与原始资料/02_原始数据/附件.xlsx").is_file():
            return candidate
    raise FileNotFoundError("无法从脚本位置向上找到含原始附件的C题论文工作区")


工作区 = 定位工作区(项目目录)
输出根目录 = 项目目录 / "正式候选输出"
独立复核目录 = 项目目录 / "独立复核"
独立复核目录.mkdir(parents=True, exist_ok=True)

题目路径 = 工作区 / "00_题目与原始资料/01_题目原文/C题.pdf"
原始工作簿路径 = 工作区 / "00_题目与原始资料/02_原始数据/附件.xlsx"
第一问事件源路径 = 工作区 / "01_第一问/04_代码/关系建模/第一问建模完整复现包_20260825/00_共同口径/冻结数据/第一问抽血事件层冻结样本.csv"
第一问记录源路径 = 工作区 / "01_第一问/04_代码/关系建模/第一问建模完整复现包_20260825/00_共同口径/冻结数据/第一问记录层冻结样本.csv"
第二问运行清单路径 = 工作区 / "02_第二问/04_代码/第二问无自拟参数完整复现包_20260825/正式候选输出/04_复现/第二问运行清单.json"
建模脚本路径 = 项目目录 / "第三问多因素达标比例建模.py"
一键脚本路径 = 项目目录 / "一键运行第三问多因素建模.ps1"

达标阈值 = 0.04
数值概率下限 = np.finfo(float).eps


def 文件哈希(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def 读JSON(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def 含中文(value: str) -> bool:
    return any("\u4e00" <= char <= "\u9fff" for char in str(value))


验收记录: list[dict[str, object]] = []


def 验收(name: str, condition: bool, evidence: str):
    验收记录.append(
        {
            "验收项": name,
            "通过标志": bool(condition),
            "状态": "通过" if condition else "失败",
            "独立证据": evidence,
        }
    )


def 独立技术复测(records: pd.DataFrame, event_keys: set[str]):
    selected = records.loc[records["抽血事件键"].isin(event_keys)].copy()
    columns = ["孕妇代码", "抽血次数", "检测日期规范值", "孕周原始值"]
    residual_sum = 0.0
    degrees = 0
    groups = 0
    record_count = 0
    crossing = 0
    for _, group in selected.groupby(columns, dropna=False, sort=False):
        if len(group) < 2:
            continue
        values = group["Y染色体浓度"].to_numpy(float)
        mean = float(values.mean())
        residual_sum += float(np.sum((values - mean) ** 2))
        degrees += len(values) - 1
        groups += 1
        record_count += len(values)
        crossing += int(float(values.min()) < 达标阈值 <= float(values.max()))
    return {
        "组数": groups,
        "记录数": record_count,
        "自由度": degrees,
        "合并组内标准差": math.sqrt(residual_sum / degrees),
        "跨阈值组数": crossing,
    }


def 拟合独立主模型(events: pd.DataFrame, references: pd.DataFrame):
    ref = references.set_index("变量")
    week = (events["孕周数"] - float(ref.loc["孕周数", "中心_中位数"])) / float(
        ref.loc["孕周数", "尺度_四分位距"]
    )
    X = pd.DataFrame(
        {
            "截距": np.ones(len(events)),
            "孕周标准化": week,
            "首次BMI标准化": (events["首次BMI"] - float(ref.loc["首次BMI", "中心_中位数"]))
            / float(ref.loc["首次BMI", "尺度_四分位距"]),
            "首次年龄标准化": (events["首次年龄"] - float(ref.loc["首次年龄", "中心_中位数"]))
            / float(ref.loc["首次年龄", "尺度_四分位距"]),
            "首次身高标准化": (events["首次身高"] - float(ref.loc["首次身高", "中心_中位数"]))
            / float(ref.loc["首次身高", "尺度_四分位距"]),
            "首次生产次数": events["首次生产次数"].to_numpy(float),
        }
    )
    fits = []
    for method in ["lbfgs", "powell"]:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = sm.MixedLM(
                logit(events["Y染色体浓度均值"].to_numpy(float)),
                X,
                groups=events["孕妇代码"].astype(str),
                exog_re=np.column_stack([np.ones(len(events)), week.to_numpy(float)]),
            ).fit(reml=False, method=method, maxiter=2000, disp=False)
        if np.isfinite(result.llf):
            fits.append((method, result))
    if not fits:
        raise RuntimeError("独立重拟合的两个公开优化器均未返回有限似然")
    method, result = max(fits, key=lambda item: float(item[1].llf))
    result._独立复核优化器 = method
    return result, X, ref


def 独立预测概率(events: pd.DataFrame, result, ref, bmi_delta: float | None = None):
    week = (events["孕周数"] - float(ref.loc["孕周数", "中心_中位数"])) / float(
        ref.loc["孕周数", "尺度_四分位距"]
    )
    X = pd.DataFrame(
        {
            "截距": np.ones(len(events)),
            "孕周标准化": week,
            "首次BMI标准化": (events["首次BMI"] - float(ref.loc["首次BMI", "中心_中位数"]))
            / float(ref.loc["首次BMI", "尺度_四分位距"]),
            "首次年龄标准化": (events["首次年龄"] - float(ref.loc["首次年龄", "中心_中位数"]))
            / float(ref.loc["首次年龄", "尺度_四分位距"]),
            "首次身高标准化": (events["首次身高"] - float(ref.loc["首次身高", "中心_中位数"]))
            / float(ref.loc["首次身高", "尺度_四分位距"]),
            "首次生产次数": events["首次生产次数"].to_numpy(float),
        }
    )
    beta = pd.Series(np.asarray(result.fe_params), index=X.columns)
    eta = X.to_numpy(float) @ beta.to_numpy(float)
    if bmi_delta is not None:
        eta = eta - float(beta["首次BMI标准化"]) * X["首次BMI标准化"].to_numpy(float) + bmi_delta
    random_X = np.column_stack([np.ones(len(events)), week.to_numpy(float)])
    variance = float(result.scale) + np.einsum(
        "ij,jk,ik->i", random_X, np.asarray(result.cov_re), random_X
    )
    return stats.norm.cdf((eta - logit(达标阈值)) / np.sqrt(variance))


def 独立分段(events: pd.DataFrame, result, ref):
    beta = pd.Series(
        np.asarray(result.fe_params),
        index=["截距", "孕周标准化", "首次BMI标准化", "首次年龄标准化", "首次身高标准化", "首次生产次数"],
    )
    week = (events["孕周数"] - float(ref.loc["孕周数", "中心_中位数"])) / float(
        ref.loc["孕周数", "尺度_四分位距"]
    )
    X = pd.DataFrame(
        {
            "截距": np.ones(len(events)),
            "孕周标准化": week,
            "首次BMI标准化": (events["首次BMI"] - float(ref.loc["首次BMI", "中心_中位数"]))
            / float(ref.loc["首次BMI", "尺度_四分位距"]),
            "首次年龄标准化": (events["首次年龄"] - float(ref.loc["首次年龄", "中心_中位数"]))
            / float(ref.loc["首次年龄", "尺度_四分位距"]),
            "首次身高标准化": (events["首次身高"] - float(ref.loc["首次身高", "中心_中位数"]))
            / float(ref.loc["首次身高", "尺度_四分位距"]),
            "首次生产次数": events["首次生产次数"].to_numpy(float),
        }
    )
    eta_without_bmi = X.to_numpy(float) @ beta.to_numpy(float) - float(beta["首次BMI标准化"]) * X[
        "首次BMI标准化"
    ].to_numpy(float)
    random_X = np.column_stack([np.ones(len(events)), week.to_numpy(float)])
    sd = np.sqrt(float(result.scale) + np.einsum("ij,jk,ik->i", random_X, np.asarray(result.cov_re), random_X))
    baseline = events.drop_duplicates("孕妇代码", keep="first")[["孕妇代码", "首次BMI"]]
    baseline = baseline.sort_values(["首次BMI", "孕妇代码"]).reset_index(drop=True)
    unique_bmi = np.sort(baseline["首次BMI"].unique())
    delta_grid = float(beta["首次BMI标准化"]) * (
        (unique_bmi - float(ref.loc["首次BMI", "中心_中位数"]))
        / float(ref.loc["首次BMI", "尺度_四分位距"])
    )
    y = events["达到4%标志"].to_numpy(int)
    woman_values = events["孕妇代码"].astype(str).to_numpy()
    person_cost = []
    for woman in baseline["孕妇代码"].astype(str):
        index = np.flatnonzero(woman_values == woman)
        eta = eta_without_bmi[index, None] + delta_grid[None, :]
        probability = stats.norm.cdf((eta - logit(达标阈值)) / sd[index, None])
        probability = np.clip(probability, 数值概率下限, 1 - 数值概率下限)
        loss = -(
            y[index, None] * np.log(probability)
            + (1 - y[index, None]) * np.log(1 - probability)
        )
        person_cost.append(np.mean(loss, axis=0))
    person_cost = np.vstack(person_cost)
    person_bmi = baseline["首次BMI"].to_numpy(float)
    counts = np.array([(person_bmi == value).sum() for value in unique_bmi], dtype=int)
    starts = np.r_[0, np.cumsum(counts)[:-1]]
    ends = np.cumsum(counts)
    prefix = np.vstack([np.zeros((1, len(delta_grid))), np.cumsum(person_cost, axis=0)])
    m = len(unique_bmi)
    cost = np.full((m, m), np.inf)
    delta_index = np.zeros((m, m), dtype=int)
    for i in range(m):
        for j in range(i, m):
            values = prefix[ends[j]] - prefix[starts[i]]
            best = int(np.argmin(values))
            cost[i, j] = float(values[best])
            delta_index[i, j] = best
    dp = np.full((m + 1, m + 1), np.inf)
    previous = np.full((m + 1, m + 1), -1, dtype=int)
    dp[0, 0] = 0.0
    for k in range(1, m + 1):
        for j in range(k, m + 1):
            candidates = np.arange(k - 1, j)
            values = dp[k - 1, candidates] + cost[candidates, j - 1]
            best = int(np.argmin(values))
            dp[k, j] = float(values[best])
            previous[k, j] = int(candidates[best])
    rows = []
    common = 9
    n = events["孕妇代码"].nunique()
    for k in range(1, m + 1):
        parameters = common + 2 * k - 1
        rows.append(
            {
                "组数K": k,
                "孕妇等权负对数似然": float(dp[k, m]),
                "参数计数": parameters,
                "贝叶斯信息准则": float(2 * dp[k, m] + parameters * math.log(n)),
            }
        )

    def reconstruct(k: int):
        segments = []
        j = m
        for level in range(k, 0, -1):
            i = int(previous[level, j])
            segments.append((i, j - 1, int(delta_index[i, j - 1])))
            j = i
        segments.reverse()
        output = []
        for group_number, (i, j, delta_pos) in enumerate(segments, 1):
            high = float(unique_bmi[j])
            next_low = float(unique_bmi[j + 1]) if j + 1 < m else np.nan
            output.append(
                {
                    "组别": group_number,
                    "组内最小BMI": float(unique_bmi[i]),
                    "组内最大BMI": high,
                    "与下一组切点": (high + next_low) / 2 if np.isfinite(next_low) else np.nan,
                    "人数": int(counts[i : j + 1].sum()),
                    "BMI组偏移": float(delta_grid[delta_pos]),
                    "偏移对应观测BMI": float(unique_bmi[delta_pos]),
                }
            )
        return pd.DataFrame(output)

    table = pd.DataFrame(rows)
    best_k = int(table.loc[table["贝叶斯信息准则"].idxmin(), "组数K"])
    return table, reconstruct(best_k), reconstruct(2), result, ref


def 独立前沿(events: pd.DataFrame, groups: pd.DataFrame, result, ref, scheme: str):
    baseline = events.drop_duplicates("孕妇代码", keep="first").copy()
    rows = []
    ordered = groups.sort_values("组别").reset_index(drop=True)
    low = float(events["首次BMI"].min())
    support_max = float(events["首次BMI"].max())
    for _, group in ordered.iterrows():
        high = float(group["与下一组切点"]) if pd.notna(group["与下一组切点"]) else support_max
        if pd.notna(group["与下一组切点"]):
            members = baseline.loc[(baseline["首次BMI"] >= low) & (baseline["首次BMI"] < high)].copy()
        else:
            members = baseline.loc[(baseline["首次BMI"] >= low) & (baseline["首次BMI"] <= high)].copy()
        for day in range(70, 176):
            frame = members.copy()
            frame["孕周数"] = day / 7.0
            probability = 独立预测概率(frame, result, ref, float(group["BMI组偏移"]))
            rows.append(
                {
                    "分组方案": scheme,
                    "组别": int(group["组别"]),
                    "检测孕周天数": day,
                    "预计已达标比例": float(probability.mean()),
                }
            )
        low = high
    return pd.DataFrame(rows)


def 复算前缀(detail: pd.DataFrame, identifier: str, review_type: str):
    excluded = {identifier, "有效标志", "失败信息"}
    numeric = [
        column
        for column in detail.columns
        if column not in excluded and pd.api.types.is_numeric_dtype(detail[column])
    ]
    rows = []
    for prefix in [100, 200, 400]:
        subset = detail.loc[(detail[identifier] <= prefix) & detail["有效标志"].eq(1)]
        for column in numeric:
            values = subset[column].dropna().to_numpy(float)
            if not len(values):
                continue
            rows.append(
                {
                    "复核类型": review_type,
                    "请求前缀次数": prefix,
                    "有效次数": int(len(subset)),
                    "统计量": column.replace("贝叶斯信息准则", "BIC"),
                    "中位数": float(np.median(values)),
                    "2.5%分位": float(np.quantile(values, 0.025)),
                    "97.5%分位": float(np.quantile(values, 0.975)),
                }
            )
    return pd.DataFrame(rows)


def main():
    manifest_path = 输出根目录 / "04_复现/第三问运行清单.json"
    pass_path = 输出根目录 / "04_复现/第三问自审PASS记录.json"
    hash_path = 输出根目录 / "04_复现/第三问结果文件哈希.csv"
    manifest = 读JSON(manifest_path)
    pass_record = 读JSON(pass_path)

    source_map = {
        "题目SHA256": 题目路径,
        "原始工作簿SHA256": 原始工作簿路径,
        "第一问事件源SHA256": 第一问事件源路径,
        "第一问记录源SHA256": 第一问记录源路径,
        "第二问运行清单SHA256": 第二问运行清单路径,
        "本脚本SHA256": 建模脚本路径,
    }
    source_differences = {
        key: (manifest.get(key), 文件哈希(path))
        for key, path in source_map.items()
        if manifest.get(key) != 文件哈希(path)
    }
    验收("输入和代码哈希", not source_differences, f"核对6个源对象；差异={source_differences}")
    验收(
        "自审哈希链",
        pass_record["运行清单SHA256"] == 文件哈希(manifest_path)
        and pass_record["结果哈希表SHA256"] == 文件哈希(hash_path),
        "自审记录中的运行清单和结果哈希表均独立复算",
    )
    hash_table = pd.read_csv(hash_path)
    hash_differences = []
    for _, row in hash_table.iterrows():
        target = 输出根目录 / str(row["相对正式候选输出路径"])
        if not target.is_file():
            hash_differences.append(f"缺失:{target.name}")
            continue
        if 文件哈希(target) != str(row["安全散列值_SHA256"]) or target.stat().st_size != int(row["字节数"]):
            hash_differences.append(f"不一致:{target.name}")
    验收("结果逐文件哈希", not hash_differences, f"复算{len(hash_table)}个文件；差异={hash_differences}")

    csv_files = sorted(输出根目录.rglob("*.csv"))
    bad_headers = []
    duplicate_headers = []
    for path in csv_files:
        columns = list(pd.read_csv(path, nrows=0).columns)
        if any(not 含中文(column) for column in columns):
            bad_headers.append(path.name)
        if len(columns) != len(set(columns)):
            duplicate_headers.append(path.name)
    验收(
        "中文表头",
        not bad_headers and not duplicate_headers,
        f"检查{len(csv_files)}个CSV；无中文表头={bad_headers}；重复表头={duplicate_headers}",
    )

    raw_events = pd.read_csv(第一问事件源路径)
    raw_records = pd.read_csv(第一问记录源路径)
    source_events = raw_events.loc[raw_events["纳入主模型标志"].eq(1)].copy()
    exported_events = pd.read_csv(输出根目录 / "01_数据/第三问抽血事件与决策时可得变量表.csv")
    data_ok = (
        len(source_events) == len(exported_events) == 613
        and source_events["孕妇代码"].nunique() == exported_events["孕妇代码"].nunique() == 167
        and set(source_events["抽血事件键"]) == set(exported_events["抽血事件键"])
        and exported_events["抽血事件键"].is_unique
        and np.array_equal(
            exported_events["达到4%标志"].to_numpy(int),
            (exported_events["Y染色体浓度均值"].to_numpy(float) >= 达标阈值).astype(int),
        )
    )
    验收("事件层与达标标志独立重建", data_ok, "613事件、167孕妇、事件键一一对应；4%标志逐行重算")

    baseline = pd.read_csv(输出根目录 / "01_数据/第三问孕妇首次可得基线表.csv")
    formula_bmi = baseline["首次体重"] / (baseline["首次身高"] / 100.0) ** 2
    bmi_error = float(np.max(np.abs(formula_bmi - baseline["首次BMI"])))
    first_rows = exported_events.sort_values(["孕妇代码", "孕周数", "抽血次数", "抽血事件键"]).drop_duplicates(
        "孕妇代码", keep="first"
    )
    baseline_keys_ok = set(first_rows["孕妇代码"]) == set(baseline["孕妇代码"])
    验收(
        "首次可得变量与体重共线处理",
        baseline_keys_ok and bmi_error < 1e-7,
        f"基线孕妇167人；BMI与体重/身高平方最大绝对差={bmi_error:.3e}",
    )

    roles = pd.read_csv(输出根目录 / "01_数据/第三问变量可得性与角色表.csv")
    forbidden = set(roles.loc[roles["模型角色"].eq("禁止进入主预测"), "变量"])
    expected_forbidden = {"本次及未来Y浓度/Z值", "读段数、比对比例、GC与过滤比例", "后续体重、BMI及全程均值"}
    parameter_names = set(pd.read_csv(输出根目录 / "02_模型结果/第三问推荐模型参数表.csv")["参数"])
    expected_parameters = {
        "截距",
        "孕周标准化",
        "首次BMI标准化",
        "首次年龄标准化",
        "首次身高标准化",
        "首次生产次数",
        "随机截距方差",
        "随机截距与孕周斜率协方差",
        "随机孕周斜率方差",
        "残差方差",
    }
    验收(
        "未来信息防泄漏",
        forbidden == expected_forbidden and parameter_names == expected_parameters and not manifest["是否使用检测后变量"],
        f"禁止变量={sorted(forbidden)}；主模型参数仅含预先可得变量和随机结构",
    )

    replicate = 独立技术复测(raw_records, set(exported_events["抽血事件键"]))
    assertions = 读JSON(输出根目录 / "01_数据/第三问数据构造断言.json")
    replicate_ok = (
        replicate["组数"] == assertions["严格技术复测组数"] == 18
        and replicate["记录数"] == assertions["严格技术复测记录数"] == 36
        and replicate["自由度"] == assertions["误差自由度"] == 18
        and np.isclose(replicate["合并组内标准差"], assertions["合并组内标准差"], rtol=1e-12, atol=1e-15)
    )
    验收(
        "技术复测误差独立重算",
        replicate_ok,
        f"18组/36条/自由度18；合并组内标准差={replicate['合并组内标准差']:.12g}；跨阈值组={replicate['跨阈值组数']}",
    )

    candidate_detail = pd.read_csv(输出根目录 / "03_验证/第三问候选逐孕妇留一逐事件.csv")
    candidate_person = pd.read_csv(输出根目录 / "03_验证/第三问候选逐孕妇留一逐孕妇.csv")
    candidate_table = pd.read_csv(输出根目录 / "02_模型结果/第三问候选路线统一比较.csv")
    candidates = list(candidate_table["候选模型"])
    coverage_ok = True
    metric_differences = []
    for name in candidates:
        group = candidate_detail.loc[candidate_detail["候选模型"].eq(name)].copy()
        coverage_ok &= len(group) == 613 and group["孕妇代码"].nunique() == 167 and group["抽血事件键"].is_unique
        probability = group["预测达标概率"].to_numpy(float)
        y = group["实际达标标志"].to_numpy(int)
        clip = np.clip(probability, 数值概率下限, 1 - 数值概率下限)
        logloss = -(y * np.log(clip) + (1 - y) * np.log(1 - clip))
        brier = (y - probability) ** 2
        group["重算对数损失"] = logloss
        group["重算Brier"] = brier
        person = group.groupby("孕妇代码", as_index=False).agg(
            对数损失=("重算对数损失", "mean"), Brier=("重算Brier", "mean")
        )
        counts = group.groupby("孕妇代码").size()
        weights = group["孕妇代码"].map(lambda code: 1.0 / counts[code]).to_numpy(float)
        row = candidate_table.loc[candidate_table["候选模型"].eq(name)].iloc[0]
        differences = [
            float(person["对数损失"].mean()) - float(row["逐孕妇平均对数损失"]),
            float(person["Brier"].mean()) - float(row["逐孕妇平均Brier分数"]),
            roc_auc_score(y, probability, sample_weight=weights) - float(row["孕妇等权ROC曲线下面积"]),
            average_precision_score(y, probability, sample_weight=weights) - float(row["孕妇等权PR曲线下面积"]),
        ]
        metric_differences.extend(differences)
    验收(
        "逐孕妇留一覆盖与评分复算",
        coverage_ok and max(abs(value) for value in metric_differences) < 1e-12,
        f"五个候选各613事件/167孕妇；四类评分最大绝对差={max(abs(value) for value in metric_differences):.3e}",
    )

    curves = pd.read_csv(输出根目录 / "03_验证/第三问候选全样本时点边界曲线.csv")
    selected = str(manifest["入选模型"])
    selected_row = candidate_table.loc[candidate_table["候选模型"].eq(selected)].iloc[0]
    eligible = candidate_table.loc[
        candidate_table["是否多因素"].eq(1)
        & candidate_table["全样本收敛标志"].eq(1)
        & candidate_table["有效孕妇数"].eq(167)
        & candidate_table["时点曲线边界通过标志"].eq(1)
        & candidate_table["全部概率合法标志"].eq(1)
    ].sort_values(["逐孕妇平均对数损失", "逐孕妇平均Brier分数", "模型参数数", "候选模型"])
    quadratic = candidate_table.loc[candidate_table["候选模型"].eq("第三问多因素_二次浓度混合")].iloc[0]
    selection_ok = (
        selected == str(eligible.iloc[0]["候选模型"]) == "第三问多因素_线性浓度混合"
        and bool(selected_row["时点曲线边界通过标志"])
        and not bool(quadratic["时点曲线边界通过标志"])
        and float(quadratic["10至25周最小相邻日达标比例变化"]) < 0
    )
    验收(
        "候选选择与边界否决",
        selection_ok,
        f"合格候选主评分最优={eligible.iloc[0]['候选模型']}；二次模型最小相邻日变化={float(quadratic['10至25周最小相邻日达标比例变化']):.6g}<0",
    )

    improvement = pd.read_csv(输出根目录 / "03_验证/第三问多因素相对第二问信息集增量检验.csv")
    person_pivot_log = candidate_person.pivot(index="孕妇代码", columns="候选模型", values="孕妇内平均对数损失")
    person_pivot_brier = candidate_person.pivot(index="孕妇代码", columns="候选模型", values="孕妇内平均Brier分数")
    improvement_differences = []
    for row_number, pivot in enumerate([person_pivot_log, person_pivot_brier]):
        difference = pivot[selected] - pivot["第二问信息集_线性浓度混合"]
        interval = stats.t.interval(
            0.95, len(difference) - 1, loc=float(difference.mean()), scale=float(stats.sem(difference))
        )
        row = improvement.iloc[row_number]
        improvement_differences += [
            float(difference.mean()) - float(row["多因素减同路线第二问信息集基准_均值"]),
            float(interval[0]) - float(row["差值95%区间下限"]),
            float(interval[1]) - float(row["差值95%区间上限"]),
            int((difference < 0).sum()) - int(row["多因素损失更低孕妇数"]),
        ]
    验收(
        "多因素增量独立复算",
        max(abs(value) for value in improvement_differences) < 1e-12
        and float(improvement.iloc[0]["差值95%区间上限"]) < 0
        and float(improvement.iloc[1]["差值95%区间上限"]) > 0,
        f"复算差异最大={max(abs(value) for value in improvement_differences):.3e}；对数损失区间完全低于0，Brier区间跨0",
    )

    references = pd.read_csv(输出根目录 / "02_模型结果/第三问标准化参照表.csv")
    fit, X, ref = 拟合独立主模型(exported_events, references)
    parameter_table = pd.read_csv(输出根目录 / "02_模型结果/第三问推荐模型参数表.csv")
    expected_values = {
        **dict(zip(X.columns, np.asarray(fit.fe_params))),
        "随机截距方差": float(np.asarray(fit.cov_re)[0, 0]),
        "随机截距与孕周斜率协方差": float(np.asarray(fit.cov_re)[0, 1]),
        "随机孕周斜率方差": float(np.asarray(fit.cov_re)[1, 1]),
        "残差方差": float(fit.scale),
    }
    parameter_differences = []
    for name, value in expected_values.items():
        reported = float(parameter_table.loc[parameter_table["参数"].eq(name), "估计值"].iloc[0])
        parameter_differences.append(value - reported)
    验收(
        "主模型参数独立重拟合",
        bool(fit.converged) and max(abs(value) for value in parameter_differences) < 1e-8,
        f"独立重跑lbfgs/Powell并按似然择优={fit._独立复核优化器}；10个参数最大绝对差={max(abs(value) for value in parameter_differences):.3e}",
    )

    selected_curve = curves.loc[curves["候选模型"].eq(selected)].sort_values("检测孕周天数")
    baseline_for_curve = baseline.copy()
    curve_differences = []
    for _, row in selected_curve.iterrows():
        frame = baseline_for_curve.copy()
        frame["孕周数"] = float(row["检测孕周天数"]) / 7.0
        prediction = 独立预测概率(frame, fit, ref)
        curve_differences.append(float(prediction.mean()) - float(row["预计达标比例"]))
    验收(
        "主模型概率曲线独立复算",
        max(abs(value) for value in curve_differences) < 1e-8
        and np.all(np.diff(selected_curve["预计达标比例"].to_numpy(float)) >= 0),
        f"106个逐日概率最大绝对差={max(abs(value) for value in curve_differences):.3e}；主曲线单调不降",
    )

    independent_bic, independent_main, independent_two, fit, ref = 独立分段(exported_events, fit, ref)
    reported_bic = pd.read_csv(输出根目录 / "02_模型结果/第三问全部BMI组数BIC比较.csv")
    bic_difference = np.max(
        np.abs(independent_bic["贝叶斯信息准则"].to_numpy(float) - reported_bic["贝叶斯信息准则"].to_numpy(float))
    )
    reported_main = pd.read_csv(输出根目录 / "02_模型结果/第三问主BMI分组.csv")
    reported_two = pd.read_csv(输出根目录 / "02_模型结果/第三问两组分组敏感性.csv")
    support_min = float(exported_events["首次BMI"].min())
    support_max = float(exported_events["首次BMI"].max())
    reported_cut = float(reported_two.iloc[0]["与下一组切点"])
    expected_intervals = [
        f"[{support_min:.6f},{reported_cut:.6f})",
        f"[{reported_cut:.6f},{support_max:.6f}]",
    ]
    group_ok = (
        int(independent_bic.loc[independent_bic["贝叶斯信息准则"].idxmin(), "组数K"]) == 1
        and np.isclose(float(independent_two.iloc[0]["与下一组切点"]), float(reported_two.iloc[0]["与下一组切点"]), atol=1e-10)
        and independent_two["人数"].astype(int).tolist() == reported_two["人数"].astype(int).tolist() == [151, 16]
        and int(reported_main["人数"].sum()) == 167
    )
    验收(
        "全部BMI组数与分段独立复算",
        len(reported_bic) == exported_events["首次BMI"].nunique() == 139
        and bic_difference < 1e-8
        and group_ok,
        f"K=1至139全部重算；BIC最大绝对差={bic_difference:.3e}；两组切点={float(independent_two.iloc[0]['与下一组切点']):.9f}，人数151/16",
    )
    验收(
        "BMI决策区间连续覆盖",
        reported_two["BMI区间"].tolist() == expected_intervals,
        f"实得={reported_two['BMI区间'].tolist()}；期望={expected_intervals}",
    )

    independent_frontier = pd.concat(
        [
            独立前沿(exported_events, independent_main, fit, ref, "主分组K=1"),
            独立前沿(exported_events, independent_two, fit, ref, "固定两组敏感性"),
        ],
        ignore_index=True,
    )
    reported_frontier = pd.read_csv(输出根目录 / "02_模型结果/第三问各BMI组时点与尚未达标概率Pareto前沿.csv")
    merged_frontier = independent_frontier.merge(
        reported_frontier[["分组方案", "组别", "检测孕周天数", "预计已达标比例", "预计尚未达标比例", "是否帕累托非支配点"]],
        on=["分组方案", "组别", "检测孕周天数"],
        how="outer",
        suffixes=("_独立", "_报告"),
        indicator=True,
    )
    frontier_difference = np.max(
        np.abs(
            merged_frontier["预计已达标比例_独立"].to_numpy(float)
            - merged_frontier["预计已达标比例_报告"].to_numpy(float)
        )
    )
    验收(
        "BMI组逐日前沿独立复算",
        len(reported_frontier) == 318
        and merged_frontier["_merge"].eq("both").all()
        and frontier_difference < 1e-8
        and np.allclose(
            reported_frontier["预计已达标比例"] + reported_frontier["预计尚未达标比例"], 1.0, atol=1e-12
        )
        and reported_frontier["是否帕累托非支配点"].eq(1).all(),
        f"主1组和两组敏感性共318点；概率最大绝对差={frontier_difference:.3e}；全部为真实非支配点",
    )
    two_bounds = (
        reported_frontier.loc[reported_frontier["分组方案"].eq("固定两组敏感性")]
        .groupby("组别", sort=True)[["BMI区间下限", "BMI区间上限"]]
        .first()
        .to_numpy(float)
    )
    验收(
        "前沿BMI边界连续覆盖",
        np.allclose(two_bounds, [[support_min, reported_cut], [reported_cut, support_max]], atol=1e-12),
        f"两组前沿边界={two_bounds.tolist()}；共同切点={reported_cut:.9f}",
    )

    measurement = pd.read_csv(输出根目录 / "03_验证/第三问检测误差传播逐次.csv")
    bootstrap = pd.read_csv(输出根目录 / "03_验证/第三问孕妇整簇自助逐次.csv")
    measurement_reported = pd.read_csv(输出根目录 / "03_验证/第三问检测误差传播次数收敛.csv")
    bootstrap_reported = pd.read_csv(输出根目录 / "03_验证/第三问孕妇整簇自助次数收敛.csv")
    measurement_independent = 复算前缀(measurement, "重复序号", "检测误差传播")
    bootstrap_independent = 复算前缀(bootstrap, "自助序号", "孕妇整簇自助")
    measurement_merge = measurement_independent.merge(
        measurement_reported,
        on=["复核类型", "请求前缀次数", "有效次数", "统计量"],
        suffixes=("_独立", "_报告"),
        how="outer",
        indicator=True,
    )
    bootstrap_merge = bootstrap_independent.merge(
        bootstrap_reported,
        on=["复核类型", "请求前缀次数", "有效次数", "统计量"],
        suffixes=("_独立", "_报告"),
        how="outer",
        indicator=True,
    )
    convergence_difference = 0.0
    for merged in [measurement_merge, bootstrap_merge]:
        for column in ["中位数", "2.5%分位", "97.5%分位"]:
            convergence_difference = max(
                convergence_difference,
                float(np.nanmax(np.abs(merged[f"{column}_独立"] - merged[f"{column}_报告"]))),
            )
    验收(
        "误差传播与整簇自助汇总复算",
        len(measurement) == len(bootstrap) == 400
        and measurement["有效标志"].eq(1).all()
        and bootstrap["有效标志"].eq(1).all()
        and measurement_merge["_merge"].eq("both").all()
        and bootstrap_merge["_merge"].eq("both").all()
        and convergence_difference < 1e-12,
        f"检测误差400/400、整簇自助400/400有效；100/200/400前缀最大差={convergence_difference:.3e}",
    )
    bootstrap_two_rate = float(bootstrap["贝叶斯信息准则两组挑战胜出标志"].mean())
    measurement_two_rate = float(measurement["贝叶斯信息准则两组挑战胜出标志"].mean())
    height_ci = np.quantile(bootstrap["固定效应_首次身高标准化"], [0.025, 0.975])
    bmi_ci = np.quantile(bootstrap["固定效应_首次BMI标准化"], [0.025, 0.975])
    model_card = (输出根目录 / "02_模型结果/第三问推荐模型卡.md").read_text(encoding="utf-8")
    验收(
        "不确定性与局限性披露",
        np.isclose(bootstrap_two_rate, 0.135)
        and np.isclose(measurement_two_rate, 0.0)
        and height_ci[0] < 0 < height_ci[1]
        and bmi_ci[1] < 0
        and "身高区间跨0" in model_card
        and "Brier" in model_card
        and "区间跨0" in model_card,
        f"两组挑战胜出率：整簇自助={bootstrap_two_rate:.3f}、检测误差={measurement_two_rate:.3f}；身高区间={height_ci.tolist()}跨0，BMI区间={bmi_ci.tolist()}全负",
    )

    sensitivity = pd.read_csv(输出根目录 / "03_验证/第三问数据质量口径敏感性.csv")
    parameter_sources = pd.read_csv(输出根目录 / "02_模型结果/第三问参数来源表.csv")
    required_source_rows = {
        "Y染色体浓度达标线 c",
        "决策日网格",
        "题面风险分界",
        "主预测变量集合",
        "体重处理",
        "候选选择评分",
        "不确定性报告水平α",
        "外层验证折数",
        "BMI组数K",
        "标准化中心与尺度",
        "BMI组偏移候选集合",
        "分组BIC参数计数",
        "技术测量误差标准差",
        "自助与误差传播重复次数B",
        "随机种子",
        "数值优化与概率保护",
    }
    验收(
        "敏感性和参数来源完整性",
        len(sensitivity) == 5
        and sensitivity["拟合状态"].eq("通过").all()
        and required_source_rows.issubset(set(parameter_sources["参数名称和符号"]))
        and parameter_sources["审核状态"].eq("通过").all()
        and parameter_sources["参数名称和符号"].is_unique,
        f"五个质量口径均通过；参数来源表{len(parameter_sources)}行；缺失={sorted(required_source_rows-set(parameter_sources['参数名称和符号']))}",
    )

    验收(
        "无自拟政策参数",
        not manifest["是否设置q"]
        and not manifest["是否设置风险权重"]
        and not manifest["是否设置最小组人数"]
        and not manifest["是否设置候选组数上限"]
        and "不输出虚构" not in model_card or "不给出虚构的唯一最佳日" in model_card,
        "运行清单中q、风险权重、最小组人数和组数上限均为false；只输出逐日前沿",
    )

    image_suffixes = {".png", ".jpg", ".jpeg", ".svg", ".gif", ".bmp", ".webp"}
    images = [path for path in 项目目录.rglob("*") if path.is_file() and path.suffix.lower() in image_suffixes]
    prompts = sorted((输出根目录 / "05_图表提示词").glob("*.txt"))
    bad_prompts = [
        path.name
        for path in prompts
        if "MATLAB" not in path.read_text(encoding="utf-8-sig").upper()
        or "SVG" not in path.read_text(encoding="utf-8-sig").upper()
    ]
    验收(
        "图形交付约束",
        not images and len(prompts) == 3 and not bad_prompts,
        f"新图像={len(images)}；MATLAB/SVG提示词={len(prompts)}；不合格={bad_prompts}",
    )
    验收(
        "Windows复现入口",
        一键脚本路径.read_bytes().startswith(b"\xef\xbb\xbf"),
        "一键PowerShell脚本为UTF-8 BOM",
    )
    验收(
        "模型卡公式与风险措辞",
        "b_{1i}s_{ij}" in model_card
        and "不是因果效应" in model_card
        and "10周概率只作题目窗口内外推" in model_card,
        "随机斜率明确作用于标准化孕周；相关非因果；10周标为外推",
    )

    table = pd.DataFrame(验收记录)
    status = "PASS" if table["通过标志"].all() else "REJECTED"
    table.to_csv(独立复核目录 / "第三问独立总控验收清单.csv", index=False, encoding="utf-8-sig")
    failures = table.loc[~table["通过标志"], ["验收项", "独立证据"]].to_dict("records")

    report_lines = [
        "# 第三问独立总控复核报告",
        "",
        f"- 独立审核状态：**{status}**",
        f"- 验收项：{len(table)}项；失败：{len(failures)}项。",
        "- 复核方式：不调用建模脚本自审函数，从第一问冻结源、逐事件留一预测、主模型参数、动态分段和哈希反向重建。",
        "",
        "## 关键独立结论",
        "",
        "- 数据层：613个抽血事件、167名孕妇；技术复测18组、36条、自由度18。",
        f"- 入选路线：{selected}；五个候选均完成167名孕妇逐人留一。",
        f"- 多因素相对同路线基准的对数损失差95%区间为[{float(improvement.iloc[0]['差值95%区间下限']):.6f}, {float(improvement.iloc[0]['差值95%区间上限']):.6f}]；Brier差区间跨0。",
        "- 二次孕周路线虽有更低平均预测损失，但10至25周产生早期反向段，被边界闸门正确否决。",
        f"- BMI分组：全部139个组数重算后BIC选择1组；两组挑战切点{float(independent_two.iloc[0]['与下一组切点']):.6f}、人数151/16。",
        f"- 两组挑战胜出率：孕妇整簇自助{bootstrap_two_rate:.3f}，检测误差传播{measurement_two_rate:.3f}；不支持把两组挑战升级为稳定主结论。",
        "- 题面缺少代价比或最低达标比例，材料没有设置q或风险权重，只输出时点—尚未达标概率前沿。",
        "",
        "## 归档裁决",
        "",
        "全部独立检查通过，允许进入正式目录。" if status == "PASS" else "存在失败项，必须打回，禁止归档。",
    ]
    if failures:
        report_lines.extend(["", "## 失败项", ""] + [f"- {row['验收项']}：{row['独立证据']}" for row in failures])
    report_path = 独立复核目录 / "第三问独立总控复核报告.md"
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    record = {
        "生成时间": datetime.now().astimezone().isoformat(timespec="seconds"),
        "问题": "第三问",
        "状态": status,
        "独立验收项数": int(len(table)),
        "失败项数": int(len(failures)),
        "建模运行清单SHA256": 文件哈希(manifest_path),
        "建模结果哈希表SHA256": 文件哈希(hash_path),
        "独立复核脚本SHA256": 文件哈希(脚本路径),
        "独立验收清单SHA256": 文件哈希(独立复核目录 / "第三问独立总控验收清单.csv"),
        "独立复核报告SHA256": 文件哈希(report_path),
    }
    (独立复核目录 / "第三问独立审核PASS记录.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(record, ensure_ascii=False, indent=2))
    if status != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
