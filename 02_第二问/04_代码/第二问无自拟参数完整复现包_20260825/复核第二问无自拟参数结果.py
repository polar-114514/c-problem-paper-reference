from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import expit


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
建模脚本路径 = 项目目录 / "第二问无自拟参数重构.py"
一键脚本路径 = 项目目录 / "一键运行第二问无自拟参数重构.ps1"


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


def 构造独立删失区间(events: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for woman, group in events.groupby("孕妇代码", sort=False):
        group = group.sort_values(["孕周数", "抽血次数", "抽血事件键"]).reset_index(drop=True)
        hits = np.flatnonzero(group["达到4%标志"].to_numpy(int) == 1)
        if len(hits) == 0:
            left, right, censor, first_hit, post_drop = (
                float(group["孕周数"].iloc[-1]),
                np.inf,
                "右删失",
                np.nan,
                0,
            )
        elif int(hits[0]) == 0:
            left, right, censor = 0.0, float(group["孕周数"].iloc[0]), "左删失"
            first_hit = right
            post_drop = int((group["Y染色体浓度均值"].iloc[1:] < 0.04).any())
        else:
            pos = int(hits[0])
            left = float(group["孕周数"].iloc[pos - 1])
            right = float(group["孕周数"].iloc[pos])
            censor = "区间删失"
            first_hit = right
            post_drop = int((group["Y染色体浓度均值"].iloc[pos + 1 :] < 0.04).any())
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


def 分布函数(z: np.ndarray, family: str) -> np.ndarray:
    if family == "对数正态":
        return stats.norm.cdf(z)
    if family == "对数逻辑斯蒂":
        return expit(z)
    if family == "Weibull":
        return 1.0 - np.exp(-np.exp(np.clip(z, -700, 700)))
    raise ValueError(f"未知分布：{family}")


def 独立负对数似然(row: pd.Series, data: pd.DataFrame) -> float:
    bmi = data["首次BMI"].to_numpy(float)
    mu = float(row["截距"]) + float(row["BMI系数"]) * (
        (bmi - float(row["BMI中心"])) / float(row["BMI尺度"])
    )
    sigma = float(row["尺度参数σ"])
    left = data["删失左端点"].to_numpy(float)
    right = data["删失右端点"].to_numpy(float)
    censor = data["删失类型"].to_numpy(str)
    z_left = (np.log(np.maximum(left, np.finfo(float).tiny)) - mu) / sigma
    f_left = 分布函数(z_left, str(row["分布"]))
    f_right = np.ones(len(data))
    finite = np.isfinite(right)
    f_right[finite] = 分布函数((np.log(right[finite]) - mu[finite]) / sigma, str(row["分布"]))
    probability = np.where(
        censor == "左删失",
        f_right,
        np.where(censor == "区间删失", f_right - f_left, 1.0 - f_left),
    )
    probability = np.maximum(probability, np.finfo(float).tiny)
    return float(-np.log(probability).sum())


def main():
    运行清单路径 = 输出根目录 / "04_复现/第二问运行清单.json"
    自审记录路径 = 输出根目录 / "04_复现/第二问自审PASS记录.json"
    哈希表路径 = 输出根目录 / "04_复现/第二问结果文件哈希.csv"
    运行清单 = 读JSON(运行清单路径)
    自审记录 = 读JSON(自审记录路径)

    源文件对应 = {
        "题目SHA256": 题目路径,
        "原始工作簿SHA256": 原始工作簿路径,
        "第一问事件源SHA256": 第一问事件源路径,
        "第一问记录源SHA256": 第一问记录源路径,
        "本脚本SHA256": 建模脚本路径,
    }
    源哈希差异 = {
        key: (运行清单.get(key), 文件哈希(path))
        for key, path in 源文件对应.items()
        if 运行清单.get(key) != 文件哈希(path)
    }
    验收("输入与建模脚本哈希", not 源哈希差异, f"核对5个源对象；差异={源哈希差异}")
    验收(
        "自审记录链",
        自审记录.get("运行清单SHA256") == 文件哈希(运行清单路径)
        and 自审记录.get("结果哈希表SHA256") == 文件哈希(哈希表路径),
        "自审记录中的运行清单和结果哈希表散列值均重新计算",
    )

    哈希表 = pd.read_csv(哈希表路径)
    哈希差异 = []
    for _, row in 哈希表.iterrows():
        target = 输出根目录 / str(row["相对正式候选输出路径"])
        if not target.is_file():
            哈希差异.append(f"缺失:{target.name}")
            continue
        actual = 文件哈希(target)
        if actual != str(row["安全散列值_SHA256"]) or target.stat().st_size != int(row["字节数"]):
            哈希差异.append(f"不一致:{target.name}")
    验收("结果文件逐项哈希", not 哈希差异, f"独立复算{len(哈希表)}个结果文件；差异={哈希差异}")

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
        f"检查{len(csv_files)}个CSV；无中文表头文件={bad_headers}；重复表头={duplicate_headers}",
    )

    raw_events = pd.read_csv(第一问事件源路径)
    raw_records = pd.read_csv(第一问记录源路径)
    source_events = raw_events.loc[raw_events["纳入主模型标志"].eq(1)].copy()
    source_events = source_events.sort_values(["孕妇代码", "孕周数", "抽血次数", "抽血事件键"]).reset_index(drop=True)
    first_bmi = (
        source_events.groupby("孕妇代码", as_index=False, sort=False).first()[["孕妇代码", "孕妇体质指数_BMI"]]
        .rename(columns={"孕妇体质指数_BMI": "首次BMI"})
    )
    source_events = source_events.merge(first_bmi, on="孕妇代码", how="left", validate="many_to_one")
    source_events["达到4%标志"] = (source_events["Y染色体浓度均值"] >= 0.04).astype(int)
    exported_events = pd.read_csv(输出根目录 / "01_数据/第二问抽血事件纵向表.csv")
    验收(
        "事件层独立重建",
        len(exported_events) == len(source_events) == 613
        and exported_events["孕妇代码"].nunique() == source_events["孕妇代码"].nunique() == 167
        and exported_events["抽血事件键"].is_unique
        and set(exported_events["抽血事件键"]) == set(source_events["抽血事件键"]),
        "从第一问冻结源重新筛选：613个事件、167名孕妇，事件键一一对应",
    )

    interval_source = exported_events.copy()
    independent_intervals = 构造独立删失区间(interval_source)
    exported_intervals = pd.read_csv(输出根目录 / "01_数据/第二问孕妇首次达标删失区间表.csv")
    interval_same = True
    interval_error = ""
    try:
        pd.testing.assert_frame_equal(
            independent_intervals,
            exported_intervals[independent_intervals.columns],
            check_exact=False,
            rtol=1e-12,
            atol=1e-12,
            check_dtype=False,
        )
    except AssertionError as exc:
        interval_same = False
        interval_error = str(exc).splitlines()[0]
    censor_counts = independent_intervals["删失类型"].value_counts().to_dict()
    验收(
        "删失区间独立重建",
        interval_same and censor_counts == {"左删失": 118, "区间删失": 40, "右删失": 9},
        f"左/区间/右删失={censor_counts}；逐行差异={interval_error or '无'}",
    )
    验收(
        "纵向异常事实",
        int(independent_intervals["达标后回落标志"].sum()) == 32
        and int(independent_intervals["仅一次抽血标志"].sum()) == 7,
        f"达标后回落={int(independent_intervals['达标后回落标志'].sum())}人；仅一次抽血={int(independent_intervals['仅一次抽血标志'].sum())}人",
    )

    selected_events = set(source_events["抽血事件键"])
    selected_records = raw_records.loc[raw_records["抽血事件键"].isin(selected_events)].copy()
    group_columns = ["孕妇代码", "抽血次数", "检测日期规范值", "孕周原始值"]
    residual_sum = 0.0
    degrees = 0
    replicate_groups = 0
    replicate_records = 0
    crossing_groups = 0
    for _, group in selected_records.groupby(group_columns, dropna=False, sort=False):
        if len(group) < 2:
            continue
        values = group["Y染色体浓度"].to_numpy(float)
        mean = float(values.mean())
        residual_sum += float(np.sum((values - mean) ** 2))
        degrees += len(values) - 1
        replicate_groups += 1
        replicate_records += len(values)
        crossing_groups += int(float(values.min()) < 0.04 <= float(values.max()))
    pooled_sd = math.sqrt(residual_sum / degrees)
    assertions = 读JSON(输出根目录 / "01_数据/第二问数据构造断言.json")
    验收(
        "技术复测误差独立重算",
        replicate_groups == assertions["严格技术复测组数"] == 18
        and replicate_records == assertions["严格技术复测记录数"] == 36
        and degrees == assertions["误差自由度"] == 18
        and crossing_groups == assertions["跨越4%阈值组数"]
        and np.isclose(pooled_sd, assertions["合并组内标准差"], rtol=1e-12, atol=1e-15),
        f"18组/36条/自由度18；合并组内标准差={pooled_sd:.12g}；跨阈值组={crossing_groups}",
    )

    candidates = pd.read_csv(输出根目录 / "02_模型结果/第二问AFT候选统一比较.csv")
    nll_differences = {}
    information_ok = True
    for _, row in candidates.iterrows():
        independent_nll = 独立负对数似然(row, exported_intervals)
        nll_differences[str(row["分布"])] = independent_nll - float(row["负对数似然"])
        information_ok &= np.isclose(float(row["赤池信息准则"]), 2 * independent_nll + 2 * 3, atol=1e-8)
        information_ok &= np.isclose(
            float(row["贝叶斯信息准则"]), 2 * independent_nll + 3 * math.log(len(exported_intervals)), atol=1e-8
        )
    验收(
        "AFT似然独立复算",
        max(abs(value) for value in nll_differences.values()) < 1e-8 and information_ok,
        f"三种分布的负对数似然最大绝对差={max(abs(value) for value in nll_differences.values()):.3e}；AIC/BIC公式复算通过={information_ok}",
    )
    loo_best_family = str(candidates.loc[candidates["留一负对数似然_每人"].idxmin(), "分布"])
    bic_best_family = str(candidates.loc[candidates["贝叶斯信息准则"].idxmin(), "分布"])
    验收(
        "AFT候选同目标外层选择",
        运行清单["入选分布"] == loo_best_family == bic_best_family == "对数逻辑斯蒂"
        and candidates["留一验证人数"].eq(167).all(),
        f"留一最优={loo_best_family}；BIC最优={bic_best_family}；各候选留一人数={sorted(candidates['留一验证人数'].unique())}",
    )

    group_bic = pd.read_csv(输出根目录 / "02_模型结果/第二问全部组数BIC比较.csv")
    unique_bmi = int(exported_intervals["首次BMI"].nunique())
    expected_k = np.arange(1, unique_bmi + 1)
    group_formula_ok = np.array_equal(group_bic["组数K"].to_numpy(int), expected_k)
    group_formula_ok &= np.allclose(group_bic["参数计数"].to_numpy(float), 2 * expected_k)
    group_formula_ok &= np.allclose(
        group_bic["贝叶斯信息准则"].to_numpy(float),
        2 * group_bic["负对数似然"].to_numpy(float) + 2 * expected_k * math.log(len(exported_intervals)),
        atol=1e-9,
    )
    bic_best_k = int(group_bic.loc[group_bic["贝叶斯信息准则"].idxmin(), "组数K"])
    decision = 读JSON(输出根目录 / "02_模型结果/第二问组数选择裁决.json")
    验收(
        "全部组数与BIC公式",
        group_formula_ok and bic_best_k == decision["BIC最优组数"] == 运行清单["主分组数"] == 1,
        f"首次BMI不同取值={unique_bmi}，逐一检查K=1至{unique_bmi}；BIC最优K={bic_best_k}",
    )
    验收(
        "无隐藏分组门槛",
        运行清单["主组数选择准则"] == "BIC"
        and not 运行清单["是否设置最小组人数"]
        and not 运行清单["是否设置候选组数上限"]
        and not decision["是否设置最小组人数"]
        and not decision["是否设置候选组数上限"],
        "运行清单和裁决文件均声明：BIC主准则、无最小组人数、无候选组数上限",
    )

    loo_summary = pd.read_csv(输出根目录 / "03_验证/第二问全部组数留一验证汇总.csv")
    loo_detail = pd.read_csv(输出根目录 / "03_验证/第二问全部组数留一验证逐孕妇.csv")
    loo_best_k = int(loo_summary.loc[loo_summary["留一负对数似然_每人"].idxmin(), "组数K"])
    pivot = loo_detail.loc[loo_detail["组数K"].isin([bic_best_k, loo_best_k])].pivot(
        index="孕妇代码", columns="组数K", values="留出负对数似然"
    )
    difference = pivot[loo_best_k] - pivot[bic_best_k]
    difference_ci = stats.t.interval(
        0.95,
        len(difference) - 1,
        loc=float(difference.mean()),
        scale=float(stats.sem(difference)),
    )
    验收(
        "留一检验独立汇总",
        loo_best_k == decision["留一预测最优组数"] == 2
        and np.isclose(float(difference.mean()), decision["留一最优减BIC最优的逐孕妇负对数似然差均值"], atol=1e-12)
        and np.allclose(difference_ci, decision["差值95%区间"], atol=1e-12)
        and int((difference < 0).sum()) == decision["留一最优逐孕妇损失更低人数"],
        f"留一最优K={loo_best_k}；K2-K1均值={float(difference.mean()):.12g}；95%区间={tuple(float(x) for x in difference_ci)}；K2更低损失={int((difference < 0).sum())}/167",
    )

    main_groups = pd.read_csv(输出根目录 / "02_模型结果/第二问主分组与中位达标时点.csv")
    two_groups = pd.read_csv(输出根目录 / "02_模型结果/第二问固定两组分组敏感性.csv")
    observed_bmi = np.sort(exported_intervals["首次BMI"].unique())
    cut = float(two_groups.iloc[0]["与下一组切点"])
    left_bmi = observed_bmi[observed_bmi < cut].max()
    right_bmi = observed_bmi[observed_bmi >= cut].min()
    groups_ok = (
        len(main_groups) == 1
        and int(main_groups["人数"].sum()) == 167
        and np.isclose(float(main_groups.iloc[0]["组内最小BMI"]), float(observed_bmi.min()))
        and np.isclose(float(main_groups.iloc[0]["组内最大BMI"]), float(observed_bmi.max()))
        and len(two_groups) == 2
        and int(two_groups["人数"].sum()) == 167
        and np.isclose(cut, (left_bmi + right_bmi) / 2, atol=1e-12)
    )
    验收(
        "主分组与两组敏感性边界",
        groups_ok,
        f"主方案覆盖167人；两组敏感性人数={two_groups['人数'].astype(int).tolist()}；切点={cut:.12g}，相邻观测BMI={left_bmi:.12g}/{right_bmi:.12g}",
    )
    support_min = float(observed_bmi.min())
    support_max = float(observed_bmi.max())
    expected_intervals = [f"[{support_min:.6f},{cut:.6f})", f"[{cut:.6f},{support_max:.6f}]"]
    验收(
        "BMI决策区间连续覆盖",
        two_groups["BMI区间"].tolist() == expected_intervals,
        f"实得={two_groups['BMI区间'].tolist()}；期望={expected_intervals}",
    )

    frontier = pd.read_csv(输出根目录 / "02_模型结果/第二问各组时点与尚未达标概率Pareto前沿.csv")
    frontier_ok = True
    group_counts = []
    for keys, group in frontier.groupby(["分组方案", "组别"], sort=False):
        group = group.sort_values("检测孕周天数")
        group_counts.append(len(group))
        frontier_ok &= np.array_equal(group["检测孕周天数"].to_numpy(int), np.arange(70, 176))
        frontier_ok &= np.all(np.diff(group["预计尚未达标比例"].to_numpy(float)) <= 1e-12)
        frontier_ok &= np.allclose(
            group["预计已达标比例"].to_numpy(float) + group["预计尚未达标比例"].to_numpy(float),
            1.0,
            atol=1e-12,
        )
        frontier_ok &= group["是否帕累托非支配点"].eq(1).all()
    验收(
        "时点前沿完整性",
        frontier_ok
        and sorted(group_counts) == [106, 106, 106]
        and np.allclose(
            frontier.loc[frontier["分组方案"].eq("固定两组敏感性")]
            .groupby("组别", sort=True)[["BMI区间下限", "BMI区间上限"]]
            .first()
            .to_numpy(float),
            [[support_min, cut], [cut, support_max]],
            atol=1e-12,
        ),
        f"主1组和两组敏感性共3条前沿，每条106个逐日点；概率和为1且尚未达标概率单调不升",
    )
    验收(
        "无虚构唯一最优参数",
        not 运行清单["是否设置q"]
        and not 运行清单["是否设置风险权重"]
        and "不输出虚构的唯一最优日" in (输出根目录 / "02_模型结果/第二问无自拟参数模型卡.md").read_text(encoding="utf-8"),
        "运行清单中q=false、风险权重=false；模型卡只输出完整前沿和中位分布统计量",
    )

    measurement = pd.read_csv(输出根目录 / "03_验证/第二问检测误差传播逐次.csv")
    bootstrap = pd.read_csv(输出根目录 / "03_验证/第二问孕妇整簇自助逐次.csv")
    measurement_convergence = pd.read_csv(输出根目录 / "03_验证/第二问检测误差传播次数收敛.csv")
    bootstrap_convergence = pd.read_csv(输出根目录 / "03_验证/第二问孕妇整簇自助次数收敛.csv")
    prefix_ok = set(measurement_convergence["请求前缀次数"].astype(int)) == {100, 200, 400}
    prefix_ok &= set(bootstrap_convergence["请求前缀次数"].astype(int)) == {100, 200, 400}
    验收(
        "检测误差与整簇自助有效性",
        len(measurement) == 400
        and measurement["有效标志"].eq(1).all()
        and len(bootstrap) == 400
        and bootstrap["有效标志"].eq(1).all()
        and prefix_ok,
        "检测误差400/400、孕妇整簇自助400/400有效；100/200/400前缀均输出",
    )
    bootstrap_multiplier = bootstrap["BMI每增加1单位的时间尺度倍数"].to_numpy(float)
    bootstrap_cut = bootstrap["两组切点"].to_numpy(float)
    multiplier_ci = np.quantile(bootstrap_multiplier, [0.025, 0.975])
    cut_ci = np.quantile(bootstrap_cut, [0.025, 0.975])
    two_group_rate = float(bootstrap["贝叶斯信息准则选择两组标志"].mean())
    model_card = (输出根目录 / "02_模型结果/第二问无自拟参数模型卡.md").read_text(encoding="utf-8")
    验收(
        "不确定性不被隐藏",
        multiplier_ci[0] <= 1 <= multiplier_ci[1]
        and np.isclose(two_group_rate, 0.4)
        and "不能表述成稳健显著关联" in model_card
        and "离散切点并不稳定" in model_card,
        f"BMI时间尺度倍数95%区间={multiplier_ci.tolist()}含1；两组切点95%区间={cut_ci.tolist()}；BIC选两组比例={two_group_rate:.3f}",
    )

    sensitivity = pd.read_csv(输出根目录 / "03_验证/第二问数据质量口径敏感性.csv")
    验收(
        "质量口径只作敏感性",
        len(sensitivity) == 5 and sensitivity.iloc[0]["敏感性口径"] == "主口径",
        f"共{len(sensitivity)}个口径，均单列于敏感性文件；主口径位于首行",
    )
    parameter_table = pd.read_csv(输出根目录 / "02_模型结果/第二问参数来源表.csv")
    required_parameters = {
        "Y染色体浓度达标线 c",
        "NIPT可讨论日网格",
        "题面风险分界",
        "BMI中心与尺度",
        "组数K",
        "技术测量误差标准差",
        "显著性报告水平α",
        "自助与误差传播重复次数B",
        "随机种子",
        "分段位置参数候选网格",
        "分组BIC参数计数 p_K",
        "AFT数值优化方案",
        "AFT初值构造",
        "似然概率数值下限",
        "AFT截距β0",
        "AFT的BMI系数β1",
        "AFT尺度σ",
    }
    验收(
        "参数来源完整性",
        required_parameters.issubset(set(parameter_table["参数名称和符号"]))
        and parameter_table["审核状态"].eq("通过").all()
        and parameter_table["参数名称和符号"].is_unique,
        f"参数表{len(parameter_table)}行；必要参数缺失={sorted(required_parameters - set(parameter_table['参数名称和符号']))}",
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
        f"新图像文件={len(images)}；MATLAB/SVG提示词={len(prompts)}；不合格提示词={bad_prompts}",
    )
    验收(
        "Windows复现入口编码",
        一键脚本路径.read_bytes().startswith(b"\xef\xbb\xbf"),
        "一键PowerShell脚本为UTF-8 BOM，可由Windows PowerShell 5读取中文",
    )

    table = pd.DataFrame(验收记录)
    status = "PASS" if table["通过标志"].all() else "REJECTED"
    table.to_csv(独立复核目录 / "第二问独立总控验收清单.csv", index=False, encoding="utf-8-sig")

    failures = table.loc[~table["通过标志"], ["验收项", "独立证据"]].to_dict("records")
    report_lines = [
        "# 第二问独立总控复核报告",
        "",
        f"- 独立审核状态：**{status}**",
        f"- 验收项：{len(table)}项；失败：{len(failures)}项。",
        "- 复核方式：不调用建模脚本自审函数，从第一问冻结源、第二问CSV/JSON和哈希反向重建。",
        "",
        "## 关键独立结论",
        "",
        f"- 数据层：613个抽血事件、167名孕妇；左/区间/右删失为118/40/9。",
        f"- 技术复测：18组、36条、自由度18；合并组内标准差为{pooled_sd:.8f}。",
        f"- AFT：{运行清单['入选分布']}同时取得留一预测和BIC最优；三种候选似然均已独立复算。",
        f"- 分组：BIC主结果为1组，留一预测为2组；两组切点自助95%区间为[{cut_ci[0]:.6f}, {cut_ci[1]:.6f}]，不稳定，故只作敏感性。",
        f"- BMI时间尺度倍数整簇自助95%区间为[{multiplier_ci[0]:.6f}, {multiplier_ci[1]:.6f}]，包含1，不能声称稳健显著。",
        "- 决策：题面没有数值代价比或最低达标比例，当前材料未设置q或风险权重，完整输出时点—尚未达标概率前沿。",
        "",
        "## 归档裁决",
        "",
        "通过条件为全部独立检查通过。" if status == "PASS" else "存在失败项，必须打回修正，禁止归档。",
    ]
    if failures:
        report_lines.extend(["", "## 失败项", ""] + [f"- {row['验收项']}：{row['独立证据']}" for row in failures])
    (独立复核目录 / "第二问独立总控复核报告.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    record = {
        "生成时间": datetime.now().astimezone().isoformat(timespec="seconds"),
        "问题": "第二问",
        "状态": status,
        "独立验收项数": int(len(table)),
        "失败项数": int(len(failures)),
        "建模运行清单SHA256": 文件哈希(运行清单路径),
        "建模结果哈希表SHA256": 文件哈希(哈希表路径),
        "独立复核脚本SHA256": 文件哈希(脚本路径),
        "独立验收清单SHA256": 文件哈希(独立复核目录 / "第二问独立总控验收清单.csv"),
        "独立复核报告SHA256": 文件哈希(独立复核目录 / "第二问独立总控复核报告.md"),
    }
    (独立复核目录 / "第二问独立审核PASS记录.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(record, ensure_ascii=False, indent=2))
    if status != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
