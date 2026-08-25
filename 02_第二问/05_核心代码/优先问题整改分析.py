from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import sklearn
import scipy
import statsmodels
from scipy import stats
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold


预期附件哈希 = "14827156218bd4f7e4f16db4aa6d9f757c6648379e038ae6c6b58383648614af"
达标阈值 = 0.04
检测起始天 = 70
检测结束天 = 175
分析结束天_开区间 = 182

内部列名 = [
    "序号",
    "孕妇代码",
    "年龄",
    "身高",
    "体重",
    "末次月经原始值",
    "受孕方式",
    "检测日期原始值",
    "抽血次数",
    "孕周原始值",
    "BMI",
    "原始读段数",
    "比对比例",
    "重复读段比例",
    "唯一比对读段数",
    "GC含量",
    "13号染色体Z值",
    "18号染色体Z值",
    "21号染色体Z值",
    "X染色体Z值",
    "Y染色体Z值",
    "Y染色体浓度",
    "X染色体浓度",
    "13号染色体GC含量",
    "18号染色体GC含量",
    "21号染色体GC含量",
    "过滤读段比例",
    "非整倍体",
    "怀孕次数",
    "生产次数",
    "胎儿是否健康",
]


def 定位工作区(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / "00_题目与原始资料/02_原始数据/附件.xlsx").is_file():
            return candidate
    raise FileNotFoundError("无法定位C题论文工作区")


脚本目录 = Path(__file__).resolve().parent
工作区 = 定位工作区(脚本目录)
默认附件 = 工作区 / "00_题目与原始资料/02_原始数据/附件.xlsx"
默认输出目录 = 脚本目录.parent / "02_计算输出"


def 文件哈希(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def 写CSV(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def 写JSON(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def 解析孕周天数(value: Any) -> float:
    text = str(value).strip().lower()
    match = re.fullmatch(r"(\d+)w(?:\+(\d+))?", text)
    if match is None:
        return np.nan
    return float(int(match.group(1)) * 7 + int(match.group(2) or 0))


def 周天文本(days: float) -> str:
    value = int(round(days))
    return f"{value // 7}周{value % 7}天"


def 读取男胎记录(path: Path) -> pd.DataFrame:
    actual_hash = 文件哈希(path)
    if actual_hash.lower() != 预期附件哈希:
        raise RuntimeError(f"附件哈希变化：期望{预期附件哈希}，实际{actual_hash}")
    frame = pd.read_excel(path, sheet_name=0, engine="openpyxl")
    if frame.shape != (1082, 31):
        raise RuntimeError(f"男胎原始表尺寸异常：{frame.shape}")
    frame.columns = 内部列名
    frame["孕周天数"] = frame["孕周原始值"].map(解析孕周天数)
    if frame["孕周天数"].isna().any():
        raise RuntimeError("存在无法解析的孕周")
    frame["孕周数"] = frame["孕周天数"] / 7.0
    frame["数据段"] = np.where(pd.to_numeric(frame["序号"]) < 683, "序号683前", "序号683后")
    frame["达标标志"] = (pd.to_numeric(frame["Y染色体浓度"]) >= 达标阈值).astype(int)
    return frame


def 构造事件层(records: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    source = records.loc[
        (records["孕周天数"] >= 检测起始天)
        & (records["孕周天数"] < 分析结束天_开区间)
    ].copy()
    rows: list[dict[str, Any]] = []
    for (woman, draw), group in source.groupby(["孕妇代码", "抽血次数"], sort=True):
        week_values = sorted(pd.to_numeric(group["孕周天数"], errors="coerce").dropna().unique())
        ambiguous = int(len(week_values) != 1)
        week_days = np.nan if ambiguous else float(week_values[0])
        y = pd.to_numeric(group["Y染色体浓度"], errors="coerce")
        segment_values = group["数据段"].dropna().astype(str).unique().tolist()
        if len(segment_values) != 1:
            raise RuntimeError(f"事件跨越683边界：{woman}#{draw}")
        rows.append(
            {
                "孕妇代码": str(woman),
                "抽血次数": int(draw),
                "抽血事件键": f"{woman}#{int(draw)}",
                "数据段": segment_values[0],
                "序号最小值": int(pd.to_numeric(group["序号"]).min()),
                "序号最大值": int(pd.to_numeric(group["序号"]).max()),
                "记录数": int(len(group)),
                "孕周歧义标志": ambiguous,
                "孕周天数": week_days,
                "孕周数": week_days / 7.0 if np.isfinite(week_days) else np.nan,
                "年龄": float(pd.to_numeric(group["年龄"], errors="coerce").mean()),
                "身高": float(pd.to_numeric(group["身高"], errors="coerce").mean()),
                "体重": float(pd.to_numeric(group["体重"], errors="coerce").mean()),
                "BMI": float(pd.to_numeric(group["BMI"], errors="coerce").mean()),
                "生产次数": float(pd.to_numeric(group["生产次数"], errors="coerce").mean()),
                "怀孕次数": float(pd.to_numeric(group["怀孕次数"], errors="coerce").mean()),
                "Y浓度": float(y.mean()),
                "Y浓度事件内标准差": float(y.std(ddof=1)) if len(y) > 1 else 0.0,
                "GC含量": float(pd.to_numeric(group["GC含量"], errors="coerce").mean()),
                "原始读段数": float(pd.to_numeric(group["原始读段数"], errors="coerce").mean()),
                "比对比例": float(pd.to_numeric(group["比对比例"], errors="coerce").mean()),
                "重复读段比例": float(pd.to_numeric(group["重复读段比例"], errors="coerce").mean()),
                "过滤读段比例": float(pd.to_numeric(group["过滤读段比例"], errors="coerce").mean()),
            }
        )
    all_events = pd.DataFrame(rows).sort_values(["孕妇代码", "孕周天数", "抽血次数"]).reset_index(drop=True)
    excluded = all_events.loc[all_events["孕周歧义标志"].eq(1)].copy()
    events = all_events.loc[all_events["孕周歧义标志"].eq(0)].copy().reset_index(drop=True)
    if not events["Y浓度"].between(0, 1, inclusive="neither").all():
        raise RuntimeError("事件Y浓度不全在(0,1)")
    events["Y浓度logit"] = np.log(events["Y浓度"] / (1.0 - events["Y浓度"]))
    events["达标标志"] = (events["Y浓度"] >= 达标阈值).astype(int)
    events["孕妇平均BMI"] = events.groupby("孕妇代码")["BMI"].transform("mean")
    events["BMI个体内偏差"] = events["BMI"] - events["孕妇平均BMI"]
    events["孕妇事件数"] = events.groupby("孕妇代码")["抽血事件键"].transform("size")
    first = (
        events.sort_values(["孕妇代码", "孕周天数", "抽血次数"])
        .groupby("孕妇代码", as_index=False)
        .first()[["孕妇代码", "BMI", "年龄", "身高", "体重", "生产次数"]]
        .rename(
            columns={
                "BMI": "首次BMI",
                "年龄": "首次年龄",
                "身高": "首次身高",
                "体重": "首次体重",
                "生产次数": "首次生产次数",
            }
        )
    )
    events = events.merge(first, on="孕妇代码", how="left", validate="many_to_one")
    return events, excluded


def 题面BMI组(series: pd.Series) -> pd.Series:
    return pd.cut(
        series,
        bins=[20.0, 28.0, 32.0, 36.0, 40.0, np.inf],
        right=False,
        labels=["[20,28)", "[28,32)", "[32,36)", "[36,40)", "[40,+∞)"],
    )


def 样本摘要(records: pd.DataFrame, events: pd.DataFrame, excluded: pd.DataFrame) -> pd.DataFrame:
    rows = []
    scopes = {
        "序号683前": events.loc[events["数据段"].eq("序号683前")],
        "序号683后": events.loc[events["数据段"].eq("序号683后")],
        "全样本": events,
    }
    for name, scope in scopes.items():
        persons = scope.sort_values(["孕妇代码", "孕周天数"]).drop_duplicates("孕妇代码")
        raw = records.loc[
            (records["孕周天数"] >= 检测起始天)
            & (records["孕周天数"] < 分析结束天_开区间)
        ]
        if name != "全样本":
            raw = raw.loc[raw["数据段"].eq(name)]
        excluded_scope = excluded if name == "全样本" else excluded.loc[excluded["数据段"].eq(name)]
        rows.append(
            {
                "样本范围": name,
                "原始记录数": int(len(raw)),
                "孕妇数": int(scope["孕妇代码"].nunique()),
                "有效抽血事件数": int(len(scope)),
                "孕周歧义排除事件数": int(len(excluded_scope)),
                "多记录事件数": int(scope["记录数"].gt(1).sum()),
                "多记录事件所含记录数": int(scope.loc[scope["记录数"].gt(1), "记录数"].sum()),
                "达标事件数": int(scope["达标标志"].sum()),
                "达标事件比例": float(scope["达标标志"].mean()),
                "BMI最小值": float(persons["首次BMI"].min()),
                "BMI中位数": float(persons["首次BMI"].median()),
                "BMI最大值": float(persons["首次BMI"].max()),
                "最早孕周": float(scope["孕周数"].min()),
                "最晚孕周": float(scope["孕周数"].max()),
            }
        )
    return pd.DataFrame(rows)


def BMI组人数表(events: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for scope_name, scope in {
        "序号683前": events.loc[events["数据段"].eq("序号683前")],
        "序号683后": events.loc[events["数据段"].eq("序号683后")],
        "全样本": events,
    }.items():
        persons = (
            scope.sort_values(["孕妇代码", "孕周天数", "抽血次数"])
            .drop_duplicates("孕妇代码")
            .copy()
        )
        persons["题面BMI组"] = 题面BMI组(persons["首次BMI"])
        for group_name in ["[20,28)", "[28,32)", "[32,36)", "[36,40)", "[40,+∞)"]:
            group = persons.loc[persons["题面BMI组"].astype(str).eq(group_name)]
            rows.append(
                {
                    "样本范围": scope_name,
                    "题面BMI组": group_name,
                    "孕妇数": int(len(group)),
                    "首次BMI最小值": float(group["首次BMI"].min()) if len(group) else np.nan,
                    "首次BMI最大值": float(group["首次BMI"].max()) if len(group) else np.nan,
                }
            )
    return pd.DataFrame(rows)


def 非吸收态审计(events: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for scope_name, scope in {
        "序号683前": events.loc[events["数据段"].eq("序号683前")],
        "序号683后": events.loc[events["数据段"].eq("序号683后")],
        "全样本": events,
    }.items():
        woman_rows = []
        for woman, group in scope.groupby("孕妇代码", sort=False):
            group = group.sort_values(["孕周天数", "抽血次数"])
            status = group["达标标志"].to_numpy(dtype=int)
            hits = np.flatnonzero(status == 1)
            first_hit = int(hits[0]) if len(hits) else None
            post_drop = int(first_hit is not None and np.any(status[first_hit + 1 :] == 0))
            woman_rows.append(
                {
                    "孕妇代码": woman,
                    "首次观测即达标": int(first_hit == 0),
                    "观测期内由未达标转为达标": int(first_hit is not None and first_hit > 0),
                    "观测期内从未达标": int(first_hit is None),
                    "达标后回落": post_drop,
                    "仅一次事件": int(len(group) == 1),
                }
            )
        woman_table = pd.DataFrame(woman_rows)
        rows.append(
            {
                "样本范围": scope_name,
                "孕妇数": int(len(woman_table)),
                "首次观测即达标人数": int(woman_table["首次观测即达标"].sum()),
                "观测期内由未达标转为达标人数": int(woman_table["观测期内由未达标转为达标"].sum()),
                "观测期内从未达标人数": int(woman_table["观测期内从未达标"].sum()),
                "达标后回落人数": int(woman_table["达标后回落"].sum()),
                "仅一次事件人数": int(woman_table["仅一次事件"].sum()),
                "严格吸收态被观测反例否决标志": int(woman_table["达标后回落"].sum() > 0),
            }
        )
    return pd.DataFrame(rows)


@dataclass
class MixedFit:
    result: Any
    method: str
    warning_text: str


def 拟合混合模型(formula: str, data: pd.DataFrame, random_formula: str = "1 + 孕周中心") -> MixedFit:
    model = smf.mixedlm(
        formula,
        data=data,
        groups=data["孕妇代码"],
        re_formula=random_formula,
    )
    messages: list[str] = []
    last_error: Exception | None = None
    for method in ("lbfgs", "powell"):
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                result = model.fit(reml=False, method=method, maxiter=3000, disp=False)
            messages.extend(str(item.message) for item in caught)
            if bool(getattr(result, "converged", False)) and np.isfinite(result.llf):
                return MixedFit(result, method, "；".join(dict.fromkeys(messages)))
        except Exception as exc:
            last_error = exc
            messages.append(f"{method}:{type(exc).__name__}:{exc}")
    raise RuntimeError(f"混合模型未收敛：{last_error}；{'；'.join(messages)}")


def 构造第一问变量(events: pd.DataFrame, centers: dict[str, float]) -> pd.DataFrame:
    data = events.copy()
    data["孕妇平均BMI"] = data.groupby("孕妇代码")["BMI"].transform("mean")
    data["BMI个体内偏差"] = data["BMI"] - data["孕妇平均BMI"]
    data["孕周中心"] = data["孕周数"] - centers["孕周"]
    data["妇间BMI中心"] = data["孕妇平均BMI"] - centers["妇间BMI"]
    data["年龄中心"] = data["年龄"] - centers["年龄"]
    data["生产次数中心"] = data["生产次数"] - centers["生产次数"]
    data["后段标志"] = data["数据段"].eq("序号683后").astype(int)
    return data


第一问公式 = (
    "Y浓度logit ~ 孕周中心 + I(孕周中心 ** 2) + 妇间BMI中心 + "
    "BMI个体内偏差 + 年龄中心 + 生产次数中心"
)


def 第一问683敏感性(events: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    persons = events.sort_values(["孕妇代码", "孕周天数", "抽血次数"]).drop_duplicates("孕妇代码")
    centers = {
        "孕周": float(events["孕周数"].mean()),
        "妇间BMI": float(persons["孕妇平均BMI"].mean()),
        "年龄": float(persons["年龄"].mean()),
        "生产次数": float(persons["生产次数"].mean()),
    }
    parameter_rows = []
    prediction_rows = []
    fits: dict[str, MixedFit] = {}
    for scope_name, scope in {
        "序号683前": events.loc[events["数据段"].eq("序号683前")],
        "序号683后": events.loc[events["数据段"].eq("序号683后")],
        "全样本": events,
    }.items():
        data = 构造第一问变量(scope, centers)
        fit = 拟合混合模型(第一问公式, data)
        fits[scope_name] = fit
        result = fit.result
        fixed = result.fe_params
        fixed_se = result.bse_fe
        for term in fixed.index:
            estimate = float(fixed[term])
            se = float(fixed_se[term])
            diagnostic_failure = (
                "not positive definite" in fit.warning_text
                or "Gradient optimization failed" in fit.warning_text
                or not bool(result.converged)
            )
            parameter_rows.append(
                {
                    "模型角色": "第一问混合模型敏感性；全样本分段GEE另作主模型",
                    "样本范围": scope_name,
                    "参数": term,
                    "估计值": estimate,
                    "标准误": se,
                    "Wald95%下限": estimate - 1.959963984540054 * se,
                    "Wald95%上限": estimate + 1.959963984540054 * se,
                    "优化器": fit.method,
                    "收敛标志": int(result.converged),
                    "警告": fit.warning_text,
                    "推断可用标志": int(not diagnostic_failure),
                    "诊断说明": (
                        "Hessian/梯度诊断未通过，系数和Wald区间不得作有效推断"
                        if diagnostic_failure
                        else "数值诊断未触发硬失败；仍按敏感性解释"
                    ),
                    "AIC": float(result.aic),
                    "BIC": float(result.bic),
                    "对数似然": float(result.llf),
                    "孕妇数": int(data["孕妇代码"].nunique()),
                    "事件数": int(len(data)),
                }
            )
        b1 = float(fixed.get("孕周中心", np.nan))
        b2 = float(fixed.get("I(孕周中心 ** 2)", np.nan))
        turning = centers["孕周"] - b1 / (2.0 * b2) if np.isfinite(b2) and abs(b2) > 1e-12 else np.nan
        turning_type = "最低点" if b2 > 0 else ("最高点" if b2 < 0 else "无二次转折")
        for week in (12.0, 16.0, 20.0, 24.0):
            row = pd.DataFrame(
                {
                    "孕周中心": [week - centers["孕周"]],
                    "妇间BMI中心": [0.0],
                    "BMI个体内偏差": [0.0],
                    "年龄中心": [0.0],
                    "生产次数中心": [0.0],
                }
            )
            logit_value = float(result.predict(row).iloc[0])
            prediction_rows.append(
                {
                    "样本范围": scope_name,
                    "孕周": week,
                    "参考条件固定效应预测Y浓度": 1.0 / (1.0 + math.exp(-logit_value)),
                    "链接尺度转折点孕周": turning,
                    "转折类型": turning_type,
                }
            )

    full_data = 构造第一问变量(events, centers)
    base_with_segment = 拟合混合模型(第一问公式 + " + 后段标志", full_data)
    interaction_formula = (
        第一问公式
        + " + 后段标志 + 后段标志:孕周中心 + 后段标志:I(孕周中心 ** 2)"
        + " + 后段标志:妇间BMI中心 + 后段标志:BMI个体内偏差"
    )
    interaction = 拟合混合模型(interaction_formula, full_data)
    lr = 2.0 * (interaction.result.llf - base_with_segment.result.llf)
    df = int(len(interaction.result.fe_params) - len(base_with_segment.result.fe_params))
    p_value = float(stats.chi2.sf(max(lr, 0.0), df))
    comparison = pd.DataFrame(
        [
            {
                "比较": "前后段关键关系是否相同",
                "简化模型": "共同斜率+后段截距",
                "扩展模型": "后段与孕周一次/二次、妇间BMI、个体内BMI交互",
                "似然比统计量": lr,
                "自由度": df,
                "P值": p_value,
                "简化模型AIC": float(base_with_segment.result.aic),
                "扩展模型AIC": float(interaction.result.aic),
                "简化模型BIC": float(base_with_segment.result.bic),
                "扩展模型BIC": float(interaction.result.bic),
                "简化模型优化器": base_with_segment.method,
                "扩展模型优化器": interaction.method,
            }
        ]
    )
    return pd.DataFrame(parameter_rows), pd.DataFrame(prediction_rows), comparison


概率候选 = [
    "线性浓度混合",
    "二次浓度混合",
    "线性达标GEE",
    "单调三次Bernstein达标模型",
]

候选简约顺序 = {
    "线性达标GEE": 1,
    "线性浓度混合": 2,
    "单调三次Bernstein达标模型": 3,
    "二次浓度混合": 4,
}


def 稳健尺度(series: pd.Series) -> tuple[float, float]:
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    center = float(np.nanmedian(values))
    q25, q75 = np.nanquantile(values, [0.25, 0.75])
    scale = float(q75 - q25)
    if not np.isfinite(scale) or scale <= 0:
        scale = float(np.nanstd(values, ddof=0))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError(f"变量{series.name}没有可用尺度")
    return center, scale


def 角色变量(role: str) -> list[tuple[str, str]]:
    if role == "第二问":
        return [("首次BMI", "BMI标准化")]
    if role == "第三问":
        return [
            ("首次BMI", "BMI标准化"),
            ("首次年龄", "年龄标准化"),
            ("首次身高", "身高标准化"),
            ("首次生产次数", "生产次数标准化"),
        ]
    raise ValueError(f"未知角色：{role}")


def 构造概率变量(
    frame: pd.DataFrame,
    role: str,
    scales: dict[str, tuple[float, float]] | None = None,
) -> tuple[pd.DataFrame, dict[str, tuple[float, float]]]:
    data = frame.copy()
    definitions = [("孕周数", "孕周标准化"), *角色变量(role)]
    if scales is None:
        scales = {source: 稳健尺度(data[source]) for source, _ in definitions}
    for source, target in definitions:
        center, scale = scales[source]
        data[target] = (pd.to_numeric(data[source], errors="coerce") - center) / scale
    data["后段标志"] = data["数据段"].eq("序号683后").astype(int)
    return data, scales


def 二分类稀疏诊断(data: pd.DataFrame, role: str) -> dict[str, Any]:
    hit = data["达标标志"].eq(1)
    hit_women = int(data.loc[hit, "孕妇代码"].nunique())
    miss_women = int(data.loc[~hit, "孕妇代码"].nunique())
    diagnostic: dict[str, Any] = {
        "达标事件数": int(hit.sum()),
        "未达标事件数": int((~hit).sum()),
        "达标孕妇数": hit_women,
        "未达标孕妇数": miss_women,
        "两类结局均至少来自两个孕妇标志": int(min(hit_women, miss_women) >= 2),
        "生产次数水平单一结局标志": 0,
        "生产次数分布": "",
    }
    if role == "第三问":
        table = pd.crosstab(data["首次生产次数"], data["达标标志"], dropna=False)
        for outcome in (0, 1):
            if outcome not in table.columns:
                table[outcome] = 0
        table = table[[0, 1]].sort_index()
        diagnostic["生产次数水平单一结局标志"] = int(
            ((table[0] == 0) | (table[1] == 0)).any()
        )
        diagnostic["生产次数分布"] = json.dumps(
            {
                str(index): {"未达标": int(row[0]), "达标": int(row[1])}
                for index, row in table.iterrows()
            },
            ensure_ascii=False,
        )
    diagnostic["推断可用标志"] = int(
        diagnostic["两类结局均至少来自两个孕妇标志"] == 1
    )
    diagnostic["诊断说明"] = (
        "两类结局均有至少两个独立孕妇提供信息"
        if diagnostic["推断可用标志"] == 1
        else "至少一类结局仅来自0或1名孕妇；系数、P值和时点只作不稳定敏感性，不作有效推断"
    )
    return diagnostic


def 人等权(frame: pd.DataFrame) -> np.ndarray:
    counts = frame.groupby("孕妇代码")["孕妇代码"].transform("size").to_numpy(dtype=float)
    return 1.0 / counts


def Bernstein矩阵(week: np.ndarray, degree: int = 3) -> np.ndarray:
    u = np.clip((np.asarray(week, dtype=float) - 10.0) / 16.0, 0.0, 1.0)
    columns = []
    for k in range(degree + 1):
        columns.append(math.comb(degree, k) * (u**k) * ((1.0 - u) ** (degree - k)))
    return np.column_stack(columns)


def Bernstein设计(data: pd.DataFrame, role: str) -> tuple[np.ndarray, np.ndarray]:
    basis = Bernstein矩阵(data["孕周数"].to_numpy(dtype=float), degree=3)
    covariate_names = [target for _, target in 角色变量(role)]
    covariates = data[covariate_names].to_numpy(dtype=float)
    return basis, covariates


def Bernstein系数(theta: np.ndarray) -> np.ndarray:
    return np.cumsum(np.asarray(theta[:4], dtype=float))


def 拟合Bernstein(data: pd.DataFrame, role: str) -> dict[str, Any]:
    basis, covariates = Bernstein设计(data, role)
    target = data["达标标志"].to_numpy(dtype=float)
    weights = 人等权(data)
    weighted_rate = float(np.average(target, weights=weights))
    weighted_rate = float(np.clip(weighted_rate, 1e-6, 1.0 - 1e-6))
    initial = np.zeros(4 + covariates.shape[1], dtype=float)
    initial[0] = math.log(weighted_rate / (1.0 - weighted_rate))

    def objective(theta: np.ndarray) -> float:
        time_coefs = Bernstein系数(theta)
        eta = basis @ time_coefs + covariates @ theta[4:]
        return float(np.sum(weights * (np.logaddexp(0.0, eta) - target * eta)))

    bounds = [(None, None), (0.0, None), (0.0, None), (0.0, None)] + [
        (None, None)
    ] * covariates.shape[1]
    from scipy.optimize import minimize

    result = minimize(
        objective,
        initial,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 3000, "ftol": 1e-12},
    )
    if not bool(result.success) or not np.isfinite(result.fun):
        raise RuntimeError(f"Bernstein模型未收敛：{result.message}")
    return {
        "类型": "Bernstein",
        "结果": result,
        "角色": role,
        "参数": np.asarray(result.x, dtype=float),
        "收敛": True,
    }


def 拟合概率候选(name: str, raw: pd.DataFrame, role: str) -> dict[str, Any]:
    data, scales = 构造概率变量(raw, role)
    covariates = [target for _, target in 角色变量(role)]
    covariate_rhs = " + ".join(covariates)
    if name == "线性浓度混合":
        formula = f"Y浓度logit ~ 孕周标准化 + {covariate_rhs}"
        fit = 拟合混合模型(formula, data, random_formula="1 + 孕周标准化")
        return {"类型": "浓度混合", "结果": fit.result, "拟合": fit, "尺度": scales, "角色": role, "名称": name}
    if name == "二次浓度混合":
        formula = f"Y浓度logit ~ 孕周标准化 + I(孕周标准化 ** 2) + {covariate_rhs}"
        fit = 拟合混合模型(formula, data, random_formula="1 + 孕周标准化")
        return {"类型": "浓度混合", "结果": fit.result, "拟合": fit, "尺度": scales, "角色": role, "名称": name}
    if name == "线性达标GEE":
        formula = f"达标标志 ~ 孕周标准化 + {covariate_rhs}"
        model = smf.gee(
            formula,
            groups=data["孕妇代码"],
            data=data,
            family=sm.families.Binomial(),
            cov_struct=sm.cov_struct.Exchangeable(),
        )
        result = model.fit(maxiter=1000)
        if not bool(getattr(result, "converged", True)):
            raise RuntimeError("GEE未收敛")
        return {
            "类型": "GEE",
            "结果": result,
            "尺度": scales,
            "角色": role,
            "名称": name,
            "诊断": 二分类稀疏诊断(data, role),
        }
    if name == "单调三次Bernstein达标模型":
        model = 拟合Bernstein(data, role)
        model.update({"尺度": scales, "名称": name})
        return model
    raise ValueError(f"未知候选：{name}")


def 拟合段调整GEE(raw: pd.DataFrame, role: str) -> dict[str, Any]:
    data, scales = 构造概率变量(raw, role)
    covariates = [target for _, target in 角色变量(role)]
    formula = f"达标标志 ~ 孕周标准化 + {' + '.join(covariates)} + 后段标志"
    model = smf.gee(
        formula,
        groups=data["孕妇代码"],
        data=data,
        family=sm.families.Binomial(),
        cov_struct=sm.cov_struct.Exchangeable(),
    )
    result = model.fit(maxiter=1000)
    if not bool(getattr(result, "converged", True)):
        raise RuntimeError("全样本段调整GEE未收敛")
    return {
        "类型": "GEE",
        "结果": result,
        "尺度": scales,
        "角色": role,
        "名称": "全样本段调整线性达标GEE",
        "诊断": 二分类稀疏诊断(data, role),
    }


def 预测概率(model: dict[str, Any], raw: pd.DataFrame) -> np.ndarray:
    data, _ = 构造概率变量(raw, model["角色"], model["尺度"])
    if model["类型"] == "浓度混合":
        result = model["结果"]
        mean = np.asarray(result.predict(data), dtype=float)
        z = np.column_stack([np.ones(len(data)), data["孕周标准化"].to_numpy(dtype=float)])
        covariance = np.asarray(result.cov_re, dtype=float)
        variance = np.einsum("ij,jk,ik->i", z, covariance, z) + float(result.scale)
        threshold_logit = math.log(达标阈值 / (1.0 - 达标阈值))
        return stats.norm.sf((threshold_logit - mean) / np.sqrt(np.maximum(variance, 1e-12)))
    if model["类型"] == "GEE":
        return np.asarray(model["结果"].predict(data), dtype=float)
    if model["类型"] == "Bernstein":
        basis, covariates = Bernstein设计(data, model["角色"])
        theta = model["参数"]
        eta = basis @ Bernstein系数(theta) + covariates @ theta[4:]
        return 1.0 / (1.0 + np.exp(-np.clip(eta, -40.0, 40.0)))
    raise ValueError(model["类型"])


def 分组交叉验证(events: pd.DataFrame, role: str, repeat_count: int = 5) -> tuple[pd.DataFrame, pd.DataFrame]:
    seed_base = int(预期附件哈希[:8], 16)
    predictions = []
    failures = []
    for repeat in range(repeat_count):
        splitter = GroupKFold(n_splits=5, shuffle=True, random_state=seed_base + repeat)
        for fold, (train_index, test_index) in enumerate(
            splitter.split(events, groups=events["孕妇代码"]), start=1
        ):
            train = events.iloc[train_index].copy()
            test = events.iloc[test_index].copy()
            for candidate in 概率候选:
                try:
                    model = 拟合概率候选(candidate, train, role)
                    probability = np.clip(预测概率(model, test), 1e-12, 1.0 - 1e-12)
                    for local_index, (_, row) in enumerate(test.iterrows()):
                        predictions.append(
                            {
                                "问题": role,
                                "候选模型": candidate,
                                "重复": repeat + 1,
                                "折": fold,
                                "孕妇代码": row["孕妇代码"],
                                "抽血事件键": row["抽血事件键"],
                                "实际达标标志": int(row["达标标志"]),
                                "预测达标概率": float(probability[local_index]),
                            }
                        )
                except Exception as exc:
                    failures.append(
                        {
                            "问题": role,
                            "候选模型": candidate,
                            "重复": repeat + 1,
                            "折": fold,
                            "错误类型": type(exc).__name__,
                            "错误": str(exc),
                        }
                    )
    prediction_columns = [
        "问题", "候选模型", "重复", "折", "孕妇代码", "抽血事件键",
        "实际达标标志", "预测达标概率",
    ]
    failure_columns = ["问题", "候选模型", "重复", "折", "错误类型", "错误"]
    return (
        pd.DataFrame(predictions, columns=prediction_columns),
        pd.DataFrame(failures, columns=failure_columns),
    )


def 全样本段调整交叉验证(
    events: pd.DataFrame,
    role: str,
    repeat_count: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    seed_base = int(预期附件哈希[48:56], 16)
    prediction_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []
    for repeat in range(repeat_count):
        splitter = GroupKFold(n_splits=5, shuffle=True, random_state=seed_base + repeat)
        for fold, (train_index, test_index) in enumerate(
            splitter.split(events, groups=events["孕妇代码"]), start=1
        ):
            train = events.iloc[train_index].copy()
            test = events.iloc[test_index].copy()
            for model_name in ("全样本未调段线性达标GEE", "全样本段调整线性达标GEE"):
                try:
                    model = (
                        拟合概率候选("线性达标GEE", train, role)
                        if model_name == "全样本未调段线性达标GEE"
                        else 拟合段调整GEE(train, role)
                    )
                    probability = np.clip(预测概率(model, test), 1e-12, 1.0 - 1e-12)
                    for local_index, (_, row) in enumerate(test.iterrows()):
                        prediction_rows.append(
                            {
                                "问题": role,
                                "覆盖性模型": model_name,
                                "重复": repeat + 1,
                                "折": fold,
                                "数据段": row["数据段"],
                                "孕妇代码": row["孕妇代码"],
                                "抽血事件键": row["抽血事件键"],
                                "实际达标标志": int(row["达标标志"]),
                                "预测达标概率": float(probability[local_index]),
                            }
                        )
                except Exception as exc:
                    failure_rows.append(
                        {
                            "问题": role,
                            "覆盖性模型": model_name,
                            "重复": repeat + 1,
                            "折": fold,
                            "错误类型": type(exc).__name__,
                            "错误": str(exc),
                        }
                    )
    predictions = pd.DataFrame(
        prediction_rows,
        columns=[
            "问题", "覆盖性模型", "重复", "折", "数据段", "孕妇代码",
            "抽血事件键", "实际达标标志", "预测达标概率",
        ],
    )
    failures = pd.DataFrame(
        failure_rows,
        columns=["问题", "覆盖性模型", "重复", "折", "错误类型", "错误"],
    )
    summary_rows = []
    for model_name, model_frame in predictions.groupby("覆盖性模型", sort=False):
        averaged = (
            model_frame.groupby(
                ["数据段", "孕妇代码", "抽血事件键", "实际达标标志"], as_index=False
            )["预测达标概率"].mean()
        )
        for scope_name, scope in [
            ("全样本", averaged),
            ("序号683前", averaged.loc[averaged["数据段"].eq("序号683前")]),
            ("序号683后", averaged.loc[averaged["数据段"].eq("序号683后")]),
        ]:
            counts = scope.groupby("孕妇代码")["孕妇代码"].transform("size").to_numpy(dtype=float)
            weights = 1.0 / counts
            y = scope["实际达标标志"].to_numpy(dtype=float)
            p = np.clip(scope["预测达标概率"].to_numpy(dtype=float), 1e-12, 1.0 - 1e-12)
            summary_rows.append(
                {
                    "问题": role,
                    "覆盖性模型": model_name,
                    "评估范围": scope_name,
                    "孕妇数": int(scope["孕妇代码"].nunique()),
                    "事件数": int(len(scope)),
                    "孕妇等权实际达标比例": float(np.average(y, weights=weights)),
                    "孕妇等权平均预测概率": float(np.average(p, weights=weights)),
                    "孕妇等权对数损失": float(log_loss(y, p, sample_weight=weights, labels=[0, 1])),
                    "孕妇等权Brier分数": float(np.average((y - p) ** 2, weights=weights)),
                    "解释边界": "段调整只控制数据段基线差；683后失败结局仅来自1名孕妇，后段斜率不可稳定识别",
                }
            )
    return predictions, failures, pd.DataFrame(summary_rows)


def 汇总交叉验证(predictions: pd.DataFrame, failures: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    expected_repeats = int(predictions["重复"].max()) if len(predictions) else 0
    rows = []
    for candidate in 概率候选:
        candidate_predictions = predictions.loc[predictions["候选模型"].eq(candidate)].copy()
        failure_count = int(failures["候选模型"].eq(candidate).sum()) if len(failures) else 0
        if len(candidate_predictions) == 0:
            rows.append({"候选模型": candidate, "失败折数": failure_count})
            continue
        averaged = (
            candidate_predictions.groupby(
                ["孕妇代码", "抽血事件键", "实际达标标志"], as_index=False
            )["预测达标概率"]
            .mean()
        )
        counts = averaged.groupby("孕妇代码")["孕妇代码"].transform("size").to_numpy(dtype=float)
        weights = 1.0 / counts
        y = averaged["实际达标标志"].to_numpy(dtype=int)
        p = np.clip(averaged["预测达标概率"].to_numpy(dtype=float), 1e-12, 1.0 - 1e-12)
        rows.append(
            {
                "候选模型": candidate,
                "失败折数": failure_count,
                "每事件获得预测重复数": int(round(len(candidate_predictions) / max(len(averaged), 1))),
                "预期重复数": expected_repeats,
                "孕妇数": int(averaged["孕妇代码"].nunique()),
                "事件数": int(len(averaged)),
                "孕妇等权对数损失": float(log_loss(y, p, sample_weight=weights, labels=[0, 1])),
                "孕妇等权Brier分数": float(np.average((y - p) ** 2, weights=weights)),
                "孕妇等权ROC曲线下面积": float(roc_auc_score(y, p, sample_weight=weights)),
                "孕妇等权PR曲线下面积": float(average_precision_score(y, p, sample_weight=weights)),
                "孕妇等权实际达标比例": float(np.average(y, weights=weights)),
                "孕妇等权平均预测概率": float(np.average(p, weights=weights)),
            }
        )
    return pd.DataFrame(rows)


def 候选逐孕妇损失比较(
    predictions: pd.DataFrame,
    eligible_names: list[str],
    bootstrap_count: int,
    role: str,
) -> tuple[pd.DataFrame, str, str]:
    person_losses: dict[str, pd.Series] = {}
    mean_losses: dict[str, float] = {}
    for candidate in eligible_names:
        frame = predictions.loc[predictions["候选模型"].eq(candidate)].copy()
        averaged = frame.groupby(
            ["孕妇代码", "抽血事件键", "实际达标标志"], as_index=False
        )["预测达标概率"].mean()
        y = averaged["实际达标标志"].to_numpy(dtype=float)
        p = np.clip(averaged["预测达标概率"].to_numpy(dtype=float), 1e-12, 1.0 - 1e-12)
        averaged["事件对数损失"] = -(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))
        person_loss = averaged.groupby("孕妇代码")["事件对数损失"].mean().sort_index()
        person_losses[candidate] = person_loss
        mean_losses[candidate] = float(person_loss.mean())
    raw_best = min(eligible_names, key=lambda item: (mean_losses[item], 候选简约顺序[item]))
    rows = []
    rng = np.random.default_rng(int(预期附件哈希[32:40], 16) + (0 if role == "第二问" else 1))
    equivalent = []
    for candidate in eligible_names:
        joined = pd.concat(
            [person_losses[candidate].rename("候选"), person_losses[raw_best].rename("最低损失路线")],
            axis=1,
            join="inner",
        ).dropna()
        difference = (joined["候选"] - joined["最低损失路线"]).to_numpy(dtype=float)
        if candidate == raw_best:
            lower = 0.0
            upper = 0.0
        else:
            bootstrap = []
            for _ in range(bootstrap_count):
                index = rng.integers(0, len(difference), size=len(difference))
                bootstrap.append(float(np.mean(difference[index])))
            lower, upper = np.quantile(bootstrap, [0.025, 0.975])
        includes_zero = int(lower <= 0.0 <= upper)
        if includes_zero:
            equivalent.append(candidate)
        rows.append(
            {
                "问题": role,
                "最低样本外损失路线": raw_best,
                "比较候选": candidate,
                "简约顺序": 候选简约顺序[candidate],
                "逐孕妇平均对数损失": mean_losses[candidate],
                "候选减最低路线损失差": float(np.mean(difference)),
                "损失差整簇重采样95%下限": float(lower),
                "损失差整簇重采样95%上限": float(upper),
                "95%区间包含0标志": includes_zero,
                "重采样次数": bootstrap_count,
            }
        )
    selected = min(equivalent, key=lambda item: 候选简约顺序[item])
    reason = (
        f"最低样本外损失路线为{raw_best}；{selected}相对该路线的逐孕妇损失差95%区间包含0，"
        "没有稳定劣化证据，按预声明简约顺序入选"
        if selected != raw_best
        else f"{raw_best}同时取得最低样本外损失且没有更简洁的等效路线"
    )
    comparison = pd.DataFrame(rows)
    comparison["最终入选标志"] = comparison["比较候选"].eq(selected).astype(int)
    return comparison, selected, reason


def 风险等级(day: int) -> str:
    if day <= 84:
        return "12周以内：题面称风险较低"
    if day < 91:
        return "12周后至13周前：题面未单列"
    return "13至27周：题面称风险高"


def 严格技术复测误差(records: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    source = records.loc[
        (records["孕周天数"] >= 检测起始天)
        & (records["孕周天数"] < 分析结束天_开区间)
    ].copy()
    source["检测日期键"] = source["检测日期原始值"].astype(str)
    rows = []
    sum_squares = 0.0
    degrees = 0
    for keys, group in source.groupby(
        ["孕妇代码", "抽血次数", "检测日期键", "孕周原始值"],
        dropna=False,
        sort=True,
    ):
        if len(group) < 2:
            continue
        values = pd.to_numeric(group["Y染色体浓度"], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(values).all():
            continue
        group_df = len(values) - 1
        group_variance = float(np.var(values, ddof=1))
        sum_squares += group_df * group_variance
        degrees += group_df
        rows.append(
            {
                "孕妇代码": str(keys[0]),
                "抽血次数": int(keys[1]),
                "检测日期": str(keys[2]),
                "孕周原始值": str(keys[3]),
                "数据段": str(group["数据段"].iloc[0]),
                "记录数": int(len(group)),
                "Y浓度均值": float(np.mean(values)),
                "Y浓度组内标准差": math.sqrt(group_variance),
                "组内自由度": group_df,
            }
        )
    if degrees <= 0:
        raise RuntimeError("没有严格技术复测可估计测量误差")
    pooled_variance = sum_squares / degrees
    summary = {
        "严格技术复测组数": float(len(rows)),
        "严格技术复测记录数": float(sum(row["记录数"] for row in rows)),
        "合并组内方差": float(pooled_variance),
        "合并组内标准差": float(math.sqrt(pooled_variance)),
        "自由度": float(degrees),
    }
    return pd.DataFrame(rows), summary


def 概率模型参数表(model: dict[str, Any], scope_name: str, role: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    parameter_rows = []
    scale_rows = []
    for source, (center, scale) in model["尺度"].items():
        scale_rows.append(
            {
                "样本范围": scope_name,
                "问题": role,
                "候选模型": model["名称"],
                "原变量": source,
                "中心值_中位数": center,
                "尺度_四分位距或退化时标准差": scale,
            }
        )
    if model["类型"] == "Bernstein":
        theta = np.asarray(model["参数"], dtype=float)
        time_coefs = Bernstein系数(theta)
        for index, value in enumerate(time_coefs):
            parameter_rows.append(
                {
                    "样本范围": scope_name,
                    "问题": role,
                    "候选模型": model["名称"],
                    "参数": f"三次Bernstein时间系数a{index}",
                    "估计值": float(value),
                    "参数来源": "附件数据约束极大似然估计",
                    "约束": "a0<=a1<=a2<=a3",
                }
            )
        for index, value in enumerate(theta[1:4], start=1):
            parameter_rows.append(
                {
                    "样本范围": scope_name,
                    "问题": role,
                    "候选模型": model["名称"],
                    "参数": f"相邻时间系数增量delta{index}",
                    "估计值": float(value),
                    "参数来源": "附件数据约束极大似然估计",
                    "约束": ">=0",
                }
            )
        for (_, target), value in zip(角色变量(role), theta[4:], strict=True):
            parameter_rows.append(
                {
                    "样本范围": scope_name,
                    "问题": role,
                    "候选模型": model["名称"],
                    "参数": target,
                    "估计值": float(value),
                    "参数来源": "附件数据约束极大似然估计",
                    "约束": "无符号约束",
                }
            )
    elif model["类型"] in {"浓度混合", "GEE"}:
        result = model["结果"]
        for term, value in result.params.items():
            standard_error = float(result.bse[term]) if term in result.bse.index else np.nan
            p_value = float(result.pvalues[term]) if term in result.pvalues.index else np.nan
            row = {
                    "样本范围": scope_name,
                    "问题": role,
                    "候选模型": model["名称"],
                    "参数": str(term),
                    "估计值": float(value),
                    "标准误": standard_error,
                    "Wald95%下限": float(value) - 1.959963984540054 * standard_error,
                    "Wald95%上限": float(value) + 1.959963984540054 * standard_error,
                    "P值": p_value,
                    "参数来源": "附件数据估计",
                    "约束": "按候选公式",
                }
            if model["类型"] == "GEE":
                row.update(model.get("诊断", {}))
            parameter_rows.append(row)
    return pd.DataFrame(parameter_rows), pd.DataFrame(scale_rows)


def 整簇重采样(events: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    women = events["孕妇代码"].drop_duplicates().to_numpy(dtype=object)
    sampled = rng.choice(women, size=len(women), replace=True)
    parts = []
    for replicate_index, woman in enumerate(sampled):
        part = events.loc[events["孕妇代码"].eq(woman)].copy()
        part["孕妇代码"] = f"{woman}@重采样{replicate_index}"
        part["抽血事件键"] = part["抽血事件键"].astype(str) + f"@重采样{replicate_index}"
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


def 提取折中时点结果(selected: pd.DataFrame, replicate: int, kind: str) -> list[dict[str, Any]]:
    rows = []
    for _, row in selected.iterrows():
        rows.append(
            {
                "不确定性来源": kind,
                "重复序号": replicate,
                "问题": row["问题"],
                "题面BMI组": row["题面BMI组"],
                "组内孕妇数": row["孕妇数"],
                "折中时点_天": row.get("折中时点_天", np.nan),
                "折中时点预计达标比例": row.get("折中时点预计达标比例", np.nan),
                "有效标志": int(pd.notna(row.get("折中时点_天", np.nan))),
            }
        )
    return rows


def 汇总不确定性(detail: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, group in detail.groupby(["不确定性来源", "问题", "题面BMI组"], sort=False):
        valid = group.loc[group["有效标志"].eq(1)].copy()
        rows.append(
            {
                "不确定性来源": keys[0],
                "问题": keys[1],
                "题面BMI组": keys[2],
                "请求重复数": int(group["重复序号"].nunique()),
                "有效重复数": int(len(valid)),
                "有效率": float(len(valid) / len(group)) if len(group) else np.nan,
                "折中时点2.5%分位_天": float(valid["折中时点_天"].quantile(0.025)) if len(valid) else np.nan,
                "折中时点中位数_天": float(valid["折中时点_天"].median()) if len(valid) else np.nan,
                "折中时点97.5%分位_天": float(valid["折中时点_天"].quantile(0.975)) if len(valid) else np.nan,
                "达标比例2.5%分位": float(valid["折中时点预计达标比例"].quantile(0.025)) if len(valid) else np.nan,
                "达标比例中位数": float(valid["折中时点预计达标比例"].median()) if len(valid) else np.nan,
                "达标比例97.5%分位": float(valid["折中时点预计达标比例"].quantile(0.975)) if len(valid) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def 构造组时点网格(
    model: dict[str, Any],
    events: pd.DataFrame,
    scope_name: str,
    role: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    persons = (
        events.sort_values(["孕妇代码", "孕周天数", "抽血次数"])
        .drop_duplicates("孕妇代码")
        .copy()
    )
    persons["题面BMI组"] = 题面BMI组(persons["首次BMI"])
    curve_rows = []
    selected_rows = []
    for group_name in ["[20,28)", "[28,32)", "[32,36)", "[36,40)", "[40,+∞)"]:
        group = persons.loc[persons["题面BMI组"].astype(str).eq(group_name)].copy()
        if len(group) == 0:
            selected_rows.append(
                {
                    "样本范围": scope_name,
                    "问题": role,
                    "题面BMI组": group_name,
                    "孕妇数": 0,
                    "折中时点_天": np.nan,
                    "折中时点_周加天": "",
                    "说明": "该样本范围内无孕妇，不估计",
                }
            )
            continue
        days = np.arange(检测起始天, 检测结束天 + 1, dtype=int)
        planned_parts = []
        for day in days:
            planned_day = group.copy()
            planned_day["孕周天数"] = float(day)
            planned_day["孕周数"] = float(day) / 7.0
            planned_parts.append(planned_day)
        planned = pd.concat(planned_parts, ignore_index=True)
        predicted = 预测概率(model, planned).reshape(len(days), len(group))
        p = np.mean(predicted, axis=1)
        delay_regret = (days - 检测起始天) / (检测结束天 - 检测起始天)
        failure_regret = 1.0 - p
        max_regret = np.maximum(delay_regret, failure_regret)
        selected_index = int(np.flatnonzero(max_regret == np.nanmin(max_regret))[0])

        observed_start_day = int(math.ceil(float(events["孕周天数"].min())))
        support_mask = days >= observed_start_day
        support_delay = np.full_like(p, np.nan, dtype=float)
        support_max_regret = np.full_like(p, np.nan, dtype=float)
        support_delay[support_mask] = (
            (days[support_mask] - observed_start_day) / (检测结束天 - observed_start_day)
        )
        support_max_regret[support_mask] = np.maximum(
            support_delay[support_mask], failure_regret[support_mask]
        )
        support_selected_index = int(
            np.flatnonzero(support_max_regret == np.nanmin(support_max_regret))[0]
        )

        # 被否决旧规则只作敏感性：它把所有组在10周的准确性遗憾都归一为1，
        # 会抹去不同BMI组在窗口起点的绝对未达标概率差异。
        denominator = float(p[-1] - p[0])
        if denominator > np.finfo(float).eps:
            normalized_failure_regret = (p[-1] - p) / denominator
            normalized_max_regret = np.maximum(delay_regret, normalized_failure_regret)
            normalized_selected_index = int(
                np.flatnonzero(normalized_max_regret == np.nanmin(normalized_max_regret))[0]
            )
        else:
            normalized_failure_regret = np.full_like(p, np.nan)
            normalized_max_regret = np.full_like(p, np.nan)
            normalized_selected_index = None
        for index, day in enumerate(days):
            curve_rows.append(
                {
                    "样本范围": scope_name,
                    "问题": role,
                    "题面BMI组": group_name,
                    "孕妇数": int(len(group)),
                    "孕周天数": int(day),
                    "孕周": float(day) / 7.0,
                    "孕周文本": 周天文本(day),
                    "预计达标比例": float(p[index]),
                    "预计尚未达标比例": float(1.0 - p[index]),
                    "等待遗憾": float((day - 检测起始天) / (检测结束天 - 检测起始天)),
                    "绝对未达标遗憾": float(failure_regret[index]),
                    "最大遗憾": float(max_regret[index]) if np.isfinite(max_regret[index]) else np.nan,
                    "折中时点标志": int(selected_index is not None and index == selected_index),
                    "观测支持起点_天": observed_start_day,
                    "观测支持起点等待遗憾_敏感性": (
                        float(support_delay[index]) if np.isfinite(support_delay[index]) else np.nan
                    ),
                    "观测支持起点最大遗憾_敏感性": (
                        float(support_max_regret[index])
                        if np.isfinite(support_max_regret[index])
                        else np.nan
                    ),
                    "观测支持起点折中时点标志_敏感性": int(index == support_selected_index),
                    "旧归一化改善遗憾_敏感性": (
                        float(normalized_failure_regret[index])
                        if np.isfinite(normalized_failure_regret[index])
                        else np.nan
                    ),
                    "旧归一化改善最大遗憾_敏感性": (
                        float(normalized_max_regret[index])
                        if np.isfinite(normalized_max_regret[index])
                        else np.nan
                    ),
                    "旧归一化改善折中时点标志_敏感性": int(
                        normalized_selected_index is not None and index == normalized_selected_index
                    ),
                    "题面风险等级": 风险等级(int(day)),
                    "10周外推标志": int(day < int(round(events["孕周天数"].min()))),
                }
            )
        if selected_index is None:
            selected_rows.append(
                {
                    "样本范围": scope_name,
                    "问题": role,
                    "题面BMI组": group_name,
                    "孕妇数": int(len(group)),
                    "折中时点_天": np.nan,
                    "折中时点_周加天": "",
                    "说明": "10至25周预计达标比例没有可识别增益，不输出虚构时点",
                }
            )
        else:
            day = int(days[selected_index])
            selected_rows.append(
                {
                    "样本范围": scope_name,
                    "问题": role,
                    "题面BMI组": group_name,
                    "孕妇数": int(len(group)),
                    "组内首次BMI最小值": float(group["首次BMI"].min()),
                    "组内首次BMI最大值": float(group["首次BMI"].max()),
                    "折中时点_天": day,
                    "折中时点_周加天": 周天文本(day),
                    "折中时点预计达标比例": float(p[selected_index]),
                    "折中时点预计尚未达标比例": float(1.0 - p[selected_index]),
                    "折中时点最大遗憾": float(max_regret[selected_index]),
                    "题面风险等级": 风险等级(day),
                    "旧归一化改善折中时点_天_敏感性": (
                        int(days[normalized_selected_index])
                        if normalized_selected_index is not None
                        else np.nan
                    ),
                    "旧归一化改善折中时点_周加天_敏感性": (
                        周天文本(int(days[normalized_selected_index]))
                        if normalized_selected_index is not None
                        else ""
                    ),
                    "观测支持起点_天": observed_start_day,
                    "观测支持起点折中时点_天_敏感性": int(days[support_selected_index]),
                    "观测支持起点折中时点_周加天_敏感性": 周天文本(
                        int(days[support_selected_index])
                    ),
                    "说明": "等待窗口占比与绝对未达标概率的无权重最小最大遗憾统计折中；不是临床唯一最佳时点",
                }
            )
    return pd.DataFrame(curve_rows), pd.DataFrame(selected_rows)


def 对比第二三问增益(
    selected_predictions: dict[str, pd.DataFrame],
    selected_names: dict[str, str],
    bootstrap_count: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    averaged: dict[str, pd.DataFrame] = {}
    for role in ("第二问", "第三问"):
        frame = selected_predictions[role]
        candidate = selected_names[role]
        frame = frame.loc[frame["候选模型"].eq(candidate)].copy()
        averaged[role] = (
            frame.groupby(["孕妇代码", "抽血事件键", "实际达标标志"], as_index=False)[
                "预测达标概率"
            ]
            .mean()
            .rename(columns={"预测达标概率": f"{role}预测概率"})
        )
    merged = averaged["第二问"].merge(
        averaged["第三问"],
        on=["孕妇代码", "抽血事件键", "实际达标标志"],
        how="inner",
        validate="one_to_one",
    )
    y = merged["实际达标标志"].to_numpy(dtype=float)
    for role in ("第二问", "第三问"):
        p = np.clip(merged[f"{role}预测概率"].to_numpy(dtype=float), 1e-12, 1.0 - 1e-12)
        merged[f"{role}事件对数损失"] = -(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))
        merged[f"{role}事件Brier分数"] = (y - p) ** 2
    merged["第三问减第二问对数损失"] = merged["第三问事件对数损失"] - merged["第二问事件对数损失"]
    merged["第三问减第二问Brier分数"] = merged["第三问事件Brier分数"] - merged["第二问事件Brier分数"]
    person = merged.groupby("孕妇代码", as_index=False)[
        ["第三问减第二问对数损失", "第三问减第二问Brier分数"]
    ].mean()
    rng = np.random.default_rng(int(预期附件哈希[8:16], 16))
    logloss_boot = []
    brier_boot = []
    values_log = person["第三问减第二问对数损失"].to_numpy(dtype=float)
    values_brier = person["第三问减第二问Brier分数"].to_numpy(dtype=float)
    for _ in range(bootstrap_count):
        index = rng.integers(0, len(person), size=len(person))
        logloss_boot.append(float(np.mean(values_log[index])))
        brier_boot.append(float(np.mean(values_brier[index])))
    summary = pd.DataFrame(
        [
            {
                "比较": "第三问主模型减第二问主模型",
                "孕妇数": int(len(person)),
                "第二问主模型": selected_names["第二问"],
                "第三问主模型": selected_names["第三问"],
                "逐孕妇平均对数损失差": float(np.mean(values_log)),
                "对数损失差整簇自助95%下限": float(np.quantile(logloss_boot, 0.025)),
                "对数损失差整簇自助95%上限": float(np.quantile(logloss_boot, 0.975)),
                "逐孕妇平均Brier差": float(np.mean(values_brier)),
                "Brier差整簇自助95%下限": float(np.quantile(brier_boot, 0.025)),
                "Brier差整簇自助95%上限": float(np.quantile(brier_boot, 0.975)),
                "重采样次数": bootstrap_count,
                "解释": "负值表示第三问多因素模型更好；有限重采样只报告区间",
            }
        ]
    )
    return person, summary


def 运行主路线不确定性(
    events: pd.DataFrame,
    selected_models: dict[str, str],
    measurement_summary: dict[str, float],
    bootstrap_count: int,
    error_count: int,
    输出目录: Path,
) -> None:
    bootstrap_rng = np.random.default_rng(int(预期附件哈希[16:24], 16))
    error_rng = np.random.default_rng(int(预期附件哈希[24:32], 16))
    bootstrap_rows: list[dict[str, Any]] = []
    error_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []

    for replicate in range(1, bootstrap_count + 1):
        sample = 整簇重采样(events, bootstrap_rng)
        for role in ("第二问", "第三问"):
            try:
                model = 拟合概率候选(selected_models[role], sample, role)
                _, selected = 构造组时点网格(model, sample, "序号683前主分析整簇重采样", role)
                bootstrap_rows.extend(提取折中时点结果(selected, replicate, "孕妇整簇重采样"))
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
        standard_error = np.sqrt(sampled_variance / perturbed["记录数"].to_numpy(dtype=float))
        perturbed_y = perturbed["Y浓度"].to_numpy(dtype=float) + error_rng.normal(
            0.0, standard_error, size=len(perturbed)
        )
        perturbed_y = np.clip(perturbed_y, np.finfo(float).eps, 1.0 - np.finfo(float).eps)
        perturbed["Y浓度"] = perturbed_y
        perturbed["Y浓度logit"] = np.log(perturbed_y / (1.0 - perturbed_y))
        perturbed["达标标志"] = (perturbed_y >= 达标阈值).astype(int)
        for role in ("第二问", "第三问"):
            try:
                model = 拟合概率候选(selected_models[role], perturbed, role)
                _, selected = 构造组时点网格(model, perturbed, "序号683前主分析检测误差传播", role)
                error_rows.extend(提取折中时点结果(selected, replicate, "检测误差传播"))
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
    写CSV(bootstrap_detail, 输出目录 / "17_主路线孕妇整簇重采样逐次.csv")
    写CSV(error_detail, 输出目录 / "18_主路线检测误差传播逐次.csv")
    写CSV(failures, 输出目录 / "19_主路线不确定性拟合失败记录.csv")
    combined = pd.concat([bootstrap_detail, error_detail], ignore_index=True)
    写CSV(汇总不确定性(combined), 输出目录 / "20_主路线时点不确定性汇总.csv")

    convergence_tables = []
    for source, detail, requested in (
        ("孕妇整簇重采样", bootstrap_detail, bootstrap_count),
        ("检测误差传播", error_detail, error_count),
    ):
        prefixes = sorted(set(value for value in (100, 200, requested) if value <= requested))
        for prefix in prefixes:
            summary = 汇总不确定性(detail.loc[detail["重复序号"].le(prefix)].copy())
            summary["前缀重复数"] = prefix
            summary["不确定性来源"] = source
            convergence_tables.append(summary)
    写CSV(pd.concat(convergence_tables, ignore_index=True), 输出目录 / "21_主路线不确定性次数收敛.csv")


def 概率模型联合整改(
    events: pd.DataFrame,
    measurement_summary: dict[str, float],
    输出目录: Path,
    bootstrap_count: int,
    error_count: int,
) -> None:
    reference_events = events.loc[events["数据段"].eq("序号683前")].copy()
    all_candidate_summaries = []
    selected_models: dict[str, str] = {}
    selected_predictions: dict[str, pd.DataFrame] = {}
    parameter_tables = []
    scale_tables = []
    paired_candidate_tables = []
    selection_reasons: dict[str, str] = {}
    sparse_diagnostics = []
    coverage_prediction_tables = []
    coverage_failure_tables = []
    coverage_summary_tables = []
    for role in ("第二问", "第三问"):
        predictions, failures = 分组交叉验证(reference_events, role=role, repeat_count=5)
        summary = 汇总交叉验证(predictions, failures, reference_events)
        summary["分析样本"] = "序号683前主分析；683后失败结局仅来自1名孕妇，不能稳定识别时点模型"
        写CSV(predictions, 输出目录 / f"08_{role}候选分组交叉验证逐事件.csv")
        写CSV(failures, 输出目录 / f"09_{role}候选拟合失败记录.csv")

        boundary_rows = []
        full_models: dict[str, dict[str, Any]] = {}
        for candidate in 概率候选:
            try:
                model = 拟合概率候选(candidate, reference_events, role)
                full_models[candidate] = model
                curves, _ = 构造组时点网格(model, reference_events, "序号683前主分析", role)
                min_change = float(
                    curves.groupby("题面BMI组", observed=True)["预计达标比例"]
                    .diff()
                    .dropna()
                    .min()
                )
                boundary_rows.append(
                    {
                        "候选模型": candidate,
                        "全样本拟合成功标志": 1,
                        "10至25周最小相邻日达标比例变化": min_change,
                        "时点曲线不下降标志": int(min_change >= -1e-10),
                        "边界说明": "-1e-10仅为浮点数值容差，不是科学阈值",
                    }
                )
            except Exception as exc:
                boundary_rows.append(
                    {
                        "候选模型": candidate,
                        "全样本拟合成功标志": 0,
                        "10至25周最小相邻日达标比例变化": np.nan,
                        "时点曲线不下降标志": 0,
                        "边界说明": f"{type(exc).__name__}:{exc}",
                    }
                )
        boundary = pd.DataFrame(boundary_rows)
        summary = summary.merge(boundary, on="候选模型", how="left", validate="one_to_one")
        summary["问题"] = role
        eligible = summary.loc[
            summary["失败折数"].fillna(1).eq(0)
            & summary["全样本拟合成功标志"].eq(1)
            & summary["时点曲线不下降标志"].eq(1)
        ].copy()
        if len(eligible) == 0:
            raise RuntimeError(f"{role}没有通过共同闸门的候选")
        paired_comparison, selected_name, selection_reason = 候选逐孕妇损失比较(
            predictions,
            eligible["候选模型"].astype(str).tolist(),
            bootstrap_count,
            role,
        )
        paired_candidate_tables.append(paired_comparison)
        selection_reasons[role] = selection_reason
        selected_models[role] = selected_name
        selected_predictions[role] = predictions
        summary["主模型标志"] = summary["候选模型"].eq(selected_name).astype(int)
        all_candidate_summaries.append(summary)

        main_model = 拟合概率候选(selected_name, reference_events, role)
        main_parameters, main_scales = 概率模型参数表(
            main_model, "序号683前主分析", role
        )
        parameter_tables.append(main_parameters)
        scale_tables.append(main_scales)
        sparse_diagnostics.append(
            {
                "样本范围": "序号683前主分析",
                "问题": role,
                "候选模型": selected_name,
                **main_model.get("诊断", {}),
            }
        )
        main_curves, main_selected = 构造组时点网格(
            main_model, reference_events, "序号683前主分析", role
        )
        写CSV(main_curves, 输出目录 / f"11_{role}主分析题面分组逐日概率与遗憾.csv")
        写CSV(main_selected, 输出目录 / f"12_{role}主分析题面分组折中时点.csv")

        # 后段单独模型只用于暴露稀疏/分离，不生成精确时点结论。
        post_events = events.loc[events["数据段"].eq("序号683后")].copy()
        post_model = 拟合概率候选(selected_name, post_events, role)
        post_parameters, post_scales = 概率模型参数表(
            post_model, "序号683后不稳定敏感性", role
        )
        parameter_tables.append(post_parameters)
        scale_tables.append(post_scales)
        sparse_diagnostics.append(
            {
                "样本范围": "序号683后不稳定敏感性",
                "问题": role,
                "候选模型": selected_name,
                **post_model.get("诊断", {}),
            }
        )

        # 全样本只做“段基线调整”的覆盖性分析；段内失败信息不足，不把它包装为通用政策。
        adjusted_model = 拟合段调整GEE(events, role)
        adjusted_parameters, adjusted_scales = 概率模型参数表(
            adjusted_model, "全样本段基线调整覆盖性敏感性", role
        )
        parameter_tables.append(adjusted_parameters)
        scale_tables.append(adjusted_scales)
        sparse_diagnostics.append(
            {
                "样本范围": "全样本段基线调整覆盖性敏感性",
                "问题": role,
                "候选模型": adjusted_model["名称"],
                **adjusted_model.get("诊断", {}),
            }
        )
        adjusted_curve_tables = []
        adjusted_selected_tables = []
        for scope_name, scope in (
            ("段调整模型_条件于序号683前", reference_events),
            ("段调整模型_条件于序号683后", post_events),
        ):
            curves, selected = 构造组时点网格(adjusted_model, scope, scope_name, role)
            selected["推断地位"] = (
                "覆盖性敏感性"
                if scope_name.endswith("683前")
                else "失败结局仅来自1名孕妇，不输出为稳定推荐"
            )
            adjusted_curve_tables.append(curves)
            adjusted_selected_tables.append(selected)
        写CSV(
            pd.concat(adjusted_curve_tables, ignore_index=True),
            输出目录 / f"11_{role}全样本段调整覆盖性逐日概率与遗憾.csv",
        )
        写CSV(
            pd.concat(adjusted_selected_tables, ignore_index=True),
            输出目录 / f"12_{role}全样本段调整覆盖性折中时点.csv",
        )

        coverage_predictions, coverage_failures, coverage_summary = 全样本段调整交叉验证(
            events, role=role, repeat_count=5
        )
        coverage_prediction_tables.append(coverage_predictions)
        coverage_failure_tables.append(coverage_failures)
        coverage_summary_tables.append(coverage_summary)

    candidate_table = pd.concat(all_candidate_summaries, ignore_index=True)
    写CSV(candidate_table, 输出目录 / "10_第二三问候选统一比较.csv")
    写CSV(pd.concat(paired_candidate_tables, ignore_index=True), 输出目录 / "10_第二三问候选逐孕妇损失差复核.csv")
    写CSV(pd.concat(parameter_tables, ignore_index=True), 输出目录 / "14_第二三问主路线三套样本参数表.csv")
    写CSV(pd.concat(scale_tables, ignore_index=True), 输出目录 / "15_第二三问主路线标准化参数表.csv")
    写CSV(pd.DataFrame(sparse_diagnostics), 输出目录 / "15_第二三问数据段稀疏与分离诊断.csv")
    写CSV(
        pd.concat(coverage_prediction_tables, ignore_index=True),
        输出目录 / "28_第二三问全样本段调整覆盖性验证逐事件.csv",
    )
    写CSV(
        pd.concat(coverage_failure_tables, ignore_index=True),
        输出目录 / "28_第二三问全样本段调整覆盖性失败记录.csv",
    )
    写CSV(
        pd.concat(coverage_summary_tables, ignore_index=True),
        输出目录 / "28_第二三问全样本段调整覆盖性验证汇总.csv",
    )
    person_diff, comparison = 对比第二三问增益(
        selected_predictions,
        selected_models,
        bootstrap_count=bootstrap_count,
    )
    写CSV(person_diff, 输出目录 / "16_第三问相对第二问逐孕妇预测损失差.csv")
    写CSV(comparison, 输出目录 / "16_第三问相对第二问预测增益汇总.csv")
    运行主路线不确定性(
        reference_events,
        selected_models,
        measurement_summary,
        bootstrap_count,
        error_count,
        输出目录,
    )
    写JSON(
        {
            "状态": "DRAFT_COMPUTED_NOT_REVIEWED",
            "主分析样本": "序号683前；不是判定683后无效，而是683后399事件仅3次未达标且只来自1名孕妇，不能稳定识别失败概率及时点",
            "覆盖性分析": "全样本加入后段标志基线调整；按前后段分别报告，不把观测段混合概率解释为通用概率",
            "第二问主路线": selected_models["第二问"],
            "第三问主路线": selected_models["第三问"],
            "选择规则": "在序号683前可识别参考样本中，先通过孕妇分组交叉验证完整性、拟合和10至25周不下降边界；再按预声明简约裁决选择；区间包含0只表示当前样本未分辨稳定差异，不称统计等效",
            "第二问裁决理由": selection_reasons["第二问"],
            "第三问裁决理由": selection_reasons["第三问"],
            "时点规则": "等待窗口占比与绝对未达标概率的无权重最小最大遗憾；并列取最早日",
            "被否决时点规则": "按组内10至25周概率改善幅度归一化的遗憾规则；因抹去组间绝对未达标差异，仅保留敏感性",
        },
        输出目录 / "13_第二三问暂定主路线.json",
    )


def 主程序(
    附件: Path,
    输出目录: Path,
    bootstrap_count: int,
    error_count: int,
) -> None:
    输出目录.mkdir(parents=True, exist_ok=True)
    records = 读取男胎记录(附件)
    events, excluded = 构造事件层(records)
    写CSV(events, 输出目录 / "01_全样本抽血事件表.csv")
    写CSV(excluded, 输出目录 / "02_孕周歧义排除事件.csv")
    写CSV(样本摘要(records, events, excluded), 输出目录 / "03_三套样本结构摘要.csv")
    写CSV(BMI组人数表(events), 输出目录 / "04_题面BMI分组人数.csv")
    写CSV(非吸收态审计(events), 输出目录 / "04_首次达标非吸收态审计.csv")

    strict_detail, measurement_summary = 严格技术复测误差(records)
    写CSV(strict_detail, 输出目录 / "04_严格技术复测明细.csv")
    写JSON(measurement_summary, 输出目录 / "04_严格技术复测误差摘要.json")

    parameters, predictions, interaction = 第一问683敏感性(events)
    写CSV(parameters, 输出目录 / "05_第一问683敏感性参数比较.csv")
    写CSV(predictions, 输出目录 / "06_第一问683敏感性参考预测.csv")
    写CSV(interaction, 输出目录 / "07_第一问前后段交互检验.csv")

    概率模型联合整改(
        events,
        measurement_summary,
        输出目录,
        bootstrap_count,
        error_count,
    )

    core_output_names = [
        "01_全样本抽血事件表.csv",
        "02_孕周歧义排除事件.csv",
        "03_三套样本结构摘要.csv",
        "04_题面BMI分组人数.csv",
        "04_首次达标非吸收态审计.csv",
        "04_严格技术复测明细.csv",
        "04_严格技术复测误差摘要.json",
        "05_第一问683敏感性参数比较.csv",
        "06_第一问683敏感性参考预测.csv",
        "07_第一问前后段交互检验.csv",
        "08_第二问候选分组交叉验证逐事件.csv",
        "08_第三问候选分组交叉验证逐事件.csv",
        "09_第二问候选拟合失败记录.csv",
        "09_第三问候选拟合失败记录.csv",
        "10_第二三问候选统一比较.csv",
        "10_第二三问候选逐孕妇损失差复核.csv",
        "11_第二问主分析题面分组逐日概率与遗憾.csv",
        "11_第三问主分析题面分组逐日概率与遗憾.csv",
        "11_第二问全样本段调整覆盖性逐日概率与遗憾.csv",
        "11_第三问全样本段调整覆盖性逐日概率与遗憾.csv",
        "12_第二问主分析题面分组折中时点.csv",
        "12_第三问主分析题面分组折中时点.csv",
        "12_第二问全样本段调整覆盖性折中时点.csv",
        "12_第三问全样本段调整覆盖性折中时点.csv",
        "13_第二三问暂定主路线.json",
        "14_第二三问主路线三套样本参数表.csv",
        "15_第二三问主路线标准化参数表.csv",
        "15_第二三问数据段稀疏与分离诊断.csv",
        "16_第三问相对第二问逐孕妇预测损失差.csv",
        "16_第三问相对第二问预测增益汇总.csv",
        "17_主路线孕妇整簇重采样逐次.csv",
        "18_主路线检测误差传播逐次.csv",
        "19_主路线不确定性拟合失败记录.csv",
        "20_主路线时点不确定性汇总.csv",
        "21_主路线不确定性次数收敛.csv",
        "28_第二三问全样本段调整覆盖性验证逐事件.csv",
        "28_第二三问全样本段调整覆盖性失败记录.csv",
        "28_第二三问全样本段调整覆盖性验证汇总.csv",
    ]
    missing_outputs = [name for name in core_output_names if not (输出目录 / name).is_file()]
    if missing_outputs:
        raise RuntimeError(f"核心输出缺失：{missing_outputs}")
    manifest = {
        "状态": "DRAFT_COMPUTED_NOT_REVIEWED",
        "原始附件": str(附件),
        "原始附件SHA256": 文件哈希(附件),
        "数据窗口": "孕周[10,26)",
        "事件定义": "孕妇代码+抽血次数；技术复测事件内均值",
        "孕妇整簇重采样次数": bootstrap_count,
        "检测误差传播次数": error_count,
        "交叉验证": "按孕妇分组，5折×5次重复",
        "随机种子": {
            "交叉验证基础种子": int(预期附件哈希[:8], 16),
            "第二问候选损失差种子": int(预期附件哈希[32:40], 16),
            "第三问候选损失差种子": int(预期附件哈希[32:40], 16) + 1,
            "第二三问增益种子": int(预期附件哈希[8:16], 16),
            "孕妇整簇重采样种子": int(预期附件哈希[16:24], 16),
            "检测误差传播种子": int(预期附件哈希[24:32], 16),
        },
        "统计报告约定": "95% Wald或整簇重采样区间；不是检测可靠性阈值",
        "数值容差": "概率裁剪1e-12仅防止log(0)；单调检查-1e-10仅为浮点容差",
        "运行命令": (
            f'python "{Path(__file__).resolve()}" --附件 "{附件}" --输出目录 "{输出目录}" '
            f'--孕妇整簇重采样次数 {bootstrap_count} --检测误差传播次数 {error_count}'
        ),
        "运行环境": {
            "Python": sys.version,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "statsmodels": statsmodels.__version__,
            "scikit-learn": sklearn.__version__,
        },
        "生成脚本": str(Path(__file__).resolve()),
        "生成脚本SHA256": 文件哈希(Path(__file__).resolve()),
        "核心输出SHA256": {
            name: 文件哈希(输出目录 / name) for name in core_output_names
        },
        "核心输出文件": core_output_names,
        "目录中其他文件不属于本清单": sorted(
            path.name for path in 输出目录.glob("*")
            if path.name not in set(core_output_names) | {"00_运行清单.json"}
        ),
    }
    写JSON(manifest, 输出目录 / "00_运行清单.json")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="第一至第三问优先问题整改分析；不生成图片、不修改正式归档。")
    parser.add_argument("--附件", type=Path, default=默认附件)
    parser.add_argument("--输出目录", type=Path, default=默认输出目录)
    parser.add_argument("--孕妇整簇重采样次数", type=int, default=400)
    parser.add_argument("--检测误差传播次数", type=int, default=400)
    args = parser.parse_args()
    主程序(
        args.附件.resolve(),
        args.输出目录.resolve(),
        args.孕妇整簇重采样次数,
        args.检测误差传播次数,
    )
