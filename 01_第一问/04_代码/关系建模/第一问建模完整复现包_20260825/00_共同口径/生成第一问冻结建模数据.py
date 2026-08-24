from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


预期原始工作簿哈希 = "14827156218bd4f7e4f16db4aa6d9f757c6648379e038ae6c6b58383648614af"

脚本目录 = Path(__file__).resolve().parent


def 定位工作区(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / "00_题目与原始资料/02_原始数据/附件.xlsx").is_file():
            return candidate
    raise FileNotFoundError("无法从脚本位置向上找到含原始附件的C题论文工作区")


工作区目录 = 定位工作区(脚本目录)
默认源快照 = 脚本目录 / "源数据快照" / "01_男胎检测数据.json"
默认原始工作簿 = 工作区目录 / "00_题目与原始资料/02_原始数据/附件.xlsx"
默认输出目录 = 脚本目录 / "冻结数据"


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


def 文件哈希(path: Path) -> str:
    摘要 = hashlib.sha256()
    with path.open("rb") as 文件:
        for 数据块 in iter(lambda: 文件.read(1024 * 1024), b""):
            摘要.update(数据块)
    return 摘要.hexdigest()


def 解析孕周天数(value: Any) -> float:
    文本 = str(value).strip().lower()
    匹配 = re.fullmatch(r"(\d+)w(?:\+(\d+))?", 文本)
    if not 匹配:
        return np.nan
    return float(int(匹配.group(1)) * 7 + int(匹配.group(2) or 0))


def 解析日期(value: Any) -> pd.Timestamp | pd.NaT:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return pd.NaT
    if isinstance(value, (int, float, np.integer, np.floating)):
        数字 = int(value)
        if 19_000_000 <= 数字 <= 21_001_231:
            return pd.to_datetime(str(数字), format="%Y%m%d", errors="coerce")
        if 1 <= 数字 <= 100_000:
            return pd.Timestamp(datetime(1899, 12, 30) + timedelta(days=数字))
    文本 = str(value).strip()
    if re.fullmatch(r"\d{8}", 文本):
        return pd.to_datetime(文本, format="%Y%m%d", errors="coerce")
    return pd.to_datetime(文本, errors="coerce")


def 日期文本(value: Any) -> str:
    日期 = 解析日期(value)
    return "" if pd.isna(日期) else 日期.strftime("%Y-%m-%d")


def 唯一文本(series: pd.Series) -> str:
    values = []
    for value in series:
        if pd.isna(value):
            continue
        text = str(value)
        if text not in values:
            values.append(text)
    return ";".join(values)


def 唯一整数文本(series: pd.Series) -> str:
    return ";".join(str(int(value)) for value in sorted(series.dropna().unique()))


def 算术均值(series: pd.Series) -> float:
    return float(pd.to_numeric(series, errors="coerce").mean())


def 主程序(源快照: Path, 原始工作簿: Path, 输出目录: Path) -> None:
    输出目录.mkdir(parents=True, exist_ok=True)

    if not 源快照.exists():
        raise FileNotFoundError(f"找不到男胎数据只读快照：{源快照}")
    if not 原始工作簿.exists():
        raise FileNotFoundError(f"找不到原始附件：{原始工作簿}")

    工作簿哈希 = 文件哈希(原始工作簿)
    if 工作簿哈希.lower() != 预期原始工作簿哈希:
        raise RuntimeError(
            "原始附件SHA256与审计冻结值不一致，停止生成。"
            f"\n期望：{预期原始工作簿哈希}\n实际：{工作簿哈希}"
        )

    快照对象 = json.loads(源快照.read_text(encoding="utf-8"))
    values = 快照对象["values"]
    if len(values) != 1083 or len(values[0]) != 31:
        raise RuntimeError(f"源快照尺寸异常：{len(values)}×{len(values[0])}")

    全部记录 = pd.DataFrame(values[1:], columns=内部列名)
    全部记录["孕周天数"] = 全部记录["孕周原始值"].map(解析孕周天数)
    if 全部记录["孕周天数"].isna().any():
        错误序号 = 全部记录.loc[全部记录["孕周天数"].isna(), "序号"].tolist()
        raise RuntimeError(f"存在无法解析的孕周，序号：{错误序号}")
    全部记录["孕周数"] = 全部记录["孕周天数"] / 7.0
    全部记录["检测日期规范值"] = 全部记录["检测日期原始值"].map(日期文本)
    全部记录["末次月经规范值"] = 全部记录["末次月经原始值"].map(日期文本)
    全部记录["检测日期解析值"] = 全部记录["检测日期原始值"].map(解析日期)
    全部记录["末次月经解析值"] = 全部记录["末次月经原始值"].map(解析日期)
    全部记录["日期推算孕周天数"] = (
        全部记录["检测日期解析值"] - 全部记录["末次月经解析值"]
    ).dt.days
    全部记录["日期孕周偏差天数"] = 全部记录["日期推算孕周天数"] - 全部记录["孕周天数"]
    全部记录["日期孕周偏差超14天标志"] = (
        全部记录["日期孕周偏差天数"].abs() > 14
    ).astype(int)
    全部记录["GC范围异常标志"] = (
        (全部记录["GC含量"] < 0.4) | (全部记录["GC含量"] > 0.6)
    ).astype(int)
    全部记录["唯一读段大于原始读段标志"] = (
        全部记录["唯一比对读段数"] > 全部记录["原始读段数"]
    ).astype(int)
    全部记录["抽血事件键"] = (
        全部记录["孕妇代码"].astype(str) + "#" + 全部记录["抽血次数"].astype(int).astype(str)
    )
    全部记录["检测会话键"] = 全部记录["抽血事件键"] + "#" + 全部记录["检测日期规范值"]

    记录层 = 全部记录.loc[
        (全部记录["序号"] < 683)
        & (全部记录["孕周天数"] >= 70)
        & (全部记录["孕周天数"] < 182)
    ].copy()
    记录层["纳入截至25周0天敏感性标志"] = (记录层["孕周天数"] <= 175).astype(int)
    记录层 = 记录层.sort_values("序号").reset_index(drop=True)

    不输出辅助列 = ["检测日期解析值", "末次月经解析值"]
    记录层输出 = 记录层.drop(columns=不输出辅助列).rename(
        columns={"BMI": "孕妇体质指数_BMI"}
    )
    记录层输出.to_csv(
        输出目录 / "第一问记录层冻结样本.csv", index=False, encoding="utf-8-sig"
    )

    事件行 = []
    for (孕妇代码, 抽血次数), 组 in 记录层.groupby(["孕妇代码", "抽血次数"], sort=True):
        孕周唯一值 = sorted(pd.to_numeric(组["孕周天数"], errors="coerce").dropna().unique())
        孕周歧义 = int(len(孕周唯一值) != 1)
        孕周天数 = np.nan if 孕周歧义 else float(孕周唯一值[0])
        y = pd.to_numeric(组["Y染色体浓度"], errors="coerce")
        y_mean = float(y.mean())
        if not (0 < y_mean < 1):
            raise RuntimeError(f"抽血事件{孕妇代码}#{抽血次数}的Y浓度均值不在(0,1)")
        受孕方式 = str(组["受孕方式"].iloc[0])
        事件行.append(
            {
                "孕妇代码": 孕妇代码,
                "抽血次数": int(抽血次数),
                "抽血事件键": f"{孕妇代码}#{int(抽血次数)}",
                "序号集合": 唯一整数文本(组["序号"]),
                "记录数": int(len(组)),
                "检测会话数": int(组["检测会话键"].nunique()),
                "检测日期集合": 唯一文本(组["检测日期规范值"]),
                "孕周原始值集合": 唯一文本(组["孕周原始值"]),
                "孕周唯一值数": int(len(孕周唯一值)),
                "孕周歧义标志": 孕周歧义,
                "孕周天数": 孕周天数,
                "孕周数": 孕周天数 / 7.0 if not pd.isna(孕周天数) else np.nan,
                "年龄": 算术均值(组["年龄"]),
                "身高": 算术均值(组["身高"]),
                "体重": 算术均值(组["体重"]),
                "孕妇体质指数_BMI": 算术均值(组["BMI"]),
                "BMI唯一值数": int(组["BMI"].nunique(dropna=True)),
                "受孕方式": 受孕方式,
                "辅助生殖标志": int(受孕方式 != "自然受孕"),
                "怀孕次数": 算术均值(组["怀孕次数"]),
                "生产次数": 算术均值(组["生产次数"]),
                "Y染色体浓度均值": y_mean,
                "Y染色体浓度中位数": float(y.median()),
                "Y染色体浓度标准差": float(y.std(ddof=1)) if len(y) > 1 else 0.0,
                "Y染色体浓度最小值": float(y.min()),
                "Y染色体浓度最大值": float(y.max()),
                "Y浓度对数几率": float(math.log(y_mean / (1.0 - y_mean))),
                "GC含量均值": 算术均值(组["GC含量"]),
                "原始读段数均值": 算术均值(组["原始读段数"]),
                "唯一比对读段数均值": 算术均值(组["唯一比对读段数"]),
                "比对比例均值": 算术均值(组["比对比例"]),
                "重复读段比例均值": 算术均值(组["重复读段比例"]),
                "过滤读段比例均值": 算术均值(组["过滤读段比例"]),
                "X染色体浓度均值": 算术均值(组["X染色体浓度"]),
                "任一记录GC范围异常标志": int(组["GC范围异常标志"].max()),
                "任一记录日期孕周偏差超14天标志": int(组["日期孕周偏差超14天标志"].max()),
                "任一记录唯一读段大于原始读段标志": int(组["唯一读段大于原始读段标志"].max()),
                "纳入主模型标志": int(not 孕周歧义),
                "纳入截至25周0天敏感性标志": int((not 孕周歧义) and 孕周天数 <= 175),
            }
        )

    事件层 = pd.DataFrame(事件行).sort_values(["孕妇代码", "抽血次数"]).reset_index(drop=True)
    事件层["孕妇抽血事件数"] = 事件层.groupby("孕妇代码")["抽血事件键"].transform("size")
    事件层["孕妇主模型事件数"] = 事件层.groupby("孕妇代码")["纳入主模型标志"].transform("sum")
    事件层["孕妇平均BMI"] = 事件层.groupby("孕妇代码")["孕妇体质指数_BMI"].transform("mean")
    事件层["BMI个体内偏差"] = 事件层["孕妇体质指数_BMI"] - 事件层["孕妇平均BMI"]
    主事件临时 = 事件层.loc[事件层["纳入主模型标志"] == 1]
    孕妇平均孕周 = 主事件临时.groupby("孕妇代码")["孕周数"].mean()
    事件层["孕妇平均孕周"] = 事件层["孕妇代码"].map(孕妇平均孕周)
    事件层["孕周个体内偏差"] = 事件层["孕周数"] - 事件层["孕妇平均孕周"]

    事件层.to_csv(
        输出目录 / "第一问抽血事件层冻结样本.csv", index=False, encoding="utf-8-sig"
    )
    主模型样本 = 事件层.loc[事件层["纳入主模型标志"] == 1].copy()
    主模型样本.to_csv(
        输出目录 / "第一问主模型冻结样本.csv", index=False, encoding="utf-8-sig"
    )
    排除事件 = 事件层.loc[事件层["纳入主模型标志"] == 0].copy()
    排除事件.to_csv(
        输出目录 / "第一问主模型排除事件.csv", index=False, encoding="utf-8-sig"
    )

    多记录组 = 事件层.loc[事件层["记录数"] > 1]
    核对项 = [
        ("683前且孕周在[10,26)的记录数", 674, len(记录层)),
        ("记录层孕妇数", 167, 记录层["孕妇代码"].nunique()),
        ("B+I抽血事件数", 614, len(事件层)),
        ("B+I多记录事件组数", 39, len(多记录组)),
        ("B+I多记录事件涉及记录数", 99, int(多记录组["记录数"].sum())),
        ("孕周歧义事件数", 1, int(事件层["孕周歧义标志"].sum())),
        ("主模型抽血事件数", 613, len(主模型样本)),
        ("主模型孕妇数", 167, 主模型样本["孕妇代码"].nunique()),
        ("截至25周0天敏感性记录数", 670, int(记录层["纳入截至25周0天敏感性标志"].sum())),
        (
            "截至25周0天B+I事件数（含孕周歧义事件）",
            611,
            int(
                事件层.loc[
                    (事件层["孕周唯一值数"] == 1) & (事件层["孕周天数"] <= 175)
                ].shape[0]
                + 事件层["孕周歧义标志"].sum()
            ),
        ),
        (
            "截至25周0天可拟合事件数",
            610,
            int(事件层["纳入截至25周0天敏感性标志"].sum()),
        ),
        ("A055第3次抽血歧义事件数", 1, int(((事件层["孕妇代码"] == "A055") & (事件层["抽血次数"] == 3) & (事件层["孕周歧义标志"] == 1)).sum())),
    ]
    核对表 = pd.DataFrame(核对项, columns=["核对项目", "期望值", "实际值"])
    核对表["结论"] = np.where(核对表["期望值"] == 核对表["实际值"], "通过", "失败")
    核对表.to_csv(输出目录 / "冻结样本核对摘要.csv", index=False, encoding="utf-8-sig")
    if (核对表["结论"] != "通过").any():
        raise RuntimeError("冻结样本核对失败：\n" + 核对表.to_string(index=False))

    变量角色 = pd.DataFrame(
        [
            ("Y染色体浓度均值", "因变量", "抽血事件内算术平均；主拟合可比较原尺度与logit尺度", "浓度为(0,1)比例；先消除同一抽血事件内重复记录的伪独立"),
            ("孕周数", "核心解释变量", "连续变量；比较线性与低自由度样条", "回答孕周相关特性，禁止任意分箱替代主模型"),
            ("孕妇平均BMI", "核心妇间解释变量", "连续变量", "表示不同孕妇长期BMI水平差异"),
            ("BMI个体内偏差", "核心个体内解释变量", "连续变量", "避免把同一孕妇体重变化与孕妇之间差异混为一个效应"),
            ("年龄", "临床辅助变量", "预先指定调整项", "题目所称其他指标之一，且与BMI不构成定义性重复"),
            ("辅助生殖标志", "稀疏敏感性变量", "不进入主调整块；只作稳健性调整", "主模型样本仅11个事件、3名孕妇属于辅助生殖，无法稳定估计独立系数"),
            ("生产次数", "临床辅助变量", "进入辅助调整块并检验共线性", "无缺失，可作为既往生育史的简洁调整项"),
            ("怀孕次数", "缺失敏感性变量", "不进入主调整块；仅作缺失模式或完整病例敏感性分析", "主模型事件中缺失167/613，直接完整病例分析会整组损失部分孕妇"),
            ("GC与读段质量变量", "质量敏感性变量", "只在扩展模型成块调整", "683断点审计显示测序关系可能受数据机制影响，不作生物学解释"),
            ("身高、体重", "排除的替代表达", "不与BMI同时进入主模型", "BMI由身高和体重定义，同时解释会产生共线性和含义重叠"),
            ("Y染色体Z值", "禁止预测变量", "不进入关系模型", "与Y浓度来自同一检测信号，存在信息泄漏和循环解释"),
            ("序号", "禁止预测变量", "仅用于溯源", "序号不是生物学变量，且会代理数据生成机制断点"),
            ("683后记录", "机制核验样本", "不进入主关系模型", "审计发现个体内BMI-Y关系近乎确定性，不能作生物学解释"),
        ],
        columns=["变量或变量组", "建模角色", "主模型处理", "理由"],
    )
    变量角色.to_csv(输出目录 / "第一问建模变量角色表.csv", index=False, encoding="utf-8-sig")

    清单路径 = 输出目录 / "冻结数据清单.json"
    输出文件 = sorted(
        path for path in 输出目录.iterdir() if path.is_file() and path.resolve() != 清单路径.resolve()
    )
    清单 = {
        "生成时间": datetime.now().astimezone().isoformat(timespec="seconds"),
        "源男胎快照": str(源快照),
        "源男胎快照SHA256": 文件哈希(源快照),
        "原始工作簿": str(原始工作簿),
        "原始工作簿SHA256": 工作簿哈希,
        "主参考记录数": int(len(记录层)),
        "抽血事件数": int(len(事件层)),
        "主模型事件数": int(len(主模型样本)),
        "主模型孕妇数": int(主模型样本["孕妇代码"].nunique()),
        "聚合口径": "孕妇代码B+抽血次数I；同一事件的Y浓度与质量变量取算术均值；孕周不一致时不平均而排除主拟合",
        "输出文件SHA256": {path.name: 文件哈希(path) for path in 输出文件},
    }
    清单路径.write_text(
        json.dumps(清单, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(清单, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="生成第一问统一冻结建模数据，不绘图。")
    parser.add_argument("--源快照", type=Path, default=默认源快照)
    parser.add_argument("--原始工作簿", type=Path, default=默认原始工作簿)
    parser.add_argument("--输出目录", type=Path, default=默认输出目录)
    args = parser.parse_args()
    主程序(args.源快照, args.原始工作簿, args.输出目录)
