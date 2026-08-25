from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


def 定位工作区(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / "00_题目与原始资料/02_原始数据/附件.xlsx").is_file():
            return candidate
    raise FileNotFoundError("无法定位C题论文工作区")


def 文件哈希(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def 写CSV(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")


脚本路径 = Path(__file__).resolve()
工作区 = 定位工作区(脚本路径.parent)


def 生成第一问中心化常数() -> tuple[list[Path], list[Path]]:
    source = (
        工作区
        / "01_第一问/03_图表与提示词/01_已有数据审计图/02_制图数据/第一问_全样本抽血事件表.csv"
    )
    events = pd.read_csv(source, encoding="utf-8-sig")
    persons = (
        events.sort_values(["孕妇代码", "孕周天数", "抽血次数"])
        .drop_duplicates("孕妇代码")
    )
    rows = [
        {
            "变量": "孕周中心",
            "中心化常数": float(events["孕周数"].mean()),
            "单位": "周",
            "计算口径": "1012个有效抽血事件的算术平均",
        },
        {
            "变量": "妇间BMI中心",
            "中心化常数": float(persons["孕妇平均BMI"].mean()),
            "单位": "kg/m^2",
            "计算口径": "每名孕妇先求平均BMI，再对267名孕妇等权平均",
        },
        {
            "变量": "年龄中心",
            "中心化常数": float(persons["年龄"].mean()),
            "单位": "岁",
            "计算口径": "267名孕妇等权算术平均",
        },
        {
            "变量": "生产次数中心",
            "中心化常数": float(persons["生产次数"].mean()),
            "单位": "次",
            "计算口径": "267名孕妇等权算术平均",
        },
    ]
    frame = pd.DataFrame(rows)
    frame["孕妇数"] = int(persons["孕妇代码"].nunique())
    frame["抽血事件数"] = int(len(events))
    frame["来源文件"] = str(source.relative_to(工作区))
    output = 工作区 / "01_第一问/02_核心结果/第一问_中心化常数.csv"
    写CSV(frame, output)
    return [source], [output]


def 生成第二问分组稳定性证据() -> tuple[list[Path], list[Path]]:
    fold_path = 工作区 / "02_第二问/02_核心结果/第二问_单切点折间稳定性.csv"
    bootstrap_path = 工作区 / "02_第二问/02_核心结果/第二问_BMI切点整簇重采样次数收敛.csv"
    folds = pd.read_csv(fold_path, encoding="utf-8-sig")
    convergence = pd.read_csv(bootstrap_path, encoding="utf-8-sig")
    final = convergence.sort_values("前缀重复数").iloc[-1]
    models = ["不使用BMI的一组模型", "连续BMI模型", "单切点两组模型"]
    fold_counts = folds["折内BIC最优模型"].value_counts().to_dict()
    bootstrap_columns = {
        "不使用BMI的一组模型": "一组模型入选比例",
        "连续BMI模型": "连续BMI模型入选比例",
        "单切点两组模型": "单切点模型入选比例",
    }
    cut = folds["单切点最优BMI切点"].dropna()
    rows: list[dict[str, object]] = []
    for model in models:
        rows.append(
            {
                "证据层级": "25个按孕妇分组训练折",
                "候选模型": model,
                "总次数": int(len(folds)),
                "入选次数": int(fold_counts.get(model, 0)),
                "入选比例": float(fold_counts.get(model, 0) / len(folds)),
                "单切点2.5%分位": float(cut.quantile(0.025)) if model == "单切点两组模型" else None,
                "单切点中位数": float(cut.quantile(0.5)) if model == "单切点两组模型" else None,
                "单切点97.5%分位": float(cut.quantile(0.975)) if model == "单切点两组模型" else None,
                "分析样本": "序号683前主分析",
                "解释": "训练折仅用于稳定性审计，不把漂移切点作为最终数据规律",
            }
        )
    for model in models:
        ratio = float(final[bootstrap_columns[model]])
        total = int(final["有效重复数"])
        rows.append(
            {
                "证据层级": "孕妇整簇重采样",
                "候选模型": model,
                "总次数": total,
                "入选次数": int(round(total * ratio)),
                "入选比例": ratio,
                "单切点2.5%分位": float(final["单切点位置2.5%分位"]) if model == "单切点两组模型" else None,
                "单切点中位数": float(final["单切点位置中位数"]) if model == "单切点两组模型" else None,
                "单切点97.5%分位": float(final["单切点位置97.5%分位"]) if model == "单切点两组模型" else None,
                "分析样本": str(final["分析样本"]),
                "解释": "按孕妇整簇抽样；题面五组承担实施输出，不声称由此自动识别",
            }
        )
    output = 工作区 / "02_第二问/02_核心结果/第二问_BMI分组稳定性完整证据.csv"
    写CSV(pd.DataFrame(rows), output)
    return [fold_path, bootstrap_path], [output]


def 生成第四问补充证据() -> tuple[list[Path], list[Path]]:
    support = 工作区 / "04_第四问/02_核心结果/复核支撑数据"
    difference_path = support / "第四问_候选路线差异整簇自助逐次.csv"
    threshold_path = support / "第四问_候选路线训练内阈值逐孕妇.csv"
    differences = pd.read_csv(difference_path, encoding="utf-8-sig")
    difference_rows = []
    for column in differences.columns:
        if column == "自助序号":
            continue
        values = pd.to_numeric(differences[column], errors="coerce").dropna()
        difference_rows.append(
            {
                "比较对象": "贝叶斯信息准则剪枝树减L1正则多因素逻辑回归",
                "统计量": column.replace("剪枝树减逻辑回归_", ""),
                "有效整簇重采样次数": int(len(values)),
                "中位数": float(values.quantile(0.5)),
                "2.5%分位": float(values.quantile(0.025)),
                "97.5%分位": float(values.quantile(0.975)),
                "区间解释": "孕妇整簇重采样95%区间；差值为剪枝树减逻辑回归",
            }
        )
    difference_output = 工作区 / "04_第四问/02_核心结果/第四问_候选路线差异95%区间.csv"
    写CSV(pd.DataFrame(difference_rows), difference_output)

    thresholds = pd.read_csv(threshold_path, encoding="utf-8-sig")
    thresholds = thresholds.loc[thresholds["目标"].eq("任一T13_T18_T21异常")].copy()
    threshold_rows = []
    for route, group in thresholds.groupby("路线", sort=False):
        values = pd.to_numeric(group["训练内阈值"], errors="coerce").dropna()
        threshold_rows.append(
            {
                "目标": "任一T13_T18_T21异常",
                "路线": route,
                "阈值记录数": int(len(values)),
                "训练内阈值最小值": float(values.min()),
                "训练内阈值中位数": float(values.median()),
                "训练内阈值最大值": float(values.max()),
                "阈值来源": "；".join(sorted(group["阈值来源"].dropna().astype(str).unique())),
                "使用边界": (
                    "固定文献Z值规则，不是概率阈值"
                    if route == "三标准差Z值规则基准"
                    else "每个外层测试孕妇只使用其余孕妇训练内选择；不是临床唯一阈值"
                ),
            }
        )
    threshold_output = 工作区 / "04_第四问/02_核心结果/第四问_外层训练内阈值摘要.csv"
    写CSV(pd.DataFrame(threshold_rows), threshold_output)
    return [difference_path, threshold_path], [difference_output, threshold_output]


def main() -> None:
    sources: list[Path] = []
    outputs: list[Path] = []
    for generator in (
        生成第一问中心化常数,
        生成第二问分组稳定性证据,
        生成第四问补充证据,
    ):
        generated_sources, generated_outputs = generator()
        sources.extend(generated_sources)
        outputs.extend(generated_outputs)
    record = {
        "状态": "PASS",
        "说明": "只生成论文证据索引表，不重新拟合或改变四问模型",
        "生成脚本": str(脚本路径.relative_to(工作区)),
        "生成脚本安全散列值_SHA256": 文件哈希(脚本路径),
        "来源文件安全散列值_SHA256": {
            str(path.relative_to(工作区)): 文件哈希(path) for path in sources
        },
        "输出文件安全散列值_SHA256": {
            str(path.relative_to(工作区)): 文件哈希(path) for path in outputs
        },
    }
    record_path = 工作区 / "05_公共材料/06_核心代码索引/论文补充证据生成记录.json"
    record_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(record, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
