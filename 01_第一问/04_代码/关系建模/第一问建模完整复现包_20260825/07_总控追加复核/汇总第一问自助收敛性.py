from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


本目录 = Path(__file__).resolve().parent
输入根目录 = 本目录 / "自助收敛性"
输出目录 = 本目录 / "自助收敛性复核"


def 区间状态(lower: float, upper: float) -> str:
    if lower > 0:
        return "区间全为正"
    if upper < 0:
        return "区间全为负"
    return "区间跨0"


def 主程序() -> None:
    rows: list[dict] = []
    for requested in (200, 400, 800):
        path = 输入根目录 / f"B{requested}" / "主线候选_簇自助置信区间.csv"
        table = pd.read_csv(path)
        for _, row in table.iterrows():
            lower = float(row["自助95%区间下限"])
            upper = float(row["自助95%区间上限"])
            rows.append(
                {
                    "请求重复次数": requested,
                    "有效重复次数": int(row["有效重复数"]),
                    "参数": row["参数"],
                    "自助均值": float(row["自助均值"]),
                    "自助标准误": float(row["自助标准误"]),
                    "95%区间下限": lower,
                    "95%区间上限": upper,
                    "区间相对0状态": 区间状态(lower, upper),
                }
            )
    combined = pd.DataFrame(rows)
    pivot_lower = combined.pivot(index="参数", columns="请求重复次数", values="95%区间下限")
    pivot_upper = combined.pivot(index="参数", columns="请求重复次数", values="95%区间上限")
    pivot_state = combined.pivot(index="参数", columns="请求重复次数", values="区间相对0状态")
    comparison_rows = []
    for parameter in pivot_lower.index:
        comparison_rows.append(
            {
                "参数": parameter,
                "B200区间下限": float(pivot_lower.loc[parameter, 200]),
                "B200区间上限": float(pivot_upper.loc[parameter, 200]),
                "B400区间下限": float(pivot_lower.loc[parameter, 400]),
                "B400区间上限": float(pivot_upper.loc[parameter, 400]),
                "B800区间下限": float(pivot_lower.loc[parameter, 800]),
                "B800区间上限": float(pivot_upper.loc[parameter, 800]),
                "B400至B800下限变化": float(pivot_lower.loc[parameter, 800] - pivot_lower.loc[parameter, 400]),
                "B400至B800上限变化": float(pivot_upper.loc[parameter, 800] - pivot_upper.loc[parameter, 400]),
                "B200状态": pivot_state.loc[parameter, 200],
                "B400状态": pivot_state.loc[parameter, 400],
                "B800状态": pivot_state.loc[parameter, 800],
                "三档定性结论一致": int(len({pivot_state.loc[parameter, 200], pivot_state.loc[parameter, 400], pivot_state.loc[parameter, 800]}) == 1),
            }
        )
    comparison = pd.DataFrame(comparison_rows)
    core = comparison.loc[comparison["参数"].isin(["孕周一次项", "孕周二次项", "妇间BMI效应", "个体内BMI效应"])]
    audit = {
        "重复次数序列": [200, 400, 800],
        "抽样单位": "孕妇整簇",
        "随机种子": 20250824,
        "核心参数三档定性结论全部一致": bool(core["三档定性结论一致"].eq(1).all()),
        "年龄效应说明": "年龄效应在B200下区间全负、B400与B800下跨0，故只能继续报告为边缘且不稳健证据。",
        "用途": "验证有限重抽样次数不改变孕周、妇间BMI和个体内BMI三项核心结论；不据此把任一重复次数解释为科学常数。",
    }
    输出目录.mkdir(parents=True, exist_ok=True)
    combined.to_csv(输出目录 / "第一问自助收敛性长表.csv", index=False, encoding="utf-8-sig")
    comparison.to_csv(输出目录 / "第一问自助收敛性区间对比.csv", index=False, encoding="utf-8-sig")
    (输出目录 / "第一问自助收敛性复核.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    主程序()
