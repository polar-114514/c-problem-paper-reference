from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold


本目录 = Path(__file__).resolve().parent
复核根目录 = 本目录.parent
主审脚本 = 复核根目录 / "04_主审验收" / "主审统一检验.py"
数据路径 = 复核根目录 / "00_共同口径" / "冻结数据" / "第一问主模型冻结样本.csv"
输出目录 = 本目录 / "交叉验证设置稳定性"


def 加载主审模块():
    spec = importlib.util.spec_from_file_location("第一问主审统一检验", 主审脚本)
    if spec is None or spec.loader is None:
        raise RuntimeError("无法加载第一问主审脚本。")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def 主程序() -> None:
    module = 加载主审模块()
    data = module.读取并准备(数据路径)
    base = "bmi_between_c + bmi_within + age_c + parity"
    candidates = [
        {"名称": "原尺度二次随机斜率", "公式": f"y_mean ~ week_c + I(week_c**2) + {base}", "响应尺度": "原Y尺度", "随机公式": "1+week_c"},
        {"名称": "logit线性随机斜率", "公式": f"logit_y ~ week_c + {base}", "响应尺度": "logit尺度", "随机公式": "1+week_c"},
        {"名称": "logit二次随机截距", "公式": f"logit_y ~ week_c + I(week_c**2) + {base}", "响应尺度": "logit尺度", "随机公式": "1"},
        {"名称": "logit二次随机斜率", "公式": f"logit_y ~ week_c + I(week_c**2) + {base}", "响应尺度": "logit尺度", "随机公式": "1+week_c"},
        {"名称": "logit样条随机斜率", "公式": f"logit_y ~ cr(week, df=3, constraints='center') + {base}", "响应尺度": "logit尺度", "随机公式": "1+week_c"},
    ]

    # 3 至 10 折覆盖每折约 17 至 56 名孕妇；这里只检查计算设置敏感性，
    # 不把任何一个折数当作科学参数。两个固定种子用于检查随机分配变化。
    fold_counts = list(range(3, 11))
    seeds = [20250824, 20250825]
    rows: list[dict] = []
    for fold_count in fold_counts:
        for seed in seeds:
            splitter = GroupKFold(n_splits=fold_count, shuffle=True, random_state=seed)
            splits = list(splitter.split(data, groups=data["woman"]))
            for candidate in candidates:
                for fold_index, (train_index, test_index) in enumerate(splits, 1):
                    train = data.iloc[train_index].copy()
                    test = data.iloc[test_index].copy()
                    try:
                        result, method, warning = module.拟合混合模型(
                            candidate["公式"], train, candidate["随机公式"], reml=False
                        )
                        prediction = module.原尺度预测(result, test, candidate["响应尺度"])
                        row = {
                            "折数": fold_count,
                            "随机种子": seed,
                            "折号": fold_index,
                            "候选模型": candidate["名称"],
                            "测试孕妇数": int(test["woman"].nunique()),
                            "测试事件数": int(len(test)),
                            "收敛": 1,
                            "优化器": method,
                            "均方根误差_RMSE": float(math.sqrt(mean_squared_error(test["y_mean"], prediction))),
                            "平均绝对误差_MAE": float(mean_absolute_error(test["y_mean"], prediction)),
                            "决定系数_R2": float(r2_score(test["y_mean"], prediction)),
                            "异常或提示": warning,
                        }
                    except Exception as exc:
                        row = {
                            "折数": fold_count,
                            "随机种子": seed,
                            "折号": fold_index,
                            "候选模型": candidate["名称"],
                            "测试孕妇数": int(test["woman"].nunique()),
                            "测试事件数": int(len(test)),
                            "收敛": 0,
                            "优化器": "",
                            "均方根误差_RMSE": np.nan,
                            "平均绝对误差_MAE": np.nan,
                            "决定系数_R2": np.nan,
                            "异常或提示": f"{type(exc).__name__}:{exc}",
                        }
                    rows.append(row)

    details = pd.DataFrame(rows)
    summary = (
        details.groupby(["折数", "候选模型"], as_index=False)
        .agg(
            **{
                "验证折数": ("折号", "size"),
                "收敛折数": ("收敛", "sum"),
                "RMSE均值": ("均方根误差_RMSE", "mean"),
                "RMSE标准差": ("均方根误差_RMSE", "std"),
                "MAE均值": ("平均绝对误差_MAE", "mean"),
                "R2均值": ("决定系数_R2", "mean"),
            }
        )
    )
    summary["RMSE组内排名"] = summary.groupby("折数")["RMSE均值"].rank(method="min")

    logit_subset = summary.loc[summary["候选模型"].str.startswith("logit")].copy()
    logit_winners = (
        logit_subset.sort_values(["折数", "RMSE均值", "候选模型"])
        .groupby("折数", as_index=False)
        .first()[["折数", "候选模型", "RMSE均值"]]
        .rename(
            columns={
                "候选模型": "Logit候选中RMSE最低模型",
                "RMSE均值": "Logit候选最低RMSE",
            }
        )
    )
    quadratic = summary.loc[summary["候选模型"] == "logit二次随机斜率", ["折数", "RMSE均值", "收敛折数", "验证折数"]]
    spline = summary.loc[summary["候选模型"] == "logit样条随机斜率", ["折数", "RMSE均值"]].rename(columns={"RMSE均值": "样条RMSE均值"})
    stability = quadratic.merge(spline, on="折数", how="left").merge(logit_winners, on="折数", how="left")
    stability["二次减样条RMSE"] = stability["RMSE均值"] - stability["样条RMSE均值"]
    stability["二次模型全部收敛"] = (stability["收敛折数"] == stability["验证折数"]).astype(int)

    OUTPUT_ENCODING = "utf-8-sig"
    输出目录.mkdir(parents=True, exist_ok=True)
    details.to_csv(输出目录 / "第一问交叉验证设置稳定性逐折.csv", index=False, encoding=OUTPUT_ENCODING)
    summary.to_csv(输出目录 / "第一问交叉验证设置稳定性汇总.csv", index=False, encoding=OUTPUT_ENCODING)
    stability.to_csv(输出目录 / "第一问推荐模型稳定性复核.csv", index=False, encoding=OUTPUT_ENCODING)

    audit = {
        "独立单位": "孕妇",
        "折数范围": fold_counts,
        "随机分配种子": seeds,
        "用途": "计算设置敏感性；不把折数或种子解释为科学参数",
        "全部拟合数": int(len(details)),
        "失败拟合数": int((details["收敛"] == 0).sum()),
        "二次Logit随机斜率全部折数均全折收敛": bool(stability["二次模型全部收敛"].eq(1).all()),
        "各折数Logit候选中RMSE最低模型": logit_winners.to_dict(orient="records"),
        "解释边界": "RMSE敏感性只检查计算设置；主模型仍同时依据比例边界、异方差、层级结构和复杂度选择。",
    }
    (输出目录 / "第一问交叉验证设置稳定性复核.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    主程序()
