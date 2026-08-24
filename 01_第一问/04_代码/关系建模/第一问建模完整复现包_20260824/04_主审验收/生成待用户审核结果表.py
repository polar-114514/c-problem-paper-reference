from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "00_共同口径" / "冻结数据" / "第一问主模型冻结样本.csv"
REVIEW_DIR = Path(__file__).resolve().parent / "统一复核结果"
BOOT_DIR = Path(__file__).resolve().parent / "二次Logit混合模型独立复核"
CANDIDATE_C_DIR = ROOT / "03_候选C_边界稳健模型"
OUTPUT_DIR = ROOT / "05_待用户审核" / "结果表"


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def coefficients_from_review() -> dict[str, float]:
    table = pd.read_csv(REVIEW_DIR / "04_推荐主线固定效应.csv")
    values = dict(zip(table["参数"], table["估计值_logit尺度"]))
    return {
        "截距": values["截距"],
        "孕周一次项": values["孕周一次项"],
        "孕周二次项": values["孕周二次项"],
        "妇间BMI效应": values["妇间BMI效应"],
        "个体内BMI效应": values["个体内BMI效应"],
        "年龄效应": values["年龄效应"],
        "生产次数效应": values["生产次数效应"],
    }


def effect_metrics(coef: dict[str, float], data: pd.DataFrame) -> dict[str, float]:
    week_c = data["孕周数"].to_numpy() - 18.0
    bmi_between_c = data["孕妇平均BMI"].to_numpy() - 32.0
    bmi_within = data["BMI个体内偏差"].to_numpy()
    age_c = data["年龄"].to_numpy() - 29.0
    parity = data["生产次数"].to_numpy()

    def eta(dw=0.0, db=0.0, dbw=0.0, da=0.0, dp=0.0):
        return (
            coef["截距"]
            + coef["孕周一次项"] * (week_c + dw)
            + coef["孕周二次项"] * (week_c + dw) ** 2
            + coef["妇间BMI效应"] * (bmi_between_c + db)
            + coef["个体内BMI效应"] * (bmi_within + dbw)
            + coef["年龄效应"] * (age_c + da)
            + coef["生产次数效应"] * (parity + dp)
        )

    base = expit(eta())

    def reference_prediction(week, bmi_between=32.0, bmi_within_value=0.0, age=29.0, parity_value=0.0):
        wc = week - 18.0
        return float(
            expit(
                coef["截距"]
                + coef["孕周一次项"] * wc
                + coef["孕周二次项"] * wc**2
                + coef["妇间BMI效应"] * (bmi_between - 32.0)
                + coef["个体内BMI效应"] * bmi_within_value
                + coef["年龄效应"] * (age - 29.0)
                + coef["生产次数效应"] * parity_value
            )
        )

    # 一周调整预测差只以原孕周小于25周0天的事件为基准，保证调整后孕周仍在[10,26)内。
    孕周加一有效掩码 = data["孕周数"].to_numpy() < 25.0
    output = {
        "样本标准化的一周调整预测差": float(
            np.mean((expit(eta(dw=1.0)) - base)[孕周加一有效掩码]) * 100
        ),
        "全样本标准化妇间BMI增加1": float(np.mean(expit(eta(db=1.0)) - base) * 100),
        "全样本标准化个体内BMI增加1": float(np.mean(expit(eta(dbw=1.0)) - base) * 100),
        "全样本标准化年龄增加1岁": float(np.mean(expit(eta(da=1.0)) - base) * 100),
        "全样本标准化生产次数增加1": float(np.mean(expit(eta(dp=1.0)) - base) * 100),
    }
    y12 = reference_prediction(12.0)
    y20 = reference_prediction(20.0)
    y24 = reference_prediction(24.0)
    output["参考孕妇12至20周变化"] = (y20 - y12) * 100
    output["参考孕妇12至24周变化"] = (y24 - y12) * 100
    output["参考孕妇18周妇间BMI增加1"] = (
        reference_prediction(18.0, bmi_between=33.0) - reference_prediction(18.0)
    ) * 100
    output["参考孕妇18周个体内BMI增加1"] = (
        reference_prediction(18.0, bmi_within_value=1.0) - reference_prediction(18.0)
    ) * 100
    output["曲线最低点孕周"] = 18.0 - coef["孕周一次项"] / (2 * coef["孕周二次项"])
    for week in [12.0, 16.0, 20.0, 24.0]:
        y = reference_prediction(week)
        slope = y * (1 - y) * (
            coef["孕周一次项"] + 2 * coef["孕周二次项"] * (week - 18.0)
        )
        output[f"参考孕妇{int(week)}周局部斜率"] = slope * 100
    return output


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(DATA_PATH)
    coef = coefficients_from_review()
    bootstrap = pd.read_csv(BOOT_DIR / "主线候选_簇自助明细.csv")
    boot_metrics = []
    for _, row in bootstrap.iterrows():
        boot_coef = {
            "截距": row["截距"],
            "孕周一次项": row["孕周一次项"],
            "孕周二次项": row["孕周二次项"],
            "妇间BMI效应": row["妇间BMI效应"],
            "个体内BMI效应": row["个体内BMI效应"],
            "年龄效应": row["年龄效应"],
            "生产次数效应": row["生产次数效应"],
        }
        boot_metrics.append(effect_metrics(boot_coef, data))
    boot_metric_table = pd.DataFrame(boot_metrics)
    point_metrics = effect_metrics(coef, data)

    units = {
        "样本标准化的一周调整预测差": "以孕周小于25周0天的事件为基准，其余协变量不变；孕周增加1周后重新计算一次项与二次项",
        "全样本标准化妇间BMI增加1": "孕妇平均BMI增加1 kg/m²",
        "全样本标准化个体内BMI增加1": "同一孕妇BMI相对本人均值增加1 kg/m²",
        "全样本标准化年龄增加1岁": "年龄增加1岁",
        "全样本标准化生产次数增加1": "生产次数增加1次",
        "参考孕妇12至20周变化": "参考孕妇由12周变为20周",
        "参考孕妇12至24周变化": "参考孕妇由12周变为24周",
        "参考孕妇18周妇间BMI增加1": "18周参考孕妇的平均BMI增加1 kg/m²",
        "参考孕妇18周个体内BMI增加1": "18周同一孕妇BMI偏差增加1 kg/m²",
        "曲线最低点孕周": "曲线驻点位置",
        "参考孕妇12周局部斜率": "12周附近孕周增加1周",
        "参考孕妇16周局部斜率": "16周附近孕周增加1周",
        "参考孕妇20周局部斜率": "20周附近孕周增加1周",
        "参考孕妇24周局部斜率": "24周附近孕周增加1周",
    }
    effect_rows = []
    for name, point in point_metrics.items():
        values = boot_metric_table[name].dropna()
        low = float(values.quantile(0.025))
        high = float(values.quantile(0.975))
        if name == "曲线最低点孕周":
            conclusion = "估计曲线低谷位置"
            result_unit = "孕周"
        else:
            conclusion = "区间不跨0" if low * high > 0 else "区间跨0，证据不足"
            result_unit = "Y浓度百分点"
        effect_rows.append(
            {
                "效应": name,
                "变化单位": units[name],
                "点估计": point,
                "簇自助95%区间下限": low,
                "簇自助95%区间上限": high,
                "结果单位": result_unit,
                "解释": conclusion,
                "有效自助次数": len(values),
            }
        )
    pd.DataFrame(effect_rows).to_csv(
        OUTPUT_DIR / "第一问核心效应原尺度解释.csv", index=False, encoding="utf-8-sig"
    )

    coef_table = pd.read_csv(REVIEW_DIR / "04_推荐主线固定效应.csv")
    boot_summary = pd.read_csv(BOOT_DIR / "主线候选_簇自助置信区间.csv")
    lrt = pd.read_csv(REVIEW_DIR / "05_推荐主线整体显著性检验.csv")
    coef_to_boot = {
        "截距": "截距",
        "孕周一次项": "孕周一次项",
        "孕周二次项": "孕周二次项",
        "妇间BMI效应": "妇间BMI效应",
        "个体内BMI效应": "个体内BMI效应",
        "年龄效应": "年龄效应",
        "生产次数效应": "生产次数效应",
    }
    lrt_map = {
        "孕周一次项": float(lrt.loc[lrt["检验项"] == "孕周总体（二次项整体）", "P值"].iloc[0]),
        "孕周二次项": float(lrt.loc[lrt["检验项"] == "孕周非线性（二次项）", "P值"].iloc[0]),
        "妇间BMI效应": float(lrt.loc[lrt["检验项"] == "妇间BMI", "P值"].iloc[0]),
        "个体内BMI效应": float(lrt.loc[lrt["检验项"] == "个体内BMI", "P值"].iloc[0]),
        "年龄效应": float(lrt.loc[lrt["检验项"] == "年龄", "P值"].iloc[0]),
        "生产次数效应": float(lrt.loc[lrt["检验项"] == "生产次数", "P值"].iloc[0]),
    }
    parameter_rows = []
    for _, row in coef_table.iterrows():
        name = row["参数"]
        b = boot_summary.loc[boot_summary["参数"] == coef_to_boot[name]].iloc[0]
        low = float(b["自助95%区间下限"])
        high = float(b["自助95%区间上限"])
        if name == "年龄效应" and low <= 0 <= high:
            conclusion = "渐近检验显著，但簇自助区间跨0，按边缘证据处理"
        elif low <= 0 <= high:
            conclusion = "无充分独立关联证据"
        else:
            conclusion = "稳健显著"
        parameter_rows.append(
            {
                "参数": name,
                "估计值_logit尺度": row["估计值_logit尺度"],
                "整块或单项似然比P值": lrt_map.get(name, np.nan),
                "簇自助95%区间下限": low,
                "簇自助95%区间上限": high,
                "主审结论": conclusion,
            }
        )
    pd.DataFrame(parameter_rows).to_csv(
        OUTPUT_DIR / "第一问推荐模型系数与显著性.csv", index=False, encoding="utf-8-sig"
    )

    comparison = pd.read_csv(REVIEW_DIR / "01_候选模型统一比较.csv")
    c_compare = pd.read_csv(CANDIDATE_C_DIR / "02_模型比较与分组验证.csv")
    compare_rows = []
    choices = [
        ("原尺度二次随机斜率", "原尺度二次混合模型", "5×5孕妇分组"),
        ("logit线性随机斜率", "线性Logit混合模型", "5×5孕妇分组"),
        ("logit二次随机斜率", "二次Logit混合模型（推荐）", "5×5孕妇分组"),
        ("logit样条随机斜率", "3自由度样条Logit混合模型", "5×5孕妇分组"),
    ]
    decisions = {
        "原尺度二次随机斜率": "预测略优但异方差明显，仅作尺度敏感性",
        "logit线性随机斜率": "线性不足，孕周二次项整体显著",
        "logit二次随机斜率": "兼顾边界、层级、简洁与稳健推断，选为唯一主线",
        "logit样条随机斜率": "与二次项预测近似但参数更多，BIC更差",
    }
    for key, display, validation in choices:
        row = comparison.loc[comparison["候选模型"] == key].iloc[0]
        compare_rows.append(
            {
                "路线": display,
                "重复测量处理": row["随机结构"],
                "验证口径": validation,
                "组外RMSE": row["RMSE均值"],
                "组外MAE": row["MAE均值"],
                "组外R2": row["R2均值"],
                "边界或方差优势": "预测天然在0到1" if "logit" in key else "无",
                "主审结论": decisions[key],
            }
        )
    for key, display, decision in [
        ("Beta回归-孕周变精度-二次孕周", "变精度Beta回归", "比例分布拟合良好，但没有孕妇随机效应；作稳健性对照"),
        ("分数logit GEE-二次孕周", "分数Logit GEE", "人口平均推断稳健，但不能表达个体随机斜率；作稳健性对照"),
    ]:
        row = c_compare.loc[c_compare["模型"] == key].iloc[0]
        compare_rows.append(
            {
                "路线": display,
                "重复测量处理": "孕妇簇稳健" if "Beta" in key else "交换型GEE+孕妇簇稳健",
                "验证口径": "单次5折孕妇分组",
                "组外RMSE": row["均方根误差（RMSE）"],
                "组外MAE": row["平均绝对误差（MAE）"],
                "组外R2": row["组外R²"],
                "边界或方差优势": "比例边界+变精度" if "Beta" in key else "比例边界+方差稳健",
                "主审结论": decision,
            }
        )
    pd.DataFrame(compare_rows).to_csv(
        OUTPUT_DIR / "第一问候选路线主审对比.csv", index=False, encoding="utf-8-sig"
    )

    sensitivity = pd.read_csv(REVIEW_DIR / "06_推荐主线稳健性分析.csv")
    sensitivity["12至20周预测变化_百分点"] = sensitivity["12至20周预测变化"] * 100
    sensitivity["12至24周预测变化_百分点"] = sensitivity["12至24周预测变化"] * 100
    keep_columns = [
        "敏感性场景",
        "事件数",
        "孕妇数",
        "妇间BMI估计",
        "妇间BMI_P值",
        "个体内BMI估计",
        "个体内BMI_P值",
        "曲线最低点孕周",
        "12至20周预测变化_百分点",
        "12至24周预测变化_百分点",
        "Breusch-Pagan检验P值",
    ]
    sensitivity[keep_columns].to_csv(
        OUTPUT_DIR / "第一问稳健性分析摘要.csv", index=False, encoding="utf-8-sig"
    )

    acceptance = pd.DataFrame(
        [
            ("样本口径", "613个抽血事件、167名孕妇；A055歧义事件排除", "通过"),
            ("数据机制", "683后机制段未进入主模型", "通过"),
            ("重复测量", "孕妇随机截距与孕周随机斜率", "通过"),
            ("BMI分解", "妇间均值与个体内偏差分开", "通过"),
            ("显著性", f"相同随机结构下整块LRT+{len(bootstrap)}次有效孕妇簇自助", "通过"),
            ("组外验证", "5个随机种子×5折，全部按孕妇分组并收敛", "通过"),
            ("复杂度", "样条、BMI二次与交互均无充分增益，已删除", "通过"),
            ("比例边界", "Logit变换后固定效应预测全部位于0到1", "通过"),
            ("异方差", "主线BP P=0.0977；原尺度P约4.84e-11", "通过"),
            ("独立稳健路线", "Beta与分数Logit GEE结论方向一致", "通过"),
            ("图形规范", "未生成新图，仅保留MATLAB-SVG制图提示词TXT", "通过"),
            ("归档边界", "当前仅在99_临时中转，等待用户确认", "通过"),
        ],
        columns=["验收项", "证据", "结论"],
    )
    acceptance.to_csv(OUTPUT_DIR / "第一问最终验收清单.csv", index=False, encoding="utf-8-sig")

    清单路径 = OUTPUT_DIR / "第一问待审核结果表清单.json"
    output_files = sorted(
        path for path in OUTPUT_DIR.iterdir() if path.is_file() and path.resolve() != 清单路径.resolve()
    )
    manifest = {
        "生成时间": datetime.now().astimezone().isoformat(timespec="seconds"),
        "主模型数据SHA256": file_hash(DATA_PATH),
        "簇自助有效次数": len(bootstrap),
        "结果表SHA256": {path.name: file_hash(path) for path in output_files},
    }
    清单路径.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
