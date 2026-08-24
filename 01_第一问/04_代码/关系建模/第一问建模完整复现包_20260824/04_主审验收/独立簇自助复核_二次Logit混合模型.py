from __future__ import annotations

import argparse
import hashlib
import json
import math
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.special import expit


默认数据 = (
    Path(__file__).resolve().parents[1]
    / "00_共同口径"
    / "冻结数据"
    / "第一问主模型冻结样本.csv"
)
默认输出目录 = Path(__file__).resolve().parent / "二次Logit混合模型独立复核"
公式 = "响应 ~ 孕周中心 + I(孕周中心**2) + 妇间BMI中心 + 个体内BMI偏差 + 年龄中心 + 生产次数"


def 文件哈希(path: Path) -> str:
    摘要 = hashlib.sha256()
    with path.open("rb") as 文件:
        for 数据块 in iter(lambda: 文件.read(1024 * 1024), b""):
            摘要.update(数据块)
    return 摘要.hexdigest()


def 准备数据(path: Path) -> pd.DataFrame:
    data = pd.read_csv(path)
    必需列 = [
        "孕妇代码",
        "孕周数",
        "孕妇平均BMI",
        "BMI个体内偏差",
        "年龄",
        "生产次数",
        "Y染色体浓度均值",
    ]
    缺列 = [列 for 列 in 必需列 if 列 not in data.columns]
    if 缺列:
        raise RuntimeError(f"冻结数据缺少列：{缺列}")
    if len(data) != 613 or data["孕妇代码"].nunique() != 167:
        raise RuntimeError("主模型冻结样本不是613个事件、167名孕妇，停止复核。")
    if data[必需列].isna().any().any():
        raise RuntimeError("主模型必需变量存在缺失，停止复核。")
    data = data.copy()
    data["响应"] = np.log(data["Y染色体浓度均值"] / (1 - data["Y染色体浓度均值"]))
    data["孕周中心"] = data["孕周数"] - 18.0
    data["妇间BMI中心"] = data["孕妇平均BMI"] - 32.0
    data["个体内BMI偏差"] = data["BMI个体内偏差"]
    data["年龄中心"] = data["年龄"] - 29.0
    return data


def 拟合(data: pd.DataFrame, group_col: str):
    model = smf.mixedlm(
        公式,
        data,
        groups=data[group_col],
        re_formula="1 + 孕周中心",
        missing="raise",
    )
    try:
        result = model.fit(reml=False, method="lbfgs", maxiter=3000, disp=False)
    except Exception:
        result = model.fit(reml=False, method="powell", maxiter=5000, disp=False)
    if not result.converged:
        raise RuntimeError("混合模型未收敛")
    return result


def 百分位区间(values: pd.Series) -> tuple[float, float]:
    return float(values.quantile(0.025)), float(values.quantile(0.975))


def 主程序(数据路径: Path, 输出目录: Path, 重复次数: int, 随机种子: int) -> None:
    warnings.filterwarnings("ignore")
    输出目录.mkdir(parents=True, exist_ok=True)
    data = 准备数据(数据路径)
    result = 拟合(data, "孕妇代码")

    参数中文 = {
        "Intercept": "截距",
        "孕周中心": "孕周一次项",
        "I(孕周中心 ** 2)": "孕周二次项",
        "妇间BMI中心": "妇间BMI效应",
        "个体内BMI偏差": "个体内BMI效应",
        "年龄中心": "年龄效应",
        "生产次数": "生产次数效应",
    }
    渐近表 = []
    conf = result.conf_int()
    for 参数名, 估计值 in result.fe_params.items():
        渐近表.append(
            {
                "参数": 参数中文.get(参数名, 参数名),
                "内部参数名": 参数名,
                "估计值": float(估计值),
                "标准误": float(result.bse_fe[参数名]),
                "渐近P值": float(result.pvalues[参数名]),
                "渐近95%置信区间下限": float(conf.loc[参数名, 0]),
                "渐近95%置信区间上限": float(conf.loc[参数名, 1]),
                "比值比": float(math.exp(估计值)),
            }
        )
    pd.DataFrame(渐近表).to_csv(
        输出目录 / "主线候选_渐近系数表.csv", index=False, encoding="utf-8-sig"
    )

    rng = np.random.default_rng(随机种子)
    women = data["孕妇代码"].drop_duplicates().to_numpy()
    bootstrap_rows: list[dict[str, float | int]] = []
    失败次数 = 0
    for 重复 in range(1, 重复次数 + 1):
        sampled = rng.choice(women, size=len(women), replace=True)
        blocks = []
        for 抽样序号, woman in enumerate(sampled):
            block = data.loc[data["孕妇代码"] == woman].copy()
            block["自助孕妇簇"] = f"{woman}__{抽样序号}"
            blocks.append(block)
        boot = pd.concat(blocks, ignore_index=True)
        try:
            fitted = 拟合(boot, "自助孕妇簇")
        except Exception:
            失败次数 += 1
            continue
        row: dict[str, float | int] = {"重复序号": 重复}
        for 参数名, 估计值 in fitted.fe_params.items():
            row[参数中文.get(参数名, 参数名)] = float(估计值)
        二次项 = float(fitted.fe_params["I(孕周中心 ** 2)"])
        row["曲线最低点孕周"] = (
            18.0 - float(fitted.fe_params["孕周中心"]) / (2 * 二次项)
            if 二次项 != 0
            else np.nan
        )
        bootstrap_rows.append(row)
        if 重复 % 25 == 0:
            pd.DataFrame(bootstrap_rows).to_csv(
                输出目录 / "主线候选_簇自助明细_检查点.csv",
                index=False,
                encoding="utf-8-sig",
            )

    bootstrap = pd.DataFrame(bootstrap_rows)
    if len(bootstrap) < max(100, int(重复次数 * 0.9)):
        raise RuntimeError(
            f"簇自助有效重复仅{len(bootstrap)}/{重复次数}，低于90%，停止出具区间。"
        )
    bootstrap.to_csv(
        输出目录 / "主线候选_簇自助明细.csv", index=False, encoding="utf-8-sig"
    )

    summary_rows = []
    for column in [*参数中文.values(), "曲线最低点孕周"]:
        values = bootstrap[column].dropna()
        low, high = 百分位区间(values)
        if column == "曲线最低点孕周":
            sign_p = np.nan
        else:
            sign_p = min(1.0, 2 * min(float((values <= 0).mean()), float((values >= 0).mean())))
        summary_rows.append(
            {
                "参数": column,
                "有效重复数": int(len(values)),
                "自助均值": float(values.mean()),
                "自助标准误": float(values.std(ddof=1)),
                "自助95%区间下限": low,
                "自助95%区间上限": high,
                "符号双侧P值": sign_p,
            }
        )
    pd.DataFrame(summary_rows).to_csv(
        输出目录 / "主线候选_簇自助置信区间.csv", index=False, encoding="utf-8-sig"
    )

    prediction_rows = []
    for bmi_between in [28.0, 32.0, 36.0, 40.0]:
        for week in np.round(np.arange(11.0, 25.71, 0.25), 2):
            eta = (
                float(result.fe_params["Intercept"])
                + float(result.fe_params["孕周中心"]) * (week - 18.0)
                + float(result.fe_params["I(孕周中心 ** 2)"]) * (week - 18.0) ** 2
                + float(result.fe_params["妇间BMI中心"]) * (bmi_between - 32.0)
            )
            boot_eta = (
                bootstrap["截距"]
                + bootstrap["孕周一次项"] * (week - 18.0)
                + bootstrap["孕周二次项"] * (week - 18.0) ** 2
                + bootstrap["妇间BMI效应"] * (bmi_between - 32.0)
            )
            boot_y = expit(boot_eta.to_numpy())
            prediction_rows.append(
                {
                    "孕周数": week,
                    "孕妇平均BMI": bmi_between,
                    "BMI个体内偏差": 0.0,
                    "年龄": 29.0,
                    "生产次数": 0,
                    "典型孕妇预测Y浓度": float(expit(eta)),
                    "固定效应95%区间下限": float(np.quantile(boot_y, 0.025)),
                    "固定效应95%区间上限": float(np.quantile(boot_y, 0.975)),
                }
            )
    pd.DataFrame(prediction_rows).to_csv(
        输出目录 / "主线候选_预测关系网格.csv", index=False, encoding="utf-8-sig"
    )

    cov_re = result.cov_re.to_numpy()
    随机效应表 = pd.DataFrame(
        [
            ("随机截距标准差", math.sqrt(cov_re[0, 0])),
            ("随机孕周斜率标准差", math.sqrt(cov_re[1, 1])),
            (
                "随机截距与随机斜率相关系数",
                cov_re[0, 1] / math.sqrt(cov_re[0, 0] * cov_re[1, 1]),
            ),
            ("残差标准差（logit尺度）", math.sqrt(result.scale)),
        ],
        columns=["指标", "数值"],
    )
    随机效应表.to_csv(
        输出目录 / "主线候选_随机效应摘要.csv", index=False, encoding="utf-8-sig"
    )

    清单路径 = 输出目录 / "主线候选_运行清单.json"
    输出文件 = sorted(
        path for path in 输出目录.iterdir() if path.is_file() and path.resolve() != 清单路径.resolve()
    )
    清单 = {
        "运行时间": datetime.now().astimezone().isoformat(timespec="seconds"),
        "输入文件": str(数据路径),
        "输入文件SHA256": 文件哈希(数据路径),
        "样本抽血事件数": int(len(data)),
        "孕妇数": int(data["孕妇代码"].nunique()),
        "固定效应公式": 公式,
        "随机效应": "孕妇随机截距+孕周随机斜率",
        "簇自助请求次数": int(重复次数),
        "簇自助有效次数": int(len(bootstrap)),
        "簇自助失败次数": int(失败次数),
        "随机种子": int(随机种子),
        "模型收敛": bool(result.converged),
        "对数似然": float(result.llf),
        "赤池信息准则_AIC": float(result.aic),
        "贝叶斯信息准则_BIC": float(result.bic),
        "输出文件SHA256": {path.name: 文件哈希(path) for path in 输出文件},
    }
    清单路径.write_text(
        json.dumps(清单, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(清单, ensure_ascii=False, indent=2))


def 仅刷新现有清单(数据路径: Path, 输出目录: Path) -> None:
    """不重跑模型，只修复现有清单的输入与输出哈希。"""
    清单路径 = 输出目录 / "主线候选_运行清单.json"
    if not 清单路径.exists():
        raise FileNotFoundError(f"找不到待刷新的运行清单：{清单路径}")
    清单 = json.loads(清单路径.read_text(encoding="utf-8"))
    清单["输入文件"] = str(数据路径)
    清单["输入文件SHA256"] = 文件哈希(数据路径)
    if "AIC" in 清单:
        清单["赤池信息准则_AIC"] = 清单.pop("AIC")
    if "BIC" in 清单:
        清单["贝叶斯信息准则_BIC"] = 清单.pop("BIC")
    输出文件 = sorted(
        path for path in 输出目录.iterdir() if path.is_file() and path.resolve() != 清单路径.resolve()
    )
    清单["输出文件SHA256"] = {path.name: 文件哈希(path) for path in 输出文件}
    清单["清单哈希刷新时间"] = datetime.now().astimezone().isoformat(timespec="seconds")
    清单路径.write_text(json.dumps(清单, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(清单, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="独立复核二次Logit随机斜率混合模型，不绘图。")
    parser.add_argument("--数据", type=Path, default=默认数据)
    parser.add_argument("--输出目录", type=Path, default=默认输出目录)
    parser.add_argument("--重复次数", type=int, default=400)
    parser.add_argument("--随机种子", type=int, default=20250824)
    parser.add_argument("--仅刷新清单", action="store_true")
    args = parser.parse_args()
    if args.仅刷新清单:
        仅刷新现有清单(args.数据, args.输出目录)
    else:
        主程序(args.数据, args.输出目录, args.重复次数, args.随机种子)
