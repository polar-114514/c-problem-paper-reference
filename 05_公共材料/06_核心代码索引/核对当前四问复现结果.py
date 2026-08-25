from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


预期附件安全散列值 = "14827156218bd4f7e4f16db4aa6d9f757c6648379e038ae6c6b58383648614af"


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


脚本路径 = Path(__file__).resolve()
工作区 = 定位工作区(脚本路径.parent)


第一至三问映射 = [
    ("第一问", "05_第一问683敏感性参数比较.csv", "01_第一问/02_核心结果/第一问_683敏感性参数比较.csv"),
    ("第一问", "22_第一问主模型分段GEE参数.csv", "01_第一问/02_核心结果/第一问_分段GEE参数.csv"),
    ("第一问", "23_第一问主模型前后段派生效应.csv", "01_第一问/02_核心结果/第一问_前后段派生效应.csv"),
    ("第一问", "24_第一问主模型分段异质性Wald检验.csv", "01_第一问/02_核心结果/第一问_分段异质性Wald检验.csv"),
    ("第二问", "10_第二三问候选统一比较.csv", "02_第二问/02_核心结果/第二三问_候选统一比较.csv"),
    ("第二问", "14_第二三问主路线三套样本参数表.csv", "02_第二问/02_核心结果/第二三问_主路线参数表.csv"),
    ("第二问", "25_第二问BMI统计分组支持度.csv", "02_第二问/02_核心结果/第二问_BMI统计分组支持度.csv"),
    ("第二问", "26_第二问单切点折间稳定性.csv", "02_第二问/02_核心结果/第二问_单切点折间稳定性.csv"),
    ("第二问", "27_第二问统计主分组与折中时点.csv", "02_第二问/02_核心结果/第二问_统计主分组与折中时点.csv"),
    ("第三问", "10_第二三问候选统一比较.csv", "03_第三问/02_核心结果/第二三问_候选统一比较.csv"),
    ("第三问", "14_第二三问主路线三套样本参数表.csv", "03_第三问/02_核心结果/第二三问_主路线参数表.csv"),
    ("第三问", "15_第二三问主路线标准化参数表.csv", "03_第三问/02_核心结果/第三问_主路线标准化参数表.csv"),
    ("第三问", "16_第三问相对第二问预测增益汇总.csv", "03_第三问/02_核心结果/第三问_相对第二问预测增益汇总.csv"),
    ("第三问", "27_第三问统计主分组与折中时点.csv", "03_第三问/02_核心结果/第三问_统计主分组与折中时点.csv"),
]


第四问映射 = [
    ("01_数据/第四问变量角色与泄漏禁用表.csv", "04_第四问/02_核心结果/第四问_变量角色与泄漏禁用表.csv"),
    ("01_数据/第四问标签与样本审计摘要.csv", "04_第四问/02_核心结果/第四问_标签与样本审计摘要.csv"),
    ("02_模型结果/第四问参数来源表.csv", "04_第四问/02_核心结果/第四问_参数来源表.csv"),
    ("02_模型结果/第四问各异常类型识别指标.csv", "04_第四问/02_核心结果/第四问_各异常类型识别指标.csv"),
    ("02_模型结果/第四问候选路线统一比较.csv", "04_第四问/02_核心结果/第四问_候选路线统一比较.csv"),
    ("02_模型结果/第四问入选模型留一系数稳定性.csv", "04_第四问/02_核心结果/第四问_入选模型留一系数稳定性.csv"),
    ("02_模型结果/第四问入选模型全样本参数表.csv", "04_第四问/02_核心结果/第四问_入选模型全样本参数表.csv"),
    ("02_模型结果/第四问入选模型校准十分位表.csv", "04_第四问/02_核心结果/第四问_入选模型校准十分位表.csv"),
    ("02_模型结果/第四问入选模型预处理参照表.csv", "04_第四问/02_核心结果/第四问_入选模型预处理参照表.csv"),
    ("02_模型结果/第四问入选模型阈值性能完整曲线.csv", "04_第四问/02_核心结果/第四问_入选模型阈值性能完整曲线.csv"),
    ("03_验证/第四问候选路线指标95%区间.csv", "04_第四问/02_核心结果/第四问_候选路线指标95%区间.csv"),
    ("03_验证/第四问各异常类型指标95%区间.csv", "04_第四问/02_核心结果/第四问_各异常类型指标95%区间.csv"),
]


def 加入文件核对(
    rows: list[dict[str, object]],
    question: str,
    generated: Path,
    approved: Path,
) -> None:
    generated_exists = generated.is_file()
    approved_exists = approved.is_file()
    generated_hash = 文件哈希(generated) if generated_exists else ""
    approved_hash = 文件哈希(approved) if approved_exists else ""
    rows.append(
        {
            "问题": question,
            "检查项": "复现文件与当前批准核心表逐字节一致",
            "复现文件": str(generated),
            "批准文件": str(approved.relative_to(工作区)),
            "复现文件安全散列值_SHA256": generated_hash,
            "批准文件安全散列值_SHA256": approved_hash,
            "状态": "PASS" if generated_exists and approved_exists and generated_hash == approved_hash else "FAIL",
        }
    )


def main(q123_output: Path, q4_output: Path, report_directory: Path) -> int:
    q123_output = q123_output.resolve()
    q4_output = q4_output.resolve()
    report_directory = report_directory.resolve()
    report_directory.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []

    attachment = 工作区 / "00_题目与原始资料/02_原始数据/附件.xlsx"
    attachment_hash = 文件哈希(attachment)
    rows.append(
        {
            "问题": "公共",
            "检查项": "原始附件安全散列值",
            "复现文件": str(attachment.relative_to(工作区)),
            "批准文件": "预期附件安全散列值",
            "复现文件安全散列值_SHA256": attachment_hash,
            "批准文件安全散列值_SHA256": 预期附件安全散列值,
            "状态": "PASS" if attachment_hash == 预期附件安全散列值 else "FAIL",
        }
    )

    for question, generated_relative, approved_relative in 第一至三问映射:
        加入文件核对(
            rows,
            question,
            q123_output / generated_relative,
            工作区 / approved_relative,
        )
    for generated_relative, approved_relative in 第四问映射:
        加入文件核对(
            rows,
            "第四问",
            q4_output / generated_relative,
            工作区 / approved_relative,
        )

    q4_pass_path = q4_output / "04_复现/第四问自审PASS记录.json"
    q4_status = "MISSING"
    if q4_pass_path.is_file():
        q4_status = str(json.loads(q4_pass_path.read_text(encoding="utf-8"))["状态"])
    rows.append(
        {
            "问题": "第四问",
            "检查项": "第四问内部总控状态",
            "复现文件": str(q4_pass_path),
            "批准文件": "要求状态PASS",
            "复现文件安全散列值_SHA256": 文件哈希(q4_pass_path) if q4_pass_path.is_file() else "",
            "批准文件安全散列值_SHA256": "不适用",
            "状态": "PASS" if q4_status == "PASS" else "FAIL",
        }
    )

    checklist = pd.DataFrame(rows)
    checklist_path = report_directory / "四问复现核对清单.csv"
    checklist.to_csv(checklist_path, index=False, encoding="utf-8-sig")
    failed = int(checklist["状态"].ne("PASS").sum())
    status = {
        "状态": "PASS" if failed == 0 else "REJECTED",
        "检查项": int(len(checklist)),
        "通过项": int(checklist["状态"].eq("PASS").sum()),
        "失败项": failed,
        "核对范围": "原始附件、第一至第三问14张核心表、第四问12张核心表及第四问内部总控状态",
        "清单安全散列值_SHA256": 文件哈希(checklist_path),
    }
    status_path = report_directory / "四问复现核对状态.json"
    status_path.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(status, ensure_ascii=False, indent=2))
    return 0 if failed == 0 else 2


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="核对当前四问复现结果与批准核心表")
    parser.add_argument("--第一至三问输出", type=Path, required=True)
    parser.add_argument("--第四问输出", type=Path, required=True)
    parser.add_argument("--报告目录", type=Path, required=True)
    args = parser.parse_args()
    raise SystemExit(main(args.第一至三问输出, args.第四问输出, args.报告目录))
