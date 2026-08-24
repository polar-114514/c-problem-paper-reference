from __future__ import annotations

import csv
import hashlib
import json
import re
from pathlib import Path


仓库根目录 = Path(__file__).resolve().parent.parent
清单文件 = 仓库根目录 / "08_发布与复核" / "已审核材料清单.csv"
原题文件 = 仓库根目录 / "00_题目与原始资料" / "01_题目原文" / "C题.pdf"
附件文件 = 仓库根目录 / "00_题目与原始资料" / "02_原始数据" / "附件.xlsx"

预期原题哈希 = "d54400f4bb9920c0d37f15f5b6bc161aa281c4b32c03c063efdeb739716eac08"
预期附件哈希 = "14827156218bd4f7e4f16db4aa6d9f757c6648379e038ae6c6b58383648614af"

必要路径 = [
    原题文件,
    附件文件,
    仓库根目录 / "README_建模材料总索引.md",
    仓库根目录 / "01_第一问" / "README_当前有效版本.md",
    仓库根目录 / "02_第二问" / "README_当前有效版本.md",
    仓库根目录 / "03_第三问" / "README_当前有效版本.md",
    仓库根目录 / "04_第四问" / "README_当前有效版本.md",
    仓库根目录 / "01_第一问" / "04_代码" / "关系建模" / "第一问建模完整复现包_20260825" / "一键运行第一问完整复现.ps1",
    仓库根目录 / "02_第二问" / "04_代码" / "第二问无自拟参数完整复现包_20260825" / "一键运行第二问无自拟参数重构.ps1",
    仓库根目录 / "03_第三问" / "04_代码" / "第三问多因素达标比例完整复现包_20260825" / "一键运行第三问多因素建模.ps1",
    仓库根目录 / "04_第四问" / "04_代码" / "第四问女胎异常判定完整复现包_20260825" / "一键运行第四问女胎异常判定.ps1",
    仓库根目录 / "05_公共材料" / "四问联合总审_20260825" / "四问联合PASS记录.json",
    仓库根目录 / "05_公共材料" / "最终工作区复核_20260825" / "最终工作区PASS记录.json",
]

禁止目录名 = {"node_modules", "__pycache__", "99_临时中转", "历史PNG图_格式规范变更前"}
禁止扩展名 = {".pyc", ".zip"}
文本扩展名 = {".md", ".txt", ".py", ".ps1", ".mjs", ".m", ".wl", ".json", ".csv"}
敏感模式 = {
    "GitHub令牌": re.compile(r"(?:ghp_|github_pat_)[A-Za-z0-9_]{20,}"),
    "OpenAI密钥": re.compile(r"sk-[A-Za-z0-9_-]{20,}"),
    "AWS访问键": re.compile(r"AKIA[0-9A-Z]{16}"),
    "电子邮箱": re.compile(r"(?<![\w.-])[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}(?![\w.-])"),
}


def 文件哈希(路径: Path) -> str:
    摘要 = hashlib.sha256()
    with 路径.open("rb") as 文件:
        for 数据块 in iter(lambda: 文件.read(1024 * 1024), b""):
            摘要.update(数据块)
    return 摘要.hexdigest()


def 主程序() -> None:
    错误: list[str] = []
    for 路径 in 必要路径:
        if not 路径.is_file():
            错误.append(f"缺少必要文件：{路径.relative_to(仓库根目录)}")

    if 原题文件.is_file() and 文件哈希(原题文件) != 预期原题哈希:
        错误.append("原题PDF哈希与冻结值不一致")
    if 附件文件.is_file() and 文件哈希(附件文件) != 预期附件哈希:
        错误.append("原始附件哈希与冻结值不一致")

    联合记录路径 = 仓库根目录 / "05_公共材料" / "四问联合总审_20260825" / "四问联合PASS记录.json"
    最终记录路径 = 仓库根目录 / "05_公共材料" / "最终工作区复核_20260825" / "最终工作区PASS记录.json"
    if 联合记录路径.is_file():
        联合记录 = json.loads(联合记录路径.read_text(encoding="utf-8-sig"))
        if not (联合记录.get("状态") == "PASS" and 联合记录.get("联合检查项") == 52 and 联合记录.get("失败项") == 0):
            错误.append("四问联合总审记录不是52/52通过")
    if 最终记录路径.is_file():
        最终记录 = json.loads(最终记录路径.read_text(encoding="utf-8-sig"))
        if not (最终记录.get("状态") == "PASS" and 最终记录.get("检查项") == 50 and 最终记录.get("失败项") == 0):
            错误.append("最终工作区复核记录不是50/50通过")

    正式包 = [
        仓库根目录 / "01_第一问" / "04_代码" / "关系建模" / "第一问建模完整复现包_20260825",
        仓库根目录 / "02_第二问" / "04_代码" / "第二问无自拟参数完整复现包_20260825",
        仓库根目录 / "03_第三问" / "04_代码" / "第三问多因素达标比例完整复现包_20260825",
        仓库根目录 / "04_第四问" / "04_代码" / "第四问女胎异常判定完整复现包_20260825",
    ]
    图片后缀 = {".png", ".jpg", ".jpeg", ".svg", ".gif", ".webp", ".bmp", ".tif", ".tiff"}
    for 序号, 包 in enumerate(正式包, start=1):
        图片 = [路径 for 路径 in 包.rglob("*") if 路径.is_file() and 路径.suffix.lower() in 图片后缀]
        提示词数 = sum(1 for 路径 in 包.rglob("*.txt") if "提示词" in 路径.name)
        if 图片:
            错误.append(f"第{序号}问正式包含图片：{[str(p.relative_to(包)) for p in 图片[:5]]}")
        if 提示词数 < 3:
            错误.append(f"第{序号}问正式包制图提示词不足3份")

    实际文件 = sorted(
        路径
        for 路径 in 仓库根目录.rglob("*")
        if 路径.is_file() and ".git" not in 路径.parts and 路径.resolve() != 清单文件.resolve()
    )
    for 路径 in 实际文件:
        相对路径 = 路径.relative_to(仓库根目录)
        if any(部分 in 禁止目录名 for 部分 in 相对路径.parts):
            错误.append(f"发现禁止目录文件：{相对路径}")
        if 路径.suffix.lower() in 禁止扩展名:
            错误.append(f"发现禁止扩展名文件：{相对路径}")
        if 路径.stat().st_size > 20 * 1024 * 1024:
            错误.append(f"文件超过20MB：{相对路径}")
        if 路径.suffix.lower() in 文本扩展名:
            文本 = 路径.read_text(encoding="utf-8-sig", errors="ignore")
            for 名称, 模式 in 敏感模式.items():
                if 模式.search(文本):
                    错误.append(f"发现高置信敏感模式[{名称}]：{相对路径}")

    if not 清单文件.is_file():
        错误.append("缺少已审核材料清单")
        清单行: list[dict[str, str]] = []
    else:
        with 清单文件.open("r", encoding="utf-8-sig", newline="") as 文件:
            清单行 = list(csv.DictReader(文件))

    清单路径集合 = {行["相对仓库路径"] for 行 in 清单行}
    实际路径集合 = {路径.relative_to(仓库根目录).as_posix() for 路径 in 实际文件}
    if 清单路径集合 != 实际路径集合:
        错误.append(
            f"清单与实际文件集合不一致：仅清单{sorted(清单路径集合-实际路径集合)}；"
            f"仅实际{sorted(实际路径集合-清单路径集合)}"
        )
    for 行 in 清单行:
        路径 = 仓库根目录 / 行["相对仓库路径"]
        if 路径.is_file():
            if 路径.stat().st_size != int(行["文件大小_字节"]):
                错误.append(f"文件大小与清单不一致：{行['相对仓库路径']}")
            if 文件哈希(路径) != 行["SHA256"]:
                错误.append(f"文件哈希与清单不一致：{行['相对仓库路径']}")

    结果 = {
        "验证状态": "通过" if not 错误 else "失败",
        "清单条目数": len(清单行),
        "实际受检文件数_不含清单": len(实际文件),
        "原题SHA256": 文件哈希(原题文件) if 原题文件.is_file() else None,
        "附件SHA256": 文件哈希(附件文件) if 附件文件.is_file() else None,
        "错误数": len(错误),
        "错误": 错误,
    }
    print(json.dumps(结果, ensure_ascii=False, indent=2))
    if 错误:
        raise SystemExit(1)


if __name__ == "__main__":
    主程序()
