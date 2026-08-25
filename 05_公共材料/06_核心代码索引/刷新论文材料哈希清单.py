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


脚本路径 = Path(__file__).resolve()
工作区 = 定位工作区(脚本路径.parent)
清单路径 = 工作区 / "05_公共材料/04_写作规范与检查/精简论文材料SHA256.csv"
状态路径 = 工作区 / "05_公共材料/04_写作规范与检查/精简论文材料状态.json"


def main() -> None:
    candidates = [工作区 / "README_建模材料总索引.md"]
    for directory_name in (
        "01_第一问",
        "02_第二问",
        "03_第三问",
        "04_第四问",
        "05_公共材料",
        "06_论文总稿",
        "07_最终提交",
    ):
        directory = 工作区 / directory_name
        candidates.extend(path for path in directory.rglob("*") if path.is_file())
    excluded = {清单路径.resolve(), 状态路径.resolve()}
    files = sorted(
        {path.resolve() for path in candidates if path.resolve() not in excluded},
        key=lambda path: str(path.relative_to(工作区)).casefold(),
    )
    rows = [
        {
            "相对路径": str(path.relative_to(工作区)),
            "文件安全散列值_SHA256": 文件哈希(path),
            "字节数": int(path.stat().st_size),
        }
        for path in files
    ]
    pd.DataFrame(rows).to_csv(清单路径, index=False, encoding="utf-8-sig")
    status = {
        "状态": "PASS",
        "清单覆盖文件数": len(rows),
        "清单安全散列值_SHA256": 文件哈希(清单路径),
        "覆盖范围": "README_建模材料总索引.md及01至07目录；不包含原始资料、99临时中转、GitHub发布、清单本身和状态文件",
        "生成脚本": str(脚本路径.relative_to(工作区)),
        "生成脚本安全散列值_SHA256": 文件哈希(脚本路径),
    }
    状态路径.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(status, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
