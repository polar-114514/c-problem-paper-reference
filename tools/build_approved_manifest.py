from __future__ import annotations

import csv
import hashlib
from pathlib import Path


仓库根目录 = Path(__file__).resolve().parent.parent
输出文件 = 仓库根目录 / "08_发布与复核" / "已审核材料清单.csv"


def 文件哈希(路径: Path) -> str:
    摘要 = hashlib.sha256()
    with 路径.open("rb") as 文件:
        for 数据块 in iter(lambda: 文件.read(1024 * 1024), b""):
            摘要.update(数据块)
    return 摘要.hexdigest()


def 来源与状态(相对路径: Path) -> tuple[str, str]:
    if 相对路径.parts[0] == "00_题目与原始资料":
        return "本地题目原始资料", "用户授权纳入私有仓库"
    if 相对路径.parts[0] == "01_第一问":
        return "第一问正式归档", "已审核通过"
    return "GitHub仓库框架", "发布辅助材料"


def 主程序() -> None:
    文件列表 = sorted(
        路径
        for 路径 in 仓库根目录.rglob("*")
        if 路径.is_file()
        and ".git" not in 路径.parts
        and 路径.resolve() != 输出文件.resolve()
    )
    输出文件.parent.mkdir(parents=True, exist_ok=True)
    with 输出文件.open("w", encoding="utf-8-sig", newline="") as 文件:
        写入器 = csv.DictWriter(
            文件,
            fieldnames=["相对仓库路径", "文件大小_字节", "SHA256", "来源类型", "审核状态"],
        )
        写入器.writeheader()
        for 路径 in 文件列表:
            相对路径 = 路径.relative_to(仓库根目录)
            来源类型, 审核状态 = 来源与状态(相对路径)
            写入器.writerow(
                {
                    "相对仓库路径": 相对路径.as_posix(),
                    "文件大小_字节": 路径.stat().st_size,
                    "SHA256": 文件哈希(路径),
                    "来源类型": 来源类型,
                    "审核状态": 审核状态,
                }
            )
    print(f"已生成清单：{输出文件}；条目数：{len(文件列表)}")


if __name__ == "__main__":
    主程序()
