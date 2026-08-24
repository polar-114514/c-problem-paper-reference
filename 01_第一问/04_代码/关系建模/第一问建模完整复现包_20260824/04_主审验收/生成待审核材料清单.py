from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path


脚本目录 = Path(__file__).resolve().parent
候选根目录 = 脚本目录.parent
待审核目录 = 候选根目录 / "05_待用户审核"
制图提示目录 = 候选根目录 / "06_制图提示词"
输出清单 = 待审核目录 / "待用户审核材料清单.json"
图像扩展名 = {".svg", ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tif", ".tiff", ".webp", ".pdf"}


def sha256(路径: Path) -> str:
    摘要 = hashlib.sha256()
    with 路径.open("rb") as 文件:
        for 数据块 in iter(lambda: 文件.read(1024 * 1024), b""):
            摘要.update(数据块)
    return 摘要.hexdigest()


def 收集文件(目录: Path) -> list[dict[str, object]]:
    结果: list[dict[str, object]] = []
    for 路径 in sorted(目录.rglob("*")):
        if not 路径.is_file() or 路径.resolve() == 输出清单.resolve():
            continue
        结果.append(
            {
                "相对候选根目录路径": str(路径.relative_to(候选根目录)),
                "文件大小_字节": 路径.stat().st_size,
                "SHA256": sha256(路径),
            }
        )
    return 结果


def main() -> None:
    待审核目录.mkdir(parents=True, exist_ok=True)
    文件记录 = 收集文件(待审核目录) + 收集文件(制图提示目录)
    图像文件 = [
        str(路径.relative_to(候选根目录))
        for 路径 in 候选根目录.rglob("*")
        if 路径.is_file() and 路径.suffix.lower() in 图像扩展名
    ]
    if 图像文件:
        raise RuntimeError(f"候选目录中发现不应存在的图像文件：{图像文件}")

    清单 = {
        "生成时间": datetime.now().astimezone().isoformat(timespec="seconds"),
        "候选根目录": str(候选根目录),
        "材料状态": "待用户审核，未正式归档",
        "待审核及制图提示文件数": len(文件记录),
        "图像文件数": 0,
        "文件": 文件记录,
    }
    输出清单.write_text(json.dumps(清单, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"输出清单": str(输出清单), "文件数": len(文件记录), "图像文件数": 0}, ensure_ascii=False))


if __name__ == "__main__":
    main()
