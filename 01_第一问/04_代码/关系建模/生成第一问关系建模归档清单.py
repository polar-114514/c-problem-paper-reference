from __future__ import annotations

import csv
import hashlib
import json
import os
import re
from datetime import datetime
from pathlib import Path


脚本路径 = Path(__file__).resolve()
第一问目录 = 脚本路径.parents[2]
临时快照环境值 = os.environ.get("Q1_TEMP_SNAPSHOT")
临时快照目录 = Path(临时快照环境值) if 临时快照环境值 else None
完整复现包目录 = 脚本路径.parent / "第一问建模完整复现包_20260824"
输出清单 = 第一问目录 / "08_复核记录" / "关系建模" / "第一问关系建模归档清单.json"

关系建模目录 = [
    第一问目录 / "01_分析思路" / "关系建模",
    第一问目录 / "02_数据处理" / "关系建模",
    第一问目录 / "03_模型与推导" / "关系建模",
    第一问目录 / "04_代码" / "关系建模",
    第一问目录 / "05_结果数据" / "关系建模",
    第一问目录 / "06_图表" / "关系建模",
    第一问目录 / "07_手写正文素材" / "关系建模",
    第一问目录 / "08_复核记录" / "关系建模",
]
图像扩展名 = {".svg", ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tif", ".tiff", ".webp"}


def 文件哈希(路径: Path) -> str:
    摘要 = hashlib.sha256()
    with 路径.open("rb") as 文件:
        for 数据块 in iter(lambda: 文件.read(1024 * 1024), b""):
            摘要.update(数据块)
    return 摘要.hexdigest()


def 文件映射(目录: Path) -> dict[str, str]:
    return {
        str(路径.relative_to(目录)): 文件哈希(路径)
        for 路径 in sorted(目录.rglob("*"))
        if 路径.is_file()
    }


def 主程序() -> None:
    缺失目录 = [str(目录) for 目录 in 关系建模目录 if not 目录.is_dir()]
    if 缺失目录:
        raise FileNotFoundError(f"正式归档目录不完整：{缺失目录}")

    文件列表: list[Path] = []
    for 目录 in 关系建模目录:
        文件列表.extend(
            路径
            for 路径 in 目录.rglob("*")
            if 路径.is_file() and 路径.resolve() != 输出清单.resolve()
        )
    文件列表 = sorted(set(文件列表))

    无中文语义表头: list[dict[str, object]] = []
    for 路径 in (路径 for 路径 in 文件列表 if 路径.suffix.lower() == ".csv"):
        with 路径.open("r", encoding="utf-8-sig", newline="") as 文件:
            表头 = next(csv.reader(文件))
        异常列 = [列 for 列 in 表头 if not re.search(r"[\u4e00-\u9fff]", 列)]
        if 异常列:
            无中文语义表头.append(
                {"文件": str(路径.relative_to(第一问目录)), "异常列": 异常列}
            )

    图像文件 = [
        str(路径.relative_to(第一问目录))
        for 路径 in 文件列表
        if 路径.suffix.lower() in 图像扩展名
    ]
    缓存文件 = [
        str(路径.relative_to(第一问目录))
        for 路径 in 文件列表
        if 路径.suffix.lower() == ".pyc" or "__pycache__" in 路径.parts
    ]

    复现包映射 = 文件映射(完整复现包目录)
    if 临时快照目录 is not None and 临时快照目录.is_dir():
        临时映射 = 文件映射(临时快照目录)
        复现包差异 = {
            "仅临时快照存在": sorted(set(临时映射) - set(复现包映射)),
            "仅正式复现包存在": sorted(set(复现包映射) - set(临时映射)),
            "同路径哈希不同": sorted(
                路径
                for 路径 in set(临时映射) & set(复现包映射)
                if 临时映射[路径] != 复现包映射[路径]
            ),
        }
    else:
        复现包差异 = "临时快照当前不存在，未执行实时逐文件比较"

    if 无中文语义表头:
        raise RuntimeError(f"发现无中文语义的CSV表头：{无中文语义表头}")
    if 图像文件:
        raise RuntimeError(f"关系建模归档中发现本阶段不应生成的图像：{图像文件}")
    if 缓存文件:
        raise RuntimeError(f"关系建模归档中发现Python缓存：{缓存文件}")
    if isinstance(复现包差异, dict) and any(复现包差异.values()):
        raise RuntimeError(f"正式复现包与临时审核快照不一致：{复现包差异}")

    清单 = {
        "生成时间": datetime.now().astimezone().isoformat(timespec="seconds"),
        "归档状态": "主审及用户审核通过，已正式归档",
        "第一问目录": str(第一问目录),
        "正式归档文件数_不含本清单": len(文件列表),
        "完整复现包文件数": len(复现包映射),
        "完整复现包与临时快照差异": 复现包差异,
        "CSV无中文语义表头文件数": 0,
        "本阶段新增图像文件数": 0,
        "Python缓存文件数": 0,
        "文件": [
            {
                "相对第一问目录路径": str(路径.relative_to(第一问目录)),
                "文件大小_字节": 路径.stat().st_size,
                "SHA256": 文件哈希(路径),
            }
            for 路径 in 文件列表
        ],
    }
    输出清单.write_text(json.dumps(清单, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "输出清单": str(输出清单),
                "归档文件数": len(文件列表),
                "复现包文件数": len(复现包映射),
                "表头异常": 0,
                "图像文件": 0,
                "缓存文件": 0,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    主程序()
