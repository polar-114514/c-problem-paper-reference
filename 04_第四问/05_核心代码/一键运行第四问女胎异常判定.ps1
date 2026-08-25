param(
    [string]$PythonExe = "",
    [string]$输出目录 = ""
)

$ErrorActionPreference = "Stop"
$env:PYTHONDONTWRITEBYTECODE = "1"
$脚本目录 = Split-Path -Parent $MyInvocation.MyCommand.Path
$工作区 = (Resolve-Path (Join-Path $脚本目录 "..\.." )).Path

if ([string]::IsNullOrWhiteSpace($PythonExe)) {
    $PythonExe = (Get-Command python -ErrorAction Stop).Source
}
if ([string]::IsNullOrWhiteSpace($输出目录)) {
    $输出目录 = Join-Path $工作区 "99_临时中转\当前四问一键复现\第四问"
}

$附件 = Join-Path $工作区 "00_题目与原始资料\02_原始数据\附件.xlsx"
$题目 = Join-Path $工作区 "00_题目与原始资料\01_题目原文\C题.pdf"
$合同 = Join-Path $脚本目录 "第四问题意合同与候选设计.md"
$主脚本 = Join-Path $脚本目录 "第四问女胎异常判定建模.py"

& $PythonExe $主脚本 --附件 $附件 --题目 $题目 --合同 $合同 --输出目录 $输出目录
if ($LASTEXITCODE -ne 0) {
    throw "第四问建模失败，退出码：$LASTEXITCODE"
}

Write-Output "第四问复现完成：$输出目录"
