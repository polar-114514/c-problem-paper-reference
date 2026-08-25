param(
    [string]$PythonExe = "",
    [string]$输出根目录 = ""
)

$ErrorActionPreference = "Stop"
$env:PYTHONDONTWRITEBYTECODE = "1"
$脚本目录 = Split-Path -Parent $MyInvocation.MyCommand.Path
$工作区 = (Resolve-Path (Join-Path $脚本目录 "..\.." )).Path

if ([string]::IsNullOrWhiteSpace($PythonExe)) {
    $PythonExe = (Get-Command python -ErrorAction Stop).Source
}
if ([string]::IsNullOrWhiteSpace($输出根目录)) {
    $运行批次 = Get-Date -Format "yyyyMMdd_HHmmss"
    $输出根目录 = Join-Path $工作区 "99_临时中转\当前四问一键复现\$运行批次"
}

$附件 = Join-Path $工作区 "00_题目与原始资料\02_原始数据\附件.xlsx"
$第一至三问输出 = Join-Path $输出根目录 "第一至三问"
$第四问输出 = Join-Path $输出根目录 "第四问"
$主脚本 = Join-Path $工作区 "01_第一问\05_核心代码\优先问题整改分析.py"
$补充脚本 = Join-Path $工作区 "01_第一问\05_核心代码\补充主审整改.py"
$AFT证据目录 = Join-Path $工作区 "05_公共材料\06_核心代码索引\复现依赖\第二问AFT历史敏感性"
$第四问一键脚本 = Join-Path $工作区 "04_第四问\05_核心代码\一键运行第四问女胎异常判定.ps1"
$核对脚本 = Join-Path $脚本目录 "核对当前四问复现结果.py"

New-Item -ItemType Directory -Path $输出根目录 -Force | Out-Null

Write-Output "[1/4] 运行第一至第三问主分析"
& $PythonExe $主脚本 --附件 $附件 --输出目录 $第一至三问输出 --孕妇整簇重采样次数 400 --检测误差传播次数 400
if ($LASTEXITCODE -ne 0) {
    throw "第一至第三问主分析失败，退出码：$LASTEXITCODE"
}

Write-Output "[2/4] 运行第一至第三问补充审计"
& $PythonExe $补充脚本 --输出目录 $第一至三问输出 --孕妇整簇重采样次数 400 --检测误差传播次数 400 --AFT证据目录 $AFT证据目录
if ($LASTEXITCODE -ne 0) {
    throw "第一至第三问补充审计失败，退出码：$LASTEXITCODE"
}

Write-Output "[3/4] 运行第四问完整建模"
& $第四问一键脚本 -PythonExe $PythonExe -输出目录 $第四问输出

Write-Output "[4/4] 核对复现输出与当前批准核心表"
& $PythonExe $核对脚本 --第一至三问输出 $第一至三问输出 --第四问输出 $第四问输出 --报告目录 (Join-Path $输出根目录 "联合核对")
if ($LASTEXITCODE -ne 0) {
    throw "复现输出与当前核心材料不一致，退出码：$LASTEXITCODE"
}

Write-Output "四问复现及核对完成：$输出根目录"
