$ErrorActionPreference = 'Stop'
$脚本目录 = Split-Path -Parent $MyInvocation.MyCommand.Path
$候选目录 = Split-Path -Parent $脚本目录
$模型脚本 = Join-Path $候选目录 '01_代码\候选B_受控样条混合模型.py'
$冻结文件 = Join-Path (Split-Path -Parent $候选目录) '00_共同口径\冻结数据\第一问主模型冻结样本.csv'

$Python命令 = Get-Command py -ErrorAction Stop
$env:PYTHONDONTWRITEBYTECODE = '1'
& $Python命令.Source -3 $模型脚本 --input $冻结文件 --output $候选目录
if ($LASTEXITCODE -ne 0) {
    throw "候选B模型运行失败，退出码：$LASTEXITCODE"
}

Write-Host '候选B模型运行完成；请查看 02_结果表 与 03_模型报告。'
