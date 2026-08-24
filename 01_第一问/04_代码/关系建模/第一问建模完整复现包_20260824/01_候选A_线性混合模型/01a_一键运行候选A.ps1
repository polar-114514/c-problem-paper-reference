Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$scriptPath = Join-Path $PSScriptRoot '01_运行候选A模型.py'
$pythonCommand = Get-Command python -ErrorAction Stop

& $pythonCommand.Source -c "import numpy,pandas,scipy,statsmodels,patsy,sklearn" 
if ($LASTEXITCODE -ne 0) {
    throw '当前 Python 缺少候选A所需依赖：numpy、pandas、scipy、statsmodels、patsy、scikit-learn。'
}

& $pythonCommand.Source $scriptPath
if ($LASTEXITCODE -ne 0) {
    throw "候选A模型脚本运行失败，退出码：$LASTEXITCODE"
}

Write-Host '候选A已重算完成；本流程不会生成任何图像。'
