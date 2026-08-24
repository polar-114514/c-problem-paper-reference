$ErrorActionPreference = 'Stop'
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$modelScript = Join-Path $scriptDir '第二问无自拟参数重构.py'

& python $modelScript
if ($LASTEXITCODE -ne 0) {
    throw "第二问无自拟参数重构失败，退出码：$LASTEXITCODE"
}

Write-Host '第二问重构、验证、自审和哈希已完成；本流程未生成任何图片。'
