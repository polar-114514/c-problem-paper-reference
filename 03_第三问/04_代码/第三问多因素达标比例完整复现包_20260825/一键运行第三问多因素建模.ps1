$ErrorActionPreference = 'Stop'
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$modelScript = Join-Path $scriptDir '第三问多因素达标比例建模.py'

& python $modelScript
if ($LASTEXITCODE -ne 0) {
    throw "第三问多因素达标比例建模失败，退出码：$LASTEXITCODE"
}

Write-Host '第三问建模、验证、自审和哈希已完成；本流程未生成任何图片。'
