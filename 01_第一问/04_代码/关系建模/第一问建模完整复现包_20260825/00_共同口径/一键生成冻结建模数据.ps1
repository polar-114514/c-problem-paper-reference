$ErrorActionPreference = 'Stop'
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$pythonScript = Join-Path $scriptDir '生成第一问冻结建模数据.py'

python $pythonScript
if ($LASTEXITCODE -ne 0) {
    throw "第一问冻结建模数据生成失败，退出码：$LASTEXITCODE"
}

Write-Host "第一问冻结建模数据已生成并通过硬断言。"
