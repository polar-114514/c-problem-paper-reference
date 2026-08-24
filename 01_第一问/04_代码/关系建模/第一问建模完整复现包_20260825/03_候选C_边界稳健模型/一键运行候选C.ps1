$ErrorActionPreference = 'Stop'
$scriptPath = Join-Path $PSScriptRoot '运行候选C_边界稳健模型.py'
python $scriptPath
if ($LASTEXITCODE -ne 0) {
    throw "候选C运行失败，退出码：$LASTEXITCODE"
}
