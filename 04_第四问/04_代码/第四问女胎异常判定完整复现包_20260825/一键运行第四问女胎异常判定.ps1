$ErrorActionPreference = 'Stop'
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)
$scriptDirectory = Split-Path -Parent $MyInvocation.MyCommand.Path
$pythonCommand = Get-Command python -ErrorAction Stop
& $pythonCommand.Source (Join-Path $scriptDirectory '第四问女胎异常判定建模.py')
if ($LASTEXITCODE -ne 0) {
    throw "第四问建模或内部审核未通过，退出码：$LASTEXITCODE"
}
Write-Output '第四问建模、验证、自审和哈希已完成；本流程未生成任何图片。'
