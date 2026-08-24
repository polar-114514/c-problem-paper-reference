param(
    [switch]$跳过边界稳健对照
)

$ErrorActionPreference = 'Stop'
$主审目录 = Split-Path -Parent $MyInvocation.MyCommand.Path
$候选根目录 = Split-Path -Parent $主审目录
$共同口径目录 = Join-Path $候选根目录 '00_共同口径'
$候选C目录 = Join-Path $候选根目录 '03_候选C_边界稳健模型'

function Invoke-PythonChecked {
    param(
        [Parameter(Mandatory = $true)][string]$脚本,
        [string[]]$参数 = @()
    )
    & python $脚本 @参数
    if ($LASTEXITCODE -ne 0) {
        throw "Python脚本执行失败：$脚本；退出码：$LASTEXITCODE"
    }
}

Write-Host '[1/6] 重建并核对第一问冻结数据'
& (Join-Path $共同口径目录 '一键生成冻结建模数据.ps1')

if (-not $跳过边界稳健对照) {
    Write-Host '[2/6] 重跑Beta/GEE边界稳健对照'
    & (Join-Path $候选C目录 '一键运行候选C.ps1')
} else {
    Write-Host '[2/6] 已按参数跳过Beta/GEE边界稳健对照'
}

Write-Host '[3/6] 统一拟合、显著性检验、诊断和5×5孕妇分组交叉验证'
Invoke-PythonChecked -脚本 (Join-Path $主审目录 '主审统一检验.py')

Write-Host '[4/6] 执行孕妇整簇自助：请求400次'
Invoke-PythonChecked -脚本 (Join-Path $主审目录 '独立簇自助复核_二次Logit混合模型.py') -参数 @('--重复次数', '400', '--随机种子', '20250824')

Write-Host '[5/6] 生成全中文待审核结果表'
Invoke-PythonChecked -脚本 (Join-Path $主审目录 '生成待用户审核结果表.py')

Write-Host '[6/6] 生成待审核材料哈希清单并检查未产生图像'
Invoke-PythonChecked -脚本 (Join-Path $主审目录 '生成待审核材料清单.py')

Write-Host '第一问推荐主线已复现完成；未生成图像，未写入正式归档目录。'
