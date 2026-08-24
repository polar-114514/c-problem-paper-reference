$ErrorActionPreference = 'Stop'
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)
$packageDirectory = Split-Path -Parent $MyInvocation.MyCommand.Path
$reviewDirectory = Join-Path $packageDirectory '04_主审验收'
$extraDirectory = Join-Path $packageDirectory '07_总控追加复核'
$bootstrapScript = Join-Path $reviewDirectory '独立簇自助复核_二次Logit混合模型.py'
$dataPath = Join-Path $packageDirectory '00_共同口径\冻结数据\第一问主模型冻结样本.csv'

function Invoke-PythonChecked {
    param(
        [Parameter(Mandatory = $true)][string]$Script,
        [string[]]$Arguments = @()
    )
    & python $Script @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Python脚本执行失败：$Script；退出码：$LASTEXITCODE"
    }
}

Write-Output '[1/7] 重跑推荐主线、候选对照、主交叉验证和400次主簇自助'
& (Join-Path $reviewDirectory '一键复现第一问推荐主线.ps1')
if ($LASTEXITCODE -ne 0) {
    throw "第一问推荐主线复现失败，退出码：$LASTEXITCODE"
}

Write-Output '[2/7] 重跑200次孕妇整簇自助'
Invoke-PythonChecked -Script $bootstrapScript -Arguments @('--数据', $dataPath, '--输出目录', (Join-Path $extraDirectory '自助收敛性\B200'), '--重复次数', '200', '--随机种子', '20250824')

Write-Output '[3/7] 重跑400次孕妇整簇自助'
Invoke-PythonChecked -Script $bootstrapScript -Arguments @('--数据', $dataPath, '--输出目录', (Join-Path $extraDirectory '自助收敛性\B400'), '--重复次数', '400', '--随机种子', '20250824')

Write-Output '[4/7] 重跑800次孕妇整簇自助'
Invoke-PythonChecked -Script $bootstrapScript -Arguments @('--数据', $dataPath, '--输出目录', (Join-Path $extraDirectory '自助收敛性\B800'), '--重复次数', '800', '--随机种子', '20250824')

Write-Output '[5/7] 汇总200/400/800次自助收敛性'
Invoke-PythonChecked -Script (Join-Path $extraDirectory '汇总第一问自助收敛性.py')

Write-Output '[6/7] 重跑3至10折计算设置敏感性'
Invoke-PythonChecked -Script (Join-Path $extraDirectory '检验第一问计算设置稳定性.py')

Write-Output '[7/7] 生成参数来源、验收清单、哈希和PASS记录'
Invoke-PythonChecked -Script (Join-Path $extraDirectory '生成第一问总控复核材料.py') -Arguments @('--要求正式一致')

Write-Output '第一问完整复现和总控审核完成；本流程未生成图片。'
