param(
    [string]$InputXlsx = "",
    [string]$OutputRoot = "",
    [string]$PythonExe = "C:\python\python.exe",
    [string]$NodeExe = "node",
    [string]$MatlabExe = "C:\Program Files\MATLAB\R2021b\bin\matlab.exe",
    [string]$ExpectedSourceSha256 = "14827156218bd4f7e4f16db4aa6d9f757c6648379e038ae6c6b58383648614af"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Assert-FileExists {
    param([string]$Path, [string]$Label)
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw "$($Label)不存在：$Path"
    }
}

function Invoke-NativeStep {
    param(
        [string]$Label,
        [string]$FilePath,
        [string[]]$Arguments
    )
    Write-Host ""
    Write-Host "[$Label]"
    & $FilePath @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$($Label)失败，退出码：$LASTEXITCODE"
    }
}

$CodeDir = $PSScriptRoot
$WorkspaceRoot = [System.IO.Path]::GetFullPath((Join-Path $CodeDir "..\..\.."))
if ([string]::IsNullOrWhiteSpace($InputXlsx)) {
    $InputXlsx = Join-Path $WorkspaceRoot "00_题目与原始资料\02_原始数据\附件.xlsx"
}
if ([string]::IsNullOrWhiteSpace($OutputRoot)) {
    $stamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $OutputRoot = Join-Path $WorkspaceRoot "99_临时中转\第一问数据审计一键复现_$stamp"
}

$InputXlsx = [System.IO.Path]::GetFullPath($InputXlsx)
$OutputRoot = [System.IO.Path]::GetFullPath($OutputRoot)

if (Test-Path -LiteralPath $OutputRoot) {
    throw "输出目录已存在，为避免覆盖已有结果，请换一个新目录：$OutputRoot"
}

$scripts = [ordered]@{
    "只读提取" = Join-Path $CodeDir "只读提取脚本_extract_source.mjs"
    "主审计" = Join-Path $CodeDir "主审计脚本_audit_q1.py"
    "独立数据审计" = Join-Path $CodeDir "独立数据审计_q1_audit.py"
    "独立JS复算" = Join-Path $CodeDir "独立复算脚本_verify_audit.mjs"
    "中文表头规范化" = Join-Path $CodeDir "中文表头规范化_normalize_headers.mjs"
    "MATLAB绘图数据导出" = Join-Path $CodeDir "导出MATLAB绘图数据.py"
    "MATLAB SVG制图" = Join-Path $CodeDir "build_q1_svgs.m"
    "一键复现入口" = $PSCommandPath
    "Python依赖锁定" = Join-Path $CodeDir "requirements_数据审计.txt"
}

Assert-FileExists $InputXlsx "原始Excel"
Assert-FileExists $PythonExe "Python"
Assert-FileExists $NodeExe "Node.js"
Assert-FileExists $MatlabExe "MATLAB"
foreach ($item in $scripts.GetEnumerator()) {
    Assert-FileExists $item.Value $item.Key
}

$artifactPackage = Join-Path $CodeDir "node_modules\@oai\artifact-tool\package.json"
Assert-FileExists $artifactPackage "@oai/artifact-tool"

$actualSourceSha256 = (Get-FileHash -LiteralPath $InputXlsx -Algorithm SHA256).Hash.ToLowerInvariant()
if (-not [string]::IsNullOrWhiteSpace($ExpectedSourceSha256) -and
    $actualSourceSha256 -ne $ExpectedSourceSha256.ToLowerInvariant()) {
    throw "原始Excel的SHA-256与本次审计快照不一致。实际：$actualSourceSha256；期望：$ExpectedSourceSha256"
}

Invoke-NativeStep "Python依赖检查" $PythonExe @(
    "-c",
    "import numpy,pandas,scipy,matplotlib,openpyxl; print('Python dependencies available')"
)
Invoke-NativeStep "Node.js检查" $NodeExe @("--version")

$sourceSnapshot = Join-Path $OutputRoot "01_源数据快照"
$mainAudit = Join-Path $OutputRoot "02_主审计原始输出"
$independentAudit = Join-Path $OutputRoot "03_独立数据审计"
$verification = Join-Path $OutputRoot "04_独立复算校验"
$chineseRoot = Join-Path $OutputRoot "05_中文结果表"
$processingDir = Join-Path $chineseRoot "02_数据处理\数据质量与结构审计"
$resultDir = Join-Path $chineseRoot "05_结果数据\数据质量与结构审计"
$plotData = Join-Path $OutputRoot "06_MATLAB绘图数据"
$svgDir = Join-Path $OutputRoot "07_MATLAB_SVG图表"

foreach ($dir in @($sourceSnapshot, $mainAudit, $independentAudit, $verification, $processingDir, $resultDir, $plotData, $svgDir)) {
    New-Item -ItemType Directory -Path $dir -Force | Out-Null
}

Invoke-NativeStep "1/7 只读提取Excel" $NodeExe @(
    $scripts["只读提取"], $InputXlsx, $sourceSnapshot
)

$maleJson = Join-Path $sourceSnapshot "01_男胎检测数据.json"
$mainSummary = Join-Path $mainAudit "audit_summary.json"
Invoke-NativeStep "2/7 主审计计算" $PythonExe @(
    $scripts["主审计"], "--source-json", $maleJson, "--output", $mainAudit
)

$independentSummary = Join-Path $independentAudit "q1_audit_summary.json"
Invoke-NativeStep "3/7 独立Python审计" $PythonExe @(
    $scripts["独立数据审计"], "--input", $InputXlsx, "--output", $independentAudit
)

$jsVerification = Join-Path $verification "independent_js_verification.json"
Invoke-NativeStep "4/7 独立JavaScript复算" $NodeExe @(
    $scripts["独立JS复算"], $maleJson, $mainSummary, $jsVerification
)

$resultMap = [ordered]@{
    "batch_comparison.csv" = "683前后批次对比.csv"
    "sample_counts.csv" = "不同口径样本量.csv"
    "event_hierarchy_counts.csv" = "事件层级口径对照.csv"
    "draw_number_gaps.csv" = "抽血序号缺口.csv"
    "technical_repeat_groups.csv" = "多记录抽血事件组.csv"
    "missingness.csv" = "缺失值统计.csv"
    "date_gestation_outliers_gt14d.csv" = "日期孕周偏差超过14天.csv"
    "audit_issues.csv" = "审计问题清单.csv"
    "robust_univariate_flags.csv" = "稳健单变量异常标志.csv"
    "correlation_decomposition.csv" = "相关结构分解.csv"
    "longitudinal_progression_checks.csv" = "纵向进展检查.csv"
}
foreach ($entry in $resultMap.GetEnumerator()) {
    $sourceFile = Join-Path $mainAudit $entry.Key
    Assert-FileExists $sourceFile $entry.Key
    Copy-Item -LiteralPath $sourceFile -Destination (Join-Path $resultDir $entry.Value)
}
Copy-Item -LiteralPath (Join-Path $independentAudit "q1_row_flags.csv") -Destination (Join-Path $processingDir "第一问_逐行数据质量标志.csv")

Invoke-NativeStep "5/7 中文表头规范化" $NodeExe @(
    $scripts["中文表头规范化"], "--root=$chineseRoot", "--apply"
)

Invoke-NativeStep "6/7 导出MATLAB确定性绘图数据" $PythonExe @(
    $scripts["MATLAB绘图数据导出"],
    "--source-json", $maleJson,
    "--audit-summary", $mainSummary,
    "--output", $plotData
)

$oldCodeDir = $env:Q1_CODE_DIR
$oldPlotData = $env:Q1_PLOT_DATA_DIR
$oldSvgOutput = $env:Q1_SVG_OUTPUT_DIR
try {
    $env:Q1_CODE_DIR = $CodeDir
    $env:Q1_PLOT_DATA_DIR = $plotData
    $env:Q1_SVG_OUTPUT_DIR = $svgDir
    $matlabCommand = "addpath(getenv('Q1_CODE_DIR')); build_q1_svgs(getenv('Q1_PLOT_DATA_DIR'), getenv('Q1_SVG_OUTPUT_DIR'));"
    Invoke-NativeStep "7/7 MATLAB生成纯矢量SVG" $MatlabExe @("-batch", $matlabCommand)
}
finally {
    $env:Q1_CODE_DIR = $oldCodeDir
    $env:Q1_PLOT_DATA_DIR = $oldPlotData
    $env:Q1_SVG_OUTPUT_DIR = $oldSvgOutput
}

$sourceManifest = Get-Content -LiteralPath (Join-Path $sourceSnapshot "manifest.json") -Raw -Encoding UTF8 | ConvertFrom-Json
$independent = Get-Content -LiteralPath $independentSummary -Raw -Encoding UTF8 | ConvertFrom-Json
$jsCheck = Get-Content -LiteralPath $jsVerification -Raw -Encoding UTF8 | ConvertFrom-Json
$svgCheck = Get-Content -LiteralPath (Join-Path $svgDir "SVG制图校验.json") -Raw -Encoding UTF8 | ConvertFrom-Json

if ($sourceManifest.sourceSha256.ToLowerInvariant() -ne $actualSourceSha256) { throw "提取清单中的源文件哈希不一致。" }
if ([int]$independent.workbook_structure.male_data_rows -ne 1082) { throw "男胎记录数不是1082。" }
if ([int]$independent.modelability_without_fitting.primary_rows -ne 674) { throw "主参考样本不是674条。" }
if ([int]$independent.cohorts.pre683_clinical_through_25w0_sensitivity.rows -ne 670) { throw "25周0天敏感性样本不是670条。" }
if ([int]$jsCheck.passed -ne 222 -or [int]$jsCheck.failed -ne 0) { throw "独立JS复算222项未全部通过。" }
if ([int]$svgCheck.svg_count -ne 8 -or -not [bool]$svgCheck.all_vector) { throw "MATLAB SVG校验未通过。" }

$csvFiles = @(Get-ChildItem -LiteralPath $chineseRoot -Recurse -Filter "*.csv" -File)
if ($csvFiles.Count -ne 12) { throw "中文结果CSV数量不是12个，实际：$($csvFiles.Count)" }
foreach ($csv in $csvFiles) {
    $header = (Get-Content -LiteralPath $csv.FullName -TotalCount 1 -Encoding UTF8).TrimStart([char]0xFEFF)
    if ($header -match "[A-Za-z]+_[A-Za-z0-9_]+") {
        throw "CSV表头仍包含英文代码式字段：$($csv.FullName)"
    }
}

$pythonVersionJson = & $PythonExe -c "import json,sys,numpy,pandas,scipy,matplotlib,openpyxl; print(json.dumps({'python':sys.version.split()[0],'numpy':numpy.__version__,'pandas':pandas.__version__,'scipy':scipy.__version__,'matplotlib':matplotlib.__version__,'openpyxl':openpyxl.__version__}))"
if ($LASTEXITCODE -ne 0) { throw "无法记录Python环境版本。" }
$pythonVersions = $pythonVersionJson | ConvertFrom-Json
$nodeVersion = ((& $NodeExe --version) | Select-Object -Last 1).Trim()
$artifactVersion = (Get-Content -LiteralPath $artifactPackage -Raw -Encoding UTF8 | ConvertFrom-Json).version

$codeHashes = foreach ($entry in $scripts.GetEnumerator()) {
    [ordered]@{
        用途 = $entry.Key
        文件 = [System.IO.Path]::GetFileName($entry.Value)
        SHA256 = (Get-FileHash -LiteralPath $entry.Value -Algorithm SHA256).Hash.ToLowerInvariant()
    }
}

$outputFiles = foreach ($file in Get-ChildItem -LiteralPath $OutputRoot -Recurse -File | Sort-Object FullName) {
    if ($file.Name -eq "复现结果清单.json") { continue }
    [ordered]@{
        相对路径 = [System.IO.Path]::GetRelativePath($OutputRoot, $file.FullName)
        字节数 = $file.Length
        SHA256 = (Get-FileHash -LiteralPath $file.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
    }
}

$manifest = [ordered]@{
    用途 = "第一问数据质量与结构审计的一键复现；不包含正式模型拟合"
    复现时间 = (Get-Date).ToString("o")
    源Excel = [ordered]@{
        路径 = $InputXlsx
        SHA256 = $actualSourceSha256
        工作表 = "男胎检测数据"
        使用区域 = "A1:AE1083"
    }
    环境 = [ordered]@{
        Python = $pythonVersions
        Node = $nodeVersion
        artifact_tool = $artifactVersion
        MATLAB = $svgCheck.matlab_version
        MATLAB字体 = $svgCheck.font
    }
    固定口径 = [ordered]@{
        原始记录数 = 1082
        孕妇数 = 267
        抽血事件主键 = "孕妇代码B+检测抽血次数I"
        抽血事件数 = 1021
        检测会话主键 = "孕妇代码B+检测抽血次数I+检测日期H"
        检测会话数 = 1063
        严格元数据组数_B加I加H加J = 1064
        多检测抽血事件 = "40组/101条"
        同日多记录检测会话 = "19组/38条"
        严格元数据一致重复 = "18组/36条"
        主参考样本 = "序号<683且孕周位于[10,26)，674条"
        敏感性样本 = "序号<683且孕周不超过25周0天，670条"
        硬删除记录数 = 0
    }
    验收 = [ordered]@{
        独立JS复算 = "222/222通过"
        中文表头CSV数 = $csvFiles.Count
        MATLAB_SVG数 = [int]$svgCheck.svg_count
        SVG全为纯矢量 = [bool]$svgCheck.all_vector
        正式建模已开始 = $false
    }
    代码SHA256 = $codeHashes
    产物SHA256 = $outputFiles
}

$manifestPath = Join-Path $OutputRoot "复现结果清单.json"
$manifest | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $manifestPath -Encoding UTF8

Write-Host ""
Write-Host "复现成功：$OutputRoot"
Write-Host "验收：1082条原始记录，主参考样本674条，25周0天敏感性样本670条，事件层级1021/1063/1064，JS 222/222，MATLAB纯矢量SVG 8/8。"
