import fs from "node:fs/promises";
import path from "node:path";
import crypto from "node:crypto";
import { fileURLToPath } from "node:url";
import { Workbook } from "@oai/artifact-tool";

const applyChanges = process.argv.includes("--apply");
const rootArg = process.argv.find((arg) => arg.startsWith("--root="));
const scriptDir = path.dirname(fileURLToPath(import.meta.url));
const root = rootArg
  ? path.resolve(rootArg.slice("--root=".length))
  : path.resolve(scriptDir, "../..");

const csvDirs = [
  path.join(root, "02_数据处理/数据质量与结构审计"),
  path.join(root, "05_结果数据/数据质量与结构审计"),
];

const headerMap = {
  "序号": "序号",
  "孕妇代码": "孕妇代码",
  "检测抽血次数": "检测抽血次数",
  "检测孕周": "检测孕周（原始值）",
  "孕妇BMI": "孕妇BMI（kg/m²）",
  "Y染色体浓度": "Y染色体浓度（比例，0–1）",
  seq: "序号",
  woman_id: "孕妇代码",
  draw_no: "检测抽血次数",
  row_id: "检测记录标识",
  draw_id: "抽血事件标识",
  draw_assay_count: "同一抽血事件检测记录数（条）",
  date_session_id: "孕妇－检测日期会话标识（B+H）",
  assay_session_id: "检测会话标识（B+I+H）",
  assay_session_record_count: "同一检测会话记录数（条）",
  strict_metadata_id: "严格元数据一致组标识（B+I+H+J）",
  gest_raw: "检测孕周（原始值）",
  gest_days: "解析检测孕周（天）",
  gest_weeks: "解析检测孕周（周）",
  bmi_calculated: "由身高体重重算的孕妇BMI（kg/m²）",
  bmi_abs_error: "孕妇BMI与重算BMI绝对差（kg/m²）",
  date_gest_delta_days: "日期推算孕周－记录孕周（天）",
  storage687: "序号687日期存储切换标志",
  primary_include: "截至25周0天敏感性样本纳入标志",
  sensitivity_through_25w0_include: "截至25周0天敏感性样本纳入标志",
  core_missing_or_invalid: "核心分析字段缺失或非法标志",
  outside_10w0_25w0: "孕周不在10周0天至25周0天内标志",
  outside_10w0_25w6: "孕周不在10周0天至25周6天内标志",
  lmp_unusable: "末次月经无法解析标志",
  date_delta_negative: "日期推算孕周小于记录孕周标志",
  date_delta_abs_gt21: "日期推算孕周与记录孕周绝对差大于21天标志",
  bmi_formula_abs_gt_0_01: "BMI重算绝对差大于0.01kg/m²标志",
  unique_reads_gt_raw: "唯一比对读段数大于原始读段数标志",
  read_formula_abs_error_gt2: "唯一比对读段经验式绝对误差大于2条标志",
  gc_below_40pct: "GC含量低于40%标志",
  gc_below_39pct: "GC含量低于39%标志",
  filter_ratio_above_p99: "过滤读段率高于全样本第99百分位标志",
  y_tukey_outlier: "Y染色体浓度Tukey箱线图异常值标志",
  y_below_4pct: "Y染色体浓度低于4%标志",
  same_draw_multitest: "同一抽血事件存在多条检测记录标志",
  same_session_multirecord: "同一检测会话存在多条记录标志",
  draw_gestation_conflict: "同一抽血事件孕周冲突标志",
  primary_exclusion_reasons: "参考主样本排除原因",
  mark_only_reasons: "仅标记不排除原因",
  audit_action: "审计处置分类",
  batch: "分析批次（683断点口径）",
  records: "检测记录数（条）",
  women: "孕妇人数（人）",
  draws: "抽血事件数（次）",
  week_mean: "解析检测孕周均值（周）",
  week_min: "解析检测孕周最小值（周）",
  week_max: "解析检测孕周最大值（周）",
  bmi_mean: "孕妇BMI均值（kg/m²）",
  bmi_min: "孕妇BMI最小值（kg/m²）",
  bmi_max: "孕妇BMI最大值（kg/m²）",
  y_mean: "Y染色体浓度均值（比例，0–1）",
  y_min: "Y染色体浓度最小值（比例，0–1）",
  y_max: "Y染色体浓度最大值（比例，0–1）",
  multi_record_draws: "含多条检测记录的抽血事件数（次）",
  sample: "样本口径",
  scope: "统计范围",
  key_notation: "事件键记号",
  definition: "事件键定义",
  recommended_role: "建议统计角色",
  unique_events: "唯一事件或组数（个）",
  multi_groups: "多记录组数（个）",
  multi_rows: "多记录组内记录数（条）",
  pooled_within_y_sd: "组内Y染色体浓度合并标准差（比例，0–1）",
  groups_crossing_4pct: "跨越4%阈值的组数（个）",
  size_distribution: "组大小分布（JSON）",
  observed: "实际出现的抽血序号列表",
  expected: "按跨度应有抽血序号列表",
  seqs: "源数据序号列表",
  gest_days_nunique: "组内解析检测孕周唯一值数（个）",
  test_date_nunique: "组内检测日期唯一值数（个）",
  weight_kg_nunique: "组内体重唯一值数（个）",
  bmi_nunique: "组内孕妇BMI唯一值数（个）",
  age_nunique: "组内年龄唯一值数（个）",
  height_cm_nunique: "组内身高唯一值数（个）",
  conception_nunique: "组内受孕方式唯一值数（个）",
  crosses_4pct: "组内Y染色体浓度跨越4%阈值标志",
  y_range: "组内Y染色体浓度极差（比例，0–1）",
  any_metadata_inconsistency: "组内任一元数据不一致标志",
  column: "源数据字段代码",
  label: "源数据字段中文名",
  missing: "缺失记录数（条）",
  pct: "缺失比例（0–1）",
  lmp_raw: "末次月经（原始值）",
  test_date_raw: "检测日期（原始值）",
  date_gest_error_days: "日期推算孕周－记录孕周（天）",
  severity: "严重等级",
  issue: "审计问题",
  evidence: "证据摘要",
  action: "处理建议",
  value: "异常变量原始取值",
  variable: "变量代码",
  robust_z: "稳健Z分数（无量纲）",
  level: "相关结构层级",
  method: "相关分析方法",
  n: "相关计算有效样本量",
  r: "相关系数r（无量纲）",
  p: "双侧检验P值",
  from_draw: "前一检测抽血次数",
  to_draw: "后一检测抽血次数",
  date_delta_days: "相邻抽血检测日期差（天）",
  gest_delta_days: "相邻抽血记录孕周差（天）",
  delta_mismatch_days: "相邻抽血日期差－孕周差（天）",
  gestation_nonincreasing: "相邻抽血记录孕周未递增标志",
  date_nonincreasing: "相邻抽血检测日期未递增标志",
};

function stripBom(value) {
  return String(value ?? "").replace(/^\uFEFF/, "");
}

function boolValue(value) {
  if (value === true || value === 1) return true;
  return ["true", "1", "yes", "是"].includes(String(value ?? "").trim().toLowerCase());
}

function csvCell(value) {
  if (value === null || value === undefined) return "";
  let text;
  if (typeof value === "boolean") text = value ? "True" : "False";
  else if (value instanceof Date) text = value.toISOString();
  else text = String(value);
  return /[",\r\n]/.test(text) ? `"${text.replaceAll('"', '""')}"` : text;
}

const csvLine = (values) => values.map(csvCell).join(",");
const csvDocument = (rows) => `\uFEFF${rows.map(csvLine).join("\r\n")}\r\n`;
const sha256 = (text) => crypto.createHash("sha256").update(text, "utf8").digest("hex");

function splitCsvText(text) {
  const withoutBom = text.replace(/^\uFEFF/, "");
  const match = withoutBom.match(/\r\n|\n|\r/);
  if (!match || match.index === undefined) return { eol: "\r\n", body: "" };
  return { eol: match[0], body: withoutBom.slice(match.index + match[0].length) };
}

function translateHeaders(headers, rowFlags) {
  return headers.map((raw) => {
    const clean = stripBom(raw).trim();
    if (clean === "batch683") {
      return rowFlags ? "序号683及以后批次标志" : "分析批次（683断点口径）";
    }
    if (Object.hasOwn(headerMap, clean)) return headerMap[clean];
    if (/[\u3400-\u9fff]/u.test(clean)) return clean;
    throw new Error(`缺少中文字段映射: ${clean}`);
  });
}

const files = [];
for (const dir of csvDirs) {
  const entries = await fs.readdir(dir, { withFileTypes: true });
  for (const entry of entries) {
    if (entry.isFile() && entry.name.toLowerCase().endsWith(".csv")) files.push(path.join(dir, entry.name));
  }
}
files.sort((a, b) => a.localeCompare(b, "zh-CN"));

const summaries = [];
for (const file of files) {
  const originalText = await fs.readFile(file, "utf8");
  const workbook = await Workbook.fromCSV(originalText, { sheetName: "数据" });
  const sheet = workbook.worksheets.getItem("数据");
  const values = sheet.getUsedRange(true).values;
  const originalHeaders = values.length ? values[0].map(stripBom) : [];
  let outputText = originalText;
  let translatedHeaders;
  let mainReferenceCount = null;
  let sensitivity25w0Count = null;
  let bodyPreserved = null;
  const primaryIdx = originalHeaders.indexOf("primary_include");
  const alreadyNormalizedMainIdx = originalHeaders.indexOf("参考主样本纳入标志");

  if (primaryIdx >= 0) {
    const index = Object.fromEntries(originalHeaders.map((name, idx) => [name, idx]));
    const required = ["core_missing_or_invalid", "batch683", "outside_10w0_25w0", "outside_10w0_25w6", "primary_exclusion_reasons", "mark_only_reasons", "audit_action"];
    for (const name of required) if (!(name in index)) throw new Error(`${path.basename(file)}缺少字段: ${name}`);
    const baseTranslated = translateHeaders(originalHeaders, true);
    const explicitSensitivityIdx = originalHeaders.indexOf("sensitivity_through_25w0_include");
    if (explicitSensitivityIdx >= 0) {
      translatedHeaders = [...baseTranslated];
      translatedHeaders[primaryIdx] = "参考主样本纳入标志";
      translatedHeaders[explicitSensitivityIdx] = "截至25周0天敏感性样本纳入标志";
    } else {
      translatedHeaders = [
        ...baseTranslated.slice(0, primaryIdx),
        "参考主样本纳入标志",
        "截至25周0天敏感性样本纳入标志",
        ...baseTranslated.slice(primaryIdx + 1),
      ];
    }
    const newRows = [translatedHeaders];
    mainReferenceCount = 0;
    sensitivity25w0Count = 0;
    for (const originalRow of values.slice(1)) {
      const row = [...originalRow];
      const coreInvalid = boolValue(row[index.core_missing_or_invalid]);
      const inBatch683 = boolValue(row[index.batch683]);
      const outside25w0 = boolValue(row[index.outside_10w0_25w0]);
      const outside25w6 = boolValue(row[index.outside_10w0_25w6]);
      const mainReference = !coreInvalid && !inBatch683 && !outside25w6;
      const sensitivity25w0 = !coreInvalid && !inBatch683 && !outside25w0;
      const reasons = [];
      if (coreInvalid) reasons.push("hard_core_invalid");
      if (inBatch683) reasons.push("primary_batch683");
      if (outside25w6) reasons.push("primary_outside_10w0_25w6");
      row[index.primary_exclusion_reasons] = reasons.join(";");
      const marked = String(row[index.mark_only_reasons] ?? "") !== "";
      row[index.audit_action] = coreInvalid ? "HARD_EXCLUDE" : !mainReference ? "EXCLUDE_PRIMARY_KEEP_SENSITIVITY" : marked ? "KEEP_PRIMARY_MARKED" : "KEEP_PRIMARY_CLEAN";
      if (explicitSensitivityIdx >= 0) {
        row[primaryIdx] = mainReference;
        row[explicitSensitivityIdx] = sensitivity25w0;
        newRows.push(row);
      } else {
        newRows.push([...row.slice(0, primaryIdx), mainReference, sensitivity25w0, ...row.slice(primaryIdx + 1)]);
      }
      if (mainReference) mainReferenceCount += 1;
      if (sensitivity25w0) sensitivity25w0Count += 1;
    }
    const target = sheet.getRangeByIndexes(0, 0, newRows.length, translatedHeaders.length);
    target.values = newRows;
    outputText = csvDocument(target.values);
  } else if (alreadyNormalizedMainIdx >= 0) {
    translatedHeaders = originalHeaders;
    const sensitivityIdx = originalHeaders.indexOf("截至25周0天敏感性样本纳入标志");
    mainReferenceCount = values.slice(1).filter((row) => boolValue(row[alreadyNormalizedMainIdx])).length;
    sensitivity25w0Count = values.slice(1).filter((row) => boolValue(row[sensitivityIdx])).length;
  } else {
    translatedHeaders = translateHeaders(originalHeaders, false);
    const headerRange = sheet.getRangeByIndexes(0, 0, 1, translatedHeaders.length);
    headerRange.values = [translatedHeaders];
    const parts = splitCsvText(originalText);
    outputText = `\uFEFF${csvLine(headerRange.values[0])}${parts.eol}${parts.body}`;
    bodyPreserved = sha256(parts.body) === sha256(splitCsvText(outputText).body);
  }

  const duplicates = [...new Set(translatedHeaders.filter((h, idx) => translatedHeaders.indexOf(h) !== idx))];
  if (duplicates.length) throw new Error(`${path.basename(file)}存在重复中文字段: ${duplicates.join("、")}`);
  if (applyChanges && outputText !== originalText) await fs.writeFile(file, outputText, "utf8");
  const verifyText = applyChanges ? await fs.readFile(file, "utf8") : outputText;
  const verifyWorkbook = await Workbook.fromCSV(verifyText, { sheetName: "数据" });
  const verifyValues = verifyWorkbook.worksheets.getItem("数据").getUsedRange(true).values;
  const verifiedHeaders = verifyValues[0].map(stripBom);
  if (verifiedHeaders.some((header) => header.includes("_"))) throw new Error(`${path.basename(file)}仍存在英文代码式表头`);
  summaries.push({
    file,
    dataRowsBefore: Math.max(0, values.length - 1),
    dataRowsAfter: Math.max(0, verifyValues.length - 1),
    columnsBefore: originalHeaders.length,
    columnsAfter: verifiedHeaders.length,
    headers: verifiedHeaders,
    duplicateHeaders: duplicates,
    bodyPreserved,
    mainReferenceCount,
    sensitivity25w0Count,
    changed: outputText !== originalText,
  });
}

process.stdout.write(`${JSON.stringify({ applied: applyChanges, files: summaries.length, summaries }, null, 2)}\n`);
