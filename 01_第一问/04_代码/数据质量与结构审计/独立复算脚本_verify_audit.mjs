import fs from "node:fs/promises";
import path from "node:path";

if (process.argv.length < 5) {
  throw new Error("用法: node 独立复算脚本_verify_audit.mjs <男胎JSON快照> <审计汇总JSON> <复算结果JSON>");
}
const sourcePath = path.resolve(process.argv[2]);
const summaryPath = path.resolve(process.argv[3]);
const outputPath = path.resolve(process.argv[4]);

const source = JSON.parse(await fs.readFile(sourcePath, "utf8"));
const summary = JSON.parse(await fs.readFile(summaryPath, "utf8"));
const rows = source.values.slice(1);

function blank(value) {
  return value === null || value === undefined || (typeof value === "string" && value.trim() === "");
}

function gestation(value) {
  const match = String(value).trim().match(/^(\d+)\s*[wW周]\s*(?:\+\s*(\d+))?$/);
  if (!match) return null;
  const week = Number(match[1]);
  const day = Number(match[2] ?? 0);
  if (day < 0 || day > 6) return null;
  return { week: week + day / 7, days: week * 7 + day };
}

function parseDate(value) {
  if (blank(value)) return null;
  if (typeof value === "number") {
    const integer = Math.trunc(value);
    if (integer >= 19000101) {
      const text = String(integer);
      return Date.UTC(Number(text.slice(0, 4)), Number(text.slice(4, 6)) - 1, Number(text.slice(6, 8))) / 86400000;
    }
    return Date.UTC(1899, 11, 30) / 86400000 + value;
  }
  const timestamp = Date.parse(String(value));
  return Number.isNaN(timestamp) ? null : timestamp / 86400000;
}

function mean(values) {
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

function variance(values, ddof = 1) {
  if (values.length <= ddof) return 0;
  const avg = mean(values);
  return values.reduce((sum, value) => sum + (value - avg) ** 2, 0) / (values.length - ddof);
}

function pearson(xs, ys) {
  const xMean = mean(xs);
  const yMean = mean(ys);
  let numerator = 0;
  let xDenominator = 0;
  let yDenominator = 0;
  for (let index = 0; index < xs.length; index += 1) {
    const x = xs[index] - xMean;
    const y = ys[index] - yMean;
    numerator += x * y;
    xDenominator += x * x;
    yDenominator += y * y;
  }
  return numerator / Math.sqrt(xDenominator * yDenominator);
}

const records = rows.map((row) => {
  const gest = gestation(row[9]);
  const woman = String(row[1]);
  const drawNumber = Number(row[8]);
  const testDate = parseDate(row[7]);
  const dateKey = testDate === null ? "MISSING_DATE" : String(testDate);
  const draw = `${woman}|${drawNumber}`;
  return {
    seq: Number(row[0]),
    woman,
    drawNumber,
    draw,
    dateSession: `${woman}|${dateKey}`,
    assaySession: `${draw}|${dateKey}`,
    strictMetadata: `${draw}|${dateKey}|${gest?.days ?? "MISSING_GEST"}`,
    week: gest?.week ?? null,
    gestDays: gest?.days ?? null,
    bmi: Number(row[10]),
    y: Number(row[21]),
    weight: Number(row[4]),
    height: Number(row[3]),
    lmp: parseDate(row[5]),
    testDateRaw: row[7],
    testDate,
    totalReads: Number(row[11]),
    align: Number(row[12]),
    duplicate: Number(row[13]),
    uniqueReads: Number(row[14]),
    filtered: Number(row[26]),
    batch: Number(row[0]) >= 683 ? "post" : "pre",
  };
});

const women = new Set(records.map((row) => row.woman));
const draws = new Map();
for (const row of records) {
  if (!draws.has(row.draw)) draws.set(row.draw, []);
  draws.get(row.draw).push(row);
}
const repeats = [...draws.values()].filter((group) => group.length > 1);
const repeatSizeDistribution = Object.fromEntries(
  [...new Set(repeats.map((group) => group.length))]
    .sort((a, b) => a - b)
    .map((size) => [String(size), repeats.filter((group) => group.length === size).length]),
);
const pooledNumerator = repeats.reduce((sum, group) => sum + (group.length - 1) * variance(group.map((row) => row.y)), 0);
const pooledDenominator = repeats.reduce((sum, group) => sum + group.length - 1, 0);
const pooledWithinSd = Math.sqrt(pooledNumerator / pooledDenominator);
const repeatFlips = repeats.filter((group) => Math.min(...group.map((row) => row.y)) < 0.04 && Math.max(...group.map((row) => row.y)) >= 0.04).length;

function groupMap(group, key) {
  const result = new Map();
  for (const row of group) {
    if (!result.has(row[key])) result.set(row[key], []);
    result.get(row[key]).push(row);
  }
  return result;
}

function keyMetrics(group, key) {
  const groups = groupMap(group, key);
  const repeated = [...groups.values()].filter((items) => items.length > 1);
  const denominator = repeated.reduce((sum, items) => sum + items.length - 1, 0);
  const numerator = repeated.reduce(
    (sum, items) => sum + (items.length - 1) * variance(items.map((row) => row.y)),
    0,
  );
  const sizeDistribution = {};
  for (const items of groups.values()) {
    const size = String(items.length);
    sizeDistribution[size] = (sizeDistribution[size] ?? 0) + 1;
  }
  return {
    events: groups.size,
    extraRecords: group.length - groups.size,
    multiGroups: repeated.length,
    multiRows: repeated.reduce((sum, items) => sum + items.length, 0),
    sizeDistribution,
    pooledWithinYSd: denominator ? Math.sqrt(numerator / denominator) : 0,
    groupsCrossing4pct: repeated.filter(
      (items) => Math.min(...items.map((row) => row.y)) < 0.04 && Math.max(...items.map((row) => row.y)) >= 0.04,
    ).length,
  };
}

const clinical = records.filter((row) => row.week >= 10 && row.week < 26);
const through25 = records.filter((row) => row.week >= 10 && row.week <= 25);
const pre = records.filter((row) => row.batch === "pre");
const post = records.filter((row) => row.batch === "post");
const primary = pre.filter((row) => row.week >= 10 && row.week < 26);
const sensitivity = pre.filter((row) => row.week >= 10 && row.week <= 25);

const eventScopes = {
  "全男胎1082条": records,
  "683前682条": pre,
  "主参考674条": primary,
  "敏感性670条": sensitivity,
};
const eventKeys = {
  "B+I": "draw",
  "B+H": "dateSession",
  "B+I+H": "assaySession",
  "B+I+H+J": "strictMetadata",
};
const eventHierarchy = Object.fromEntries(
  Object.entries(eventScopes).map(([scope, group]) => [
    scope,
    Object.fromEntries(Object.entries(eventKeys).map(([notation, key]) => [notation, keyMetrics(group, key)])),
  ]),
);

function centeredRelation(group) {
  const byWoman = new Map();
  for (const row of group) {
    if (!byWoman.has(row.woman)) byWoman.set(row.woman, []);
    byWoman.get(row.woman).push(row);
  }
  const x = [];
  const y = [];
  for (const womanRows of byWoman.values()) {
    const bmiMean = mean(womanRows.map((row) => row.bmi));
    const yMean = mean(womanRows.map((row) => row.y));
    for (const row of womanRows) {
      x.push(row.bmi - bmiMean);
      y.push(row.y - yMean);
    }
  }
  const xx = x.reduce((sum, value) => sum + value * value, 0);
  const slope = x.reduce((sum, value, index) => sum + value * y[index], 0) / xx;
  const residuals = y.map((value, index) => value - slope * x[index]);
  const sse = residuals.reduce((sum, value) => sum + value * value, 0);
  const sst = y.reduce((sum, value) => sum + value * value, 0);
  return { slope, r2: 1 - sse / sst, residualSd: Math.sqrt(variance(residuals)) };
}

function rawCorrelation(group, key) {
  return pearson(group.map((row) => row[key]), group.map((row) => row.y));
}

function withinCorrelation(group, key) {
  const byWoman = new Map();
  for (const row of group) {
    if (!byWoman.has(row.woman)) byWoman.set(row.woman, []);
    byWoman.get(row.woman).push(row);
  }
  const x = [];
  const y = [];
  for (const womanRows of byWoman.values()) {
    const xMean = mean(womanRows.map((row) => row[key]));
    const yMean = mean(womanRows.map((row) => row.y));
    for (const row of womanRows) {
      x.push(row[key] - xMean);
      y.push(row.y - yMean);
    }
  }
  return pearson(x, y);
}

function betweenCorrelation(group, key) {
  const byWoman = new Map();
  for (const row of group) {
    if (!byWoman.has(row.woman)) byWoman.set(row.woman, []);
    byWoman.get(row.woman).push(row);
  }
  const x = [];
  const y = [];
  for (const womanRows of byWoman.values()) {
    x.push(mean(womanRows.map((row) => row[key])));
    y.push(mean(womanRows.map((row) => row.y)));
  }
  return pearson(x, y);
}

const readRelation = {};
for (const [name, group] of [["pre", pre], ["post", post]]) {
  const residuals = group.map((row) => row.uniqueReads - row.totalReads * row.align * (1 - row.duplicate) * (1 - row.filtered));
  readRelation[name] = {
    matchesWithin2: residuals.filter((value) => Math.abs(value) <= 2).length,
    uniqueGreaterThanTotal: group.filter((row) => row.uniqueReads > row.totalReads).length,
  };
}

const dateErrors = records
  .filter((row) => row.lmp !== null && row.testDate !== null && row.gestDays !== null)
  .map((row) => row.testDate - row.lmp - row.gestDays);

const repeatedDrawGroups = [...groupMap(records, "draw").entries()].filter(([, group]) => group.length > 1);
const crossDateRepeatedDraws = repeatedDrawGroups.filter(([, group]) => new Set(group.map((row) => row.testDate)).size > 1);
const gestationConflictDraws = repeatedDrawGroups.filter(([, group]) => new Set(group.map((row) => row.gestDays)).size > 1);
const a055Draw3 = records
  .filter((row) => row.woman === "A055" && row.drawNumber === 3)
  .map((row) => ({ seq: row.seq, testDate: row.testDate, gestDays: row.gestDays, y: row.y }));
const dateSessionDrawConflicts = [...groupMap(records, "dateSession").values()]
  .filter((group) => new Set(group.map((row) => row.drawNumber)).size > 1).length;

function isoDateFromEpochDay(epochDay) {
  return new Date(epochDay * 86400000).toISOString().slice(0, 10);
}

function crossingGroupIds(group, key, formatter) {
  return [...groupMap(group, key).entries()]
    .filter(([, items]) => items.length > 1)
    .filter(([, items]) => Math.min(...items.map((row) => row.y)) < 0.04 && Math.max(...items.map((row) => row.y)) >= 0.04)
    .map(([groupKey, items]) => formatter(groupKey, items))
    .sort();
}

function sequenceDifference(left, right) {
  const rightSequences = new Set(right.map((row) => row.seq));
  return left.map((row) => row.seq).filter((seq) => !rightSequences.has(seq)).sort((a, b) => a - b);
}

const preWomen = new Set(pre.map((row) => row.woman));
const postWomen = new Set(post.map((row) => row.woman));
const postWomanGroups = groupMap(post, "woman");
const dateStorageFirstExcelSerialSeq = Math.min(
  ...records
    .filter((row) => typeof row.testDateRaw === "number" && row.testDateRaw < 19000101)
    .map((row) => row.seq),
);

const independentlyComputed = {
  records: records.length,
  women: women.size,
  draws: draws.size,
  repeatGroups: repeats.length,
  repeatRecords: repeats.reduce((sum, group) => sum + group.length, 0),
  repeatSizeDistribution,
  pooledWithinSd,
  repeatFlips,
  clinicalRecords10w0d25w6d: clinical.length,
  through25w0dRecords: through25.length,
  pre: { records: pre.length, women: new Set(pre.map((row) => row.woman)).size, draws: new Set(pre.map((row) => row.draw)).size },
  post: { records: post.length, women: new Set(post.map((row) => row.woman)).size, draws: new Set(post.map((row) => row.draw)).size },
  postCenteredBmiY: centeredRelation(post),
  preCenteredBmiY: centeredRelation(pre),
  readRelation,
  dateErrors: {
    n: dateErrors.length,
    absGt7: dateErrors.filter((value) => Math.abs(value) > 7).length,
    absGt14: dateErrors.filter((value) => Math.abs(value) > 14).length,
    absGt21: dateErrors.filter((value) => Math.abs(value) > 21).length,
  },
  correlations: {
    week: { raw: rawCorrelation(records, "week"), between: betweenCorrelation(records, "week"), within: withinCorrelation(records, "week") },
    bmi: { raw: rawCorrelation(records, "bmi"), between: betweenCorrelation(records, "bmi"), within: withinCorrelation(records, "bmi") },
  },
  bmiFormulaMaxError: Math.max(...records.map((row) => Math.abs(row.bmi - row.weight / (row.height / 100) ** 2))),
  eventHierarchy,
  hierarchyDiagnostics: {
    crossDateRepeatedDrawGroups: crossDateRepeatedDraws.length,
    gestationConflictDrawGroups: gestationConflictDraws.map(([key]) => key),
    dateSessionDrawConflicts,
    a055Draw3,
    drawGroupsCrossing4pct: crossingGroupIds(records, "draw", (_key, items) => `${items[0].woman}#${items[0].drawNumber}`),
    assaySessionGroupsCrossing4pct: crossingGroupIds(
      records,
      "assaySession",
      (_key, items) => `${items[0].woman}@${isoDateFromEpochDay(items[0].testDate)}`,
    ),
    primaryMinusSensitivitySequences: sequenceDifference(primary, sensitivity),
    preMinusPrimarySequences: sequenceDifference(pre, primary),
    prePostWomanOverlap: [...preWomen].filter((woman) => postWomen.has(woman)).sort(),
    postWomenWithNonFourRecords: [...postWomanGroups.entries()]
      .filter(([, items]) => items.length !== 4)
      .map(([woman, items]) => [woman, items.length]),
    dateStorageFirstExcelSerialSeq,
  },
};

const checks = [];
function check(label, actual, expected, tolerance = 0) {
  const pass = typeof actual === "number" && typeof expected === "number"
    ? Math.abs(actual - expected) <= tolerance
    : JSON.stringify(actual) === JSON.stringify(expected);
  checks.push({ label, actual, expected, tolerance, pass });
}

check("records", independentlyComputed.records, summary.basic.records);
check("women", independentlyComputed.women, summary.basic.women);
check("draws", independentlyComputed.draws, summary.basic.draws);
check("repeat_groups", independentlyComputed.repeatGroups, summary.repeat_audit.multi_record_draw_groups);
check("repeat_records", independentlyComputed.repeatRecords, summary.repeat_audit.records_in_multi_groups);
check("repeat_pooled_sd", independentlyComputed.pooledWithinSd, summary.repeat_audit.pooled_within_draw_y_sd, 1e-12);
check("repeat_flips_4pct", independentlyComputed.repeatFlips, summary.repeat_audit.groups_crossing_4pct);
check("clinical_10_25w6_records", independentlyComputed.clinicalRecords10w0d25w6d, summary.samples["题干窗口10w0d-25w6d"].records);
check("pre_records", independentlyComputed.pre.records, summary.batch_audit["683前"].records);
check("post_records", independentlyComputed.post.records, summary.batch_audit["683后"].records);
check("post_centered_slope", independentlyComputed.postCenteredBmiY.slope, summary.batch_audit["683后"].centered_bmi_y_relation.slope, 1e-12);
check("post_centered_r2", independentlyComputed.postCenteredBmiY.r2, summary.batch_audit["683后"].centered_bmi_y_relation.r2, 1e-14);
check("pre_read_relation", independentlyComputed.readRelation.pre.matchesWithin2, summary.quality_audit.read_relation_by_batch["683前"].match_within_2_reads);
check("post_read_relation", independentlyComputed.readRelation.post.matchesWithin2, summary.quality_audit.read_relation_by_batch["683后"].match_within_2_reads);
check("unique_gt_total", independentlyComputed.readRelation.post.uniqueGreaterThanTotal, summary.quality_audit.unique_reads_gt_total);
check("date_error_n", independentlyComputed.dateErrors.n, summary.date_audit.date_gestation_error.n);
check("date_abs_gt14", independentlyComputed.dateErrors.absGt14, summary.date_audit.abs_error_gt14);
check("raw_week_corr", independentlyComputed.correlations.week.raw, summary.correlations["全部"].raw.gest_week.pearson.r, 1e-12);
check("within_week_corr", independentlyComputed.correlations.week.within, summary.correlations["全部"].within.gest_week.r, 1e-12);
check("between_week_corr", independentlyComputed.correlations.week.between, summary.correlations["全部"].between.gest_week.r, 1e-12);
check("raw_bmi_corr", independentlyComputed.correlations.bmi.raw, summary.correlations["全部"].raw.bmi.pearson.r, 1e-12);
check("within_bmi_corr", independentlyComputed.correlations.bmi.within, summary.correlations["全部"].within.bmi.r, 1e-12);
check("between_bmi_corr", independentlyComputed.correlations.bmi.between, summary.correlations["全部"].between.bmi.r, 1e-12);
check("bmi_formula_max_error", independentlyComputed.bmiFormulaMaxError, summary.body_audit.bmi_formula_abs_error.max, 1e-12);

const expectedHierarchy = {
  "全男胎1082条": {
    "B+I": [1021, 40, 101, 0.006100848548, 8],
    "B+H": [1063, 19, 38, 0.004711098811, 4],
    "B+I+H": [1063, 19, 38, 0.004711098811, 4],
    "B+I+H+J": [1064, 18, 36, 0.004733604810, 4],
  },
  "683前682条": {
    "B+I": [621, 40, 101, 0.006100848548, 8],
    "B+H": [663, 19, 38, 0.004711098811, 4],
    "B+I+H": [663, 19, 38, 0.004711098811, 4],
    "B+I+H+J": [664, 18, 36, 0.004733604810, 4],
  },
  "主参考674条": {
    "B+I": [614, 39, 99, 0.006054942527, 8],
    "B+H": [655, 19, 38, 0.004711098811, 4],
    "B+I+H": [655, 19, 38, 0.004711098811, 4],
    "B+I+H+J": [656, 18, 36, 0.004733604810, 4],
  },
  "敏感性670条": {
    "B+I": [611, 38, 97, 0.006102817208, 8],
    "B+H": [651, 19, 38, 0.004711098811, 4],
    "B+I+H": [651, 19, 38, 0.004711098811, 4],
    "B+I+H+J": [652, 18, 36, 0.004733604810, 4],
  },
};

for (const [scope, keys] of Object.entries(expectedHierarchy)) {
  for (const [notation, expected] of Object.entries(keys)) {
    const actual = independentlyComputed.eventHierarchy[scope][notation];
    const generated = summary.event_hierarchy[scope][notation];
    for (const [metric, index] of [["events", 0], ["multiGroups", 1], ["multiRows", 2]]) {
      check(`baseline_${scope}_${notation}_${metric}`, actual[metric], expected[index]);
      const generatedMetric = metric === "multiGroups" ? "multi_groups" : metric === "multiRows" ? "multi_rows" : metric;
      check(`generated_${scope}_${notation}_${metric}`, actual[metric], generated[generatedMetric]);
    }
    check(`baseline_${scope}_${notation}_pooled_sd`, actual.pooledWithinYSd, expected[3], 1e-12);
    check(`generated_${scope}_${notation}_pooled_sd`, actual.pooledWithinYSd, generated.pooled_within_y_sd, 1e-12);
    check(`baseline_${scope}_${notation}_flips`, actual.groupsCrossing4pct, expected[4]);
    check(`generated_${scope}_${notation}_flips`, actual.groupsCrossing4pct, generated.groups_crossing_4pct);
    check(
      `identity_${scope}_${notation}_records_equals_events_plus_extras`,
      eventScopes[scope].length,
      actual.events + actual.extraRecords,
    );
  }
}

check("baseline_cross_date_repeated_draw_groups", independentlyComputed.hierarchyDiagnostics.crossDateRepeatedDrawGroups, 39);
check("baseline_gestation_conflict_draw_groups", independentlyComputed.hierarchyDiagnostics.gestationConflictDrawGroups, ["A055|3"]);
check("baseline_date_session_draw_conflicts", independentlyComputed.hierarchyDiagnostics.dateSessionDrawConflicts, 0);
check("baseline_sequence_is_exactly_1_to_1082", records.every((row, index) => row.seq === index + 1), true);
check("baseline_primary_minus_sensitivity_sequences", independentlyComputed.hierarchyDiagnostics.primaryMinusSensitivitySequences, [111, 112, 411, 509]);
check("baseline_pre_minus_primary_sequences", independentlyComputed.hierarchyDiagnostics.preMinusPrimarySequences, [87, 188, 348, 383, 462, 463, 613, 618]);
check("baseline_pre_post_woman_overlap", independentlyComputed.hierarchyDiagnostics.prePostWomanOverlap, []);
check("baseline_post_100_women", postWomen.size, 100);
check("baseline_post_each_woman_has_four_records", independentlyComputed.hierarchyDiagnostics.postWomenWithNonFourRecords, []);
check("baseline_date_storage_first_excel_serial_seq", independentlyComputed.hierarchyDiagnostics.dateStorageFirstExcelSerialSeq, 687);
check("baseline_pre_read_relation_matches_within_2", independentlyComputed.readRelation.pre.matchesWithin2, 682);
check("baseline_post_read_relation_matches_within_2", independentlyComputed.readRelation.post.matchesWithin2, 0);
check("baseline_pre_unique_greater_than_total", independentlyComputed.readRelation.pre.uniqueGreaterThanTotal, 0);
check("baseline_post_unique_greater_than_total", independentlyComputed.readRelation.post.uniqueGreaterThanTotal, 71);
check("baseline_post_centered_bmi_y_r2_gate", independentlyComputed.postCenteredBmiY.r2 > 1 - 1e-12, true);
check("baseline_post_centered_bmi_y_residual_sd_gate", independentlyComputed.postCenteredBmiY.residualSd < 1e-9, true);
check(
  "baseline_draw_groups_crossing_4pct_ids",
  independentlyComputed.hierarchyDiagnostics.drawGroupsCrossing4pct,
  ["A023#3", "A029#2", "A035#3", "A041#4", "A066#1", "A114#2", "A147#3", "A155#2"],
);
check(
  "baseline_assay_session_groups_crossing_4pct_ids",
  independentlyComputed.hierarchyDiagnostics.assaySessionGroupsCrossing4pct,
  ["A023@2023-06-16", "A041@2023-06-27", "A066@2023-04-29", "A114@2023-06-21"],
);
check("baseline_A055_draw3_seqs", independentlyComputed.hierarchyDiagnostics.a055Draw3.map((row) => row.seq), [240, 241]);
check("baseline_A055_draw3_dates", independentlyComputed.hierarchyDiagnostics.a055Draw3.map((row) => row.testDate), [19560, 19560]);
check("baseline_A055_draw3_gest_days", independentlyComputed.hierarchyDiagnostics.a055Draw3.map((row) => row.gestDays), [148, 143]);
check("baseline_A055_draw3_gest_gap", Math.abs(independentlyComputed.hierarchyDiagnostics.a055Draw3[0].gestDays - independentlyComputed.hierarchyDiagnostics.a055Draw3[1].gestDays), 5);

const result = {
  independentlyComputed,
  checks,
  passed: checks.filter((item) => item.pass).length,
  failed: checks.filter((item) => !item.pass).length,
};
await fs.mkdir(path.dirname(outputPath), { recursive: true });
await fs.writeFile(outputPath, JSON.stringify(result, null, 2), "utf8");
console.log(JSON.stringify({ passed: result.passed, failed: result.failed, failures: checks.filter((item) => !item.pass) }, null, 2));
