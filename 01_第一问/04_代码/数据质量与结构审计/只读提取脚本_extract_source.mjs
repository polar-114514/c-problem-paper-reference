import fs from "node:fs/promises";
import path from "node:path";
import crypto from "node:crypto";
import { fileURLToPath } from "node:url";
import { FileBlob, SpreadsheetFile } from "@oai/artifact-tool";

const scriptDir = path.dirname(fileURLToPath(import.meta.url));
const workspaceRoot = path.resolve(scriptDir, "../../..");
const inputPath = path.resolve(
  process.argv[2] ?? path.join(workspaceRoot, "00_题目与原始资料/02_原始数据/附件.xlsx"),
);
const outputDir = path.resolve(
  process.argv[3] ?? path.join(workspaceRoot, "99_临时中转/第一问数据审计复现/source_snapshot"),
);

await fs.mkdir(outputDir, { recursive: true });

const inputBytes = await fs.readFile(inputPath);
const sourceSha256 = crypto.createHash("sha256").update(inputBytes).digest("hex");
const input = await FileBlob.load(inputPath);
const workbook = await SpreadsheetFile.importXlsx(input);

const overview = await workbook.inspect({
  kind: "workbook,sheet,definedName,drawing",
  maxChars: 8000,
  tableMaxRows: 5,
  tableMaxCols: 8,
  tableMaxCellChars: 120,
});

const manifest = {
  inputPath,
  sourceSha256,
  extractedAt: new Date().toISOString(),
  workbookOverview: overview.ndjson,
  sheets: [],
};

function cellType(value) {
  if (value === null || value === undefined) return "blank";
  if (value instanceof Date) return "date";
  if (typeof value === "string" && value.trim() === "") return "blank-string";
  return typeof value;
}

for (let index = 0; index < workbook.worksheets.items.length; index += 1) {
  const sheet = workbook.worksheets.getItemAt(index);
  const used = sheet.getUsedRange();
  const values = used ? used.values : [];
  const formulas = used ? used.formulas : [];
  const types = values.map((row) => row.map(cellType));
  const rowCount = values.length;
  const columnCount = rowCount ? Math.max(...values.map((row) => row.length)) : 0;
  const safeName = sheet.name.replace(/[\\/:*?"<>|]/g, "_");
  const formulaCellCount = formulas.reduce(
    (total, row) => total + row.filter((value) => typeof value === "string" && value.startsWith("=")).length,
    0,
  );

  const payload = {
    sheetName: sheet.name,
    usedRange: used?.address ?? null,
    rowCount,
    columnCount,
    values,
    formulas,
    types,
  };
  const fileName = `${String(index + 1).padStart(2, "0")}_${safeName}.json`;
  await fs.writeFile(path.join(outputDir, fileName), JSON.stringify(payload), "utf8");
  manifest.sheets.push({
    index,
    name: sheet.name,
    fileName,
    usedRange: payload.usedRange,
    rowCount,
    columnCount,
    formulaCellCount,
  });
}

await fs.writeFile(path.join(outputDir, "manifest.json"), JSON.stringify(manifest, null, 2), "utf8");
console.log(JSON.stringify(manifest, null, 2));
