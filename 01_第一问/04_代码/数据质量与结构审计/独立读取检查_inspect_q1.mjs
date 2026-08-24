import { FileBlob, SpreadsheetFile } from "@oai/artifact-tool";

const input = await FileBlob.load("C:/Users/15599/Desktop/C题/附件.xlsx");
const workbook = await SpreadsheetFile.importXlsx(input);

const overview = await workbook.inspect({
  kind: "workbook,sheet,table",
  maxChars: 10000,
  tableMaxRows: 5,
  tableMaxCols: 31,
  tableMaxCellChars: 100,
});
console.log(overview.ndjson);

const maleTail = await workbook.inspect({
  kind: "region",
  sheetId: "男胎检测数据",
  range: "A680:AE690",
  maxChars: 18000,
});
console.log(maleTail.ndjson);
