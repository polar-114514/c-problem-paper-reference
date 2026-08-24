import crypto from "node:crypto";
import fs from "node:fs/promises";
import path from "node:path";

if (process.argv.length < 4) {
  throw new Error("用法: node 生成Wolfram事件层级载荷.mjs <男胎JSON快照> <输出载荷JSON>");
}

const sourcePath = path.resolve(process.argv[2]);
const outputPath = path.resolve(process.argv[3]);
const source = JSON.parse(await fs.readFile(sourcePath, "utf8"));

if (!Array.isArray(source.values) || source.values.length !== 1083) {
  throw new Error("男胎JSON快照结构异常：预期1行表头和1082条记录。");
}

const records = source.values.slice(1).map((row) => [
  Number(row[0]),
  String(row[1]),
  Number(row[8]),
  row[7],
  String(row[9]),
  Number(row[21]),
]);

const payload = {
  载荷用途: "Wolfram Language独立复核第一问事件层级；不用于正式模型拟合",
  字段顺序: ["序号", "孕妇代码", "检测抽血次数", "检测日期原始值", "检测孕周原始值", "Y染色体浓度（比例，0–1）"],
  记录: records,
};
// Wolfram 15.0 的远端 RawJSON 导入器在当前接口中不能稳定接收JSON内的
// 直写非ASCII字符，因此只改变JSON字符表示，将它们写成标准 \uXXXX 转义；
// 导入后的中文键和值不变，载荷本身保持纯ASCII，便于跨接口复现。
const text = JSON.stringify(payload).replace(
  /[\u007f-\uffff]/g,
  (character) => `\\u${character.charCodeAt(0).toString(16).padStart(4, "0")}`,
);
const sha256 = crypto.createHash("sha256").update(text, "utf8").digest("hex");

await fs.mkdir(path.dirname(outputPath), { recursive: true });
await fs.writeFile(outputPath, text, "utf8");
console.log(JSON.stringify({ 输出文件: outputPath, 记录数: records.length, 载荷SHA256: sha256 }, null, 2));
