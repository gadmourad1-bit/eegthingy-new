import fs from "node:fs/promises";
import path from "node:path";
import { pathToFileURL } from "node:url";

const [artifactModule, jsonPath, xlsxPath] = process.argv.slice(2);
if (!artifactModule || !jsonPath || !xlsxPath) {
  throw new Error("usage: export_mirepnet_online_excel.mjs ARTIFACT_MODULE INPUT.json OUTPUT.xlsx");
}
const { SpreadsheetFile, Workbook } = await import(pathToFileURL(artifactModule).href);
const payload = JSON.parse(await fs.readFile(jsonPath, "utf8"));
const workbook = Workbook.create();
const summary = workbook.worksheets.add("Session Summary");
const windows = workbook.worksheets.add("Live Windows");
const navy = "#16324F";
const paleBlue = "#EAF2FF";
const header = { fill: navy, font: { bold: true, color: "#FFFFFF" } };

summary.showGridLines = false;
summary.getRange("A1:F1").merge();
summary.getRange("A1").values = [["MIRepNet Online Session"]];
summary.getRange("A1").format = {
  fill: navy, font: { bold: true, color: "#FFFFFF", size: 18 }, rowHeight: 32,
};
const meta = payload.metadata ?? {};
summary.getRange("A3:B10").values = [
  ["Created", payload.created_at ?? ""],
  ["Mode", meta.mode ?? "online"],
  ["Input device", meta.input_device ?? ""],
  ["Checkpoint", meta.checkpoint ?? ""],
  ["Decoder", meta.decoder ?? "MIRepNet"],
  ["Decision windows", (payload.windows ?? []).length],
  ["Window stride (seconds)", Number(meta.stride_seconds ?? 0)],
  ["Labels available", "No — live predictions only"],
];
summary.getRange("A3:A10").format = { fill: paleBlue, font: { bold: true, color: navy } };
summary.getRange("A:B").format.autofitColumns();
summary.getRange("B:B").format.columnWidth = 60;

windows.showGridLines = false;
const headers = [
  "Window", "Timestamp", "Log odds", "Adaptive center", "Recentered margin",
  "Prediction", "Confidence", "Committed class",
];
windows.getRange("A1:H1").values = [headers];
windows.getRange("A1:H1").format = header;
const rows = (payload.windows ?? []).map((row) => [
  Number(row.window), row.timestamp, Number(row.log_odds), Number(row.adaptive_center),
  Number(row.recentered_margin), Number(row.prediction), Number(row.confidence),
  Number(row.committed_class),
]);
if (rows.length) {
  windows.getRangeByIndexes(1, 0, rows.length, headers.length).values = rows;
  windows.getRange(`C2:E${rows.length + 1}`).format.numberFormat = "0.0000";
  windows.getRange(`G2:G${rows.length + 1}`).format.numberFormat = "0.00%";
  windows.getRange(`G2:G${rows.length + 1}`).conditionalFormats.add("colorScale", {
    thresholds: ["min", "50%", "max"], colors: ["#F8696B", "#FFEB84", "#63BE7B"],
  });
  const table = windows.tables.add(`A1:H${rows.length + 1}`, true, "LiveWindowsTable");
  table.style = "TableStyleMedium2";
}
windows.freezePanes.freezeRows(1);
windows.freezePanes.freezeColumns(2);
windows.getRange("A:H").format.autofitColumns();
windows.getRange("B:B").format.columnWidth = 25;

await fs.mkdir(path.dirname(xlsxPath), { recursive: true });
const output = await SpreadsheetFile.exportXlsx(workbook);
await output.save(xlsxPath);
console.log(xlsxPath);
