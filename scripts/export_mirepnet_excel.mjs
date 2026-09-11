import fs from "node:fs/promises";
import path from "node:path";
import { pathToFileURL } from "node:url";

const [artifactModule, jsonPath, xlsxPath, previewPath = ""] = process.argv.slice(2);
if (!artifactModule || !jsonPath || !xlsxPath) {
  throw new Error("usage: export_mirepnet_excel.mjs ARTIFACT_MODULE INPUT.json OUTPUT.xlsx [PREVIEW.png]");
}

const { SpreadsheetFile, Workbook } = await import(pathToFileURL(artifactModule).href);
const payload = JSON.parse(await fs.readFile(jsonPath, "utf8"));
const workbook = Workbook.create();

const navy = "#16324F";
const blue = "#2563EB";
const paleBlue = "#EAF2FF";
const paleGreen = "#E7F6EC";
const paleRed = "#FDECEC";
const gray = "#667085";
const lightBorder = "#D0D5DD";
const headerFormat = {
  fill: navy,
  font: { bold: true, color: "#FFFFFF" },
  verticalAlignment: "center",
};

function applyTableStyle(sheet, address, tableName) {
  const table = sheet.tables.add(address, true, tableName);
  table.style = "TableStyleMedium2";
  table.showBandedRows = true;
  return table;
}

function setTitle(sheet, title, subtitle, endColumn = "H") {
  sheet.showGridLines = false;
  sheet.getRange(`A1:${endColumn}1`).merge();
  sheet.getRange("A1").values = [[title]];
  sheet.getRange("A1").format = {
    fill: navy,
    font: { bold: true, color: "#FFFFFF", size: 18 },
    rowHeight: 32,
    verticalAlignment: "center",
  };
  sheet.getRange(`A2:${endColumn}2`).merge();
  sheet.getRange("A2").values = [[subtitle]];
  sheet.getRange("A2").format = { font: { color: gray, italic: true }, rowHeight: 23 };
}

const summary = workbook.worksheets.add("Run Summary");
const subjectsSheet = workbook.worksheets.add("Subject Results");
const epochsSheet = workbook.worksheets.add("Epoch Predictions");
const notesSheet = workbook.worksheets.add("Protocol Notes");

setTitle(summary, "MIRepNet Patient Test Report", "Generated automatically after the offline test run", "H");
summary.getRange("A4:B10").values = [
  ["Created", payload.created_at ?? ""],
  ["Checkpoint", payload.checkpoint ?? ""],
  ["Selected patients", (payload.subjects_requested ?? []).map((n) => `S${String(n).padStart(2, "0")}`).join(", ")],
  ["Protocol", payload.protocol ?? ""],
  ["Recording scope", payload.recording_scope ?? ""],
  ["Window mode", payload.window_mode ?? ""],
  ["Runtime (minutes)", Number(payload.run_seconds ?? 0) / 60],
];
summary.getRange("A4:A10").format = { fill: paleBlue, font: { bold: true, color: navy } };
summary.getRange("B10").format.numberFormat = "0.00";

const protocolLabels = {
  strict_zero_shot: "Strict zero-shot",
  unlabeled_ea: "Unlabeled EA",
  calibrated_raw: "Calibrated",
  calibrated_balanced_block: "Calibrated balanced block",
};
const aggregateKeys = Object.keys(payload.aggregate ?? {});
summary.getRange("A12:E12").values = [["Protocol", "Patients", "Macro accuracy", ">= 80%", ">= 90%"]];
summary.getRange("A12:E12").format = headerFormat;

const subjectResultRows = [];
for (const result of payload.results ?? []) {
  for (const [key, label] of Object.entries(protocolLabels)) {
    if (!result[key]) continue;
    const metric = result[key];
    const epochMatches = (payload.epoch_results ?? []).filter(
      (row) => row.subject === result.subject && row.protocol === key,
    );
    const correct = epochMatches.reduce((total, row) => total + (row.correct ? 1 : 0), 0);
    subjectResultRows.push([
      `S${String(result.subject).padStart(2, "0")}`,
      Boolean(result.training_seen),
      key,
      Number(metric.n_trials ?? epochMatches.length),
      correct,
      null,
      Number(metric.balanced_accuracy ?? 0),
      (result.test_files ?? []).join(", "),
    ]);
  }
}

subjectsSheet.showGridLines = false;
subjectsSheet.getRange("A1:H1").values = [[
  "Patient", "Seen in training", "Protocol", "Test epochs", "Correct epochs",
  "Accuracy", "Balanced accuracy", "Test files",
]];
subjectsSheet.getRange("A1:H1").format = headerFormat;
if (subjectResultRows.length) {
  subjectsSheet.getRangeByIndexes(1, 0, subjectResultRows.length, 8).values = subjectResultRows;
  const accuracyFormulas = subjectResultRows.map((_, index) => [`=IFERROR(E${index + 2}/D${index + 2},0)`]);
  subjectsSheet.getRangeByIndexes(1, 5, accuracyFormulas.length, 1).formulas = accuracyFormulas;
  subjectsSheet.getRange(`F2:G${subjectResultRows.length + 1}`).format.numberFormat = "0.00%";
  subjectsSheet.getRange(`F2:G${subjectResultRows.length + 1}`).conditionalFormats.add("colorScale", {
    thresholds: ["min", "50%", "max"],
    colors: ["#F8696B", "#FFEB84", "#63BE7B"],
  });
  applyTableStyle(subjectsSheet, `A1:H${subjectResultRows.length + 1}`, "SubjectResultsTable");
}
subjectsSheet.freezePanes.freezeRows(1);
subjectsSheet.getRange("A:H").format.autofitColumns();
subjectsSheet.getRange("H:H").format.columnWidth = 48;

if (aggregateKeys.length) {
  const aggregateRows = aggregateKeys.map((key) => [protocolLabels[key] ?? key, null, null, null, null]);
  summary.getRangeByIndexes(12, 0, aggregateRows.length, 5).values = aggregateRows;
  const end = subjectResultRows.length + 1;
  const formulas = aggregateKeys.map((key) => {
    return [
      `=COUNTIF('Subject Results'!$C$2:$C$${end},"${key}")`,
      `=IFERROR(AVERAGEIF('Subject Results'!$C$2:$C$${end},"${key}",'Subject Results'!$F$2:$F$${end}),0)`,
      `=COUNTIFS('Subject Results'!$C$2:$C$${end},"${key}",'Subject Results'!$F$2:$F$${end},">=0.8")`,
      `=COUNTIFS('Subject Results'!$C$2:$C$${end},"${key}",'Subject Results'!$F$2:$F$${end},">=0.9")`,
    ];
  });
  summary.getRangeByIndexes(12, 1, formulas.length, 4).formulas = formulas;
  summary.getRange(`C13:C${12 + aggregateRows.length}`).format.numberFormat = "0.00%";
  summary.getRange(`C13:C${12 + aggregateRows.length}`).conditionalFormats.add("colorScale", {
    thresholds: ["min", "50%", "max"],
    colors: ["#F8696B", "#FFEB84", "#63BE7B"],
  });
  applyTableStyle(summary, `A12:E${12 + aggregateRows.length}`, "AggregateResultsTable");
}
summary.getRange("A:N").format.autofitColumns();
summary.getRange("B:B").format.columnWidth = 55;

epochsSheet.showGridLines = false;
const epochHeaders = [
  "Run ID", "Patient", "Seen in training", "Protocol", "Recording file",
  "Epoch in file", "Test epoch", "True label", "Predicted label", "Confidence", "Correct",
];
epochsSheet.getRange("A1:K1").values = [epochHeaders];
epochsSheet.getRange("A1:K1").format = headerFormat;
const epochRows = (payload.epoch_results ?? []).map((row) => [
  row.run_id ?? "",
  `S${String(row.subject).padStart(2, "0")}`,
  Boolean(row.training_seen),
  row.protocol,
  row.file,
  Number(row.epoch_in_file),
  Number(row.test_epoch),
  Number(row.true_label),
  Number(row.predicted_label),
  Number(row.confidence),
  null,
]);
if (epochRows.length) {
  epochsSheet.getRangeByIndexes(1, 0, epochRows.length, epochHeaders.length).values = epochRows;
  const correctFormulas = epochRows.map((_, index) => [`=IF(H${index + 2}=I${index + 2},1,0)`]);
  epochsSheet.getRangeByIndexes(1, 10, correctFormulas.length, 1).formulas = correctFormulas;
  epochsSheet.getRange(`J2:J${epochRows.length + 1}`).format.numberFormat = "0.00%";
  epochsSheet.getRange(`K2:K${epochRows.length + 1}`).conditionalFormats.add("cellIs", {
    operator: "equal", formula: 1, format: { fill: paleGreen, font: { color: "#166534" } },
  });
  epochsSheet.getRange(`K2:K${epochRows.length + 1}`).conditionalFormats.add("cellIs", {
    operator: "equal", formula: 0, format: { fill: paleRed, font: { color: "#991B1B" } },
  });
  applyTableStyle(epochsSheet, `A1:K${epochRows.length + 1}`, "EpochPredictionsTable");
}
epochsSheet.freezePanes.freezeRows(1);
epochsSheet.freezePanes.freezeColumns(2);
epochsSheet.getRange("A:K").format.autofitColumns();
epochsSheet.getRange("E:E").format.columnWidth = 45;

setTitle(notesSheet, "Protocol and Reproducibility Notes", "Interpret the protocol column before comparing scores", "F");
notesSheet.getRange("A4:C8").values = [
  ["Protocol", "Uses target EEG", "Uses target labels"],
  ["Strict zero-shot", "No", "No"],
  ["Unlabeled EA", "Yes, completed target batch", "No"],
  ["Calibrated", "Yes, early calibration recordings", "Yes, calibration only"],
  ["Calibrated balanced block", "Yes", "Yes, calibration only; assumes 50/50 test schedule"],
];
notesSheet.getRange("A4:C4").format = headerFormat;
applyTableStyle(notesSheet, "A4:C8", "ProtocolNotesTable");
notesSheet.getRange("A10:B14").values = [
  ["Foundation weights updated during testing", String(Boolean(payload.foundation_weights_updated))],
  ["Test labels used for adaptation", String(Boolean(payload.protocol_contract?.test_labels_used_for_adaptation))],
  ["JSON source", jsonPath],
  ["Workbook format", "MIRepNet patient-test Excel v1"],
  ["Epoch definition", "One preprocessed MI trial presented to the model"],
];
notesSheet.getRange("A10:A14").format = { fill: paleBlue, font: { bold: true, color: navy } };
notesSheet.getRange("A:C").format.autofitColumns();
notesSheet.getRange("B:B").format.columnWidth = 65;

await fs.mkdir(path.dirname(xlsxPath), { recursive: true });
if (previewPath) {
  const preview = await workbook.render({ sheetName: "Run Summary", autoCrop: "all", scale: 1, format: "png" });
  await fs.mkdir(path.dirname(previewPath), { recursive: true });
  await fs.writeFile(previewPath, new Uint8Array(await preview.arrayBuffer()));
}
const output = await SpreadsheetFile.exportXlsx(workbook);
await output.save(xlsxPath);
console.log(xlsxPath);
