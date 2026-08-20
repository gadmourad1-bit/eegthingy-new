#!/usr/bin/env python3
"""Fail-fast structural checks for the internal TBME manuscript draft."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import re
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
ABSTRACT_LIMIT = 250
CONCLUSION_LIMIT = 300
COVER_LETTER_LIMIT = 250
TARGET_PAGES = 8
MAX_PAGES = 12
MAX_PDF_BYTES = 10_000_000


@dataclass(frozen=True)
class Finding:
    level: str
    message: str


def without_comments(source: str) -> str:
    """Remove unescaped LaTeX comments while preserving line boundaries."""

    return re.sub(r"(?<!\\)%[^\n]*", "", source)


def plain_text(fragment: str) -> str:
    """Produce conservative prose for word counts and heading checks."""

    text = without_comments(fragment)
    text = re.sub(
        r"\\begin\s*\{(?:equation\*?|align\*?|figure\*?|table\*?)\}.*?"
        r"\\end\s*\{(?:equation\*?|align\*?|figure\*?|table\*?)\}",
        " ",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    text = re.sub(r"\$\$.*?\$\$|\$.*?\$|\\\[.*?\\\]|\\\(.*?\\\)", " ", text, flags=re.DOTALL)
    text = re.sub(
        r"\\(?:label|ref|eqref|autoref|cite\w*|url)\*?\s*\{[^{}]*\}",
        " ",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"\\href\s*\{[^{}]*\}\s*\{([^{}]*)\}",
        r" \1 ",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"\\begin\s*\{[^{}]+\}|\\end\s*\{[^{}]+\}", " ", text)
    text = re.sub(r"\\[A-Za-z@]+\*?(?:\s*\[[^\]]*\])?", " ", text)
    text = re.sub(r"\\[^A-Za-z\s]", " ", text)
    text = text.replace("{", " ").replace("}", " ").replace("~", " ")
    return re.sub(r"\s+", " ", text).strip()


def words(fragment: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9]+(?:[-'’][A-Za-z0-9]+)*", plain_text(fragment))


def environment(source: str, name: str) -> list[str]:
    pattern = re.compile(
        rf"\\begin\s*\{{{re.escape(name)}\}}(.*?)"
        rf"\\end\s*\{{{re.escape(name)}\}}",
        flags=re.IGNORECASE | re.DOTALL,
    )
    return pattern.findall(without_comments(source))


def sections(source: str) -> list[tuple[str, int, int]]:
    clean = without_comments(source)
    matches = list(
        re.finditer(r"\\section\*?\s*\{([^{}]+)\}", clean, flags=re.IGNORECASE)
    )
    records: list[tuple[str, int, int]] = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(clean)
        records.append((plain_text(match.group(1)).casefold(), match.end(), end))
    return records


def pdf_page_count(path: Path) -> int:
    try:
        from pypdf import PdfReader  # type: ignore[import-not-found]

        return len(PdfReader(str(path)).pages)
    except (ImportError, OSError, ValueError):
        pass

    pdfinfo = shutil.which("pdfinfo")
    if pdfinfo:
        completed = subprocess.run(
            [pdfinfo, str(path)],
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode == 0:
            match = re.search(r"^Pages:\s*(\d+)\s*$", completed.stdout, re.MULTILINE)
            if match:
                return int(match.group(1))

    payload = path.read_bytes()
    count = len(re.findall(rb"/Type\s*/Page(?!s)\b", payload))
    if count:
        return count
    raise RuntimeError("unable to determine PDF page count; install pypdf or pdfinfo")


def check_tex(path: Path) -> tuple[list[Finding], int]:
    findings: list[Finding] = []
    source = path.read_text(encoding="utf-8")
    clean = without_comments(source)

    abstracts = environment(source, "abstract")
    if len(abstracts) != 1:
        findings.append(Finding("FAIL", f"expected one abstract environment, found {len(abstracts)}"))
    else:
        abstract_count = len(words(abstracts[0]))
        if abstract_count >= ABSTRACT_LIMIT:
            findings.append(Finding("FAIL", f"abstract has {abstract_count} words; must be under {ABSTRACT_LIMIT}"))
        else:
            findings.append(Finding("PASS", f"abstract has {abstract_count} words (<{ABSTRACT_LIMIT})"))

        abstract_text = plain_text(abstracts[0]).casefold()
        labels = ("objective", "methods", "results", "conclusion", "significance")
        positions = [abstract_text.find(label) for label in labels]
        missing = [label.title() for label, position in zip(labels, positions) if position < 0]
        if missing:
            findings.append(Finding("FAIL", "structured abstract labels missing: " + ", ".join(missing)))
        elif positions != sorted(positions):
            findings.append(Finding("FAIL", "structured abstract labels are out of order"))
        else:
            findings.append(Finding("PASS", "structured abstract labels are present in order"))

    if re.search(r"\\begin\s*\{[^{}]*keywords\}", clean, flags=re.IGNORECASE):
        findings.append(Finding("PASS", "index-terms environment is present"))
    else:
        findings.append(Finding("FAIL", "index-terms environment is missing"))

    records = sections(source)
    headings = [name for name, _, _ in records]
    requirements = (
        ("Introduction", lambda value: value == "introduction"),
        ("Methods", lambda value: value in {"methods", "methodology", "materials and methods"}),
        ("Results", lambda value: value == "results"),
        ("Discussion", lambda value: value == "discussion"),
        ("Conclusion", lambda value: value in {"conclusion", "conclusions"}),
    )
    required_positions: list[int] = []
    missing_headings: list[str] = []
    for display, predicate in requirements:
        position = next((index for index, value in enumerate(headings) if predicate(value)), -1)
        if position < 0:
            missing_headings.append(display)
        else:
            required_positions.append(position)
    if missing_headings:
        findings.append(Finding("FAIL", "required section headings missing: " + ", ".join(missing_headings)))
    elif required_positions != sorted(required_positions):
        findings.append(Finding("FAIL", "required section headings are out of order"))
    else:
        findings.append(Finding("PASS", "required section headings are present in order"))

    conclusion_record = next(
        (record for record in records if record[0] in {"conclusion", "conclusions"}),
        None,
    )
    if conclusion_record is not None:
        _, start, end = conclusion_record
        conclusion_count = len(words(clean[start:end]))
        if conclusion_count >= CONCLUSION_LIMIT:
            findings.append(Finding("FAIL", f"conclusion has {conclusion_count} words; must be under {CONCLUSION_LIMIT}"))
        else:
            findings.append(Finding("PASS", f"conclusion has {conclusion_count} words (<{CONCLUSION_LIMIT})"))

    bibliography_pattern = re.compile(
        r"\\begin\s*\{thebibliography\}|\\bibliography\s*\{|\\printbibliography\b",
        flags=re.IGNORECASE,
    )
    if bibliography_pattern.search(clean):
        findings.append(Finding("PASS", "reference section marker is present"))
    else:
        findings.append(Finding("FAIL", "reference section marker is missing"))

    if re.search(r"biograph(?:y|ies|ynophoto)", clean, flags=re.IGNORECASE):
        findings.append(Finding("FAIL", "author biography content or environment detected"))
    else:
        findings.append(Finding("PASS", "no author biography content detected"))

    todo_count = len(re.findall(r"\bTODO\b", source, flags=re.IGNORECASE))
    return findings, todo_count


def check_cover_letter(path: Path) -> tuple[list[Finding], int]:
    findings: list[Finding] = []
    source = path.read_text(encoding="utf-8")
    count = len(re.findall(r"[A-Za-z0-9]+(?:[-'’][A-Za-z0-9]+)*", source))
    if count >= COVER_LETTER_LIMIT:
        findings.append(Finding("FAIL", f"cover letter has {count} words; must be under {COVER_LETTER_LIMIT}"))
    else:
        findings.append(Finding("PASS", f"cover letter has {count} words (<{COVER_LETTER_LIMIT})"))
    return findings, len(re.findall(r"\bTODO\b", source, flags=re.IGNORECASE))


def check_pdf(path: Path) -> list[Finding]:
    findings: list[Finding] = []
    size = path.stat().st_size
    if size >= MAX_PDF_BYTES:
        findings.append(Finding("FAIL", f"PDF is {size} bytes; must be under {MAX_PDF_BYTES}"))
    else:
        findings.append(Finding("PASS", f"PDF is {size} bytes (<{MAX_PDF_BYTES})"))

    try:
        pages = pdf_page_count(path)
    except RuntimeError as error:
        findings.append(Finding("FAIL", str(error)))
    else:
        if pages > MAX_PAGES:
            findings.append(Finding("FAIL", f"PDF has {pages} pages; maximum is {MAX_PAGES}"))
        elif pages > TARGET_PAGES:
            findings.append(Finding("WARN", f"PDF has {pages} pages; target is <={TARGET_PAGES} and overlength charges may apply"))
        else:
            findings.append(Finding("PASS", f"PDF has {pages} pages (target <={TARGET_PAGES})"))
    return findings


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--tex", type=Path, default=ROOT / "main.tex")
    result.add_argument("--pdf", type=Path, default=ROOT / "output" / "TBME_Internal_Working_Draft.pdf")
    result.add_argument("--cover-letter", type=Path, default=ROOT / "cover_letter.txt")
    result.add_argument("--references", type=Path, default=ROOT / "references.bib")
    result.add_argument("--skip-pdf", action="store_true")
    result.add_argument("--fail-on-todo", action="store_true")
    return result


def main() -> int:
    args = parser().parse_args()
    findings: list[Finding] = []
    todo_counts: dict[str, int] = {}

    if not args.tex.is_file():
        findings.append(Finding("FAIL", f"manuscript source is missing: {args.tex}"))
    else:
        tex_findings, count = check_tex(args.tex)
        findings.extend(tex_findings)
        todo_counts[str(args.tex)] = count

    if not args.cover_letter.is_file():
        findings.append(Finding("FAIL", f"cover letter is missing: {args.cover_letter}"))
    else:
        cover_findings, count = check_cover_letter(args.cover_letter)
        findings.extend(cover_findings)
        todo_counts[str(args.cover_letter)] = count

    if args.references.is_file():
        reference_source = args.references.read_text(encoding="utf-8")
        todo_counts[str(args.references)] = len(
            re.findall(r"\bTODO\b", reference_source, flags=re.IGNORECASE)
        )

    if not args.skip_pdf:
        if not args.pdf.is_file():
            findings.append(Finding("FAIL", f"working PDF is missing: {args.pdf}"))
        else:
            findings.extend(check_pdf(args.pdf))

    total_todos = sum(todo_counts.values())
    details = ", ".join(f"{Path(path).name}={count}" for path, count in todo_counts.items())
    if total_todos and args.fail_on_todo:
        findings.append(Finding("FAIL", f"TODO count is {total_todos} ({details})"))
    elif total_todos:
        findings.append(Finding("WARN", f"TODO count is {total_todos} ({details})"))
    else:
        findings.append(Finding("PASS", "TODO count is 0"))

    for finding in findings:
        print(f"[{finding.level}] {finding.message}")

    failures = sum(finding.level == "FAIL" for finding in findings)
    warnings = sum(finding.level == "WARN" for finding in findings)
    print(f"summary: failures={failures} warnings={warnings} todos={total_todos}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
