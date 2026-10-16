"""CSV/TSV/XLSX supplementary tables → records (no network).

uv run python -m unittest discover tests
"""

import io
import unittest

import openpyxl

from librarian.supplementary import extract_supplementary_records


def _xlsx(sheets):
    workbook = openpyxl.Workbook()
    workbook.remove(workbook.active)
    for name, rows in sheets.items():
        sheet = workbook.create_sheet(name)
        for row in rows:
            sheet.append(row)
    buffer = io.BytesIO()
    workbook.save(buffer)
    return buffer.getvalue()


class SupplementaryTableTest(unittest.TestCase):
    def test_csv_repeats_header_in_every_record(self):
        rows = ["gene,log2FC,padj"] + [f"GENE{i},{i}.5,0.01" for i in range(60)]
        records = extract_supplementary_records(
            {"s1.csv": "\n".join(rows).encode()}, {}, max_records=40
        )
        self.assertGreater(len(records), 1)
        for record in records:
            self.assertTrue(record["text"].startswith("gene | log2FC | padj ; "))
        self.assertIn("GENE59 | 59.5 | 0.01", records[-1]["text"])

    def test_tsv_uses_tab_delimiter(self):
        records = extract_supplementary_records(
            {"s1.tsv": b"strain\tgenotype\nMG1655\twild type\n"}, {}, max_records=40
        )
        self.assertEqual(records[0]["text"], "strain | genotype ; MG1655 | wild type")

    def test_xlsx_titles_records_with_sheet_and_caps_them(self):
        data = _xlsx(
            {
                "Hits": [["gene", "score"]] + [[f"G{i}", i * 1.0] for i in range(500)],
                "Primers": [
                    ["name", "sequence"],
                    ["SOX2-F", "GGCAGCTACAGCATGATGCAGGA"],
                ],
            }
        )
        records = extract_supplementary_records(
            {"table_s1.xlsx": data}, {}, max_records=5
        )
        self.assertEqual(len(records), 5)
        self.assertTrue(records[0]["section_title"].endswith(', sheet "Hits"'))
        self.assertIn("G0 | 0", records[0]["text"])  # 0.0 shown as 0
