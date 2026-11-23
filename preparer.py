import random
from datasets import load_dataset
from openpyxl import Workbook
from openpyxl.styles import Alignment, Font
from openpyxl.worksheet.datavalidation import DataValidation

ds = load_dataset(
    "tblard/allocine",
    data_files={"test": "allocine/test-00000-of-00001.parquet"},
    split="test",
)

random.seed(42)
courts = [i for i in range(len(ds)) if 150 < len(ds[i]["review"]) < 500]
choisis = random.sample(courts, 50)

wb = Workbook()
ws = wb.active
ws.title = "annotation"
ws.append(["id", "critique", "scenario", "acteurs",
           "realisation", "musique", "duree"])
for c in ws[1]:
    c.font = Font(bold=True)

for i in choisis:
    ws.append([i, ds[i]["review"], None, None, None, None, None])

ws.column_dimensions["A"].width = 8
ws.column_dimensions["B"].width = 80
for col in "CDEFG":
    ws.column_dimensions[col].width = 14

for row in ws.iter_rows(min_row=2, max_row=51):
    row[1].alignment = Alignment(wrap_text=True, vertical="top")
    for cell in row[2:]:
        cell.alignment = Alignment(vertical="top", horizontal="center")

dv = DataValidation(type="list", formula1='"positif,negatif,neutre"',
                    allow_blank=True)
ws.add_data_validation(dv)
dv.add("C2:G51")

ws.freeze_panes = "C2"
wb.save("annotation.xlsx")
print("OK : annotation.xlsx cree avec 50 critiques")