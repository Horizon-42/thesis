"""Read the FAA ACD xlsx (first sheet) with stdlib only -> acd.json (list of dicts)."""
import json, re, zipfile, xml.etree.ElementTree as ET

from _paths import ACD_XLSX as P, WORK
NS = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
z = zipfile.ZipFile(P)
ss = [ "".join(t.text or "" for t in si.iter("{%s}t" % NS["m"])) for si in ET.fromstring(z.read("xl/sharedStrings.xml")).findall("m:si", NS)]
wb = ET.fromstring(z.read("xl/workbook.xml"))
sheets = [s.get("name") for s in wb.find("m:sheets", NS)]
print(sheets, [n for n in z.namelist() if "sheet" in n])
def col(ref):
    s = re.match(r"[A-Z]+", ref).group(); n = 0
    for ch in s: n = n * 26 + ord(ch) - 64
    return n - 1
root = ET.fromstring(z.read("xl/worksheets/sheet1.xml"))
rows = []
for r in root.iter("{%s}row" % NS["m"]):
    vals = {}
    for c in r.findall("m:c", NS):
        v = c.find("m:v", NS); t = c.get("t")
        if v is None:
            is_ = c.find("m:is", NS)
            val = "".join(x.text or "" for x in is_.iter("{%s}t" % NS["m"])) if is_ is not None else None
        else:
            val = ss[int(v.text)] if t == "s" else v.text
        vals[col(c.get("r"))] = val
    rows.append(vals)
hdr = [rows[0].get(i) for i in range(max(rows[0]) + 1)]
out = [{hdr[i]: r.get(i) for i in range(len(hdr))} for r in rows[1:] if r]
(WORK / "acd.json").write_text(json.dumps(out))
print(len(out)); print(hdr)
