#!/usr/bin/env bash
# Re-fetch every source used by README.md (URLs as of 2026-09-14). One curl per file.
# PDFs are gitignored (root .gitignore has *.pdf); the HTML pages are the text copies.
# NOT fetched here (cross-linked instead): the 7110.65BB consolidated PDF, 7210.3EE, the AIM,
# ICAO Doc 9643 (SOIR) and the Erzberger-Itoh NASA TP. Run ../runway_assignment/download.sh.
# Needs: curl; ghostscript (gs) for the two excerpts; pdftotext + python3 for the parsed CSV.
set -u
cd "$(dirname "$0")"
mkdir -p papers
UA='Mozilla/5.0 (X11; Linux x86_64; rv:128.0) Gecko/20100101 Firefox/128.0'
ATC=https://www.faa.gov/air_traffic/publications/atpubs/atc_html
FOA=https://www.faa.gov/air_traffic/publications/atpubs/foa_html
PCG=https://www.faa.gov/air_traffic/publications/atpubs/pcg_html
AIP=https://www.faa.gov/air_traffic/publications/atpubs/aip_html
ORD=https://www.faa.gov/documentLibrary/media/Order
NOT=https://www.faa.gov/documentLibrary/media/Notice

# --- FAA JO 7110.65BB, Change 3 (eff. 2026-07-09): HTML, one page per section -------------
curl -sSL -A "$UA" -o papers/7110.65BB_index.html                                        "$ATC/index.html"
curl -sSL -A "$UA" -o papers/7110.65BB_chap0_section_0_Explanation_of_Changes.html        "$ATC/chap0_section_0.html"
curl -sSL -A "$UA" -o papers/7110.65BB_chap2_section_1_General.html                       "$ATC/chap2_section_1.html"
curl -sSL -A "$UA" -o papers/7110.65BB_chap3_section_9_Departure_Procedures_and_Separation.html "$ATC/chap3_section_9.html"
curl -sSL -A "$UA" -o papers/7110.65BB_chap3_section_10_Arrival_Procedures_and_Separation.html  "$ATC/chap3_section_10.html"
curl -sSL -A "$UA" -o papers/7110.65BB_chap5_section_5_Radar_Separation.html              "$ATC/chap5_section_5.html"
curl -sSL -A "$UA" -o papers/7110.65BB_chap5_section_9_Radar_Arrivals.html                "$ATC/chap5_section_9.html"
curl -sSL -A "$UA" -o papers/7110.65BB_chap7_section_4_Approaches.html                    "$ATC/chap7_section_4.html"

# --- JO 7210.3EE (Change 3), Pilot/Controller Glossary (Change 3), US AIP GEN 1.7 ----------
curl -sSL -A "$UA" -o papers/7210.3EE_index.html                     "$FOA/"
curl -sSL -A "$UA" -o papers/7210.3EE_chap10_section_4_Services.html "$FOA/chap10_section_4.html"
curl -sSL -A "$UA" -o papers/PCG_index.html                          "$PCG/"
curl -sSL -A "$UA" -o papers/PCG_glossary-a.html                     "$PCG/glossary-a.html"
curl -sSL -A "$UA" -o papers/PCG_glossary-c.html                     "$PCG/glossary-c.html"
curl -sSL -A "$UA" -o papers/PCG_glossary-p.html                     "$PCG/glossary-p.html"
curl -sSL -A "$UA" -o papers/PCG_glossary-s.html                     "$PCG/glossary-s.html"
curl -sSL -A "$UA" -o papers/US_AIP_index.html                       "$AIP/index.html"
curl -sSL -A "$UA" -o papers/US_AIP_GEN_1.7_Differences_from_ICAO.html "$AIP/part1_gen_section_1.7.html"

# --- edition / change / notice evidence -----------------------------------------------------
curl -sSL -A "$UA" -o papers/FAA_atpubs_publications_index.html "https://www.faa.gov/air_traffic/publications/"
curl -sSL -A "$UA" -o papers/FAA_at_notices_7110.65BB.html      "https://www.faa.gov/air_traffic/publications/at_notices/?documentID=1043461"
curl -sSL -A "$UA" -o papers/FAA_at_notices_7360.1K.html        "https://www.faa.gov/air_traffic/publications/at_notices/?documentID=1043797"

# --- notices in force against 7110.65BB and 7360.1K ------------------------------------------
curl -sSL -A "$UA" -o papers/FAA_N_JO_7110.801_Interim_Helicopter_Separation_eff2026-03-18.pdf "$NOT/GENOT_N_JO_7110.801_Interim_Helicopter_Separation_Procedures.pdf"
curl -sSL -A "$UA" -o papers/FAA_N_JO_7110.802_Traffic_Advisories_eff2026-05-19.pdf           "$NOT/N_JO_7110.802_Traffic_Advisories.pdf"
curl -sSL -A "$UA" -o papers/FAA_N_JO_7110.803_Separation_eff2026-10-28.pdf                   "$NOT/2026-04-28_Notice_7110.803_Multiple_Paragraphs_Separation_v2.pdf"
curl -sSL -A "$UA" -o papers/FAA_N_JO_7110.804_Visual_Separation_eff2026-10-28.pdf            "$NOT/2026-04-24_Notice_7110.804_7-2-1_Visual_Separation.pdf"
curl -sSL -A "$UA" -o papers/FAA_N_JO_7110.805_Approaches_to_Multiple_Runways_eff2026-06-19.pdf "$NOT/2026-02-24_Notice_7110.805_Apchs_to_Multiple_Runways_FINAL_Updated.pdf"
curl -sSL -A "$UA" -o papers/FAA_N_JO_7110.806_MEARTS_3NM_eff2026-09-14.pdf                   "$NOT/2026-08-11_Notice_7110.806_MEARTS_3_NM_Separation_Utilizing_FDM_v2.pdf"
curl -sSL -A "$UA" -o papers/FAA_N_JO_7360.7_Change_to_7360.1_eff2025-10-14.pdf               "$NOT/GENOT_N_JO_7360.7_Change_to_FAA_Order_JO_7360.1_Aircraft_Type_Designators.pdf"

# --- supplementary FAA orders ------------------------------------------------------------------
curl -sSL -A "$UA" -o papers/FAA_JO_7110.308E_Simultaneous_Dependent_Approaches_CSPR_eff2023-10-05.pdf "$ORD/JO_7110.308E_Simultaneous_Dependent_Approaches_to_CSPR.pdf"
curl -sSL -A "$UA" -o papers/FAA_JO_7110.110B_DCIA_with_CRDA_eff2017-11-17.pdf "$ORD/FAA_Order_JO_7110.110B_Dependent_Converging_Instrument_Approaches_(DCIA)_with_Converging_Runway_Display_Aid_(CRDA).pdf"
curl -sSL -A "$UA" -o papers/FAA_JO_7110.663A_MCPRS_with_CRDA_eff2024-03-01.pdf "$ORD/2023-01-30_Order_7110.663A_MCPRS-CRDA_FINAL.pdf"

# --- international and secondary -------------------------------------------------------------
curl -sSL -A "$UA" -o papers/EUROCONTROL_RECAT-EU_Edition_2.0_2024-11-08.pdf "https://www.eurocontrol.int/sites/default/files/2024-12/eurocontrol-recat-eu-edition-2-0.pdf"
curl -sSL -A "$UA" -o papers/SKYbrary_separation-standards.html            "https://skybrary.aero/articles/separation-standards"
curl -sSL -A "$UA" -o papers/SKYbrary_mitigation-wake-turbulence-hazard.html "https://skybrary.aero/articles/mitigation-wake-turbulence-hazard"
curl -sSL -A "$UA" -o papers/SKYbrary_icao-wake-turbulence-category.html   "https://skybrary.aero/articles/icao-wake-turbulence-category"
# icao.int answered curl with HTTP 403 on 2026-09-14 (the copy in papers/ was saved through a
# browser-like fetch). If this line leaves an HTML error page, save the PDF from a browser.
curl -sSL -f -A "$UA" -o papers/ICAO_APAC_SAIOSEACG4_IP03_PANS-ATM_Amendment12_surveillance_minima_2025-03.pdf \
  "https://www.icao.int/sites/default/files/APAC/Meetings/2025/2025%20SAIOSEACG4/4-Information%20Papers/A4-IP03-Update-on-the-amendment-concerning-separation-minima-based-on-a.pdf" \
  || echo "FAIL (expected: icao.int blocks curl) ICAO APAC IP/03 - save it from a browser"

# --- excerpts cut from two large PDFs (full PDFs go to a temp dir and are deleted) --------------
# gs prints "pdfmark destination page N points beyond the last page": harmless (outline links
# into pages that are not in the excerpt). The excerpt's md5 changes per run; its text does not.
T=$(mktemp -d)
curl -sSL -A "$UA" -o "$T/7110.65BB_basic.pdf" "$ORD/7110.65BB_Basic_dtd_2-20-25.pdf"
gs -q -dNOPAUSE -dBATCH -dSAFER -sDEVICE=pdfwrite -sPageList=1,310-314 \
  -sOutputFile=papers/FAA_JO_7110.65BB_Basic_eff2025-02-20_EXCERPT_5-5-4_legacy_wake.pdf "$T/7110.65BB_basic.pdf"
curl -sSL -A "$UA" -o "$T/7360.1K.pdf" "$ORD/FAA_Order_JO_7360.1K_Aircraft_Type_Designators.pdf"
gs -q -dNOPAUSE -dBATCH -dSAFER -sDEVICE=pdfwrite -sPageList=1-121 \
  -sOutputFile=papers/FAA_JO_7360.1K_Aircraft_Type_Designators_eff2025-06-12_EXCERPT.pdf "$T/7360.1K.pdf"

# --- parsed CSV of 7360.1K Appendix A (PDF pp. 10-121); see README section 2.5 -------------------
# CWT / SRS / LAHSO tokens are assigned to columns by their position under each page's own
# "CWT  SRS  LAHSO" header (a lone "I" could be either column); the first token that is not a
# code, or a second code for a filled column, starts the model text.
pdftotext -layout "$T/7360.1K.pdf" "$T/7360.1K.txt"
python3 - "$T/7360.1K.txt" papers/FAA_JO_7360.1K_AppendixA_categories_parsed.csv <<'PY'
import csv, re, sys
pages = open(sys.argv[1], encoding="utf-8", errors="replace").read().split("\f")
CLASS = r"(?:[@$]?Fixed-wing|Gyroplane|Helicopter|Powered-lift)"
HEAD = re.compile(r"^\s{0,8}([A-Z0-9]{1,4}\*?)\s+(" + CLASS + r")\s+(\d{1,2}[A-Z]/[JHLS]\+?)")
WTC = re.compile(r"\s+(Light/Medium|Medium/Heavy|Light|Medium|Heavy|Super)")
PAT = {"CWT": r"[A-I]", "SRS": r"I{1,3}", "LAHSO": r"\d{1,2}"}
rows = []
for p in range(10, 122):
    lines = pages[p - 1].splitlines()
    hdr = next(l for l in lines if re.search(r"CWT\s+SRS\s+LAHSO", l))
    col = {k: hdr.index(k) for k in PAT}
    for line in lines:
        m = HEAD.match(line)
        if not m:
            continue
        pos, wtc = m.end(), ""
        mw = WTC.match(line, pos)
        if mw and mw.start(1) < col["CWT"] - 1:
            wtc, pos = mw.group(1), mw.end()
        vals = dict.fromkeys(PAT, "")
        for t in re.finditer(r"\S+", line[pos:]):
            start, tok = pos + t.start(), t.group(0)
            fits = [k for k in PAT if re.fullmatch(PAT[k], tok)]
            if not fits or start > col["LAHSO"] + 8:
                break
            key = min(fits, key=lambda k: abs(col[k] - start))
            if vals[key]:
                break
            vals[key] = tok
        rows.append([m.group(1), m.group(2), m.group(3), wtc, vals["CWT"], vals["SRS"], vals["LAHSO"], p])
with open(sys.argv[2], "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["type_designator", "class", "engine_number_type_faa_weight_class", "icao_wtc",
                "cwt", "srs", "lahso_group", "pdf_page"])
    w.writerows(rows)
print("parsed", len(rows), "Appendix A rows (expected 2653)")
PY
rm -rf "$T"
