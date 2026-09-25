#!/usr/bin/env bash
# Re-download every document cited in README.md (reduced-flap approach speeds for E75L, B737,
# A319, E170, E190; retrieved 2026-09-25) into papers/ next to this script, and print each
# file's sha256 so a populated folder can be checked against the table in README.md.
#
#   bash docs/literature/approach_speeds_reduced_flap/download.sh
#
# An existing non-empty file is NEVER overwritten: the script skips it and still prints its sha256.
# About 36 MB. PDFs are git-ignored (root .gitignore has *.pdf; the .zip and .xml are not), so run
# this after a fresh clone.
#
# Two URLs serve "the current issue", not a fixed file: the two EASA TCDS links
# (downloads/16507, downloads/7525) will return a newer issue once EASA publishes one, and the
# sha256 will then differ from README.md. The NTSB docket links need the EXACT FileName
# parameter shown here; with any other FileName the server answers HTTP 200 with an empty body.
set -u

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
D="${APPROACH_SPEEDS_PAPERS:-$HERE/papers}"
UA='Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36'
SLEEP="${FETCH_SLEEP:-2}"   # politeness delay between requests
NTSB='https://data.ntsb.gov/Docket/Document/docBLOB'
rc=0

get() {  # get <file name under papers/> <url>
    local name="$1" url="$2" out="$D/$1"
    mkdir -p "$D"
    if [ -s "$out" ]; then
        printf 'skip     %-72s %s\n' "$name" "$(sha256sum "$out" | cut -d' ' -f1)"
        return 0
    fi
    local code
    code=$(curl -sS -A "$UA" -L --compressed --retry 3 --retry-delay 5 --retry-all-errors \
                -o "$out.part" -w '%{http_code}' "$url") || code="curl-error"
    if [ "$code" != "200" ] || [ ! -s "$out.part" ]; then
        printf 'FAILED   %-72s HTTP %s (empty body counts as a failure)\n' "$name" "$code" >&2
        rm -f "$out.part"; rc=1
        sleep "$SLEEP"; return 1
    fi
    mv "$out.part" "$out"
    printf 'fetched  %-72s %s\n' "$name" "$(sha256sum "$out" | cut -d' ' -f1)"
    sleep "$SLEEP"
}

echo "== Definitions"
get FAA_InFO23001_2023-01-09.pdf \
    'https://www.faa.gov/sites/faa.gov/files/InFO23001.pdf'
# 14 CFR 97.3 as in force on 2026-09-01 (eCFR versioner API; needs --compressed)
get eCFR_14CFR_97.3_2026-09-01.xml \
    'https://www.ecfr.gov/api/versioner/v1/full/2026-09-01/title-14.xml?part=97&section=97.3'

echo
echo "== B737 (Boeing 737-700)"
get Boeing_FAA_Reference_Code_and_Approach_Speeds_2016-03-30.pdf \
    'https://www.boeing.com/content/dam/boeing/v2/airports/faq/arcandapproachspeeds.pdf'
get FAA_FSB_Boeing737_Rev17_2020-11-16.pdf \
    'https://www.faa.gov/sites/faa.gov/files/2022-08/737_FSB_Report.pdf'
get NTSB_DCA13FA131_Operations_Group_Chairman_Factual.pdf \
    "$NTSB?ID=40418230&FileExtension=.PDF&FileName=Operations%20-%20Group%20Chairman%27s%20Factual%20Report-Master.PDF"
get NTSB_DCA19IA036_Aircraft_Performance_Study.pdf \
    "$NTSB?ID=11940941&FileExtension=pdf&FileName=DCA19IA036_perfstudy_B-Rel.pdf"

echo
echo "== A319"
get Airbus_AC_A319_0624.pdf \
    'https://www.aircraft.airbus.com/sites/g/files/jlcbta126/files/2024-06/AC_A319_0624.pdf'
get FAA_FSB_A320_family_Rev7_Draft.pdf \
    'https://www.faa.gov/aircraft/draft_docs/fsb/FSBR_A320_Rev_7_Draft.pdf'
get EASA_TCDS_A064_A318-A321_Iss62_2026-06-26.pdf \
    'https://www.easa.europa.eu/en/downloads/16507/en'
get NTSB_DCA11IA040_Ops2_Att7_UAL_A319-A320_Landing_Performance.pdf \
    "$NTSB?ID=40355135&FileExtension=.PDF&FileName=Operations+2+-+Attachment+7+-+Landing+Performance+Corrections-Master.PDF"
get NTSB_DCA12IA096_Ops2_Att2_JetBlue_A320_QRH_Landing_Performance.pdf \
    "$NTSB?ID=40384278&FileExtension=.PDF&FileName=Operations+2+-+Attachment+2+-+Landing+Performance-Master.PDF"

echo
echo "== E-Jets (E170, E175 = E75L, E190)"
get Embraer_APM-1901_E190_Rev24_2024-11-29.pdf \
    'https://embraer.com/media/jwsj1u1u/e190apm_apm_190-1.pdf'
get Embraer_E175_APM_NTSB_DCA20IA014_Att2.pdf \
    "$NTSB?ID=13691419&FileExtension=pdf&FileName=DCA20IA014+Attachment+2+-+Embraer+175+Airport+Planning+Manual-Rel.pdf"
get Embraer_E175_spec.pdf \
    'https://www.embraer.com/media/o3sjzbwl/e175_spec.pdf'
get EASA_OEB_Embraer170-195_Rev5_2013-07-01.pdf \
    'https://www.easa.europa.eu/sites/default/files/dfu/20130701%20Embraer%20ERJ%20170-175-190-195%20EASA%20OEB%20Report%20Rev%205.pdf'
get TCCA_OE_Report_Embraer_E-Jets_Rev1_2023-11-03.pdf \
    'https://tc.canada.ca/sites/default/files/2023-12/OE_EMBRAER_E170_E-JETS_OE_REPORT_REVISION_1.pdf'
get EASA_TCDS_IM_A_001_ERJ170_Iss13_2022-01-31.pdf \
    'https://www.easa.europa.eu/en/downloads/7525/en'
get NTSB_AAR-08-01_Shuttle_America_6448_ERJ170.pdf \
    'https://www.ntsb.gov/investigations/AccidentReports/Reports/AAR0801.pdf'
get NTSB_DCA07MA072_Ops2_Factual.pdf \
    "$NTSB?ID=40279863&FileExtension=.PDF&FileName=Operations%202%20-%20Factual%20Report%20of%20Group%20Chairman-Master.PDF"
get NTSB_DCA07MA072_Ops2_Att5_ERJ170_POH_Normal_Procedures.pdf \
    "$NTSB?ID=40279825&FileExtension=.PDF&FileName=Operations%202%20-%20Attachment%205%20ERJ-170%20POH%20Normal%20Procedures-Master.PDF"

echo
echo "== Lead only: ICAO/ECAC ANP noise-model data (a per-flap approach CAS coefficient, not a VREF)"
get ANP_archive_v2.3.zip \
    'https://www.easa.europa.eu/en/downloads/138164/en'
get ECAC_Doc29_4th_Ed_Vol2_Dec2016.pdf \
    'https://www.ecac-ceac.org/images/documents/ECAC-Doc_29_4th_edition_Dec_2016_Volume_2.pdf'

exit "$rc"
