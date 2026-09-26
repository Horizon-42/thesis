#!/usr/bin/env bash
# Re-download every publicly downloadable source cited in README.md (roll rates of transport
# aircraft, certification roll performance, procedure-design reaction / bank-establishment
# allowances, ATC instruction -> aircraft response delays; retrieved 2026-09-26) into papers/
# next to this script, and print each file's sha256 so a populated folder can be checked
# against the table in README.md section 5.
#
#   bash docs/literature/roll_rate_and_turn_response/download.sh
#
# An existing non-empty file is NEVER overwritten: the script skips it and still prints its
# sha256. To re-fetch a file, move the old copy aside first. About 130 MB. PDFs are git-ignored
# (root .gitignore has *.pdf; the one .xml is not).
#
# Requests are sequential with a pause between them (FETCH_SLEEP, default 2 s).
#
# Server quirks (all checked 2026-09-26):
#  - rosap.ntl.bts.gov (Cardosi & Boole 1991) answers 403 to a bare curl user agent; the browser
#    user agent plus Accept headers below are needed.
#  - everyspec.com (MIL specs, a mirror, not the DoD's own ASSIST site) needs the Referer header
#    of the spec's own page.
#  - The NTSB docket link needs the EXACT FileName parameter shown; with any other FileName the
#    server answers HTTP 200 with an empty body.
#  - The EASA link (downloads/139073) is CS-25 Amendment 28 as corrected on 20/11/2025; EASA
#    replaced that file once already, so the sha256 may change without a new amendment.
#  - The UK CAA link serves ICAO Doc 8168 Vol II, 5th edition (2006) with Amendment 3, as filed
#    for airspace change proposal ACP-2015-02. It is NOT the current (7th, 2020) edition.
#  - The eCFR link is the versioner API for the text in force on 2026-09-01 (needs --compressed).
#  - Three files get a NEW sha256 on every download although the content is the same (checked by
#    fetching twice): the USPTO patent PDF (its CreationDate is the time of the request; 4 bytes
#    differ) and both everyspec MIL PDFs (about 30 bytes of the trailer ID at the end differ;
#    extracted text identical). For those three, compare size and page count, not the sha256.
set -u

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
D="${ROLL_RATE_PAPERS:-$HERE/papers}"
UA='Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36'
SLEEP="${FETCH_SLEEP:-2}"
NTSB='https://data.ntsb.gov/Docket/Document/docBLOB'
rc=0

get() {  # get <file name under papers/> <url> [extra curl args...]
    local name="$1" url="$2"; shift 2
    local out="$D/$name"
    mkdir -p "$D"
    if [ -s "$out" ]; then
        printf 'skip     %-72s %s\n' "$name" "$(sha256sum "$out" | cut -d' ' -f1)"
        return 0
    fi
    local code
    code=$(curl -sS -A "$UA" -L --compressed --retry 3 --retry-delay 5 --retry-all-errors \
                "$@" -o "$out.part" -w '%{http_code}' "$url") || code="curl-error"
    if [ "$code" != "200" ] || [ ! -s "$out.part" ]; then
        printf 'FAILED   %-72s HTTP %s (empty body counts as a failure)\n' "$name" "$code" >&2
        rm -f "$out.part"; rc=1
        sleep "$SLEEP"; return 1
    fi
    mv "$out.part" "$out"
    printf 'fetched  %-72s %s\n' "$name" "$(sha256sum "$out" | cut -d' ' -f1)"
    sleep "$SLEEP"
}

echo "== 1. Roll rates and bank in operations / under the autopilot"
get NASA_TN_D-5957_Holleman_1970_roll_requirements_transports.pdf \
    'https://ntrs.nasa.gov/api/citations/19700029309/downloads/19700029309.pdf'
get NTSB_DCA09MA026_Att23_Airbus_A320_FCOM_1.27.20_Normal_Law.pdf \
    "$NTSB?FileExtension=.PDF&FileName=Operations%2FHuman+Performance+2X+-+Attachment+23%3A+Airbus+FCOM+1+Flight+Control+Normal+Law-Master.PDF&ID=40313401"
get FAA_AC_25.1329-1C_Chg2.pdf \
    'https://www.faa.gov/documentLibrary/media/Advisory_Circular/AC_25.1329-1C_CHG2.pdf'
get FAA_AC_25.671-1.pdf \
    'https://www.faa.gov/documentLibrary/media/Advisory_Circular/AC_25.671-1.pdf'
get MIL-F-9490D_1975_via_everyspec.pdf \
    'https://everyspec.com/MIL-SPECS/MIL-SPECS-MIL-F/download.php?spec=MIL-F-9490D.026060.pdf' \
    -e 'https://everyspec.com/MIL-SPECS/MIL-SPECS-MIL-F/MIL-F-9490D_5297/'
get Honeywell_US5023796A_1991_heading_track_hold_patent.pdf \
    'https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/5023796'

echo
echo "== 2. Certification roll performance"
get eCFR_14CFR_25.147_2026-09-01.xml \
    'https://www.ecfr.gov/api/versioner/v1/full/2026-09-01/title-14.xml?part=25&section=25.147'
get FAA_AC_25-7D_Chg1.pdf \
    'https://www.faa.gov/documentLibrary/media/Advisory_Circular/AC_25-7D_Chg_1.pdf'
get EASA_CS-25_Amdt28.pdf \
    'https://www.easa.europa.eu/en/downloads/139073/en'
get MIL-F-8785C_1980_via_everyspec.pdf \
    'https://everyspec.com/MIL-SPECS/MIL-SPECS-MIL-F/download.php?spec=MIL-F-8785C.028392.pdf' \
    -e 'https://everyspec.com/MIL-SPECS/MIL-SPECS-MIL-F/MIL-F-8785C_5295/'

echo
echo "== 3. Procedure design (PANS-OPS, TERPS, PBN)"
get ICAO_Doc8168_VolII_via_UKCAA_ACP-2015-02.pdf \
    'https://www.caa.co.uk/publication/download/17061'
get FAA_Order_8260.3G.pdf \
    'https://www.faa.gov/documentLibrary/media/Order/Order_8260.3G.pdf'
get FAA_Order_8260.58D.pdf \
    'https://www.faa.gov/documentLibrary/media/Order/Order_8260.58D.pdf'

echo
echo "== 4. ATC instruction -> aircraft response"
get FAA_RD-91-20_Cardosi_Boole_1991_Pilot_Response_Time.pdf \
    'https://rosap.ntl.bts.gov/view/dot/9082/dot_9082_DS1.pdf' \
    -H 'Accept: text/html,application/xhtml+xml,application/xml;q=0.9,application/pdf,*/*;q=0.8' \
    -H 'Accept-Language: en-US,en;q=0.9'
get NASA_Lutz_Chatterji_Idris_2022_AIAA_response_times.pdf \
    'https://ntrs.nasa.gov/api/citations/20220007116/downloads/20220007116_Lutz_Aviation2022_NLP.pdf'
get Wuestenbecker2025_CEAS_pilot_response_time_DLR-elib.pdf \
    'https://elib.dlr.de/219500/1/Analysis_and_Prediction_of_Pilot_Response_Time_to_Air_Traffic_Control_Clearances.pdf'
get MITLL_ATC-265_Hollister_1998_A320_breakouts.pdf \
    'https://archive.ll.mit.edu/mission/aviation/publications/publication-files/atc-reports/Hollister_1998_ATC-265_WW-15318.pdf'
get MITLL_ATC-263_Hollister_1998_B747-400_breakouts.pdf \
    'https://archive.ll.mit.edu/mission/aviation/publications/publication-files/atc-reports/Hollister_1998_ATC-263_WW-15318.pdf'
get EUROCONTROL_GUID-159_STCA_Part_III_Ed1.0.pdf \
    'https://www.eurocontrol.int/sites/default/files/2019-09/eurocontrol-guidelines-159-part-iii-1.0.pdf'

exit "$rc"
