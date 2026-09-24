#!/usr/bin/env bash
# Re-download every public file of the aircraft-performance source survey (2026-09-23) into
# data/aircraft_performance/<source>/ (git-ignored) and print the sha256 of each file.
#
#   bash docs/literature/aircraft_performance/download.sh
#
# An existing file is NEVER overwritten: the script skips it and still prints its sha256, so a
# populated tree can be checked against the tables in README.md.
# About 45 MB, plus 80 EUROCONTROL APD pages fetched one at a time with a pause between requests.
#
# Not touched here: data/aircraft_performance/airframes/ and download_airframes.sh, which belong to
# the separate airframe-facts pack in this folder.
#
# Provenance, licence quotes and what each file was read for: README.md in this folder.
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
# Run from the main checkout. From a git worktree, point this at the main tree's data instead
# (a worktree has no data/ of its own, and the script would start a fresh download there).
D="${AIRCRAFT_PERF_DATA:-$ROOT/data/aircraft_performance}"
UA='Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36'
SLEEP="${FETCH_SLEEP:-3}"        # politeness delay between requests
rc=0

get() {  # get <relative path under data/aircraft_performance> <url>
    local rel="$1" url="$2" out="$D/$1"
    mkdir -p "$(dirname "$out")"
    if [ -s "$out" ]; then
        printf 'skip     %-78s %s\n' "$rel" "$(sha256sum "$out" | cut -d' ' -f1)"
        return 0
    fi
    local code
    code=$(curl -sS -A "$UA" -L --retry 3 --retry-delay 5 --retry-all-errors \
                -o "$out.part" -w '%{http_code}' "$url") || code="curl-error"
    if [ "$code" != "200" ] || [ ! -s "$out.part" ]; then
        printf 'FAILED   %-78s HTTP %s\n' "$rel" "$code" >&2
        rm -f "$out.part"; rc=1
        sleep "$SLEEP"; return 1
    fi
    mv "$out.part" "$out"
    printf 'fetched  %-78s %s\n' "$rel" "$(sha256sum "$out" | cut -d' ' -f1)"
    sleep "$SLEEP"
}

echo "== 1. OpenAP (PyPI files; sha256 must equal the PyPI digests quoted in README)"
PY='https://files.pythonhosted.org/packages'
get openap/openap-2.6.2-py3-none-any.whl \
    "$PY/d2/99/e1c97d17a08a64e2ae7e8118af412bbb6deab17a6fb88b08c46aff14c47f/openap-2.6.2-py3-none-any.whl"
get openap/openap-2.6.2.tar.gz \
    "$PY/b0/2f/f84ea8ee49984a41d950d5e16bfd693e883cf10c21c063e969dc99bfaf3e/openap-2.6.2.tar.gz"
get openap/openap-2.4-py3-none-any.whl \
    "$PY/bf/3c/e44ba37ec362e9bbe13ba5aa17b625a0ff955272e8b7e83e826ee7b10fa5/openap-2.4-py3-none-any.whl"
# GitHub master copy of the synonym table (changes if upstream edits it; 2026-09-23 = release 2.6.2)
get openap/gh_master_aircraft_synonym.csv \
    'https://raw.githubusercontent.com/junzis/openap/master/openap/data/aircraft/_synonym.csv'

echo
echo "== 2. pycontrails / Poll-Schumann"
get pycontrails/pycontrails-0.63.5-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl \
    "$PY/a9/31/738a8d09877603988ffb5e36f5ef66ac9369a2c9092474dce8d34888e07a/pycontrails-0.63.5-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl"
GH='https://raw.githubusercontent.com/contrailcirrus/pycontrails/main'
get pycontrails/LICENSE_main.txt "$GH/LICENSE"
get pycontrails/NOTICE_main.txt "$GH/NOTICE"
get pycontrails/pycontrails_main.bib "$GH/docs/_static/pycontrails.bib"
EL='https://elib.dlr.de'
get pycontrails/papers/Poll_Schumann_2021_Part1_aer.2020.62_DLR-elib.pdf \
    "$EL/135592/1/Poll_Schumann_estimation_method_fuel_burn_performance_aircraft_cruise_part_1_fundamentals_2020.pdf"
get pycontrails/papers/Poll_Schumann_2021_Part2_aer.2020.124_DLR-elib.pdf \
    "$EL/139037/1/Poll_Schumann_fuel_burn_aircraft_cruise_part_2__aircrafts_characteristic_parameters_Aeronautical_J_2020.pdf"
get pycontrails/papers/Poll_Schumann_2025_Part3_aer.2024.141_DLR-elib.pdf \
    "$EL/212199/1/Poll_Schumann_Fuel-burn%20performance%20civil-transport-aircraft-part-3-full-flight-profile.pdf"
get pycontrails/papers/Poll_Schumann_2024_turbofan_engine_all_phases_aer.2024.92_DLR-elib.pdf \
    "$EL/206626/1/Poll_Schumann_2024b_a-simple-model-for-the-estimation-of-turbofan-engine-performance-in-all-airborne-phases-of-flight_Aeronautical_J.pdf"

echo
echo "== 3. EUROCONTROL BADA (public pages and documents only; no licensed data)"
EC='https://www.eurocontrol.int'
get bada/eurocontrol_model_bada.html "$EC/model/bada"
get bada/bada-standard-licence-information.pdf "$EC/sites/default/files/2020-06/bada-standard-licence-information.pdf"
get bada/bada-licence-information-family-4.pdf "$EC/sites/default/files/2025-07/bada-licence-information-family-4.pdf"
get bada/bada-integrated-standard-licence-for-information.pdf \
    "$EC/sites/default/files/2020-06/bada-integrated-standard-licence-for-information.pdf"
get bada/eurocontrol-showcase-summit-bada-latest-evolutions.pdf \
    "$EC/sites/default/files/2025-11/eurocontrol-showcase-summit-bada-latest-evolutions.pdf"
get bada/news_bada_just_got_better.html \
    "$EC/news/bada-atms-most-comprehensive-aircraft-performance-model-just-got-even-better"
get bada/BADA_3.7_Synonym_Aircraft_Report_EEC-TR-2009-007.pdf "$EC/archive_download/all/node/9706"
get bada/BADA_3.8_Synonym_Aircraft_Report_EEC-TR-2010-007.pdf "$EC/archive_download/all/node/9689"
get bada/BADA_3.8_Coverage_2009_traffic.pdf "$EC/archive_download/all/node/9692"
get bada/BADA_3.8_User_Manual_EEC-TR-2010-003.pdf "$EC/archive_download/all/node/9690"
BG='https://raw.githubusercontent.com/eurocontrol-bada'
get bada/EIH-Technical-Report-121122-58-v1.4_BADA4_spec.pdf \
    "$BG/model-specifications/main/BADA_Family_4/EIH-Technical-Report-121122-58-v1.4.pdf"
get bada/model-specifications_README.md "$BG/model-specifications/main/README.md"
get bada/model-specifications_AMENDMENT_TO_EUPL_license.md "$BG/model-specifications/main/AMENDMENT_TO_EUPL_license.md"
get bada/pybada_github/SYNONYM.NEW "$BG/pybada/main/src/aircraft/BADA3/DUMMY/SYNONYM.NEW"
get bada/pybada_github/ReleaseSummary "$BG/pybada/main/src/aircraft/BADA3/DUMMY/ReleaseSummary"

echo
echo "== 4. ANP (EASA) and ECAC Doc 29"
EA='https://www.easa.europa.eu'
get anp/easa_anp_data_page.html \
    "$EA/en/domains/environment/policy-support-and-research/aircraft-noise-and-performance-anp-data"
get anp/anp_aircraft_substitutions_-_jets_heavy_props_22022018_.xlsx "$EA/en/downloads/138165/en"
get anp/easa_verified_anp_aircraft_types.xlsx "$EA/en/downloads/143284/en"
get anp/anp_database_v2.3.pdf "$EA/en/downloads/138159/en"
get anp/archive_anp_v2.3.zip "$EA/en/downloads/138164/en"
get anp/EASA_ANP_Database_Terms_and_Conditions_Issue_3.pdf "$EA/en/downloads/116585/en"
get anp/List_of_ANP_Data_Providers.pdf "$EA/en/downloads/137505/en"
get anp/easa_anp_data_request_form.html "$EA/en/anp-data-request"
get anp/easa_copyright_disclaimer.html "$EA/en/copyright-disclaimer"
get anp/ECAC-Doc_29_4th_edition_Dec_2016_Volume_2.pdf \
    'https://www.ecac-ceac.org/images/documents/ECAC-Doc_29_4th_edition_Dec_2016_Volume_2.pdf'
if [ -s "$D/anp/archive_anp_v2.3.zip" ] && [ ! -d "$D/anp/v2.3" ]; then
    mkdir -p "$D/anp/v2.3" && unzip -q -n "$D/anp/archive_anp_v2.3.zip" -d "$D/anp/v2.3" \
        && echo "unzipped anp/archive_anp_v2.3.zip -> anp/v2.3/"
fi

echo
echo "== 5. EUROCONTROL Aircraft Performance Database (one page per type; 'No' pages are the"
echo "   database's answer for a designator it does not hold, not a download failure)"
for t in A306 A30B A339 ASTR B350 B712 B722 B735 B753 B762 B764 B78X BCS1 BCS3 BE20 BE36 BE40 C208 \
         C25A C25B C25C C25M C525 C560 C56X C650 C680 C68A C700 C750 CL30 CL35 CL60 CRJ1 CRJ2 CRJ7 \
         E135 E50P E545 E55P E75S F2TH F5 F900 FA10 FA20 FA50 FA7X G150 G280 GA6C GALX GL5T GL7T \
         GLEX GLF3 GLF4 GLF5 H25A H25B H25C HDJT LJ31 LJ35 LJ40 LJ45 LJ55 LJ60 MD82 MD83 MD88 P28A \
         PC12 PC24 PRM1 S22T SF50 SR22 T38; do
    get "eurocontrol_apd/$t.html" "https://contentzone.eurocontrol.int/aircraftperformance/details.aspx?ICAO=$t"
done
get eurocontrol_apd/_help.html 'https://contentzone.eurocontrol.int/aircraftperformance/help.aspx'

echo
echo "== 6. ICAO Aircraft Engine Emissions Databank (EASA-hosted). The download id serves the"
echo "   CURRENT issue; README pins issue 32 (20 March 2026) by sha256."
get icao_emissions/easa_icao_edb_page.html "$EA/en/domains/environment/icao-aircraft-engine-emissions-databank"
get icao_emissions/edb-emissions-databank_v32__web_.xlsx "$EA/en/downloads/131424/en"
get icao_emissions/edb-introduction_text_-rev2.pdf "$EA/en/downloads/45576/en"

echo
echo "== 7. Other"
get other/aedt_purchase.html 'https://aedt.faa.gov/Purchase.aspx'
BS='https://raw.githubusercontent.com/TUDelft-CNS-ATM/bluesky/master'
get other/bluesky_BS/D328.xml "$BS/bluesky/resources/performance/BS/aircraft/D328.xml"
get other/bluesky_BS/SB20.xml "$BS/bluesky/resources/performance/BS/aircraft/SB20.xml"
get other/bluesky_BS/LICENSE_bluesky.txt "$BS/LICENSE"

echo
if [ "$rc" -eq 0 ]; then
    echo "all sources present."
else
    echo "one or more downloads FAILED -- see the lines above." >&2
fi
echo
echo "Not fetched by this script (local provenance snapshots, re-query by hand if needed):"
echo "  * openap/pypi_openap.json, openap/gh_*.json, pycontrails/pypi_pycontrails.json,"
echo "    pycontrails/gh_tree_main.json -- PyPI / GitHub API answers of 2026-09-23; they change."
echo "  * The unpacked trees openap/x262, openap/x24, pycontrails/x (unzip the wheels) and the"
echo "    .txt text layers beside the PDFs (pdftotext -layout <pdf> <txt>)."
echo "  * HTML pages are live pages: their sha256 changes whenever the site is touched, even if the"
echo "    numbers do not. excerpts/ pins the numbers that were read."
exit "$rc"
