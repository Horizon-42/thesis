#!/usr/bin/env bash
# Re-download every document of the airframe-facts pack (docs/literature/aircraft_performance/
# airframe_facts_sources.md) to the path it was read from, and print the sha256 of each file.
#
#   bash docs/literature/aircraft_performance/download_airframes.sh
#
# An existing file is NEVER overwritten: the script skips it and still prints its sha256, so a
# populated tree can be checked against the sources table. Web pages (.html) change sha256 whenever
# the site does; the excerpt files in airframe_excerpts/ pin the numbers.
# Paths are relative to the main checkout. From a git worktree (which has no data/ of its own) run
#   REPO_ROOT=/path/to/main/checkout bash .../download_airframes.sh
# Not fetched: the pycontrails Poll-Schumann file (ps_params_20250328, tier-4 cross-check only),
# which belongs to the sibling survey in this folder (download.sh).
set -u

ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
UA='Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36'
SLEEP="${FETCH_SLEEP:-3}"        # politeness delay between requests
rc=0
JAR="$(mktemp)"; trap 'rm -f "$JAR"' EXIT

report() {  # report <status> <rel>
    printf '%-8s %-96s %s\n' "$1" "$2" "$(sha256sum "$ROOT/$2" | cut -d' ' -f1)"
}

get() {  # get <path relative to ROOT> <url>
    local rel="$1" url="$2" out="$ROOT/$1" code
    mkdir -p "$(dirname "$out")"
    if [ -s "$out" ]; then report skip "$rel"; return 0; fi
    code=$(curl -sS -A "$UA" -L --retry 3 --retry-delay 5 --retry-all-errors \
                -o "$out.part" -w '%{http_code}' "$url") || code="curl-error"
    if [ "$code" != "200" ] || [ ! -s "$out.part" ]; then
        printf 'FAILED   %-96s HTTP %s\n' "$rel" "$code" >&2; rm -f "$out.part"; rc=1; sleep "$SLEEP"; return 1
    fi
    mv "$out.part" "$out"; report fetched "$rel"; sleep "$SLEEP"
}

# FAA DRS answers a plain download with HTTP 403. It serves a TCDS after GET /guest/login (keep the
# cookies), then GET /api/browse/documents/summaryguiddocview/<docUniqueId> (its JSON "id" is the file
# id), then GET /api/content/alf/<id>. The docUniqueId is the last element of each source URL.
drs_logged_in=0
get_drs() {  # get_drs <path relative to ROOT> <docUniqueId>
    local rel="$1" uid="$2" out="$ROOT/$1" fid code
    mkdir -p "$(dirname "$out")"
    if [ -s "$out" ]; then report skip "$rel"; return 0; fi
    if [ "$drs_logged_in" = 0 ]; then
        curl -sS -A "$UA" -c "$JAR" -b "$JAR" -o /dev/null https://drs.faa.gov/guest/login; drs_logged_in=1; sleep "$SLEEP"
    fi
    fid=$(curl -sS -A "$UA" -c "$JAR" -b "$JAR" "https://drs.faa.gov/api/browse/documents/summaryguiddocview/$uid" \
          | python3 -c 'import json,sys; print(json.load(sys.stdin)["id"])') || fid=""
    sleep "$SLEEP"
    if [ -z "$fid" ]; then printf 'FAILED   %-96s DRS lookup\n' "$rel" >&2; rc=1; return 1; fi
    code=$(curl -sS -A "$UA" -c "$JAR" -b "$JAR" -o "$out.part" -w '%{http_code}' \
                "https://drs.faa.gov/api/content/alf/$fid") || code="curl-error"
    if [ "$code" != "200" ] || [ ! -s "$out.part" ]; then
        printf 'FAILED   %-96s HTTP %s\n' "$rel" "$code" >&2; rm -f "$out.part"; rc=1; sleep "$SLEEP"; return 1
    fi
    mv "$out.part" "$out"; report fetched "$rel"; sleep "$SLEEP"
}

echo "== EASA type certificate data sheets (aircraft and engine)"
get data/aircraft_performance/airframes/easa/EASA_TCDS_A_008_Falcon2000_Issue13.pdf 'https://www.easa.europa.eu/en/downloads/7510/en'
get data/aircraft_performance/airframes/easa/EASA_TCDS_A_010_TBM700_Issue22.pdf 'https://www.easa.europa.eu/en/downloads/16508/en'
get data/reference_speeds/easa/EASA_TCDS_A_062_MF50_MF900_F900EX_Issue10.pdf 'https://www.easa.europa.eu/en/downloads/7401/en'
get data/aircraft_performance/airframes/easa/EASA_TCDS_A_089_PC-12_Issue11.pdf 'https://www.easa.europa.eu/en/downloads/7348/en'
get data/aircraft_performance/airframes/easa/EASA_TCDS_E_018_BR700-710_Issue16.pdf 'https://www.easa.europa.eu/en/downloads/7767/en'
get data/aircraft_performance/airframes/easa/EASA_TCDS_E_023_BR700-715_Issue01.pdf 'https://www.easa.europa.eu/en/downloads/7757/en'
get data/aircraft_performance/airframes/easa/EASA_TCDS_E_063_Tay_Issue06.pdf 'https://www.easa.europa.eu/en/downloads/7697/en'
get data/aircraft_performance/airframes/easa/EASA_TCDS_E_200_Austro_E4_Issue12.pdf 'https://www.easa.europa.eu/en/downloads/7617/en'
get data/aircraft_performance/airframes/easa/EASA_TCDS_IM_A_001_ERJ170_Issue13.pdf 'https://www.easa.europa.eu/en/downloads/7525/en'
get data/aircraft_performance/airframes/easa/EASA_TCDS_IM_A_007_Cirrus_SR2x_Issue19.pdf 'https://www.easa.europa.eu/en/downloads/7512/en'
get data/aircraft_performance/airframes/easa/EASA_TCDS_IM_A_009_BD-700_Issue14.pdf 'https://www.easa.europa.eu/en/downloads/7508/en'
get data/aircraft_performance/airframes/easa/EASA_TCDS_IM_A_013_G200_Issue05.pdf 'https://www.easa.europa.eu/en/downloads/7500/en'
get data/aircraft_performance/airframes/easa/EASA_TCDS_IM_A_022_DA40_Issue24.pdf 'https://www.easa.europa.eu/en/downloads/44254/en'
get data/aircraft_performance/airframes/easa/EASA_TCDS_IM_A_023_CL-600_Challenger_Issue19.pdf 'https://www.easa.europa.eu/en/downloads/7479/en'
get data/aircraft_performance/airframes/easa/EASA_TCDS_IM_A_033_C680_Issue9.pdf 'https://www.easa.europa.eu/en/downloads/7459/en'
get data/aircraft_performance/airframes/easa/EASA_TCDS_IM_A_070_Gulfstream_GII-GV_Issue13.pdf 'https://www.easa.europa.eu/en/downloads/7384/en'
get data/aircraft_performance/airframes/easa/EASA_TCDS_IM_A_073_Beech390_Issue2.pdf 'https://www.easa.europa.eu/en/downloads/7378/en'
get data/aircraft_performance/airframes/easa/EASA_TCDS_IM_A_078_C525_R19.pdf 'https://www.easa.europa.eu/en/downloads/7368/en'
get data/aircraft_performance/airframes/easa/EASA_TCDS_IM_A_080_BD-100_Issue11.pdf 'https://www.easa.europa.eu/en/downloads/7364/en'
get data/reference_speeds/easa/EASA_TCDS_IM_A_085_Hawker_Issue05.pdf 'https://www.easa.europa.eu/en/downloads/7356/en'
get data/aircraft_performance/airframes/easa/EASA_TCDS_IM_A_097_C750_Issue3.pdf 'https://www.easa.europa.eu/en/downloads/7335/en'
get data/aircraft_performance/airframes/easa/EASA_TCDS_IM_A_158_EMB-505_Issue11.pdf 'https://www.easa.europa.eu/en/downloads/7282/en'
get data/reference_speeds/easa/EASA_TCDS_IM_A_207_Cessna_500_560XL_Issue10.pdf 'https://www.easa.europa.eu/en/downloads/7244/en'
get data/aircraft_performance/airframes/easa/EASA_TCDS_IM_A_211_MD90_B717_Issue3.pdf 'https://www.easa.europa.eu/en/downloads/7240/en'
get data/aircraft_performance/airframes/easa/EASA_TCDS_IM_A_212_LearjetM60.pdf 'https://www.easa.europa.eu/en/downloads/7238/en'
get data/aircraft_performance/airframes/easa/EASA_TCDS_IM_A_226_C208.pdf 'https://www.easa.europa.eu/en/downloads/7230/en'
get data/aircraft_performance/airframes/easa/EASA_TCDS_IM_A_234_PA-28_Issue8.pdf 'https://www.easa.europa.eu/en/downloads/17202/en'
get data/aircraft_performance/airframes/easa/EASA_TCDS_IM_A_277_KingAir_B200_B300_Issue22.pdf 'https://www.easa.europa.eu/en/downloads/7208/en'
get data/aircraft_performance/airframes/easa/EASA_TCDS_IM_A_279_Bonanza_Issue3.pdf 'https://www.easa.europa.eu/en/downloads/7204/en'
get data/aircraft_performance/airframes/easa/EASA_TCDS_IM_A_280_Beech55-58-95_Issue4.pdf 'https://www.easa.europa.eu/en/downloads/7202/en'
get data/aircraft_performance/airframes/easa/EASA_TCDS_IM_A_348_G280_Issue6.pdf 'https://www.easa.europa.eu/en/downloads/7184/en'
get data/aircraft_performance/airframes/easa/EASA_TCDS_IM_A_503_C90_Issue9.pdf 'https://www.easa.europa.eu/en/downloads/7106/en'
get data/aircraft_performance/airframes/easa/EASA_TCDS_IM_A_526_EMB-550_Issue12.pdf 'https://www.easa.europa.eu/en/downloads/17998/en'
get data/aircraft_performance/airframes/easa/EASA_TCDS_IM_A_570_BD-500_Issue25.pdf 'https://www.easa.europa.eu/en/downloads/20964/en'
get data/aircraft_performance/airframes/easa/EASA_TCDS_IM_A_595_GVII_Issue15.pdf 'https://www.easa.europa.eu/en/downloads/104086/en'
get data/reference_speeds/easa/EASA_TCDS_IM_A_615_SF50_Issue6.pdf 'https://www.easa.europa.eu/en/downloads/24242/en'
get data/aircraft_performance/airframes/easa/EASA_TCDS_IM_A_620_M700_Issue3.pdf 'https://www.easa.europa.eu/en/downloads/129064/en'
get data/aircraft_performance/airframes/easa/EASA_TCDS_IM_A_673_CL-600_RJ_Issue3.pdf 'https://www.easa.europa.eu/en/downloads/111186/en'
get data/aircraft_performance/airframes/easa/EASA_TCDS_IM_E_008_PT6A-67_Issue06.pdf 'https://www.easa.europa.eu/en/downloads/7787/en'
get data/aircraft_performance/airframes/easa/EASA_TCDS_IM_E_011_TFE731-20-60_Issue08.pdf 'https://www.easa.europa.eu/en/downloads/7781/en'
get data/aircraft_performance/airframes/easa/EASA_TCDS_IM_E_016_FJ44_FJ33_Issue13.pdf 'https://www.easa.europa.eu/en/downloads/7771/en'
get data/aircraft_performance/airframes/easa/EASA_TCDS_IM_E_032_Lycoming_IO-360_Issue03.pdf 'https://www.easa.europa.eu/en/downloads/7741/en'
get data/aircraft_performance/airframes/easa/EASA_TCDS_IM_E_044_AE3007_Issue05.pdf 'https://www.easa.europa.eu/en/downloads/7719/en'
get data/aircraft_performance/airframes/easa/EASA_TCDS_IM_E_048_PW530_Issue4.pdf 'https://www.easa.europa.eu/en/downloads/7715/en'
get data/aircraft_performance/airframes/easa/EASA_TCDS_IM_E_051_PW306_Issue04.pdf 'https://www.easa.europa.eu/en/downloads/7711/en'
get data/aircraft_performance/airframes/easa/EASA_TCDS_IM_E_053_CF34-8_Issue04.pdf 'https://www.easa.europa.eu/en/downloads/18525/en'
get data/aircraft_performance/airframes/easa/EASA_TCDS_IM_E_057_PW308_Issue06.pdf 'https://www.easa.europa.eu/en/downloads/7705/en'
get data/aircraft_performance/airframes/easa/EASA_TCDS_IM_E_058_AS907_Issue10.pdf 'https://www.easa.europa.eu/en/downloads/7703/en'
get data/aircraft_performance/airframes/easa/EASA_TCDS_IM_E_077_JT15D_Issue01.pdf 'https://www.easa.europa.eu/en/downloads/7671/en'
get data/aircraft_performance/airframes/easa/EASA_TCDS_IM_E_078_PT6A-41.pdf 'https://www.easa.europa.eu/en/downloads/7669/en'
get data/aircraft_performance/airframes/easa/EASA_TCDS_IM_E_090_PW1500G_Issue10.pdf 'https://www.easa.europa.eu/en/downloads/20863/en'
get data/aircraft_performance/airframes/easa/EASA_TCDS_IM_E_094_PT6A-100_Issue3.pdf 'https://www.easa.europa.eu/en/downloads/16560/en'
get data/aircraft_performance/airframes/easa/EASA_TCDS_IM_E_096_PW800_Issue07.pdf 'https://www.easa.europa.eu/en/downloads/33453/en'
get data/aircraft_performance/airframes/easa/EASA_TCDS_IM_E_100_IO-550_Issue4.pdf 'https://www.easa.europa.eu/en/downloads/7645/en'
get data/aircraft_performance/airframes/easa/EASA_TCDS_IM_E_105_TSIO-550_Issue03.pdf 'https://www.easa.europa.eu/en/downloads/7639/en'
get data/aircraft_performance/airframes/easa/EASA_TCDS_IM_E_113_Passport20_Issue06.pdf 'https://www.easa.europa.eu/en/downloads/68115/en'
get data/aircraft_performance/airframes/easa/EASA_TCDS_IM_E_233_CF34-1_CF34-3_Issue03.pdf 'https://www.easa.europa.eu/en/downloads/65434/en'
echo
echo "== FAA type certificate data sheets (DRS)"
get_drs data/aircraft_performance/airframes/faa/FAA_TCDS_2A13_Rev66.pdf DRSDOCID125895332520251113164724.0001
get_drs data/aircraft_performance/airframes/faa/FAA_TCDS_A00016WI_Rev6.pdf DRSDOCID160807395920260707144812.0001
get_drs data/aircraft_performance/airframes/faa/FAA_TCDS_A16SW_Rev29.pdf 231916F7717E56DC8625811A00650FEA.0001
get_drs data/aircraft_performance/airframes/faa/FAA_TCDS_E-274_Rev22.pdf 74C397F0D8FA23B886257B63006BA6A4.0001
get_drs data/aircraft_performance/airframes/faa/FAA_TCDS_E45NE_Original.pdf D9ECD1561D65A54686257547006E5B34.0001
get_drs data/aircraft_performance/airframes/faa/FAA_TCDS_E6WE_Rev18.pdf DRSDOCID176321561520260605175830.0001
echo
echo "== Manufacturer documents and pages"
get data/reference_speeds/airbus/A220-ACP-Issue013-00-27Nov2025.pdf 'https://www.aircraft.airbus.com/sites/g/files/jlcbta126/files/2025-12/A220-ACP-Issue013-00-27Nov2025.pdf'
get data/reference_speeds/bombardier/bombardier_challenger-3500.html 'https://bombardier.com/en/aircraft/challenger-3500'
get data/reference_speeds/bombardier/bombardier_challenger-650.html 'https://bombardier.com/en/aircraft/challenger-650'
get data/reference_speeds/bombardier/CRJ700APMR15.pdf 'https://customer.aero.bombardier.com/webd/BAG/CustSite/BRAD/RACSDocument.nsf/51aae8b2b3bfdf6685256c300045ff31/ec63f8639ff3ab9d85257c1500635bd8/$FILE/ATTE8Q23.pdf/CRJ700APMR15.pdf'
get data/reference_speeds/bombardier/bombardier_global-6500.html 'https://bombardier.com/en/aircraft/global-6500'
get data/aircraft_performance/airframes/bombardier/bombardier_global-8000.html 'https://bombardier.com/en/aircraft/global-8000'
get data/reference_speeds/dassault/Falcon-2000LXS-Backgrounder.pdf 'https://www.dassaultfalcon.com/app/uploads/2023/02/Falcon-2000LXS-Backgrounder.pdf'
get data/reference_speeds/diamond/DA40_NG_Product_Folder_2025_SCREEN.pdf 'https://www.diamondaircraft.com/fileadmin/diamondaircraft/documents/da40/DA40_NG_Product_Folder_2025_SCREEN.pdf'
get data/reference_speeds/embraer/phenom-300e-electronic.pdf 'https://embraer.com/media/b4lchr5u/phenom-300e-electronic.pdf'
get data/reference_speeds/embraer/praetor-500-eletronic.pdf 'https://embraer.com/media/asbhvchi/praetor-500-eletronic.pdf'
get data/aircraft_performance/airframes/gulfstream/gulfstream_g300.html 'https://www.gulfstream.com/en/aircraft/gulfstream-g300/'
get data/aircraft_performance/airframes/gulfstream/g300_specs.pdf 'https://assets.gulfstream.aero/downloads/g300.pdf'
get data/reference_speeds/gulfstream/gulfstream_g600.html 'https://www.gulfstream.com/en/aircraft/gulfstream-g600/'
get data/aircraft_performance/airframes/gulfstream/g600_specs.pdf 'https://assets.gulfstream.aero/downloads/g600.pdf'
get data/reference_speeds/pilatus/PC-12-PRO-Just-The-Facts-2025-web.pdf 'https://www.pilatus-aircraft.com/assets/files/PC-12-PRO-Just-The-Facts-2025-web.pdf'
get data/reference_speeds/piper/2026_Archer_LX_Foldover.pdf 'https://www.piper.com/wp-content/uploads/2019/01/2026_Archer_LX_Foldover.pdf'
get data/reference_speeds/textron/baron_product_card.pdf 'https://beechcraft.txtav.com/-/media/beechcraft/files/product-card/baron_product_card.pdf'
get data/reference_speeds/textron/bonanza_product_card.pdf 'https://beechcraft.txtav.com/-/media/beechcraft/files/product-card/bonanza_product_card.pdf'
get data/reference_speeds/textron/caravan_product_card.pdf 'https://cessna.txtav.com/-/media/cessna/files/product-cards/turboprop/caravan_product_card.pdf'
get data/reference_speeds/textron/citation_cj3_gen3_product_card.pdf 'https://cessna.txtav.com/-/media/cessna/files/product-cards/citation/citation_cj3_gen3_product_card.pdf'
get data/reference_speeds/textron/citation_cj4_gen3_product_card.pdf 'https://cessna.txtav.com/-/media/cessna/files/product-cards/citation/citation_cj4_gen3_product_card.pdf'
get data/reference_speeds/textron/king_air_260_product_card.pdf 'https://beechcraft.txtav.com/-/media/beechcraft/files/product-card/king_air_260_product_card.pdf'
get data/reference_speeds/textron/king_air_360_product_card.pdf 'https://beechcraft.txtav.com/-/media/beechcraft/files/product-card/king_air_360_product_card.pdf'
get data/reference_speeds/textron/citation_latitude_product_card.pdf 'https://cessna.txtav.com/-/media/cessna/files/product-cards/citation/citation_latitude_product_card.pdf'
get data/reference_speeds/textron/citation_longitude_product_card.pdf 'https://cessna.txtav.com/-/media/cessna/files/product-cards/citation/citation_longitude_product_card.pdf'
get data/reference_speeds/textron/citation_m2_gen3_product_card.pdf 'https://cessna.txtav.com/-/media/cessna/files/product-cards/citation/citation_m2_gen3_product_card.pdf'
echo
echo "== Added 2026-09-24 for C56X (Cessna 560XL); the aircraft TCDS IM.A.207 is in the EASA list above"
get data/aircraft_performance/airframes/easa/EASA_TCDS_IM_E_013_PW545_Issue03.pdf 'https://www.easa.europa.eu/en/downloads/7777/en'
get_drs data/reference_speeds/faa/FAA_TCDS_A22CE_Rev74.pdf DRSDOCID132119292720251209201014.0001
get_drs data/aircraft_performance/airframes/faa/FAA_TCDS_E00059EN_Rev6.pdf DRSDOCID170763942020240724131938.0001
echo
if [ "$rc" -eq 0 ]; then echo "all sources present."; else echo "one or more downloads FAILED -- see above." >&2; fi
echo "Text layers (read by the excerpts) are regenerated with: pdftotext -layout <file>.pdf <file>.txt"
exit "$rc"
