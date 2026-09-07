#!/usr/bin/env bash
# Re-download every remote source of the reference-speed provenance pack into
# data/reference_speeds/ (git-ignored) and print the sha256 of each file.
#
#   bash docs/reference_speeds/fetch_sources.sh
#
# An existing file is NEVER overwritten: the script skips it and still prints its
# sha256, so a populated tree can be verified against the tables in README.md.
# Be aware this pulls ~360 MB of PDFs plus 41 EUROCONTROL pages and 5 manufacturer product pages.
#
# Provenance and the exact page/section each file was read for: docs/reference_speeds/README.md
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
D="$ROOT/data/reference_speeds"
UA='Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36'
SLEEP="${FETCH_SLEEP:-4}"        # politeness delay between requests to the same host
rc=0

get() {  # get <relative path under data/reference_speeds> <url>
    local rel="$1" url="$2" out="$D/$1"
    mkdir -p "$(dirname "$out")"
    if [ -s "$out" ]; then
        printf 'skip     %-64s %s\n' "$rel" "$(sha256sum "$out" | cut -d' ' -f1)"
        return 0
    fi
    local code
    code=$(curl -sS -A "$UA" -L --retry 3 --retry-delay 5 --retry-all-errors \
                -o "$out.part" -w '%{http_code}' "$url") || code="curl-error"
    if [ "$code" != "200" ] || [ ! -s "$out.part" ]; then
        printf 'FAILED   %-64s HTTP %s\n' "$rel" "$code" >&2
        rm -f "$out.part"; rc=1
        sleep "$SLEEP"; return 1
    fi
    mv "$out.part" "$out"
    printf 'fetched  %-64s %s\n' "$rel" "$(sha256sum "$out" | cut -d' ' -f1)"
    sleep "$SLEEP"
}

echo "== FAA Office of Airports, Aircraft Characteristics Database (October 2024)"
get faa/FAA-Aircraft-Characteristics-Database-2024-10.xlsx \
    'https://www.faa.gov/airports/engineering/aircraft_char_database/aircraft_data'

echo
echo "== Boeing, Airplane Characteristics for Airport Planning"
BO='https://www.boeing.com/content/dam/boeing/v2/airports/acaps'
for f in 737NG_REV_C.pdf 737CL_REV_E.pdf 737MAX_RevK.pdf 757_Rev_H.pdf 767_REV_K.pdf \
         777-200-200ER-300_Rev_E.pdf 777-200LR-300ER-F_Rev_G.pdf 787_ACAP_Rev_Q.pdf 717.pdf; do
    get "boeing/$f" "$BO/$f"
done

echo
echo "== Airbus, Aircraft Characteristics - Airport and Maintenance Planning"
AB='https://www.aircraft.airbus.com/sites/g/files/jlcbta126/files'
get airbus/AC_A319_0624.pdf "$AB/2024-06/AC_A319_0624.pdf"
get airbus/AC_A320_0624.pdf "$AB/2025-01/AC_A320_0624.pdf"
get airbus/ac_a321_1223.pdf "$AB/2023-12/ac_a321_1223.pdf"
get airbus/Airbus-Commercial-Aircraft-AC-A300-600-Dec-2009.pdf \
    "$AB/2023-02/Airbus-Commercial-Aircraft-AC-A300-600-Dec-2009.pdf"
get airbus/A220-ACP-Issue013-00-27Nov2025.pdf "$AB/2025-12/A220-ACP-Issue013-00-27Nov2025.pdf"

echo
echo "== Embraer 175 Airport Planning Manual (copy filed in the NTSB docket for DCA20IA014)"
get embraer/E175_APM_NTSB_DCA20IA014_Att2.pdf \
    'https://data.ntsb.gov/Docket/Document/docBLOB?ID=13691419&FileExtension=pdf&FileName=DCA20IA014+Attachment+2+-+Embraer+175+Airport+Planning+Manual-Rel.pdf'

echo
echo "== Bombardier"
BB='https://customer.aero.bombardier.com/webd/BAG/CustSite/BRAD/RACSDocument.nsf/51aae8b2b3bfdf6685256c300045ff31/ec63f8639ff3ab9d85257c1500635bd8/$FILE'
get bombardier/CRJ900APMR11.pdf "$BB/ATTQF1EY.pdf/CRJ900APMR11.pdf"
get bombardier/CRJ200APMR8.pdf  "$BB/ATT1ES4H.pdf/CRJ200APMR8.pdf"
# bombardier.com media redirector; serves 2018-10/Global_5000_Fact_Sheet_0.pdf
get bombardier/Global_5000_Fact_Sheet.pdf 'https://bombardier.com/en/media/2531/download'
get bombardier/CRJ700APMR15.pdf "$BB/ATTE8Q23.pdf/CRJ700APMR15.pdf"
# Bombardier's current fact sheets sit behind a request form; the per-model product pages
# still print the full Weights block, so the pages themselves are the source.
for m in challenger-3500 challenger-650 global-6500; do
    get "bombardier/bombardier_$m.html" "https://bombardier.com/en/aircraft/$m"
done

echo
echo "== Pilatus"
get pilatus/Pilatus-Aircraft-Ltd-PC-24-Factsheet.pdf \
    'https://www.pilatus-aircraft.com/assets/files/Brochures/PC-24/Pilatus-Aircraft-Ltd-PC-24-Factsheet.pdf'
get pilatus/PC-12-PRO-Just-The-Facts-2025-web.pdf \
    'https://www.pilatus-aircraft.com/assets/files/PC-12-PRO-Just-The-Facts-2025-web.pdf'

echo
echo "== Textron Aviation / Cessna product cards"
TX='https://cessna.txtav.com/-/media/cessna/files/product-cards'
for f in citation_ascend_product_card citation_latitude_product_card \
         citation_longitude_product_card citation_cj3_gen3_product_card \
         citation_cj4_gen3_product_card citation_m2_gen3_product_card; do
    get "textron/$f.pdf" "$TX/citation/$f.pdf"
done
get textron/caravan_product_card.pdf "$TX/turboprop/caravan_product_card.pdf"
get textron/skyhawk_product_card.pdf "$TX/piston/skyhawk_product_card.pdf"
get textron/skylane_product_card.pdf "$TX/piston/skylane_product_card.pdf"

echo
echo "== Textron Aviation / Beechcraft product cards"
BX='https://beechcraft.txtav.com/-/media/beechcraft/files/product-card'
for f in king_air_360_product_card king_air_260_product_card bonanza_product_card baron_product_card; do
    get "textron/$f.pdf" "$BX/$f.pdf"
done

echo
echo "== Cirrus Aircraft (product pages; the brochures print no weights)"
get cirrus/cirrusaircraft_sr22.html        'https://cirrusaircraft.com/aircraft/sr22/'
get cirrus/cirrusaircraft_vision-jet.html  'https://cirrusaircraft.com/aircraft/vision-jet/'

echo
echo "== Piper Aircraft"
PI='https://www.piper.com/wp-content/uploads/2019/01'
get piper/2026_Archer_LX_Foldover.pdf "$PI/2026_Archer_LX_Foldover.pdf"
get piper/2026_M500_Brochure_V2.pdf   "$PI/2026_M500_Brochure_V2.pdf"

echo
echo "== Diamond Aircraft"
get diamond/DA40_NG_Product_Folder_2025_SCREEN.pdf \
    'https://www.diamondaircraft.com/fileadmin/diamondaircraft/documents/da40/DA40_NG_Product_Folder_2025_SCREEN.pdf'

echo
echo "== Dassault Aviation"
get dassault/Falcon-2000LXS-Backgrounder.pdf \
    'https://www.dassaultfalcon.com/app/uploads/2023/02/Falcon-2000LXS-Backgrounder.pdf'

echo
echo "== EASA (used only where the FAA database publishes no landing weight: SF50)"
get easa/EASA_TCDS_IM_A_615_SF50_Issue6.pdf 'https://www.easa.europa.eu/en/downloads/24242/en'

echo
echo "== EUROCONTROL Aircraft Performance Database (one page per ICAO type)"
echo "   E75L, GLF6 and PC24 legitimately come back as 'No ICAO' -- that is the database's answer,"
echo "   not a download failure."
echo "   Only the 40 types of the original pack are fetched: the 132 types added in the"
echo "   2026-09-07 extension carry no EUROCONTROL corroboration (see README)."
for t in B38M B737 B738 E75L B739 A319 CRJ9 A321 A320 A21N B39M A20N C56X B763 B752 E170 E190 \
         GLF5 GLF6 A306 C25A LJ45 PC24 A333 C172 B772 B788 CRJ2 C525 GL5T B789 C550 A332 A359 \
         B734 B762 E145 A343 B77W B735; do
    get "eurocontrol/$t.html" \
        "https://contentzone.eurocontrol.int/aircraftperformance/details.aspx?ICAO=$t"
done

echo
if [ "$rc" -eq 0 ]; then
    echo "all sources present."
else
    echo "one or more downloads FAILED -- see the lines above." >&2
fi
echo
echo "Not fetched by this script:"
echo "  * OpenAP 2.4 aircraft YAML files. They ship inside the 'aeroviz' conda env at"
echo "    site-packages/openap/data/aircraft/<type>.yml; their values and checksums are pinned in"
echo "    docs/reference_speeds/excerpts/openap_2_4_oew.csv."
echo "  * The .txt text layers beside each PDF. Regenerate with:"
echo "        for f in \$(find \"$D\" -name '*.pdf'); do pdftotext -layout \"\$f\" \"\${f%.pdf}.txt\"; done"
exit "$rc"
