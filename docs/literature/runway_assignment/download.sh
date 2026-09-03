#!/usr/bin/env bash
# Re-fetch every document indexed in README.md from its public source (2026-09-03 URLs).
# ICAO Doc 4444 and Doc 9426 are not fetched: no authorised public copy (see README).
set -u
cd "$(dirname "$0")"
mkdir -p official nasa_eurocontrol papers
UA="Mozilla/5.0 (X11; Linux x86_64; rv:128.0) Gecko/20100101 Firefox/128.0"
get() { # get <destination> <url>
  if [ -s "$1" ]; then echo "have $1"; return; fi
  curl -sL -f --retry 2 -A "$UA" -o "$1" "$2" && echo "OK   $1" || { rm -f "$1"; echo "FAIL $1  $2"; }
}
# --- official regulatory / guidance documents ---------------------------------------------
get official/FAA_Order_JO_7110.65BB_Air_Traffic_Control_w_Chg1-3_2026-07-09.pdf "https://www.faa.gov/documentLibrary/media/Order/7110.65BB_Bsc_w_Chg_1_2_and_3_dtd_7-9-26_Final.pdf"
get official/FAA_Order_JO_7210.3EE_Facility_Operation_and_Administration_w_Chg1-3_2026-07-09.pdf "https://www.faa.gov/documentLibrary/media/Order/7210.3EE_Bsc_w_Chg_1_2_and_3_dtd_7-9-26.pdf"
get official/FAA_AIM_Aeronautical_Information_Manual_w_Chg1-3_2026-07-09.pdf "https://www.faa.gov/air_traffic/publications/media/AIM_Basic_w_Chg_1_and_2_and_3_dtd_7-9-26_FINAL.pdf"
get official/FAA_Order_8400.9_Runway_Use_Programs.pdf "https://www.faa.gov/documentLibrary/media/Order/8400-9.pdf"
get official/FAA_AC_150-5060-5_Airport_Capacity_and_Delay.pdf "https://www.faa.gov/documentlibrary/media/advisory_circular/150_5060_5.pdf"
get official/ICAO_Doc_9643_SOIR_Simultaneous_Operations_Parallel_Runways_2nd_ed_2020.pdf "https://skybrary.aero/sites/default/files/bookshelf/4647.pdf"
# --- NASA / EUROCONTROL / SESAR arrival-management reports ---------------------------------
get nasa_eurocontrol/NASA_TP-2014_Erzberger_Itoh_Arrival_Scheduling_Design_Principles.pdf "https://ntrs.nasa.gov/api/citations/20140010277/downloads/20140010277.pdf"
get nasa_eurocontrol/NASA_1989_Erzberger_Nedell_Automated_Management_of_Arrival_Traffic.pdf "https://ntrs.nasa.gov/api/citations/19890014919/downloads/19890014919.pdf"
get nasa_eurocontrol/NASA_1989_Design_of_FAST_for_TRACON.pdf "https://ntrs.nasa.gov/api/citations/19900001525/downloads/19900001525.pdf"
get nasa_eurocontrol/NASA_1995_Krzeczowski_Davis_Erzberger_Knowledge-based_Scheduling_of_Arrival_Aircraft_AIAA-95-3366.pdf "https://ntrs.nasa.gov/api/citations/19960016183/downloads/19960016183.pdf"
get nasa_eurocontrol/NASA_1995_Development_of_FAST_Controller-Engineer_Design.pdf "https://ntrs.nasa.gov/api/citations/19960001956/downloads/19960001956.pdf"
get nasa_eurocontrol/NASA_1995_ATC_Automation_Closely_Spaced_Parallel_Runways.pdf "https://ntrs.nasa.gov/api/citations/19960016111/downloads/19960016111.pdf"
get nasa_eurocontrol/NASA_1998_Passive_Final_Approach_Spacing_Tool_pFAST.pdf "https://ntrs.nasa.gov/api/citations/19980237029/downloads/19980237029.pdf"
get nasa_eurocontrol/NASA_2014_ATD-1_Concept_of_Operations_v2.pdf "https://ntrs.nasa.gov/api/citations/20140001370/downloads/20140001370.pdf"
get nasa_eurocontrol/NASA_2015_TSAS_NextGen_on_STARS.pdf "https://ntrs.nasa.gov/api/citations/20150001418/downloads/20150001418.pdf"
get nasa_eurocontrol/EUROCONTROL_2010_Arrival_Manager_Implementation_Guidelines_and_Lessons_Learned.pdf "https://skybrary.aero/sites/default/files/bookshelf/2416.pdf"
get nasa_eurocontrol/EUROCONTROL_2017_Airport_CDM_Implementation_Manual_v5.pdf "https://www.eurocontrol.int/sites/default/files/publication/files/airport-cdm-manual-2017.PDF"
get nasa_eurocontrol/SESAR_Extended_AMAN_Factsheet.pdf "https://www.sesarju.eu/sites/default/files/documents/wac2015/E-aman_factsheet_FINAL.pdf"
get nasa_eurocontrol/SESAR_Solution05_Extended_AMAN_Technical_Specification.pdf "https://www.sesarju.eu/sites/default/files/documents/solution/Sol05%205_Extended_AMAN_TS_AMAN.pdf"
# --- papers (author / open-access copies) ---------------------------------------------------
get papers/Ramanujam_Balakrishnan_2015_Airport_Configuration_Selection_THMS.pdf "https://www.mit.edu/~hamsa/pubs/RamanujamBalakrishnanTHMS2015.pdf"
get papers/Avery_Balakrishnan_2015_Predicting_Runway_Configuration_ATMSeminar.pdf "https://web.mit.edu/hamsa/www/pubs/AveryBalakrishnanATM2015.pdf"
# MIT DSpace bitstreams are resolved through the REST API (handles 1721.1/51953 and 1721.1/45254).
python3 - <<'PY'
import json, subprocess, urllib.request
UA = {"User-Agent": "Mozilla/5.0 thesis-reading-list"}
get = lambda u: json.load(urllib.request.urlopen(urllib.request.Request(u, headers=UA), timeout=60))
base = "https://dspace.mit.edu/server/api"
wanted = {"1721.1/51953": "papers/Lee_Balakrishnan_2008_Tradeoffs_in_Scheduling_Terminal-Area_Operations_ProcIEEE.pdf",
          "1721.1/45254": "papers/Lee_2008_MIT_thesis_Tradeoff_Evaluation_of_Scheduling_Algorithms_Terminal_Area.pdf"}
import os
for handle, out in wanted.items():
    if os.path.getsize(out) > 0 if os.path.exists(out) else False:
        print("have", out); continue
    d = get(base + "/discover/search/objects?query=handle:%22" + handle + "%22&size=5")
    for o in d["_embedded"]["searchResult"]["_embedded"]["objects"]:
        it = o["_embedded"]["indexableObject"]
        if it.get("handle") != handle: continue
        for b in get(f"{base}/core/items/{it['uuid']}/bundles")["_embedded"]["bundles"]:
            if b["name"] != "ORIGINAL": continue
            for bit in get(f"{base}/core/bundles/{b['uuid']}/bitstreams")["_embedded"]["bitstreams"]:
                if bit["name"].lower().endswith(".pdf"):
                    subprocess.run(["curl", "-sL", "-f", "-A", "Mozilla/5.0", "-o", out, bit["_links"]["content"]["href"]])
                    print("OK  ", out); break
PY
