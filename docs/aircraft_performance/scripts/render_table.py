"""Write the full substitution CSV and the report's main table from final_mapping.json."""
import csv
import json

from _paths import OUT, WORK

KT = 0.514444
rows = json.loads((WORK / "final_mapping.json").read_text())

SEG = {"airliner": "客机", "business_jet": "公务机", "turboprop": "涡桨", "piston_ga": "活塞",
       "rotorcraft": "旋翼", "military": "军机"}


def recommendation(r):
    """(Chinese recommendation, code the model would fly, stall margin after)."""
    v = r["verdict"]
    if v.startswith("keep"):
        return f"保留 OpenAP 的同义机型 {r['openap24']}", r["openap24"], r["r_current"]
    if v.startswith("own"):
        return "用本机参数（一手文件）", r["typecode"], r["r_own"]
    if v.startswith("PS"):
        return f"用 PS 本机参数（{r['ps_code']}）", r["ps_code"], r["r_ps"]
    if v.startswith("surrogate"):
        note = "（失速余量高于原生范围）" if "outside" in v else ""
        return f"改用 {r['sub']}{note}", r["sub"], r["r_sub"]
    if v.startswith("no airframe"):
        return "排除：模型里没有同动力、同进近类别的机型", "", None
    if v.startswith("no FAA"):
        return "排除：FAA 表没有进近速度", "", None
    if v.startswith("no acceptable"):
        return "排除：替代后仍低于失速分支", "", None
    return "排除：旋翼机 / 军机", "", None


def cur(r):
    return f"OpenAP 同义 {r['openap24']}" if r["openap24"] else "A320"


def f2(x):
    return f"{x:.2f}" if isinstance(x, (int, float)) else "—"


cols = ["typecode", "segment", "status_now", "train_flights", "val_flights",
        "faa_engine_class", "faa_aac", "faa_approach_speed_kt", "approach_speed_ms",
        "faa_mtow_kg", "faa_malw_kg", "wing_area_m2", "wing_area_source",
        "thrust_total_kN", "thrust_source", "power_total_kW", "thrust_to_weight",
        "stall_margin_now", "recommendation", "recommendation_zh", "model_code_after",
        "stall_margin_after", "stall_margin_own_mass_and_area",
        "nearest_existing", "nearest_existing_engine", "nearest_existing_aac",
        "nearest_existing_class_match", "nearest_existing_stall_margin", "second_nearest_existing",
        "similarity_distance_nearest", "distance_speed_term", "distance_thrust_to_weight_term",
        "similarity_distance_A320", "ps_model_code", "stall_margin_ps", "notes"]
with open(OUT / "2026-09-23_substitution_table.csv", "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(cols)
    for r in sorted(rows, key=lambda r: (-r["train"], r["typecode"])):
        zh, code, r_after = recommendation(r)
        v = r["faa_vapp_kt"]
        w.writerow([
            r["typecode"], r["segment"], cur(r), r["train"], r["val"],
            r["faa_engine"], r["aac"], v or "", f"{v * KT:.1f}" if v else "",
            r["mtow_kg"] or "", r["malw_kg"] or "", r["S_m2"] or "", r["S_src"],
            r["thrust_total_kN"] or "", r["thrust_src"], r["power_total_kW"] or "", r["tw"] or "",
            r["r_current"] or "", r["verdict"], zh, code, r_after or "", r["r_geom"] or "",
            r.get("sub") or "", r.get("sub_engine") or "", r.get("sub_aac") or "",
            r.get("class_match", ""), r.get("r_sub") or "", r.get("sub_second") or "",
            r.get("d", ""), r.get("d_v", ""), "" if r.get("d_tw") is None else r["d_tw"],
            r.get("d_a320", ""), r.get("ps_code") or "", r.get("r_ps") or "", r["notes"]])

lines = ["| 机型 | 类别 | 训练集航班 | FAA 进近速度 m/s（类别） | 现在 | 现在的失速余量 | 已有机型中最相似的 | 其失速余量 | 相似距离 A320 → 它 | 建议 | 建议后的失速余量 |",
         "|---|---|---:|---|---|---:|---|---:|---|---|---:|"]
for r in sorted(rows, key=lambda r: (-r["train"], r["typecode"])):
    if r["train"] < 30:
        continue
    zh, _code, r_after = recommendation(r)
    v = r["faa_vapp_kt"]
    speed = f"{v * KT:.1f}（{r['aac']}）" if v else "—"
    near = r.get("sub") or "—"
    if near != "—" and not r.get("class_match"):
        near += "（不同类，仅参考）"
    dist = f"{r['d_a320']:.2f} → {r['d']:.2f}" if r.get("d") is not None else "—"
    if r.get("d") is not None and not r.get("tw_checked") and r["faa_engine"] == "Jet":
        dist += "*"
    lines.append(f"| {r['typecode']} | {SEG.get(r['segment'], r['segment'])} | {r['train']:,} | {speed} | {cur(r)} | "
                 f"{f2(r['r_current'])} | {near} | {f2(r.get('r_sub'))} | {dist} | {zh} | {f2(r_after)} |")
(WORK / "table_main.md").write_text("\n".join(lines) + "\n")
print(len(lines) - 2, "rows in the report table")
