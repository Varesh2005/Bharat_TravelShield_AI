"""Beginner-friendly local demo for Bharat-TravelShield AI (read-only; changes no data or code).

It only CALLS the existing functions and prints a readable view:
    ml.destination_analysis.build_destination_summary   (data summary for a district)
    ml.hazard_inference.predict_for_district / predict_hazard   (the saved hazard ANNs)

Every number comes from the real processed tables in data/ and the saved models in ml/models/hazards/.
Nothing here is synthetic unless YOU edit a features file by hand (then it is your test input, not an
observation).

Run from the repository root (PowerShell):

  python -m ml.demo_local destination                      # Agra, Oct 2024, readable view
  python -m ml.demo_local destination --json               # same, full raw JSON
  python -m ml.demo_local features --hazard flood --district-code 146 --out $env:TEMP\\agra_flood.json
  python -m ml.demo_local predict  --hazard flood --features-file $env:TEMP\\agra_flood.json
  python -m ml.demo_local list                              # which hazards are supported
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:                       # lets `python ml/demo_local.py` work as well
    sys.path.insert(0, str(ROOT))

from ml import hazard_inference as hi                         # noqa: E402
from ml.destination_analysis import safe_destination_summary  # noqa: E402

LINE = "-" * 78


def _fmt(v):
    return "null" if v is None else (f"{v:.4g}" if isinstance(v, float) else str(v))


def show_record(rec: dict) -> str:
    parts = []
    for k, v in rec.items():
        if k == "data_quality":
            parts.append(f"quality={v['status']}")
        elif k == "source_state_label":
            parts.append(f"source_label={v}")
        else:
            parts.append(f"{k}={_fmt(v)}")
    return "    " + "  ".join(parts)


def show_prediction(res: dict, indent: str = "  ") -> None:
    p = res.get("prediction")
    print(f"{indent}status      : {res.get('status')}")
    if res.get("status") != "ok":
        for key in ("reason", "recovery_path", "recovery"):
            if res.get(key):
                print(f"{indent}{key:<12}: {res[key]}")
        for e in res.get("errors", []):
            print(f"{indent}error       : {e}")
        print(f"{indent}prediction  : none (nothing was invented)")
        return
    print(f"{indent}model       : {res['model_version']}")
    print(f"{indent}target      : {res['target']}")
    if "feature_year" in res:
        print(f"{indent}years       : uses data up to {res['feature_year']}, predicts {res['target_year']}")
    print(f"{indent}raw score   : {p['raw_model_score']}   (the network's 0-1 output; a ranking score)")
    if p["calibrated_probability"] is None:
        print(f"{indent}probability : WITHHELD - {p['calibration_note'].removeprefix('Withheld: ')}")
        print(f"{indent}class       : WITHHELD (no yes/no claim is made)")
    else:
        print(f"{indent}probability : {p['calibrated_probability']}   ({p['calibration_note']})")
        print(f"{indent}class       : {p['predicted_label']}  (threshold {p['decision_threshold']})")
    print(f"{indent}risk band   : {p['risk_band']}  ({p['risk_band_status']})")
    r = res["model_reliability"]
    print(f"{indent}reliability : held-out PR-AUC {r['pr_auc']}, recall {r['recall']}, precision {r['precision']}, "
          f"test-year ECE {r['ece']} (information only)")
    if "explanation" in res:
        print(f"{indent}top factors : (approximate, not causal - input vs its training median)")
        for f in res["explanation"]["top_factors"][:4]:
            print(f"{indent}   {f['feature']:<28} value {_fmt(f['value']):>10}  median {_fmt(f['training_median']):>10}"
                  f"  effect {f['effect_on_score']:+.3f}")
    print(f"{indent}limits      : {res['limitations'][0]}")


def cmd_destination(a) -> int:
    if a.district_code is None and not a.district_name:       # nothing chosen: the documented example
        a.district_name, a.state_name = "Agra", "Uttar Pradesh"
    kwargs = dict(district_code=a.district_code, district_name=None if a.district_code else a.district_name,
                  state_name=None if a.district_code else a.state_name, start_date=a.start, end_date=a.end,
                  include_experimental_ann=not a.no_ann)
    s = safe_destination_summary(**kwargs)
    if a.json:
        print(json.dumps(s, indent=2, ensure_ascii=True))
        return 1 if "error" in s else 0
    if "error" in s:
        print("ERROR:", s["error"]["code"], "-", s["error"]["message"])
        for k in ("candidates", "suggestions"):
            if k in s["error"]:
                print(" ", k, s["error"][k])
        return 1
    d, q = s["destination"], s["query_period"]
    print(LINE)
    print(f"DESTINATION: {d['district_name']} ({d['state_name']})   Census-2011 district code {d['census_district_code']}")
    print(f"  point {d['latitude']:.3f}, {d['longitude']:.3f}   population (2011) {d['population_2011']:,}")
    print(f"PERIOD     : {q['start_date']} to {q['end_date']}  -> calendar years {q['years_covered']}")
    print(f"  NOTE: {q['note']}")
    print(LINE)
    print("PART 1 - WHAT THE RECORDS SAY (real data; 'null' = unknown, never zero)")
    for name, h in s["hazards"].items():
        print(f"\n[{name}] status={h['status']}  last observed year={h['last_observed_year']}  basis={h['spatial_basis']}")
        if h.get("status_reason"):
            print(f"  reason: {h['status_reason']}")
        if name == "dengue" and h.get("state_used"):
            print(f"  state series used: {h['state_used']['state_in_source']} (mapping: {h['state_used']['mapping']})")
        for rec in h["records"]:
            print(show_record(rec))
        if name == "dengue" and h["data_quality"]["verification_pending_years"]:
            print(f"  QUALITY FLAG: verification_pending for {h['data_quality']['verification_pending_years']} - shown as reported, not verified")
        print(f"  caveat: {h['caveats'][0]}")
    if s["precautions"]:
        print("\nGENERIC PRECAUTIONS (not official advice; only where records support them):")
        for p in s["precautions"]:
            print(f"  - [{p['hazard']}] {p['text']}")
    print(LINE)
    if not a.no_ann:
        print("PART 2 - EXPERIMENTAL ANN OUTPUTS (annual, next calendar year; optional)")
        for hz, res in s["experimental_ann_predictions"]["hazards"].items():
            print(f"\n[{hz}]")
            show_prediction(res)
        print(LINE)
    print("LIMITS:", s["limitations"][0])
    print(s["disclaimer"])
    return 0


def cmd_features(a) -> int:
    res = hi.predict_for_district(a.hazard, a.district_code, a.feature_year, explain=False)
    if res.get("status") != "ok":
        print("Cannot build a real feature record:", res.get("status"), "-", res.get("reason") or res.get("errors"))
        return 1
    rec = res["input_features"]
    Path(a.out).write_text(json.dumps(rec, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote the REAL feature record for district {res['district']['census_district_code']} "
          f"({res['district']['state_name']}), feature year {res['feature_year']} (predicts {res['target_year']}).")
    print("It was built from the processed data tables by the same code used for training. Contents:")
    print(json.dumps(rec, indent=2))
    print(f"Saved to: {a.out}")
    return 0


def cmd_predict(a) -> int:
    try:
        rec = json.loads(Path(a.features_file).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        print("Could not read the features file as JSON:", type(exc).__name__)
        return 1
    res = hi.predict_hazard(a.hazard, rec, explain=not a.no_explain)
    if a.json:
        print(json.dumps(res, indent=2, ensure_ascii=True))
    else:
        print(f"DIRECT ANN INFERENCE: hazard={a.hazard}, features from {a.features_file}")
        show_prediction(res)
    return 0 if res.get("status") == "ok" else 1


def cmd_list(_a) -> int:
    for h, v in hi.list_hazards().items():
        extra = (f"probability shown: {v['calibrated_probability_available']}" if v["status"].startswith("supported")
                 else v.get("reason", ""))
        print(f"{h:<12} {v['status']:<24} {extra}")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("destination", help="readable destination summary (+ ANN outputs)")
    d.add_argument("--district-name", help="default: Agra (with state Uttar Pradesh) if nothing is given")
    d.add_argument("--state-name", help="needed when the name exists in several states, e.g. Bilaspur")
    d.add_argument("--district-code", type=int, help="use instead of name/state")
    d.add_argument("--start", default="2024-10-01", help="YYYY-MM-DD (selects calendar years; data is annual)")
    d.add_argument("--end", default="2024-10-31", help="YYYY-MM-DD")
    d.add_argument("--no-ann", action="store_true", help="skip the ANN section")
    d.add_argument("--json", action="store_true", help="print the full raw JSON instead")
    d.set_defaults(fn=cmd_destination)
    f = sub.add_parser("features", help="write the REAL feature record of a district for one hazard")
    f.add_argument("--hazard", required=True, choices=list(hi.HAZARDS))
    f.add_argument("--district-code", type=int, required=True)
    f.add_argument("--feature-year", type=int)
    f.add_argument("--out", required=True, help="path of the JSON file to write")
    f.set_defaults(fn=cmd_features)
    r = sub.add_parser("predict", help="direct ANN inference from a features JSON file")
    r.add_argument("--hazard", required=True, help="heatwave | flood | malaria (dengue/chikungunya are unsupported)")
    r.add_argument("--features-file", required=True)
    r.add_argument("--no-explain", action="store_true")
    r.add_argument("--json", action="store_true")
    r.set_defaults(fn=cmd_predict)
    l = sub.add_parser("list", help="support status of every hazard")
    l.set_defaults(fn=cmd_list)
    a = p.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    raise SystemExit(main())
