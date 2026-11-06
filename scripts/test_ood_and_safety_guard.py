"""
scripts/test_ood_and_safety_guard.py
EcoSort-Edge: Out-of-Distribution (OOD) Rejection & Bayes Risk Safety Audit
Tests:
1. Shannon Entropy Rejection Gate on ambiguous / foreign non-waste inputs.
2. Cost-Sensitive Bayes Risk Hazard Barrier on borderline battery samples.
3. Privacy-by-Design Tactile Sanitary Button Override.
"""

import sys
import os
import json
import math
from pathlib import Path

# Ensure UTF-8 output
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from policy_engine_v2 import SWM2026PolicyEngine

RESULTS_DIR = PROJECT_ROOT / "results"
OUTPUT_JSON = RESULTS_DIR / "ood_safety_guard_report.json"

def calculate_shannon_entropy(probs):
    """Calculates Shannon entropy in bits for a probability vector."""
    entropy = 0.0
    for p in probs:
        if p > 1e-9:
            entropy -= p * math.log2(p)
    return entropy

def main():
    print("\n" + "=" * 75)
    print(" [ECOSORT-EDGE] OUT-OF-DISTRIBUTION (OOD) & SAFETY POLICY AUDIT")
    print("=" * 75)

    engine = SWM2026PolicyEngine()
    test_cases = []

    # TEST CASE 1: Normal In-Distribution Clear Sample (Metal Can)
    print("\n[TEST 1] Normal In-Distribution Sample (Clean Tin Can)...")
    probs_normal = [0.01, 0.02, 0.01, 0.01, 0.92, 0.01, 0.01, 0.005, 0.005]
    ent_normal = calculate_shannon_entropy(probs_normal)
    res_normal = engine.evaluate("metal_can", 0.92, entropy=ent_normal)
    print(f"  • Predicted Class: metal_can | Conf: 92.0% | Entropy: {ent_normal:.2f} bits")
    print(f"  • Policy Decision: {res_normal['stream'].upper()} BIN ({res_normal['bin_name']})")
    assert res_normal["stream"] == "dry"
    print("  -> STATUS: PASS (Correctly routed to Dry Recyclables)")
    test_cases.append({
        "scenario": "In-Distribution Normal Waste",
        "input_class": "metal_can",
        "confidence": 0.92,
        "entropy": round(ent_normal, 2),
        "routed_stream": res_normal["stream"],
        "status": res_normal.get("status", "NORMAL"),
        "passed": True
    })

    # TEST CASE 2: Out-of-Distribution Foreign Object (The "Black Shoe" Test)
    # The probability vector is diffuse and uncertain across multiple classes
    print("\n[TEST 2] Out-of-Distribution Foreign Object (Diffuse High Entropy)...")
    probs_ood = [0.12, 0.15, 0.08, 0.10, 0.11, 0.18, 0.09, 0.07, 0.10]
    ent_ood = calculate_shannon_entropy(probs_ood)
    res_ood = engine.evaluate("multi_layer_packaging", 0.18, entropy=ent_ood)
    print(f"  • Top Candidate  : multi_layer_packaging | Conf: 18.0% | Entropy: {ent_ood:.2f} bits")
    print(f"  • Policy Decision: {res_ood['stream'].upper()} ({res_ood['status']})")
    print(f"  • Action Taken   : {res_ood['action']} ({res_ood['tip_en']})")
    assert res_ood["stream"] == "reject"
    print("  -> STATUS: PASS (Zero Random Guessing - Successfully Diverted to Manual Hold)")
    test_cases.append({
        "scenario": "Out-of-Distribution / Foreign Object (The Shoe Test)",
        "input_class": "multi_layer_packaging (ambiguous)",
        "confidence": 0.18,
        "entropy": round(ent_ood, 2),
        "routed_stream": res_ood["stream"],
        "status": res_ood["status"],
        "passed": True
    })

    # TEST CASE 3: Cost-Sensitive Hazard Interception (Borderline Battery)
    # Simulates a dirty battery with sub-threshold confidence (72%)
    print("\n[TEST 3] Life-Safety Bayes Risk Barrier (Borderline Battery at 72%)...")
    res_battery = engine.evaluate("battery", 0.72, entropy=0.85)
    print(f"  • Predicted Class: battery | Conf: 72.0% (Below 85% Safety Barrier)")
    print(f"  • Policy Decision: {res_battery['stream'].upper()} ({res_battery['status']})")
    print(f"  • Fire Safety Tip: {res_battery['tip_en']}")
    assert res_battery["stream"] == "special_care"
    print("  -> STATUS: PASS (Bayes Risk Barrier Activated - Fire Prevented)")
    test_cases.append({
        "scenario": "Borderline Battery Fire Hazard (72% Confidence)",
        "input_class": "battery",
        "confidence": 0.72,
        "entropy": 0.85,
        "routed_stream": res_battery["stream"],
        "status": res_battery["status"],
        "passed": True
    })

    # TEST CASE 4: Privacy-by-Design Tactile Sanitary Override (DPDP Act 2023)
    print("\n[TEST 4] Hardware Tactile Sanitary Override (Camera Bypassed)...")
    res_sanitary = engine.evaluate("unknown", 0.0, sanitary_button=True)
    print(f"  • Hardware Event : Physical Push-Button Actuated (Optical Sensor Bypassed)")
    print(f"  • Policy Decision: {res_sanitary['stream'].upper()} ({res_sanitary['bin_name']})")
    print(f"  • Vernacular Cue : {res_sanitary['voice_cues']['kn']} | {res_sanitary['voice_cues']['hi']}")
    assert res_sanitary["stream"] == "sanitary"
    print("  -> STATUS: PASS (100% Privacy-by-Design Enforced)")
    test_cases.append({
        "scenario": "Sanitary Privacy Mechanical Bypass (DPDP Act 2023)",
        "input_class": "tactile_override",
        "confidence": 1.0,
        "entropy": 0.0,
        "routed_stream": res_sanitary["stream"],
        "status": res_sanitary["status"],
        "passed": True
    })

    # Summary
    print("\n" + "=" * 75)
    print(" SAFETY POLICY & OOD VERIFICATION SCORECARD")
    print("=" * 75)
    passed_all = all(t["passed"] for t in test_cases)
    print(f"  • Total Scenarios Evaluated : {len(test_cases)}")
    print(f"  • All Safety Constraints Met: {'YES (100% AUDIT PASS)' if passed_all else 'NO'}")
    print("=" * 75)

    # Save to JSON
    report = {
        "timestamp": "2026-11-06T20:45:00Z",
        "audit_name": "EcoSort-Edge OOD and Safety Guard Verification",
        "policy_version": "SWM 2026 v2.0",
        "all_passed": passed_all,
        "test_results": test_cases
    }

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\n[+] Formal OOD and safety report saved to: {OUTPUT_JSON.name}\n")

if __name__ == "__main__":
    main()
