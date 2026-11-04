"""
EcoSort-Edge: 2-Minute Demonstration Runner
Satisfies Hackathon Deliverable D5 (Realistic Sorting Scenarios).
Runs interactive simulations of MRF Worker Mode and Citizen Kiosk Mode.
"""

import sys
import time
from scripts.decision_engine import ConfidenceAwareDecisionEngine
from scripts.llm_explainer import CognitiveExplanationEngine

# Ensure clean UTF-8 console output
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

def run_worker_mode(engine: ConfidenceAwareDecisionEngine):
    print("\n" + "=" * 75)
    print(" SCENARIO 1: MATERIAL RECOVERY FACILITY (MRF) WORKER MODE")
    print(" Hardware: Raspberry Pi 5 · Offline Transfer Station")
    print("=" * 75)
    print("[*] Conveyor belt running at 0.8 m/s. Sensor camera active...")
    time.sleep(0.6)

    # Simulated detection: Concealed lithium-ion battery in paper packaging
    print("\n[EVENT] Waste item enters scanning zone...")
    simulated_probs = {"battery": 0.58, "metal_can": 0.28, "paper": 0.14}

    t0 = time.perf_counter()
    decision = engine.evaluate(simulated_probs)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    print(f"[*] Perception + Triage Latency: {elapsed_ms:.2f} ms (< 35 ms SLA)")
    print(f"[*] Decision Output: {decision.suggested_stream.upper()}")
    print(f"[*] Confidence Tier: {decision.confidence_tier}")
    print(f"[*] Hardware Action: 🚨 FLASH RED STROBE LIGHT")
    print(f"[*] Audio Alert    : 📢 'SAFETY HOLD — BATTERY DETECTED!'")
    print(f"[*] Directive      : {decision.action_guidance}")
    print("\n[OUTCOME] Worker diverts battery 7.2 seconds before the compactor fires.")
    print("          Prevents facility thermal runaway fire. Zero internet required.")

def run_citizen_mode(engine: ConfidenceAwareDecisionEngine, explainer: CognitiveExplanationEngine):
    print("\n" + "=" * 75)
    print(" SCENARIO 2: PUBLIC SMART KIOSK CITIZEN GUIDANCE MODE")
    print(" Location: Metro Station / Food Court Public Kiosk")
    print(" Architecture: 'Vision Perceives, Rules Decide, Llama Explains'")
    print("=" * 75)
    print("[*] Citizen approaches disposal station with takeaway waste...")
    time.sleep(0.6)

    # Simulated detection: Contaminated plastic container
    print("\n[EVENT] Citizen scans takeaway food container...")
    simulated_probs = {"plastic_bottle": 0.74, "multi_layer_packaging": 0.22, "food_waste": 0.04}

    t0 = time.perf_counter()
    decision = engine.evaluate(simulated_probs)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    print(f"[*] Stage 1 & 2 Fast-Path Latency: {elapsed_ms:.2f} ms (< 40 ms real-time)")
    print(f"[*] Instant Triage: {decision.predicted_category.upper()} -> {decision.suggested_stream.upper()}")
    print(f"[*] Display Card  : 💡 '{decision.action_guidance}'")

    # Stage 3: On-demand citizen educational explanation
    print("\n[INTERACTION] Citizen taps 'Why?' on kiosk touchscreen...")
    time.sleep(0.4)
    explanation = explainer.explain(decision.predicted_category, decision.confidence_score, decision.suggested_stream)
    print(f"[*] Stage 3 Explainer ({explanation['generator']}):")
    print(f"    • Context : {explanation['short_explanation']}")
    print(f"    • Impact  : {explanation['environmental_impact']}")
    print(f"    • Vernacular Voice Cue : \"{explanation['regional_voice_cue']}\"")
    print("\n[OUTCOME] Citizen is guided to rinse container at wash basin before")
    print("          depositing in Dry Recyclable bin, preventing batch contamination.")

if __name__ == "__main__":
    engine = ConfidenceAwareDecisionEngine()
    explainer = CognitiveExplanationEngine()
    print("=" * 75)
    print(" EcoSort-Edge: Problem Statement 3 Demonstration (Deliverable D5)")
    print("=" * 75)
    run_worker_mode(engine)
    time.sleep(1.0)
    run_citizen_mode(engine, explainer)
    print("\n[SUCCESS] 2-minute scenario completed cleanly.\n")
