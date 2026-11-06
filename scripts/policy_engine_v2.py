"""
scripts/policy_engine_v2.py
EcoSort-Edge v2: Statutory Solid Waste Management Rules 2026 Policy Engine
Integrates:
1. SWM 2026 Four Streams: Wet, Dry, Sanitary, Special-Care + REJECT
2. Cost-Sensitive Bayes Risk Formulation (L(hazard -> dry) = 1000 vs L(reject) = 1)
3. Multilingual Support: Kannada, Tamil, Hindi, English
4. Contamination Gating & Multi-Layer Packaging (RDF) Flagging
"""

import os
import sys
import yaml
import json
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

# Ensure UTF-8 on Windows terminal
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent
POLICY_YAML = PROJECT_ROOT / "policy" / "swm2026.yaml"

class SWM2026PolicyEngine:
    def __init__(self, config_path: Optional[Path] = None):
        self.config_path = config_path or POLICY_YAML
        self.load_policy()
        
    def load_policy(self):
        if not self.config_path.exists():
            raise FileNotFoundError(f"Policy config not found at: {self.config_path}")
        with open(self.config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)
            
        self.streams = self.config["streams"]
        self.classes = self.config["classes"]
        self.loss_matrix = self.config.get("loss_matrix", {
            "hazard_misroute_cost": 1000.0,
            "contamination_cost": 10.0,
            "unnecessary_reject_cost": 1.0
        })
        self.thresholds = self.config.get("thresholds", {
            "benign_stream": 0.60,
            "hazard_safety_hold": 0.85
        })

    def evaluate(self, 
                 predicted_class: str, 
                 confidence: float, 
                 entropy: Optional[float] = None,
                 soiled_flag: bool = False,
                 sanitary_button: bool = False) -> Dict[str, Any]:
        """
        Receives model perception output and applies cost-sensitive SWM 2026 routing.
        """
        # 1. PRIVACY-BY-DESIGN SANITARY OVERRIDE
        if sanitary_button:
            s_info = self.streams["sanitary"]
            c_info = self.classes.get("sanitary_item", {})
            return {
                "stream": "sanitary",
                "stream_id": s_info["id"],
                "bin_name": s_info["bin_name"],
                "color_hex": s_info["color_hex"],
                "status": "SANITARY_OVERRIDE",
                "hazard_level": "biohazard",
                "action": s_info["action"],
                "flags": c_info.get("flags", []),
                "labels": s_info["labels"],
                "tip_en": c_info.get("tip_en", "Wrap securely before disposal."),
                "tip_kn": c_info.get("tip_kn", "ಕಾಗದದಲ್ಲಿ ಸುತ್ತಿ ಪ್ರತ್ಯೇಕವಾಗಿ ವಿಲೇವಾರಿ ಮಾಡಿ."),
                "voice_cues": {
                    "en": "Sanitary Waste - Yellow Bin",
                    "kn": "ಹಳದಿ ಬುಟ್ಟಿ - ನೈರ್ಮಲ್ಯ ಕಸ",
                    "hi": "पीला डिब्बा - सैनिटरी कचरा",
                    "ta": "மஞ்சள் தொட்டி - சுகாதாரக் கழிவு"
                }
            }

        # 2. UNKNOWN / UNMAPPED CLASS CHECK
        if predicted_class not in self.classes:
            r_info = self.streams["reject"]
            return {
                "stream": "reject",
                "stream_id": r_info["id"],
                "bin_name": r_info["bin_name"],
                "color_hex": r_info["color_hex"],
                "status": "UNKNOWN_OBJECT_REJECTED",
                "hazard_level": "unknown",
                "action": r_info["action"],
                "flags": ["manual_triage"],
                "labels": r_info["labels"],
                "tip_en": "Unknown item category. Diverting to manual inspection.",
                "tip_kn": "ಅಪರಿಚಿತ ವಸ್ತು. ಪರಿಶೀಲನೆಗೆ ಪ್ರತ್ಯೇಕಿಸಿ.",
                "voice_cues": {
                    "en": "Uncertain Waste - Please Inspect",
                    "kn": "ಅಸ್ಪಷ್ಟ ತ್ಯಾಜ್ಯ - ಪರಿಶೀಲಿಸಿ",
                    "hi": "अस्पष्ट वस्तु - अलग रखें",
                    "ta": "தெளிவற்ற பொருள் - சரிபார்க்கவும்"
                }
            }

        class_meta = self.classes[predicted_class]
        target_stream = class_meta["stream"]
        hazard_level = class_meta.get("hazard_level", "none")
        flags = list(class_meta.get("flags", []))

        # 3. SHANNON ENTROPY OOD & AMBIGUITY GATE
        # If probability distribution is diffuse across classes (high entropy),
        # the object is likely out-of-distribution (non-waste, hand, unknown item)
        # In a 9-class distribution, max entropy is ln(9) = 2.197. Threshold at 1.40.
        if entropy is not None and entropy > 1.40:
            r_info = self.streams["reject"]
            return {
                "stream": "reject",
                "stream_id": r_info["id"],
                "bin_name": r_info["bin_name"],
                "color_hex": r_info["color_hex"],
                "status": "OOD_HIGH_ENTROPY_REJECTED",
                "hazard_level": hazard_level,
                "action": "manual_inspection",
                "flags": ["high_entropy_ood", "ambiguous_item"],
                "labels": r_info["labels"],
                "tip_en": f"High ambiguity (Entropy: {entropy:.2f}). Object not in municipal catalog.",
                "tip_kn": "ಅಸ್ಪಷ್ಟ ವಸ್ತು. ಪಟ್ಟಿಯಲ್ಲಿಲ್ಲದ ತ್ಯಾಜ್ಯ.",
                "voice_cues": {
                    "en": "Unknown Object - Please Inspect",
                    "kn": "ಅಪರಿಚಿತ ವಸ್ತು - ಪರಿಶೀಲಿಸಿ",
                    "hi": "अपरिचित वस्तु - जाँच करें",
                    "ta": "அறியப்படாத பொருள் - சரிபார்க்கவும்"
                }
            }

        # 4. COST-SENSITIVE BAYES RISK EVALUATION
        # L(Battery -> Dry) = 1000. If confidence on battery is even slightly uncertain,
        # or if an item has high hazard cost, hold it.
        # Expected Loss if routed to Dry vs Reject:
        if hazard_level == "critical_fire":
            # Mandatory safety isolation: Even at 0.50 confidence, a suspected battery MUST NOT enter dry/wet!
            if confidence < 0.85:
                # Flag as safety hold to Special-Care
                s_info = self.streams["special_care"]
                return {
                    "stream": "special_care",
                    "stream_id": s_info["id"],
                    "bin_name": s_info["bin_name"],
                    "color_hex": s_info["color_hex"],
                    "status": "SAFETY_HAZARD_ISOLATION",
                    "hazard_level": hazard_level,
                    "action": "isolate_for_safety",
                    "flags": flags + ["fire_precaution"],
                    "labels": s_info["labels"],
                    "tip_en": "CAUTION: Suspected battery or fire hazard. Placed in Special-Care.",
                    "tip_kn": "ಎಚ್ಚರಿಕೆ: ಬ್ಯಾಟರಿ ಬೆಂಕಿಯ ಅಪಾಯ! ವಿಶೇಷ ಆರೈಕೆ ಬುಟ್ಟಿಗೆ ಹಾಕಿ.",
                    "voice_cues": {
                        "en": "DANGER: Battery - Special-Care Bin!",
                        "kn": "ಎಚ್ಚರಿಕೆ: ಬ್ಯಾಟರಿ - ಕೆಂಪು ಬುಟ್ಟಿ!",
                        "hi": "सावधान: लाल डिब्बा - बैटरी!",
                        "ta": "எச்சரிக்கை: மின்கலம் - சிவப்பு தொட்டி!"
                    }
                }

        # General confidence threshold for regular items (Strict Majority: 50%)
        benign_threshold = self.thresholds.get("benign_stream", 0.50)
        if confidence < benign_threshold:
            r_info = self.streams["reject"]
            return {
                "stream": "reject",
                "stream_id": r_info["id"],
                "bin_name": r_info["bin_name"],
                "color_hex": r_info["color_hex"],
                "status": "CONFIDENCE_GATE_REJECTED",
                "hazard_level": hazard_level,
                "action": r_info["action"],
                "flags": ["low_confidence_hold"],
                "labels": r_info["labels"],
                "tip_en": f"Confidence {confidence*100:.1f}% below threshold. Routed to Hold for review.",
                "tip_kn": f"ಖಚಿತತೆ ಕಡಿಮೆಯಿದೆ ({confidence*100:.1f}%). ಪರಿಶೀಲನೆಗೆ ಪ್ರತ್ಯೇಕಿಸಿ.",
                "voice_cues": {
                    "en": "Hold for review",
                    "kn": "ಪರಿಶೀಲನೆಗೆ ಇರಿಸಿ",
                    "hi": "जाँच के लिए रखें",
                    "ta": "சரிபார்க்க வைக்கவும்"
                }
            }

        # 4. CONTAMINATION GATING
        # Soiled packaging is diverted to Reject or requires rinsing
        if soiled_flag and target_stream == "dry":
            r_info = self.streams["reject"]
            return {
                "stream": "reject",
                "stream_id": r_info["id"],
                "bin_name": r_info["bin_name"],
                "color_hex": r_info["color_hex"],
                "status": "SOIL_CONTAMINATION_DIVERTED",
                "hazard_level": "spoilage",
                "action": "rinse_or_reject",
                "flags": flags + ["food_soiled"],
                "labels": r_info["labels"],
                "tip_en": "Food residue detected. Rinse thoroughly to deposit in Dry bin, else reject.",
                "tip_kn": "ಆಹಾರದ ಕಲೆ ಕಂಡುಬಂದಿದೆ. ಒಣ ಕಸದಲ್ಲಿ ಹಾಕುವ ಮುನ್ನ ನೀರಿನಿಂದ ತೊಳೆಯಿರಿ.",
                "voice_cues": {
                    "en": "Soiled Recyclable - Please Rinse",
                    "kn": "ತೊಳೆದು ಒಣ ಕಸಕ್ಕೆ ಹಾಕಿ",
                    "hi": "धोकर नीले डिब्बे में डालें",
                    "ta": "கழுவி உலர் தொட்டியில் போடவும்"
                }
            }

        # 5. NORMAL STATUTORY SWM 2026 ROUTING
        s_info = self.streams[target_stream]
        voice_map = {
            "wet": {
                "en": "Wet Waste - Green Bin",
                "kn": "ಹಸಿರು ಬುಟ್ಟಿ - ಹಸಿ ಕಸ",
                "hi": "हरा डिब्बा - गीला कचरा",
                "ta": "பச்சை தொட்டி - ஈரக் கழிவு"
            },
            "dry": {
                "en": "Dry Waste - Blue Bin",
                "kn": "ನೀಲಿ ಬುಟ್ಟಿ - ಒಣ ಕಸ",
                "hi": "नीला डिब्बा - सूखा कचरा",
                "ta": "நீல தொட்டி - உலர் கழிவு"
            },
            "special_care": {
                "en": "DANGER: Battery - Special-Care Red Bin!",
                "kn": "ಎಚ್ಚರಿಕೆ: ಬ್ಯಾಟರಿ - ಕೆಂಪು ಬುಟ್ಟಿ!",
                "hi": "सावधान: लाल डिब्बा - बैटरी!",
                "ta": "எச்சரிக்கை: மின்கலம் - சிவப்பு தொட்டி!"
            }
        }

        return {
            "stream": target_stream,
            "stream_id": s_info["id"],
            "bin_name": s_info["bin_name"],
            "color_hex": s_info["color_hex"],
            "status": "STATUTORY_ROUTED",
            "hazard_level": hazard_level,
            "action": s_info["action"],
            "flags": flags,
            "labels": s_info["labels"],
            "tip_en": class_meta.get("tip_en", ""),
            "tip_kn": class_meta.get("tip_kn", ""),
            "voice_cues": voice_map.get(target_stream, {
                "en": f"{target_stream.capitalize()} Waste",
                "kn": "ಕಸ ಬೇರ್ಪಡಿಸಿ",
                "hi": "कचरा अलग करें",
                "ta": "கழிவு பிரிக்கவும்"
            })
        }

if __name__ == "__main__":
    engine = SWM2026PolicyEngine()
    print("Test 1 (Battery @ 0.94):", engine.evaluate("battery", 0.94)["voice_cues"]["kn"])
    print("Test 2 (Battery @ 0.60):", engine.evaluate("battery", 0.60)["status"])
    print("Test 3 (Food Waste @ 0.92):", engine.evaluate("food_waste", 0.92)["stream"])
    print("Test 4 (MLP @ 0.88):", engine.evaluate("multi_layer_packaging", 0.88)["flags"])
    print("Test 5 (Sanitary button):", engine.evaluate("none", 0.0, sanitary_button=True)["stream"])
