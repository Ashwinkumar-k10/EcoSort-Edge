"""
scripts/vernacular_guidance_engine.py
EcoSort-Edge: On-Device Multi-Lingual Waste Disposal Guidance Engine

Deterministic, zero-dependency vernacular guidance for 6 Indian languages:
Hindi, Tamil, Telugu, Bengali, Marathi, English.

This is a rule-based lookup engine. It does NOT run a large language model.
It maps (predicted_class, language) -> (tip, action, rationale) in constant time.

Execution cost: <0.1 ms per query, 0 bytes additional RAM beyond the Python dict.
"""

import sys
import json
import time
import argparse
from pathlib import Path
from typing import Dict, Any, Optional

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

# 6-Language Vernacular Knowledge Base (SWM 2026 Aligned)
VERNACULAR_DATABASE = {
    "hi": {
        "language_name": "Hindi",
        "battery": {
            "tip": "सावधानी: यह लिथियम बैटरी है! इसे सूखे या गीले कचरे में कभी न डालें।",
            "action": "खतरनाक ई-कचरा केंद्र (लाल डिब्बा) में जमा करें।",
            "why": "कंपैक्टर और कचरा गाड़ियों में दबाव पड़ने पर लिथियम बैटरी में आग लग जाती है।"
        },
        "plastic_bottle": {
            "tip": "यह रिसायकल करने योग्य PET प्लास्टिक है। कृपया नीले रंग के सूखे कचरे वाले डिब्बे में डालें।",
            "action": "ढक्कन खोलें, बोतल को खाली कर पिचकाएं और नीले डिब्बे में डालें।",
            "why": "साफ और सूखी प्लास्टिक सामग्री को रीसाइक्लिंग प्लांट में आसानी से प्रोसेस किया जाता है।"
        },
        "cardboard": {
            "tip": "गत्ते का डिब्बा सूखे कचरे के नीले डिब्बे में जाएगा।",
            "action": "यदि इस पर भोजन या तेल लगा है, तो कृपया इसे साफ करें या अलग रखें।",
            "why": "गीला या तैलीय कार्डबोर्ड पेपर रीसाइक्लिंग के पूरे बैच को दूषित कर देता है।"
        },
        "food_waste": {
            "tip": "यह गीला जैविक कचरा है। कृपया बिना किसी प्लास्टिक कवर के हरे डिब्बे में डालें।",
            "action": "कचरा सीधे हरे डिब्बे में डालें, इसे खाद या सामुदायिक बायोगैस में बदला जाएगा।",
            "why": "गीला कचरा सड़कर मीथेन बनाता है, स्रोत पर अलग करने से पर्यावरण सुरक्षित रहता है।"
        },
        "vegetable_scrap": {
            "tip": "सब्जी और फलों के छिलके 100% कंपोस्टेबल गीला कचरा हैं।",
            "action": "हरे रंग के गीले कचरे वाले डिब्बे में डालें।",
            "why": "यह पोषक तत्वों से भरपूर जैविक खाद तैयार करने के लिए सबसे उपयुक्त है।"
        },
        "metal_can": {
            "tip": "एल्युमीनियम या टिन का डिब्बा मूल्यवान सूखा कचरा है।",
            "action": "खाली करके नीले सूखे कचरे के डिब्बे में डालें।",
            "why": "धातु को बार-बार बिना गुणवत्ता खोए रीसायकल किया जा सकता है।"
        },
        "glass_bottle": {
            "tip": "कांच की बोतल सूखा पुनर्चक्रण योग्य कचरा है।",
            "action": "साफ करके नीले डिब्बे में संभाल कर रखें ताकि टूटे नहीं।",
            "why": "कांच को अनंत बार पिघलाकर नए उत्पाद बनाए जा सकते हैं।"
        },
        "multi_layer_packaging": {
            "tip": "मल्टी-लेयर चिप्स/नमकीन पैकेट सूखा कचरा (RDF) है।",
            "action": "नीले डिब्बे में डालें, यह सीमेंट भट्ठों में ऊर्जा पुनर्चक्रण के लिए भेजा जाएगा।",
            "why": "मल्टी-लेयर प्लास्टिक को सामान्य रूप से रीसायकल नहीं किया जा सकता, इसलिए इसे ईंधन के रूप में उपयोग करते हैं।"
        },
        "paper": {
            "tip": "साफ कागज सूखा कचरा है।",
            "action": "नीले डिब्बे में सूखा रखें।",
            "why": "कागज को रीसायकल करने से पेड़ों की कटाई और पानी की भारी बचत होती है।"
        }
    },
    "ta": {
        "language_name": "Tamil",
        "battery": {
            "tip": "எச்சரிக்கை: இது லித்தியம் பேட்டரி! உலர் அல்லது ஈரக் கழிவுகளில் போடாதீர்கள்.",
            "action": "சிவப்பு நிற சிறப்பு ஆபத்துக் கழிவுப் பெட்டியில் ஒப்படைக்கவும்.",
            "why": "குப்பை அழுத்திகளில் பேட்டரி வெடித்து தீப்பிடிக்கும் அபாயம் உள்ளது."
        },
        "plastic_bottle": {
            "tip": "இது மறுசுழற்சி செய்யக்கூடிய PET பிளாஸ்டிக். நீல நிற உலர் கழிவு தொட்டியில் போடவும்.",
            "action": "திரவங்களை அகற்றிவிட்டு பாட்டிலை நசுக்கி நீல தொட்டியில் போடவும்.",
            "why": "சுத்தமான பிளாஸ்டிக் கழிவுகள் எளிதாக மறுசுழற்சி செய்யப்படுகின்றன."
        },
        "food_waste": {
            "tip": "இது ஈரமான கரிமக் கழிவு. பிளாஸ்டிக் பைகள் இல்லாமல் பச்சை நிற தொட்டியில் போடவும்.",
            "action": "பச்சை தொட்டியில் போடவும், இது இயற்கை உரம் மற்றும் பயோகேஸ் தயாரிக்க உதவும்.",
            "why": "ஈரக் கழிவுகள் சிறந்த இயற்கை உரத்தை வழங்குகின்றன."
        },
        "cardboard": {
            "tip": "அட்டைப் பெட்டிகளை உலர் கழிவாக நீல நிற தொட்டியில் போடவும்.",
            "action": "எண்ணெய் அல்லது உணவுப் படிவுகள் இருந்தால் சுத்தம் செய்யவும்.",
            "why": "கழிவு கலப்படம் மறுசுழற்சியைப் பாதிக்கிறது."
        }
    },
    "te": {
        "language_name": "Telugu",
        "battery": {
            "tip": "హెచ్చరిక: ఇది లిథియం బ్యాటరీ! తడి లేదా పొడి చెత్త డబ్బాలో ఎప్పుడూ వేయకండి.",
            "action": "ప్రమాదకర ఎరుపు రంగు ప్రత్యేక చెత్త డబ్బాలో మాత్రమే వేయండి.",
            "why": "కంపాక్టర్లలో ఒత్తిడి వల్ల బ్యాటరీలు పేలి అగ్ని ప్రమాదాలు జరుగుతాయి."
        },
        "plastic_bottle": {
            "tip": "ఇది రీసైకిల్ చేయగల PET ప్లాస్టిక్. నీలి రంగు పొడి చెత్త డబ్బాలో వేయండి.",
            "action": "బాటిల్‌ను ఖాళీ చేసి, నలిపి నీలి డబ్బాలో వేయండి.",
            "why": "శుభ్రమైన ప్లాస్టిక్ తిరిగి వినియోగానికి తోడ్పడుతుంది."
        },
        "food_waste": {
            "tip": "ఇది తడి సేంద్రీయ వ్యర్థం. ప్లాస్టిక్ కవర్లు లేకుండా ఆకుపచ్చ డబ్బాలో వేయండి.",
            "action": "ఆకుపచ్చ డబ్బాలో వేయండి, ఇది కంపోస్ట్ మరియు బయోగ్యాస్ తయారీకి ఉపయోగపడుతుంది.",
            "why": "తడి చెత్తను సమర్థవంతంగా పునర్వినియోగించవచ్చు."
        }
    },
    "bn": {
        "language_name": "Bengali",
        "battery": {
            "tip": "সতর্কতা: এটি লিথিয়াম ব্যাটারি! শুকনো বা ভেজা বর্জ্যে ফেলবেন না।",
            "action": "লাল রঙের বিপজ্জনক বর্জ্যের পাত্রে জমা দিন।",
            "why": "চাপের কারণে ব্যাটারিতে আগুন ধরে যেতে পারে।"
        },
        "plastic_bottle": {
            "tip": "এটি পুনর্ব্যবহারযোগ্য PET প্লাস্টিক। নীল রঙের শুকনো বর্জ্যের পাত্রে ফেলুন।",
            "action": "খালি করে নীল পাত্রে ফেলুন।",
            "why": "পরিষ্কার প্লাস্টিক সহজেই পুনর্ব্যবহার করা যায়।"
        },
        "food_waste": {
            "tip": "এটি জৈব ভেজা বর্জ্য। সবুজ রঙের বর্জ্য পাত্রে ফেলুন।",
            "action": "প্লাস্টিক ছাড়া সবুজ পাত্রে ফেলুন, এটি কম্পোস্টিংয়ের জন্য ব্যবহার হবে।",
            "why": "জৈব বর্জ্য থেকে প্রাকৃতিক সার তৈরি হয়।"
        }
    },
    "mr": {
        "language_name": "Marathi",
        "battery": {
            "tip": "धोका: ही लिथियम बॅटरी आहे! सुक्या किंवा ओल्या कचऱ्यात टाकू नका.",
            "action": "अधिकृत लाल घातक कचरा डब्यात जमा करा.",
            "why": "कचरा दाबण्याच्या यंत्रात बॅटरीमुळे आग लागू शकते."
        },
        "plastic_bottle": {
            "tip": "हे पुनर्वापर करण्यायोग्य PET प्लास्टिक आहे. निळ्या सुक्या कचऱ्याच्या डब्यात टाका.",
            "action": "बाटली रिकामी करून चपटी करा आणि निळ्या डब्यात टाका.",
            "why": "स्वच्छ प्लास्टिकचे पुनर्प्रक्रिया सुलभ होते."
        },
        "food_waste": {
            "tip": "हा ओला सेंद्रिय कचरा आहे. प्लास्टिकशिवाय हिरव्या डब्यात टाका.",
            "action": "हिरव्या डब्यात टाका, यापासून सेंद्रिय खत किंवा बायोगॅस तयार होतो.",
            "why": "ओल्या कचऱ्यापासून उत्तम सेंद्रिय खत बनते."
        }
    },
    "en": {
        "language_name": "English",
        "battery": {
            "tip": "DANGER: Lithium battery. Never dispose in dry or wet municipal bins.",
            "action": "Deposit in Special-Care Red Vault for authorised EPR take-back.",
            "why": "Lithium cells ignite under compactor pressure, causing facility fires."
        },
        "plastic_bottle": {
            "tip": "Recyclable PET plastic. Deposit in the Blue dry waste bin.",
            "action": "Empty liquids, crush container, discard in Blue Recyclable Bin.",
            "why": "Clean dry packaging feeds regional material recovery facilities."
        },
        "cardboard": {
            "tip": "Cardboard belongs in the Blue Dry Recyclable Bin.",
            "action": "If soiled with food grease, rinse or segregate to prevent contamination.",
            "why": "Grease and food residue ruin paper recycling pulp batches."
        },
        "food_waste": {
            "tip": "Organic wet waste. Deposit loose in Green Bin without plastic liners.",
            "action": "Route to community composting or anaerobic digestion facility.",
            "why": "Source segregation eliminates methane formation in landfills."
        },
        "vegetable_scrap": {
            "tip": "100% compostable organic wet waste.",
            "action": "Place in Green Wet Waste Bin.",
            "why": "High nitrogen organic material generates nutrient-rich compost."
        },
        "metal_can": {
            "tip": "Aluminium/tin container: infinitely recyclable dry item.",
            "action": "Rinse empty and deposit in the Blue Dry stream.",
            "why": "Recycling aluminium saves 95% of primary smelting energy."
        },
        "glass_bottle": {
            "tip": "Dry recyclable glass container.",
            "action": "Deposit gently in Blue Bin to prevent breakage.",
            "why": "Glass is 100% recyclable with zero loss in chemical purity."
        },
        "multi_layer_packaging": {
            "tip": "Multi-layer laminate packaging (MLP/RDF).",
            "action": "Deposit in Blue Dry Bin for cement kiln co-processing.",
            "why": "MLPs become Refuse Derived Fuel for energy recovery."
        },
        "paper": {
            "tip": "Clean dry paper.",
            "action": "Keep dry and deposit in the Blue Bin.",
            "why": "Recycling one ton of paper saves 17 trees and 26,000 litres of water."
        }
    }
}

SAMPLE_QUERIES = {
    "hi": {
        "plastic_bottle": "यह प्लास्टिक की बोतल कहाँ डालें?",
        "battery": "यह पुरानी लिथियम बैटरी कहाँ फेंकें?",
        "food_waste": "बचा हुआ खाना किस डिब्बे में जाएगा?"
    },
    "ta": {
        "plastic_bottle": "இந்த பிளாஸ்டிக் பாட்டிலை எங்கு போடுவது?",
        "battery": "இந்த பேட்டரியை எங்கே போடுவது?",
        "food_waste": "மீதமுள்ள உணவை எந்த தொட்டியில் போடுவது?"
    },
    "te": {
        "plastic_bottle": "ఈ ప్లాస్టిక్ బాటిల్‌ను ఎక్కడ వేయాలి?",
        "battery": "ఈ బ్యాటరీని ఎక్కడ పారవేయాలి?",
        "food_waste": "మిగిలిన ఆహారాన్ని ఏ డబ్బాలో వేయాలి?"
    },
    "bn": {
        "plastic_bottle": "এই প্লাস্টিকের বোতলটি কোথায় ফেলব?",
        "battery": "এই ব্যাটারিটি কোথায় ফেলব?",
        "food_waste": "উচ্ছিষ্ট খাবার কোন পাত্রে ফেলব?"
    },
    "mr": {
        "plastic_bottle": "ही प्लास्टिकची बाटली कुठे टाकावी?",
        "battery": "ही बॅटरी कुठे टाकावी?",
        "food_waste": "उरलेले अन्न कोणत्या डब्यात टाकावे?"
    },
    "en": {
        "plastic_bottle": "Where should I dispose of this plastic bottle?",
        "battery": "How do I safely discard this lithium battery?",
        "food_waste": "Which bin is for organic food leftovers?"
    }
}


class VernacularGuidanceEngine:
    """Deterministic rule-based multi-lingual guidance. No neural network involved."""

    def query(self, user_question: str, detected_class: str, lang: str = "en") -> Dict[str, Any]:
        """Look up guidance for a detected item in the requested language."""
        t0 = time.perf_counter()
        lang = lang.lower() if lang in VERNACULAR_DATABASE else "en"
        lang_db = VERNACULAR_DATABASE[lang]
        item_data = lang_db.get(detected_class, lang_db.get("plastic_bottle", {}))
        lookup_ms = (time.perf_counter() - t0) * 1000.0

        return {
            "input_query": user_question,
            "detected_item": detected_class,
            "language": lang_db["language_name"],
            "response": item_data.get("tip", ""),
            "action": item_data.get("action", ""),
            "rationale": item_data.get("why", ""),
            "engine": "deterministic_rule_lookup",
            "lookup_time_ms": round(lookup_ms, 3),
            "cloud_bandwidth_bytes": 0
        }

    def get_tip(self, predicted_class: str, lang: str = "en") -> Dict[str, str]:
        """Simple tip retrieval for embedding in evaluation JSONs."""
        lang = lang.lower() if lang in VERNACULAR_DATABASE else "en"
        lang_db = VERNACULAR_DATABASE[lang]
        item_data = lang_db.get(predicted_class, {})
        return {
            "tip": item_data.get("tip", ""),
            "action": item_data.get("action", ""),
            "why": item_data.get("why", "")
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EcoSort-Edge Vernacular Guidance Engine")
    parser.add_argument("--lang", type=str, default="hi")
    parser.add_argument("--item", type=str, default="plastic_bottle")
    args = parser.parse_args()

    engine = VernacularGuidanceEngine()

    print("\n" + "=" * 80)
    print(" [ECOSORT-EDGE] MULTI-LINGUAL VERNACULAR GUIDANCE ENGINE")
    print("=" * 80)

    # Show the Hindi plastic bottle interaction
    q = "यह प्लास्टिक की बोतल कहाँ डालें?"
    res = engine.query(q, detected_class="plastic_bottle", lang="hi")
    print(f"\n  User (Hindi)  : \"{q}\"")
    print(f"  Assistant     : \"{res['response']}\"")
    print(f"  Action Guide  : \"{res['action']}\"")
    print(f"  Lookup Time   : {res['lookup_time_ms']} ms | Cloud: 0 bytes")

    print("\n" + "-" * 80)
    print("  Battery hazard guidance across 6 languages:")
    for code in ["hi", "ta", "te", "bn", "mr", "en"]:
        b = engine.get_tip("battery", lang=code)
        name = VERNACULAR_DATABASE[code]["language_name"]
        print(f"    [{name:<8}] {b['tip']}")

    print("=" * 80 + "\n")
