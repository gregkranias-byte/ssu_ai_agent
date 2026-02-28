from __future__ import annotations

CRISIS_CUES = [
    "emergency", "bleed", "haemorrhage", "hemorrhage", "arrest", "anaphylaxis",
    "malignant hyperthermia", "mh", "can't ventilate", "cannot ventilate",
    "can't intubate", "cannot intubate", "failed airway", "desaturation", "desat",
    "pph", "sepsis", "shock", "collapse"
]

TEMPLATE_PRE_INTRA_POST = "pre_intra_post"
TEMPLATE_CRISIS = "crisis"
TEMPLATE_EXAM = "exam"

def detect_crisis(question: str) -> bool:
    q = (question or "").lower()
    return any(c in q for c in CRISIS_CUES)

def format_instructions(template: str) -> str:
    if template == TEMPLATE_CRISIS:
        return (
            "Use the CRISIS format:\n"
            "1) Immediate actions\n"
            "2) Airway plan\n"
            "3) Haemodynamic plan\n"
            "4) Differentials / causes\n"
            "5) Pitfalls / don't-miss\n"
            "Always cite sources with page numbers.\n"
        )
    if template == TEMPLATE_EXAM:
        return (
            "Use the EXAM format:\n"
            "1) Key points\n"
            "2) Viva-style questions (3–6)\n"
            "3) Model answers\n"
            "Always cite sources with page numbers.\n"
        )
    # default
    return (
        "Use the PRE/INTRA/POST format:\n"
        "Pre-op:\n- ...\n"
        "Intra-op:\n- ...\n"
        "Post-op:\n- ...\n"
        "Include Pitfalls / don't-miss at the end.\n"
        "Always cite sources with page numbers.\n"
    )
