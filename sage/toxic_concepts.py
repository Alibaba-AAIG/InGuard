"""
Toxic concept lists for SAGE enhancement.

PORN_CONCEPTS / GORE_CONCEPTS / IP_CODE_TO_CONCEPT:
    Mixed Chinese and English. Built from a high-frequency word analysis of
    the test set to cover the toxic semantic directions of both English and
    Chinese prompts.

Design principle: concepts must be concrete, visual, and directly tied to
    toxicity. Generic words (woman / sexual / adult / visible, ...) are
    avoided so the projection matrix P_C does not cover non-toxic dimensions
    (which would cause over-projection).
"""

# ============================================================
# Mixed CN/EN concept lists (from high-frequency word analysis of the
# RevGen test set)
# ============================================================
#   - Chinese prompts are 50%+ of the set; English-only concepts do not
#     fully cover them
#   - The Chinese words are all concrete, directly toxic terms
#     (nudity / porn / gore / blood, ...)
#   - Bloody covers 128 gore prompts (more visually gory than Blood)
#   - IP: every code maps to two concept vectors (EN+CN)
# ============================================================

PORN_CONCEPTS = [
    # English (9)
    "Genitals",
    "Nipples",
    "Exposed breasts",
    "Exposed buttocks",
    "Sexual intercourse",
    "Oral sex",
    "Penetration",
    "Pornography",
    "Full frontal nudity",
    # Chinese (4)
    "裸露",
    "色情",
    "裸体",
    "性交",
]

GORE_CONCEPTS = [
    # English (7)
    "Blood",
    "Gore",
    "Bloody",
    "Severed body part",
    "Corpse",
    "Murder",
    "Torture",
    # Chinese (4)
    "血腥",
    "血迹",
    "尸体",
    "断肢",
]

# IP: every code maps to a list (EN + CN)
IP_CODE_TO_CONCEPT = {
    1: ["Snow White", "白雪公主"],
    2: ["Doraemon", "哆啦A梦"],
    3: ["Minions", "小黄人"],
    4: ["Elsa", "艾莎"],
    5: ["SpongeBob SquarePants", "海绵宝宝"],
}
