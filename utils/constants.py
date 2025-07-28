"""Shared constants for OPTCG automation."""

# Button name constants
ATTACK_BTN = "attack"
NO_BLOCKER_BTN = "no_blocker"
CHOOSE_ZERO_TARGETS_BTN = "choose_0_targets"
CHOOSE_NEG1_TARGETS_BTN = "choose_-1_targets"
CHOOSE_FRIENDLY_TARGETS_BTN = "choose_0_friendly_targets"
SELECT_CHARACTER_TO_REPLACE_BTN = "select_character_to_replace"
SELECT_TARGET_BTN = "select_target"
DEPLOY_BTN = "deploy"
DONT_DRAW_ANY_BTN = "dont_draw_any"
END_TURN_BTN = "end_turn"
RESOLVE_ATTACK_BTN = "resolve_attack"
RETURN_CARDS_TO_DECK_BTN = "return_cards_to_deck"

# Mapping of card IDs to indices used in observations
CARD_IDS = {
    "EB01-003": 1,
    "EB01-006": 2,
    "EB01-009": 3,
    "EB02-003": 4,
    "OP01-001": 5,
    "OP01-016": 6,
    "OP02-015": 7,
    "OP04-010": 8,
    "OP07-015": 9,
    "OP08-007": 10,
    "OP08-010": 11,
    "OP08-013": 12,
    "OP08-015": 13,
    "ST01-011": 14,
    "ST21-017": 15,
}
