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
    "EB01-003": 0,
    "EB01-006": 1,
    "EB01-009": 2,
    "EB02-003": 3,
    "OP01-001": 4,
    "OP01-016": 5,
    "OP02-015": 6,
    "OP04-010": 7,
    "OP07-015": 8,
    "OP08-007": 9,
    "OP08-010": 10,
    "OP08-013": 11,
    "OP08-015": 12,
    "ST01-011": 13,
    "ST21-017": 14,
}
