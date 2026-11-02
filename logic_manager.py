"""Reusable test-logic helpers.

This file keeps logic selection rules outside the GUI code so the main window can
focus on UI concerns and the detection behavior can stay modular.
"""

LOGIC_NONE = "none"
LOGIC_FIRE_ALERT = "fire_alert"
LOGIC_SOUND_ALERT = "sound_alert"
LOGIC_COUNTING = "counting"


def sync_logic_checkbox_group(logic_none_var, logic_fire_var, logic_sound_var, logic_count_var=None):
    """Keep the 'None' option as the real off-state for the logic options."""
    any_logic = (
        logic_fire_var.get() or
        logic_sound_var.get()
    )
    if logic_count_var is not None:
        any_logic = any_logic or logic_count_var.get()

    if logic_none_var.get() and any_logic:
        logic_none_var.set(False)
        return

    if any_logic:
        logic_none_var.set(False)
        return

    logic_none_var.set(True)
    logic_fire_var.set(False)
    logic_sound_var.set(False)
    if logic_count_var is not None:
        logic_count_var.set(False)


def resolve_selected_logic(logic_none_var, logic_fire_var, logic_sound_var, logic_count_var=None):
    """Return the active logic set based on the checkbox state."""
    if logic_none_var.get():
        return set()

    active = set()
    if logic_fire_var.get():
        active.add(LOGIC_FIRE_ALERT)
    if logic_sound_var.get():
        active.add(LOGIC_SOUND_ALERT)
    if logic_count_var is not None and logic_count_var.get():
        active.add(LOGIC_COUNTING)
    return active
