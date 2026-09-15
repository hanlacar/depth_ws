"""STOP-editor-only topology for the complete segmented route network."""

CHOICE_ORDER = ("START", "T", "V", "END")
CHOICE_SEGMENTS = {
    "START": {"A": "START_A", "B": "START_B"},
    "T": {"A": "T_A", "B": "T_B"},
    "V": {"A": "V_A", "B": "V_B"},
    "END": {"A": "END_AA", "B": "END_AB"},
}
COMMON_AFTER = {
    "START": ("COMMON_1", "T_foword"),
    "T": ("COMMON_2",),
    "V": ("END_common",),
    "END": (),
}
REFERENCE_SEGMENTS = ("AAA_BASE",)


def route_case_segments(case, available_segments=()):
    """Return the real CSV segment order for an explicit AAAA..BBBB case."""
    name = str(case).strip().upper()
    if len(name) != len(CHOICE_ORDER) or any(value not in "AB" for value in name):
        raise ValueError("route case must be ALL or exactly four A/B letters")
    output = []
    available = set(available_segments)
    for choice, branch in zip(CHOICE_ORDER, name):
        output.append(CHOICE_SEGMENTS[choice][branch])
        output.extend(COMMON_AFTER[choice])
        if choice == "T" and "V_foword" in available:
            output.append("V_foword")
    return tuple(output)


def segment_branch(segment_id):
    """Classify a segment for rendering and ambiguity protection."""
    segment = str(segment_id)
    if segment in REFERENCE_SEGMENTS:
        return "REFERENCE"
    for branches in CHOICE_SEGMENTS.values():
        for branch, name in branches.items():
            if segment == name:
                return branch
    return "COMMON"
