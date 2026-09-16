"""One-shot segment validation result formatting and terminal completion."""


class SegmentResultLatch:
    def __init__(self, end_mode=11):
        self.end_mode = int(end_mode)
        if not 1 <= self.end_mode <= 11:
            raise ValueError("end_mode must be in [1, 11]")
        self.results = {}
        self.terminal = False

    def report(self, segment, passed, reason):
        segment = int(segment)
        if not 1 <= segment <= 11:
            raise ValueError("segment must be in [1, 11]")
        if segment in self.results:
            return None
        verdict = "COMPLETE" if bool(passed) else "FAIL"
        detail = str(reason).strip()
        line = f"[SEGMENT {segment}] {verdict}"
        if detail:
            line += " - "+detail
        self.results[segment] = line
        # FAIL is an observable validation verdict, not a motion command.  A
        # failed check must not strand the vehicle at the validation waypoint;
        # only successful completion of the selected end mode may latch the
        # early range terminal stop.  Independent safety faults remain handled
        # by SafetyGate/CommandArbiter.
        if segment == self.end_mode and bool(passed):
            self.terminal = True
        return line
