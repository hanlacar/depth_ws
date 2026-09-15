"""One-shot terminal segment result formatting."""


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
        if segment == self.end_mode:
            self.terminal = True
        return line
