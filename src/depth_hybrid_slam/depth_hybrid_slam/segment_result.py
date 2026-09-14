"""One-shot terminal segment result formatting."""


class SegmentResultLatch:
    def __init__(self):
        self.results = {}

    def report(self, segment, passed, reason):
        segment = int(segment)
        if not 1 <= segment <= 11:
            raise ValueError("segment must be in [1, 11]")
        if segment in self.results:
            return None
        verdict = "COMPLETE" if bool(passed) else "FAIL"
        line = f"[SEGMENT {segment}] {verdict} - {str(reason).strip()}"
        self.results[segment] = line
        return line
