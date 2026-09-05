from collections import defaultdict, deque
import time
import numpy as np


class UniqueFrameRates:
    def __init__(self): self.values = defaultdict(deque); self.seen = defaultdict(set)
    def mark(self, event, stamp_sec, stamp_nanosec):
        key=int(stamp_sec)*1_000_000_000+int(stamp_nanosec); now=time.perf_counter()
        if key in self.seen[event]: return False
        self.seen[event].add(key); self.values[event].append((now,key))
        while self.values[event] and self.values[event][0][0] < now-11:
            _, old=self.values[event].popleft(); self.seen[event].discard(old)
        return True
    def snapshot(self,event):
        now=time.perf_counter(); out={}
        for window in (1,5,10):
            values=[item for item in self.values[event] if item[0]>=now-window]
            if len(values)>1:
                hs=(values[-1][1]-values[0][1])/1e9; ars=values[-1][0]-values[0][0]
                gaps=np.diff([x[0] for x in values])*1000
                hf=(len(values)-1)/hs if hs>0 else 0.; af=(len(values)-1)/ars if ars>0 else 0.
                p95=float(np.percentile(gaps,95)); maximum=float(gaps.max())
            else: hf=af=p95=maximum=0.
            out[f"{window}s"]={"unique_frames":len(values),"header_fps":hf,
                               "arrival_fps":af,"arrival_gap_p95_ms":p95,
                               "arrival_gap_max_ms":maximum}
        return out
