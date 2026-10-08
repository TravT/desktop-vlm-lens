"""
In-memory store for recent screen captures, so several questions about the same screen reuse one image.

Why: the VLM server keeps the image in its prompt cache, so a follow-up on identical image bytes takes
2-4 s instead of the full vision-encode time, and every answer refers to the same pixels. Frames live
in this process's memory only (a few, short-lived) and are never written to disk.
"""

import secrets
import time
from collections import OrderedDict
from typing import Any, Dict, Optional, Tuple

MAX_FRAMES = 3
TTL_SEC = 300


class FrameStore:
    def __init__(self, max_frames: int = MAX_FRAMES, ttl_sec: float = TTL_SEC, clock=time.monotonic):
        self.max_frames = max_frames
        self.ttl_sec = ttl_sec
        self._clock = clock
        self._frames: "OrderedDict[str, Tuple[float, Any, Dict[str, Any]]]" = OrderedDict()

    def _expire(self) -> None:
        now = self._clock()
        for fid in [f for f, (t, _, _) in self._frames.items() if now - t > self.ttl_sec]:
            del self._frames[fid]

    def put(self, img, meta: Dict[str, Any]) -> str:
        self._expire()
        fid = secrets.token_hex(4)
        self._frames[fid] = (self._clock(), img, dict(meta))
        while len(self._frames) > self.max_frames:
            self._frames.popitem(last=False)
        return fid

    def get(self, frame_id: str) -> Optional[Tuple[Any, Dict[str, Any]]]:
        self._expire()
        entry = self._frames.get(frame_id)
        if not entry:
            return None
        self._frames.move_to_end(frame_id)
        return entry[1], dict(entry[2])

    def clear(self) -> None:
        self._frames.clear()

    def __len__(self) -> int:
        self._expire()
        return len(self._frames)
