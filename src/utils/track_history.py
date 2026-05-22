"""滑动窗口历史缓冲，用于需要时序信息的事件检测"""
from collections import deque
from typing import Any


class TrackHistory:
    """为每个 track_id 维护一个定长的帧级历史队列"""

    def __init__(self, maxlen: int = 30):
        self._maxlen = maxlen
        self._data: dict[int, deque] = {}

    def push(self, track_id: int, value: Any):
        if track_id not in self._data:
            self._data[track_id] = deque(maxlen=self._maxlen)
        self._data[track_id].append(value)

    def get(self, track_id: int) -> deque:
        return self._data.get(track_id, deque())

    def clean_stale(self, active_ids: set):
        stale = set(self._data.keys()) - active_ids
        for tid in stale:
            del self._data[tid]
