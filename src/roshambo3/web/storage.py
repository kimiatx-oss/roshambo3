"""Bounded sidecar ownership for long-lived and large-corpus web workflows.

Drop references, never forcibly close a memmap: an in-flight decode or block
may still hold a valid view onto an evicted group. That view owns its mapping
until consumption finishes. These adapters intentionally leave core API cache
policy unchanged.
"""
from collections import OrderedDict
from roshambo3.engine import R3gSource
from roshambo3.format import Catalog


class BoundedSource(R3gSource):
    cache_groups = 16

    def _group_sidecars(self, gi):
        # The base method holds this same lock while loading metadata. Use
        # a reentrant lock so eviction and publication remain one operation.
        with self._lock:
            if gi not in self._sidecars and len(self._sidecars) >= self.cache_groups:
                del self._sidecars[next(iter(self._sidecars))]
            return super()._group_sidecars(gi)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        import threading
        self._lock = threading.RLock()


class BoundedCatalog(Catalog):
    cache_groups = 16

    def __init__(self, corpus):
        super().__init__(corpus)
        self._recent = OrderedDict()

    def _group_sidecars(self, gi):
        if gi not in self._recent and len(self._recent) >= self.cache_groups:
            old, _ = self._recent.popitem(last=False)
            self._sidecars[old] = None
        self._recent[gi] = None
        self._recent.move_to_end(gi)
        return super()._group_sidecars(gi)
