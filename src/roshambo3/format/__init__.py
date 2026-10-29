"""Format-3 descriptors, molecule tables and self-contained group files."""
from __future__ import annotations

from .catalog import Catalog, ConformerRef
from .constants import (BLOCK_CONF, BUCKET_EDGES, COORD_SCALE, L_ALIGN, MAGIC,
                        MAX_COLOR_TYPE_COUNT, MAX_CONF_PER_GROUP, VERSION,
                        bucket_of, pad_L, quantize)
from .r3g.reader import Corpus, Group

__all__ = [
    "Corpus", "Group", "Catalog", "ConformerRef", "BLOCK_CONF", "BUCKET_EDGES",
    "COORD_SCALE", "L_ALIGN", "MAGIC", "MAX_COLOR_TYPE_COUNT",
    "MAX_CONF_PER_GROUP", "VERSION", "bucket_of", "pad_L", "quantize",
]
