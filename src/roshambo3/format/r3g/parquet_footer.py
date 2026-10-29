"""Closed format-3 Parquet footer preflight, before the Arrow parser runs.

Schema field IDs follow apache/parquet-format's parquet.thrift. Unknown
fields are refused, rather than skipped. All lengths and varints are bounded
before iteration; only the six REQUIRED flat columns can reach Arrow.
"""
from __future__ import annotations


def check_footer(data: bytes, *, n_mol: int, max_row_groups: int) -> None:
    """Raise ValueError for a footer outside FORMAT.md section 11's schema."""
    # Compact protocol: bool, byte, i16, i32, i64, binary, list, struct.
    empty = ({}, set())
    logical = ({1: empty, 10: ({1: 3, 2: 1}, {1, 2})}, set())
    schema_element = ({1: 5, 3: 5, 4: 8, 5: 5, 6: 5, 10: logical}, {4})
    size_stats = ({1: 6, 2: (6, 1), 3: (6, 1)}, set())
    encoding_stats = ({1: 5, 2: 5, 3: 5}, {1, 2, 3})
    column_meta = ({1: 5, 2: (5, 4), 3: (8, 1), 4: 5, 5: 6, 6: 6,
                    7: 6, 9: 6, 13: (encoding_stats, 4), 16: size_stats},
                   {1, 2, 3, 4, 5, 6, 7, 9})
    column_chunk = ({2: 6, 3: column_meta}, {2, 3})
    row_group = ({1: (column_chunk, 6), 2: 6, 3: 6, 5: 6, 6: 6, 7: 4}, {1, 2, 3})
    file_meta = ({1: 5, 2: (schema_element, 7), 3: 6,
                  4: (row_group, max_row_groups), 6: 8,
                  7: (({1: empty}, {1}), 6)}, {1, 2, 3, 4})
    pos = 0

    def byte():
        nonlocal pos
        if pos >= len(data):
            raise ValueError("truncated parquet footer")
        value = data[pos]
        pos += 1
        return value

    def varint(bits):
        value = 0
        for shift in range(0, bits, 7):
            b = byte()
            value |= (b & 127) << shift
            if not b & 128:
                if value >= 1 << bits:
                    break
                return value
        raise ValueError("footer varint outside its integer range")

    def integer(bits):
        z = varint(bits)
        return (z >> 1) ^ -(z & 1)

    def read(kind, wire):
        nonlocal pos
        if isinstance(kind, int):
            if wire != kind and not (kind == 1 and wire == 2):
                raise ValueError("footer field has the wrong Thrift type")
            if kind == 1:
                return wire == 1
            if kind == 3:
                return byte()
            if kind in (4, 5, 6):
                return integer({4: 16, 5: 32, 6: 64}[kind])
            length = varint(32)
            if length > len(data) - pos:
                raise ValueError("footer string outside footer")
            value = data[pos:pos + length]
            pos += length
            value.decode("utf-8")
            return value
        if isinstance(kind[0], dict):
            if wire != 12:
                raise ValueError("footer field must be a struct")
            allowed, required = kind
            values = {}
            last = 0
            while True:
                header = byte()
                if header == 0:
                    if not required <= values.keys():
                        raise ValueError("footer struct lacks required fields")
                    return values
                fid = last + (header >> 4) if header >> 4 else integer(16)
                if fid <= last or fid not in allowed:
                    raise ValueError(f"footer field {fid} repeated, out of order, or unsupported")
                last = fid
                values[fid] = read(allowed[fid], header & 15)
        if wire != 9:
            raise ValueError("footer field must be a list")
        item, cap = kind
        header = byte()
        length = header >> 4
        if length == 15:
            length = varint(32)
        if length > cap or length > len(data) - pos:
            raise ValueError("footer list exceeds its schema bound")
        return [read(item, header & 15) for _ in range(length)]

    md = read(file_meta, 12)
    if pos != len(data):
        raise ValueError("trailing bytes in parquet footer")
    names = (b"mol_id", b"ext_id", b"smiles", b"hac", b"mw", b"rotors")
    physical = (2, 6, 6, 1, 5, 1)
    schema = md[2]
    if (len(schema) != 7 or schema[0].get(5) != 6 or 1 in schema[0]
            or schema[0].get(3, 0) != 0):
        raise ValueError("footer schema must have one flat root and six columns")
    for leaf, name, typ in zip(schema[1:], names, physical):
        if leaf.get(1) != typ or leaf.get(3) != 0 or leaf[4] != name or 5 in leaf:
            raise ValueError("footer columns must match the six REQUIRED flat columns")
        if 10 in leaf and len(leaf[10]) != 1:
            raise ValueError("logical type must name exactly one type")
    if md[1] not in (1, 2) or md[3] != n_mol:
        raise ValueError("footer version or row count is invalid")
    rows = 0
    for group in md[4]:
        if not 1 <= group[3] <= 1 << 14 or len(group[1]) != 6:
            raise ValueError("footer row group has invalid dimensions")
        rows += group[3]
        for chunk, name, typ in zip(group[1], names, physical):
            col = chunk[3]
            if col[1] != typ or col[3] != [name] or col[4] not in (0, 6):
                raise ValueError("footer column type, path, or codec is unsupported")
            if col[5] != group[3] or any(col[k] < 0 for k in (6, 7, 9)):
                raise ValueError("footer column has invalid lengths")
            if not col[2] or not set(col[2]) <= {0, 3}:
                raise ValueError("footer encodings must be PLAIN/RLE")
            stats = col.get(16, {})
            if 1 in stats and (typ != 6 or stats[1] < 0):
                raise ValueError("SizeStatistics byte count is valid only for BYTE_ARRAY columns")
            for histogram in (2, 3):
                if histogram in col.get(16, {}):
                    values = col[16][histogram]
                    if len(values) > 1 or (values and values[0] < 0):
                        raise ValueError("SizeStatistics histogram is invalid for REQUIRED columns")
    if rows != n_mol:
        raise ValueError("footer row groups do not sum to n_mol")
