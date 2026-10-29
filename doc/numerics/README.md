# Frozen numerical-check protocols

Each `*_PROTOCOL.md` here was written and frozen before the check it describes
was run. The `bench/numerics_*` harnesses bind every record they write to the SHA-256
of their protocol, so these files are read at run time and **must not be
edited**: a changed byte makes every later record disagree with the earlier
ones. A revised protocol is a new file.

They live here because the harnesses read them at run time. Protocols the
harnesses merely cite, such as the directed-bias and live-optimizer protocols,
are not distributed.
