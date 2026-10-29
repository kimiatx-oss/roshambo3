# Typed-color aggregate control, preregistered before native evaluation

Scope: exact-final analytic CPU/CUDA color scoring at the identity pose, with
one shape atom and one pharmacophore per conformer, all coordinates zero.
This is a typed arithmetic control, not an ordinary molecular geometry, SEARCH
exponential, vector polynomial or FIELD qualification. Generic molecular means
remain diagnostic. All six feature types and all 36 ordered type pairs are
present, repeated 32 times without any sample-size gain.

For types a,b in 1..6 set width = 0 if (a+b)%5 == 0, otherwise
4 if (a+b)%3 == 0, otherwise 1. Height is 0 if (a+2*b)%7 == 0,
otherwise 2**(((a+b)%3)-1). This symmetric width/height matrix has noncanonical
cross-type interactions, zero-width skips and zero-height issued terms.
(The formula for zero height need not be symmetric.)

At zero distance every coordinate, scale, pose, squared-distance and exponent
operation is exact, including wire reconstruction. At nonzero width the physical
coefficient is C*h*h/width**1.5, C=(pi/2)**1.5. Width in {1,4} and dyadic heights
make the scale h*h/width**1.5 an exact power of two. Permit two binary32 ULP on
C (including its decimal representation and double coefficient formation),
plus 64 double unit roundoffs on the independent physical expression. Permit
two binary32 ULP on exp2f(0)=1, then outward-round the positive endpoint product
by a full binary32 neighbor. CPU exact-final double exp at zero is exactly one
under the C11 IEC60559 special case; the shared float interval is conservative.
CUDA's two ULP exp2f allowance is the documented CUDA13.2 accuracy assumption
already cited in BIAS_PROTOCOL.md. Multiplication by normal powers of two and
addition to zero are exact. There is only one feature pair per row: no uncertain
within-row reduction. Width-zero and height-zero contributions are exactly zero.
Every observation must satisfy its unchanged generic physical-oracle row gate.
The aggregate gate compares fsum(observed)/N to fsum(lower)/N..fsum(upper)/N,
with fp64 outward nextafter on the mean endpoints. Errors may be arbitrarily
correlated. Require aggregate detection overall and separately for each positive
ordered type-pair stratum; zero strata must stay zero. No fit to native residuals.

Before observing clean or mutant outputs, fix native coefficient plants at
multiplication by (1 + 2^-20) and (1 - 2^-20). CPU plants change both physical
color-prefactor declarations used by matrix preparation (color_scalar.cpp and
shared_pose_api.cpp). CUDA changes ROSHAMBO3_COLOR_PREFACTOR only. Each variant
must be compiled from actual source, show every original row gate passing and
the independent aggregate gate failing, then restore source, rebuild and rerun
the clean control. Preserve native/source hashes, exact patch and build logs.
A surviving plant or failing clean control is negative evidence; no panel,
interval or amplitude edits to rescue it. No production source change is shipped.

## Version 2, analytical correction before any native observation

The unchanged CPU generic singleton row allowance is about 13*u, whereas
2^-20 = 16*u. Thus version 1's amplitude cannot reliably satisfy row-pass;
this rejection follows algebraically, without inspecting a native residual.
Freeze CPU plants instead at +/-2^-21 (8*u). Use the CPU-specific IEC60559
exp(0)=1 identity to eliminate the two-float-ULP exponential uncertainty for
CPU final scoring, while retaining two coefficient ULP, reference allowance
and outward product rounding. This identity was already stated above and is
not inferred from a clean output. CUDA retains the version 1 interval and
amplitude; CUDA qualification is independent and may fail the row criterion.
No generic row bound changes. Preserve version 1 above as rejected protocol.

CPU coefficient formation additionally requires a separately executed libm
control at pow(1,1.5)=1 and pow(4,1.5)=8. Both outputs are mathematical exact
values, not production score fixtures. Record the loaded libm identity and these
controls before accepting a native receipt. This covers the two finite argument
cells used here, without assuming a full-domain CPU pow error guarantee. Double
multiplication/division by normal powers of two is exact; 64 double u covers the
declared decimal C's error and the independent physical-expression calculation.
