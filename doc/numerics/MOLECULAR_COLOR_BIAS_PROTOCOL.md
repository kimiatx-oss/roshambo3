# Canonical molecular-color aggregate control, preregistered

No native molecular output has been inspected for this protocol. Reference-only
feasibility used the published fp64 formula and the four attributed alignment
fixtures with canonical fsum/hash/endpoint safeguards unchanged. Use each complete
LIT-PCBA molecule as its own query/library at the identity pose, physical inputs
cast once to binary32; retain every shape atom and every typed color feature.
Repeat each of the four rows 32 times, with no sample-size reduction.

The existing generic physical-oracle row gate remains unchanged: argument depth
16, coefficient depth8, libm expf relative allowance2^-22, coordinate uncertainty
2*u*M (M=max absolute physical coordinate over all shape/color inputs), and a
serial summation allowance counting every possible n_query_color*n_library_color
pair. The physical oracle evaluates the actual binary32 input coordinates.

Derive an additional aggregate radius with the same preparation, exponent and
coefficient allowances, but count only issued nonzero-width color interactions:
K=sum_type n_query(type)*n_library(type), for the canonical diagonal matrix.
All off-type cells are skipped by the native canonical loops, so they execute
no addition. A serial sum of K positive terms has relative error at most
Gamma(K)=K*u/(1-K*u), under arbitrary error signs/correlation. FMA contraction
cannot increase that forward error. This is additional structural information,
not a statistical independence assumption or a residual-derived shrink factor.
Keep the generic per-row gates and compare the aggregate physical residual to
fsum(per-row K-model bounds)/N, rounded outward by one double neighbor. Evaluate
both the complete-panel mean and all four molecule strata. Within-row Gaussian
coordinate/exponential bounds are unchanged from the independent generic
forward-error contracts. Verify the actual native final issued-term census
matches repeats*sum(K) and the intended identity pose in every run.

Reference-only relative intervals (generic row, additional K model) are:
ADRB2_3p0g:(3.63889436514561e-5,1.4334151772023999e-5),
ADRB2_3pds:(4.554591450240244e-5,1.5741827930883286e-5),
ADRB2_4ldl:(3.6416418630152365e-5,1.3407927331540135e-5),
ESR1_ant_5fqv:(3.9860591262835893e-5,2.2574210107037858e-5).
These follow solely from input geometry/type counts and the declared model.

Freeze actual native CPU coefficient plants at +/-2^-15, coherently changing
all four physical C declarations (scalar matrix, shared-pose matrix, field
coefficient and canonical default parameters). Require clean/restored pass,
every unchanged generic row gate pass for both plants, overall and all four
stratum aggregate gates fail for both plants, and matching restored source and
binary hashes. Preserve actual patches, build logs, binary/source/dependency
hashes and source-specific receipts. Any failure is negative evidence; do not
change bounds, panel or amplitude to rescue the experiment.

The model retains an explicit CPU libm expf accuracy assumption over its negative
normal-output argument domain; it is not inferred from residuals. The glibc2.39
[source's expf error declaration](https://raw.githubusercontent.com/bminor/glibc/release/2.39/master/sysdeps/ieee754/flt-32/e_expf.c)
is 0.502 ULP for nearest rounding, within the conservative two-ULP relative
allowance here. Record the actual loaded libm version/hash and rounding mode.
Other library versions require their own authority or finite-domain check; do
not infer this bound merely from a library being named libm. Additive FTZ
allowance in the existing model covers subnormal terms. This is canonical CPU
exact-final color on the declared molecular panel; it does not certify generic
shape means, noncanonical matrices, vector polynomials, FIELD or optimizer paths.
