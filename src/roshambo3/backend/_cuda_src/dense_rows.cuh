#pragma once
////////////////////////////////////////////////////////////////////////////////
/// The library-row plane of a DENSE pair launch, written on the device.
///
/// A dense launch scores every query against every library row of the block,
/// query-major: row `q * rows_per_query + t` is query `q` against library row
/// `t`. The pair kernels (kernels.cu) read a library row per output row from an
/// `out_library_row` array; for a sparse launch that array is uploaded from the
/// caller's pair list, for a dense one this fills it on the device, so nothing
/// is built on the host or crosses the bus and the pair kernels -- their code
/// and their code generation -- stay exactly as they are. Deriving the row
/// inside the pair kernels instead was tried and moved score bits in one
/// register-capped cell.
///
/// Its own translation unit: it belongs to neither the pair kernels (adding it
/// to kernels.cu would put a new function in the unit whose SASS is pinned) nor
/// the top-K.
////////////////////////////////////////////////////////////////////////////////

/// Fill `d_out[q * rows_per_query + t] = t` for every `q < n_query`. A 2-D grid,
/// library row on x and query on y, so no index is divided. Asynchronous on the
/// default stream, ahead of the pair kernel that reads it.
void dense_library_rows(int * d_out, int n_query, long rows_per_query,
                        int device_id);
