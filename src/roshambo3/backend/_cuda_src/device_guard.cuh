#pragma once
////////////////////////////////////////////////////////////////////////////////
/// Make one device current for a scope, and put back what was current.
///
/// **Why this exists.** A device pointer belongs to the device it was allocated
/// on, and the runtime does not diagnose using it from another one. Every entry
/// point that allocates, frees, launches into, or copies out of a state's
/// buffers therefore has to make *that state's* device current first -- the
/// calling thread may be on any device at all. When it does not, nothing fails
/// at the point of the mistake: the fault surfaces later, as an illegal access
/// at the first place a foreign-device pointer is dereferenced, which is never
/// where it was caused.
///
/// (Twelve eager `cudaMalloc` calls in the
/// molecule-level top-K's constructor landed on whichever device the calling
/// thread happened to be on, while the kernels reading them ran on the device
/// the state was asked for. On a one-GPU host the two coincide and nothing is
/// wrong, which is why it survived to here.)
///
/// **Why a scope guard and not save/set/restore.** The hand-written form is
/// correct exactly until someone adds an allocation above it or an early
/// `return` below it. Both the top-K states had a hand-written restore in
/// `destroy` and nothing at all in the paths that allocate.
///
/// The device a state owns is `st->device_id`, never a `device_id` argument
/// passed by the caller: the state owns its allocations and owns their device,
/// and a caller that has to be told which device a state is on can be told
/// wrong. (Several bindings default that argument to 0.)
////////////////////////////////////////////////////////////////////////////////

#include <cuda_runtime.h>

#include <new>
#include <stdexcept>
#include <string>

class ScopedDevice {
public:
    /// Make `device_id` current until this object goes out of scope. A negative
    /// `device_id` means "leave the current device alone", so a state that was
    /// never given one still works.
    explicit ScopedDevice(int device_id){ enter(device_id, true); }

    /// The same, but reporting nothing. For destructor paths, where throwing
    /// would call `std::terminate` rather than report anything.
    ScopedDevice(int device_id, std::nothrow_t){ enter(device_id, false); }

    ~ScopedDevice(){
        // Deliberately unchecked: this runs while an exception may be in
        // flight, and a failure to restore is not something the caller can act
        // on. `cudaGetLastError` is *not* called -- it would swallow an error
        // the body has not checked yet.
        if (restore_) (void)cudaSetDevice(previous_);
    }

    ScopedDevice(const ScopedDevice &) = delete;
    ScopedDevice & operator=(const ScopedDevice &) = delete;

private:
    void enter(int device_id, bool report){
        cudaError_t e = cudaGetDevice(&previous_);
        if (e != cudaSuccess){
            if (report)
                throw std::runtime_error(
                    std::string("roshambo3.cuda: cudaGetDevice failed: ")
                    + cudaGetErrorString(e));
            return;
        }
        if (device_id < 0 || device_id == previous_) return;
        e = cudaSetDevice(device_id);
        if (e != cudaSuccess){
            if (report)
                throw std::runtime_error(
                    std::string("roshambo3.cuda: could not make device ")
                    + std::to_string(device_id) + " current: "
                    + cudaGetErrorString(e));
            return;
        }
        restore_ = true;
    }

    int  previous_ = 0;
    bool restore_ = false;
};
