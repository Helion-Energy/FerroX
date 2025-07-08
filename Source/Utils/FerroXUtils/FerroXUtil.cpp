/*
 * This file is part of FerroX.
 *
 * Contributor: Prabhat Kumar
 *
 */
#include <FerroXUtil.H>

using namespace amrex;

void FerroX_Util::Contains_sc(MultiFab& MaterialMask, bool& contains_SC)
{
    int has_SC = 0;

#ifdef AMREX_USE_GPU
    // Use atomic on device
    Gpu::DeviceVector<int> d_has_SC(1, 0);
    int* p_has_SC = d_has_SC.data();

    for (MFIter mfi(MaterialMask, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        const Box& bx = mfi.validbox();
        const auto mask = MaterialMask.array(mfi);

        AMREX_PARALLEL_FOR_3D(bx, i, j, k,
        {
            if (mask(i,j,k) >= 2.0) {
                Gpu::Atomic::Max(p_has_SC, 1);
            }
        });
    }

    // Copy back to host
    Gpu::copyAsync(Gpu::deviceToHost, d_has_SC.begin(), d_has_SC.end(), &has_SC);
    Gpu::streamSynchronize();
#else
    // Pure CPU path
    for (MFIter mfi(MaterialMask); mfi.isValid(); ++mfi) {
        const Box& bx = mfi.validbox();
        const auto lo = amrex::lbound(bx);
        const auto hi = amrex::ubound(bx);
        const auto mask = MaterialMask.array(mfi);

        for (int k = lo.z; k <= hi.z; ++k) {
        for (int j = lo.y; j <= hi.y; ++j) {
        for (int i = lo.x; i <= hi.x; ++i) {
            if (mask(i,j,k) >= 2.0) {
                has_SC = 1;
            }
        }}}
    }
#endif

    // Reduce across MPI ranks
    ParallelDescriptor::ReduceIntMax(has_SC);

    contains_SC = (has_SC == 1);
}

