/**
 * @file canonical.cu
 * @author Jiannan Tian
 * @brief Canonization of existing Huffman codebook.
 * @version 0.1
 * @date 2020-09-20
 * Created on 2020-04-10
 *
 * @copyright (C) 2020 by Washington State University, The University of Alabama, Argonne National Laboratory
 * See LICENSE in top-level directory
 *
 */

#include <stddef.h>
#include <stdint.h>

#ifdef __CUDACC__
#include <cooperative_groups.h>
namespace cg = cooperative_groups;
#endif

#include "canonical.cuh"

#ifdef __CUDACC__

#define KERNEL __global__
#define SUBROUTINE __host__ __device__
#define INLINE __forceinline__
#define ON_DEVICE_FALLBACK2HOST __device__

#else

#define KERNEL
#define SUBROUTINE
#define INLINE inline
#define ON_DEVICE_FALLBACK2HOST

#endif

#ifdef __CUDACC__

__device__ int max_bw = 0;

template <typename CODE, typename KEY>
KERNEL void canonize(uint8_t* singleton, int DICT_SIZE)
{
    auto  type_bw   = sizeof(CODE) * 8;
    auto  codebooks = reinterpret_cast<CODE*>(singleton);
    auto  metadata  = reinterpret_cast<int*>(singleton + sizeof(CODE) * (3 * DICT_SIZE));
    auto  keys      = reinterpret_cast<KEY*>(singleton + sizeof(CODE) * (3 * DICT_SIZE) + sizeof(int) * (4 * type_bw));
    CODE* i_cb      = codebooks;
    CODE* o_cb      = codebooks + DICT_SIZE;
    CODE* canonical = codebooks + DICT_SIZE * 2;
    auto  numl      = metadata;
    auto  iter_by_  = metadata + type_bw;
    auto  first     = metadata + type_bw * 2;
    auto  entry     = metadata + type_bw * 3;

    cg::grid_group g = cg::this_grid();

    int gid = blockDim.x * blockIdx.x + threadIdx.x;
    // TODO
    auto c  = i_cb[gid];
    int  bw = *((uint8_t*)&c + (sizeof(CODE) - 1));

    if (c != ~((CODE)0x0)) {
        atomicMax(&max_bw, bw);
        atomicAdd(&numl[bw], 1);
    }
    g.sync();

    if (gid == 0) {
        // printf("\0");
        // atomicMax(&max_bw, max_bw + 0);
        memcpy(entry + 1, numl, (type_bw - 1) * sizeof(int));
        // for (int i = 1; i < type_bw; i++) entry[i] = numl[i - 1];
        for (int i = 1; i < type_bw; i++) entry[i] += entry[i - 1];
    }
    g.sync();

    if (gid < type_bw) iter_by_[gid] = entry[gid];
    __syncthreads();
    // atomicMax(&max_bw, bw);

    if (gid == 0) {  //////// first code
        for (int l = max_bw - 1; l >= 1; l--) first[l] = static_cast<int>((first[l + 1] + numl[l + 1]) / 2.0 + 0.5);
        first[0] = 0xff;  // no off-by-one error
    }
    g.sync();

    canonical[gid] = ~((CODE)0x0);
    g.sync();
    o_cb[gid] = ~((CODE)0x0);
    g.sync();

    // Reverse Codebook Generation -- TODO isolate
    if (gid == 0) {
        // no atomicRead to handle read-after-write (true dependency)
        for (int i = 0; i < DICT_SIZE; i++) {
            auto    _c  = i_cb[i];
            uint8_t _bw = *((uint8_t*)&_c + (sizeof(CODE) - 1));

            if (_c == ~((CODE)0x0)) continue;
            canonical[iter_by_[_bw]] = static_cast<CODE>(first[_bw] + iter_by_[_bw] - entry[_bw]);
            keys[iter_by_[_bw]]      = i;

            *((uint8_t*)&canonical[iter_by_[_bw]] + sizeof(CODE) - 1) = _bw;
            iter_by_[_bw]++;
        }
    }
    g.sync();

    if (canonical[gid] == ~((CODE)0x0u)) return;
    o_cb[keys[gid]] = canonical[gid];
}

#define CANONIZE(CODE, KEY) template KERNEL void canonize<CODE, KEY>(uint8_t * singleton, int DICT_SIZE);

CANONIZE(uint32_t, uint8_t)
CANONIZE(uint32_t, uint16_t)
CANONIZE(uint32_t, uint32_t)
CANONIZE(uint64_t, uint8_t)
CANONIZE(uint64_t, uint16_t)
CANONIZE(uint64_t, uint32_t)

#else

#endif