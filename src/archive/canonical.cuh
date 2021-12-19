#ifndef CANONICAL_CUH
#define CANONICAL_CUH

/**
 * @file canonical.cuh
 * @author Jiannan Tian
 * @brief Canonization of existing Huffman codebook (header).
 * @version 0.1
 * @date 2020-09-20
 * Created on 2020-04-10
 *
 * @copyright (C) 2020 by Washington State University, The University of Alabama, Argonne National Laboratory
 * See LICENSE in top-level directory
 *
 */

#include <cstdint>

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

template <typename CODE, typename KEY>
KERNEL void canonize(uint8_t* singleton, int DICT_SIZE);

#endif

#endif
