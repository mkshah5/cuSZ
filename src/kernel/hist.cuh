/**
 * @file hist.cuh
 * @author Cody Rivera (cjrivera1@crimson.ua.edu), Megan Hickman Fulp (mlhickm@g.clemson.edu)
 * @brief Fast histogramming from [Gómez-Luna et al. 2013]
 * @version 0.1
 * @date 2020-09-20
 * Created on 2020-02-16
 *
 * @copyright (C) 2020 by Washington State University, The University of Alabama, Argonne National Laboratory
 * See LICENSE in top-level directory
 *
 */

#ifndef CUSZ_KERNEL_HIST_CUH
#define CUSZ_KERNEL_HIST_CUH

#include <cuda_runtime.h>
#include <cstdio>
#include <limits>

#include "../common.hh"
#include "../utils/timer.hh"

#define MIN(a, b) ((a) < (b)) ? (a) : (b)
const static unsigned int WARP_SIZE = 32;

#define NUM_TREES 100

// float entropies[100] = {1.016,1.075,1.131,1.184,1.236,1.286,1.335,1.382,1.428,1.474,1.518,1.563,1.607,1.651,1.695,1.738,1.782,1.826,1.869,1.913,1.957,2.0,2.044,2.088,2.131,2.175,2.219,2.262,2.306,2.35,2.393,2.437,2.481,2.524,2.568,2.611,2.655,2.699,2.742,2.786,2.83,2.873,2.917,2.961,3.004,3.048,3.092,3.135,3.179,3.223,3.266,3.31,3.354,3.397,3.441,3.484,3.528,3.572,3.615,3.659,3.703,3.746,3.79,3.834,3.877,3.921,3.965,4.008,4.052,4.096,4.139,4.183,4.227,4.27,4.314,4.357,4.401,4.445,4.488,4.532,4.576,4.619,4.663,4.707,4.75,4.794,4.838,4.881,4.925,4.969,5.012,5.056,5.099,5.143,5.187,5.23,5.274,5.318,5.361,5.405};

/** Cauchy entropies **/
// float entropies[100] = {0.32,0.345,0.372,0.401,0.432,0.465,0.5,0.537,0.577,0.62,0.664,0.712,0.762,0.815,0.871,0.929,0.991,1.055,1.122,1.191,1.264,1.339,1.416,1.495,1.577,1.66,1.745,1.831,1.918,2.005,2.094,2.182,2.271,2.359,2.446,2.533,2.619,2.703,2.786,2.868,2.949,3.028,3.106,3.182,3.257,3.33,3.403,3.474,3.545,3.615,3.683,3.752,3.819,3.887,3.954,4.02,4.087,4.153,4.219,4.285,4.35,4.416,4.482,4.547,4.612,4.678,4.743,4.808,4.873,4.938,5.003,5.068,5.132,5.197,5.262,5.326,5.39,5.455,5.519,5.583,5.647,5.71,5.774,5.838,5.901,5.964,6.027,6.09,6.153,6.216,6.278,6.34,6.402,6.464,6.526,6.588,6.649,6.71,6.771,6.832};

        

#define tix threadIdx.x
#define tiy threadIdx.y
#define tiz threadIdx.z
#define bix blockIdx.x
#define biy blockIdx.y
#define biz blockIdx.z
#define bdx blockDim.x
#define bdy blockDim.y
#define bdz blockDim.z

namespace kernel {

template <typename Input>
__global__ void NaiveHistogram(Input in_data[], int out_freq[], int N, int symbols_per_thread);

/* Copied from J. Gomez-Luna et al */
template <typename T, typename FREQ>
__global__ void p2013Histogram(T*, FREQ*, size_t, int, int);

template <typename FREQ>
__global__ void getCrossEntropy(FREQ*, float*, int, int, float*);

}  // namespace kernel

namespace kernel_wrapper {

/**
 * @brief Get frequency: a kernel wrapper
 *
 * @tparam T input type
 * @param in_data input device array
 * @param in_len input host var; len of in_data
 * @param out_freq output device array
 * @param nbin input host var; len of out_freq
 * @param milliseconds output time elapsed
 * @param stream optional stream
 */
template <typename T>
void get_frequency(
    T*           in_data,
    size_t       in_len,
    cusz::FREQ*  out_freq,
    int          nbin,
    float&       milliseconds,
    cudaStream_t stream = nullptr,
    float entropy_use = 1.016);

}  // namespace kernel_wrapper

template <typename T>
__global__ void kernel::NaiveHistogram(T in_data[], int out_freq[], int N, int symbols_per_thread)
{
    unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int j;
    if (i * symbols_per_thread < N) {  // if there is a symbol to count,
        for (j = i * symbols_per_thread; j < (i + 1) * symbols_per_thread; j++) {
            if (j < N) {
                unsigned int item = in_data[j];  // Symbol to count
                atomicAdd(&out_freq[item], 1);   // update bin count by 1
            }
        }
    }
}

template <typename T, typename FREQ>
__global__ void kernel::p2013Histogram(T* in_data, FREQ* out_freq, size_t N, int nbin, int R)
{
    // static_assert(
    //     std::numeric_limits<T>::is_integer and (not std::numeric_limits<T>::is_signed),
    //     "T must be `unsigned integer` type of {1,2,4} bytes");

    extern __shared__ int Hs[/*(nbin + 1) * R*/];

    const unsigned int warp_id     = (int)(tix / WARP_SIZE);
    const unsigned int lane        = tix % WARP_SIZE;
    const unsigned int warps_block = bdx / WARP_SIZE;
    const unsigned int off_rep     = (nbin + 1) * (tix % R);
    const unsigned int begin       = (N / warps_block) * warp_id + WARP_SIZE * blockIdx.x + lane;
    unsigned int       end         = (N / warps_block) * (warp_id + 1);
    const unsigned int step        = WARP_SIZE * gridDim.x;

    // final warp handles data outside of the warps_block partitions
    if (warp_id >= warps_block - 1) end = N;

    for (unsigned int pos = tix; pos < (nbin + 1) * R; pos += bdx) Hs[pos] = 0;
    __syncthreads();

    for (unsigned int i = begin; i < end; i += step) {
        int d = in_data[i];
        d     = d <= 0 and d >= nbin ? nbin / 2 : d;
        atomicAdd(&Hs[off_rep + d], 1);
    }
    __syncthreads();

    for (unsigned int pos = tix; pos < nbin; pos += bdx) {
        int sum = 0;
        for (int base = 0; base < (nbin + 1) * R; base += nbin + 1) { sum += Hs[base + pos]; }
        atomicAdd(out_freq + pos, sum);
    }
}

template <typename FREQ>
__global__ void kernel::getCrossEntropy(FREQ* quant_hist, float* precomputed_logqs, int num_bins, int p_total, float *cross_entropy){
    /* Be sure to allocate this on kernel launch*/
    extern __shared__ float entropy_components[];

    int bin = threadIdx.x;
    float* blk_logqs = &precomputed_logqs[blockIdx.x*num_bins];
    
    for (size_t i = bin; i < num_bins; i+=blockDim.x)
    {
        float p = (float)quant_hist[i]/ (float)p_total;
        float logq = (float)blk_logqs[i];
    
        entropy_components[i] = p*logq;
    }

    __syncthreads();

    if(threadIdx.x == 0){
        float intermediate = 0.0;
        for (size_t i = 0; i < num_bins; i++)
        {
            intermediate += entropy_components[i];
        }
        cross_entropy[blockIdx.x] = -1*intermediate;
    }

}

template <typename T>
void kernel_wrapper::get_frequency(
    T*           in_data,
    size_t       in_len,
    cusz::FREQ*  out_freq,
    int          num_buckets,
    float&       milliseconds,
    cudaStream_t stream,
    float entropy_use)
{
    // static_assert(
    //     std::numeric_limits<T>::is_integer and (not std::numeric_limits<T>::is_signed),
    //     "To get frequency, `T` must be unsigned integer type of {1,2,4} bytes");

    int device_id, max_bytes, num_SMs;
    int items_per_thread, r_per_block, grid_dim, block_dim, shmem_use;

    cudaGetDevice(&device_id);
    cudaDeviceGetAttribute(&num_SMs, cudaDevAttrMultiProcessorCount, device_id);

    auto query_maxbytes = [&]() {
        int max_bytes_opt_in;
        cudaDeviceGetAttribute(&max_bytes, cudaDevAttrMaxSharedMemoryPerBlock, device_id);

        // account for opt-in extra shared memory on certain architectures
        cudaDeviceGetAttribute(&max_bytes_opt_in, cudaDevAttrMaxSharedMemoryPerBlockOptin, device_id);
        max_bytes = std::max(max_bytes, max_bytes_opt_in);

        // config kernel attribute
        cudaFuncSetAttribute(
            kernel::p2013Histogram<T, cusz::FREQ>, cudaFuncAttributeMaxDynamicSharedMemorySize, max_bytes);
        
        cudaFuncSetAttribute(
            kernel::getCrossEntropy<cusz::FREQ>, cudaFuncAttributeMaxDynamicSharedMemorySize, max_bytes);
    };

    auto optimize_launch = [&]() {
        items_per_thread = 1;
        r_per_block      = (max_bytes / sizeof(int)) / (num_buckets + 1);
        grid_dim         = num_SMs;
        // fits to size
        block_dim = ((((in_len / (grid_dim * items_per_thread)) + 1) / 64) + 1) * 64;
        while (block_dim > 1024) {
            if (r_per_block <= 1) { block_dim = 1024; }
            else {
                r_per_block /= 2;
                grid_dim *= 2;
                block_dim = ((((in_len / (grid_dim * items_per_thread)) + 1) / 64) + 1) * 64;
            }
        }
        shmem_use = ((num_buckets + 1) * r_per_block) * sizeof(int);
    };

    query_maxbytes();
    optimize_launch();

    cuda_timer_t t;
    t.timer_start(stream);
    kernel::p2013Histogram<<<grid_dim, block_dim, shmem_use, stream>>>  //
        (in_data, out_freq, in_len, num_buckets, r_per_block);
    t.timer_end(stream);
    cudaStreamSynchronize(stream);

    milliseconds = t.get_time_elapsed();

    uint32_t* h_freq;
    h_freq = (uint32_t*)malloc(sizeof(uint32_t)*num_buckets);
    cudaMemcpy(h_freq, out_freq, sizeof(uint32_t)*num_buckets, cudaMemcpyDeviceToHost);

    // char entropy_file[100];
    // sprintf(entropy_file, "hist.data");

    // FILE *q_file = fopen(entropy_file,"wb");
    // fwrite((void *)h_freq, sizeof(uint32_t), num_buckets, q_file);
    // fclose(q_file);

    int total = 0;
    
    uint32_t *old_hfreq;
    old_hfreq = (uint32_t*)malloc(sizeof(uint32_t)*num_buckets);
    for (size_t i = 0; i < num_buckets; i++)
    {
        total+=h_freq[i];
        old_hfreq[i]=h_freq[i];
    }
    

    float entropy = 0.0;

    for (size_t i = 0; i < num_buckets; i++)
    {
        double p = (double)h_freq[i]/(double)total;

        if (p == 0)
        {
            continue;
        }
        

        entropy+= p*log2(p);
    }
    entropy = entropy*-1;

    printf("Entropy calculated: %f\n", entropy);
    uint32_t *large_hist_arr = (uint32_t *)malloc(sizeof(uint32_t)*num_buckets*NUM_TREES);
    float *large_logq_arr = (float *)malloc(sizeof(float)*num_buckets*NUM_TREES);

    /* BEGIN PREPROCESSING */

    /** Cauchy entropies  **/

    float entropy_vals[100] = {0.32,0.345,0.372,0.401,0.432,0.465,0.5,0.537,0.577,0.62,0.664,0.712,0.762,0.815,0.871,0.929,0.991,1.055,1.122,1.191,1.264,1.339,1.416,1.495,1.577,1.66,1.745,1.831,1.918,2.005,2.094,2.182,2.271,2.359,2.446,2.533,2.619,2.703,2.786,2.868,2.949,3.028,3.106,3.182,3.257,3.33,3.403,3.474,3.545,3.615,3.683,3.752,3.819,3.887,3.954,4.02,4.087,4.153,4.219,4.285,4.35,4.416,4.482,4.547,4.612,4.678,4.743,4.808,4.873,4.938,5.003,5.068,5.132,5.197,5.262,5.326,5.39,5.455,5.519,5.583,5.647,5.71,5.774,5.838,5.901,5.964,6.027,6.09,6.153,6.216,6.278,6.34,6.402,6.464,6.526,6.588,6.649,6.71,6.771,6.832};

    /** Laplace entropies **/
   // float entropy_vals[NUM_TREES] = {0.002,0.003,0.005,0.006,0.009,0.012,0.016,0.021,0.027,0.036,0.046,0.058,0.072,0.089,0.109,0.132,0.158,0.188,0.221,0.258,0.299,0.343,0.391,0.442,0.497,0.555,0.616,0.68,0.746,0.815,0.885,0.958,1.032,1.107,1.183,1.261,1.338,1.416,1.495,1.573,1.652,1.73,1.808,1.886,1.963,2.04,2.117,2.193,2.269,2.344,2.419,2.493,2.567,2.64,2.713,2.786,2.858,2.93,3.001,3.072,3.143,3.214,3.284,3.354,3.423,3.493,3.562,3.631,3.7,3.769,3.838,3.907,3.975,4.043,4.111,4.18,4.248,4.315,4.383,4.451,4.519,4.587,4.654,4.722,4.789,4.857,4.924,4.992,5.059,5.126,5.194,5.261,5.328,5.396,5.463,5.53,5.597,5.665,5.732,5.799};

    /** Gaussian entropies **/
    float entropy+vals[100] = {1.016,1.075,1.131,1.184,1.236,1.286,1.335,1.382,1.428,1.474,1.518,1.563,1.607,1.651,1.695,1.738,1.782,1.826,1.869,1.913,1.957,2.0,2.044,2.088,2.131,2.175,2.219,2.262,2.306,2.35,2.393,2.437,2.481,2.524,2.568,2.611,2.655,2.699,2.742,2.786,2.83,2.873,2.917,2.961,3.004,3.048,3.092,3.135,3.179,3.223,3.266,3.31,3.354,3.397,3.441,3.484,3.528,3.572,3.615,3.659,3.703,3.746,3.79,3.834,3.877,3.921,3.965,4.008,4.052,4.096,4.139,4.183,4.227,4.27,4.314,4.357,4.401,4.445,4.488,4.532,4.576,4.619,4.663,4.707,4.75,4.794,4.838,4.881,4.925,4.969,5.012,5.056,5.099,5.143,5.187,5.23,5.274,5.318,5.361,5.405};

    for (size_t i = 0; i < NUM_TREES; i++) // Read in every histogram and store to large hist array
    {
        char entropy_file[100];
        sprintf(entropy_file, "cauchy_hists/hist_entropy_%0.3f.data", entropy_vals[(100/NUM_TREES)*i]);

        FILE *q_file = fopen(entropy_file,"rb");
        fread((void *)&large_hist_arr[i*num_buckets], sizeof(uint32_t), num_buckets, q_file);
        fclose(q_file);


    }

    FILE *log_qfile = fopen("cauchy_logq.bin", "rb");
    fread((void*)&large_logq_arr, sizeof(float), num_buckets*NUM_TREES, log_qfile);
    fclose(log_qfile);
        
    float *d_logq_arr;
    float *d_cross_entropy, *h_cross_entropy;

    h_cross_entropy = (float *)malloc(sizeof(float)*NUM_TREES);

    cudaMalloc(&d_cross_entropy, sizeof(float)*NUM_TREES);
    cudaMalloc(&d_logq_arr, sizeof(float)*num_buckets*NUM_TREES);
    cudaMemcpy(d_logq_arr, large_logq_arr, sizeof(float)*num_buckets*NUM_TREES, cudaMemcpyHostToDevice);
    /* END PREPROCESSING */

    cuda_timer_t t1;
    t1.timer_start(stream);
    kernel::getCrossEntropy<<<NUM_TREES, 512, shmem_use, stream>>>(out_freq, d_logq_arr, num_buckets, total, d_cross_entropy);
    t1.timer_end(stream);
    cudaStreamSynchronize(stream);

    float ms_CE = t1.get_time_elapsed();
    printf("CE execute time: %f\n", ms_CE);

    cudaMemcpy(h_cross_entropy, d_cross_entropy, sizeof(float)*NUM_TREES, cudaMemcpyDeviceToHost);
    float min = h_cross_entropy[0];
    int min_ce_idx = 0;
    for (size_t i = 1; i < NUM_TREES; i++)
    {
//	printf("ent: %f\n", h_cross_entropy[i]);
        if (h_cross_entropy[i] < min)
        {
            min = h_cross_entropy[i];
            min_ce_idx = i;
	}
    }


    printf("Minimum Cross Entropy: %f\n", min);
    
    cudaMemcpy(out_freq, &large_hist_arr[min_ce_idx*num_buckets], sizeof(uint32_t)*num_buckets, cudaMemcpyHostToDevice);
    
    cudaFree(d_cross_entropy);
    cudaFree(d_logq_arr);
    free(h_cross_entropy);
    free(large_logq_arr);
    free(large_hist_arr);
    free(h_freq);
    
}

#endif
