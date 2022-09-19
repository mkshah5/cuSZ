#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <chrono>
//#include <bit>
#include <cuda.h>
#include <cuda_runtime.h>
#include "nvcomp/lz4.hpp"
#include "nvcomp.hpp"
#include "nvcomp/nvcompManagerFactory.hpp"
#include <sys/stat.h>


#include <thrust/copy.h>
#include <thrust/execution_policy.h>
#include <thrust/scan.h>
#include "cuCompactor.cuh"
#include <cub/cub.cuh>

#define TIMING

using namespace nvcomp;

/**
 * @brief TYPEDEFs required
 * 
 */
typedef union ldouble
{
    double value;
    unsigned long lvalue;
    unsigned char byte[8];
} ldouble;

typedef union lfloat
{
    float value;
    unsigned int ivalue;
    unsigned char byte[4];
} lfloat;


/**
 * @brief Globals
 * 
 */

void CUDA_CHECK_ERR(cudaError_t err){
    if (err != cudaSuccess)
    {
        printf("Error! Code: %d\n", err);
        exit(0);
    }
    
}
int littleEndian = 0;
__device__ unsigned long long int d_sigValues = 0;

struct is_nonzero
{
    __host__ __device__
    bool operator()(const float x){
        return x != 0.0;
    }
};

struct bitmap_nonzero
{
    __host__ __device__
    bool operator()(const uint8_t x){
        return x != (uint8_t)0;
    }
};
/**
 * @brief CUDA kernels
 * 
 */

__global__ void weak_threshold(float *data, float threshold, unsigned long dataLength){
    
    for (unsigned long tid = threadIdx.x+blockDim.x*blockIdx.x; tid < dataLength; tid+=blockDim.x*gridDim.x)
    {

        if (fabs(data[tid]) <= threshold)
        {
            data[tid] = 0.0;
        } else{
            atomicAdd(&d_sigValues, 1);
        }
        
    }
    
}

__global__ void apply_threshold(float *data, float threshold, unsigned long dataLength, char *bitmap){
    
    for (unsigned long tid = threadIdx.x+blockDim.x*blockIdx.x; tid < dataLength; tid+=blockDim.x*gridDim.x)
    {
        if (fabs(data[tid]) <= threshold)
        {
            data[tid] = 0.0;
            bitmap[tid] = '0';
        } else{
            atomicAdd(&d_sigValues, 1);
            bitmap[tid] = '1';
        }   
    }
}

// __global__ void gather_bitmap(char *in_bitmap, uint32_t *out_bitmap, unsigned long dataLength){

//     for(unsigned long tid = threadIdx.x+blockDim.x*blockIdx.x; tid < (dataLength/32)+1; tid+=blockDim.x*gridDim.x){

//         for (size_t i = 0; i < 32; i++)
//         {
//             if (tid)
//             {
//                 /* code */
//             }
            
//         }
        

//     }
// }
#define NUM_THREADS 64

// One thread handles 8 groups, each group has 8 data points

__global__ void compress_bitmap(char *bitmap, uint8_t *out_bitmap, uint64_t length){
    
    uint64_t chunk_width = (NUM_THREADS*64);
    __shared__ char chunk[NUM_THREADS][((8*8)+1)]; //Pad shared memory to avoid bank conflicts. 8 chars = one bit in out_bitmap, 8 bits/byte => each thread handles 8 byte
    
    unsigned long chunk_start = blockIdx.x*(chunk_width);
    unsigned long chunk_end = (blockIdx.x+1)*(chunk_width);

    unsigned long bitmap_start = blockIdx.x*(NUM_THREADS);
    if (chunk_end > length)
    {
        chunk_end = length;
    }

    uint64_t num_subchunks = (chunk_end-chunk_start)/8;    

    for (size_t i = 0; i < NUM_THREADS; i++)
    {
        for (size_t j = threadIdx.x; j < 64; j+=blockDim.x)
        {
            chunk[i][j] = bitmap[chunk_start + (i*64 +j)];
        }
    }
    __syncthreads();

    for (size_t group = 0; group < 8; group++)
    {
        int ones = 0;
        for (size_t bit = 0; bit < 8; bit++)
        {
            ones+=(int)('1'==chunk[threadIdx.x][(group*8 + bit)]);
        }
        
        int result = (int)(ones > 0);

        out_bitmap[threadIdx.x + bitmap_start] |= result << group;
    }
    
    
}

// 1 thread -> load 4 bytes -> 4 data points -> 1 byte for 8 data points -> two loads per row per thread

__global__ void compress_bitmap_uint32(char *bitmap, uint8_t *out_bitmap, uint64_t length){
    
    uint32_t *bitmap_32 = (uint32_t *) bitmap;
    uint32_t *out_bitmap_32 = (uint32_t *) out_bitmap;
    uint64_t chunk_width = (NUM_THREADS*64*sizeof(uint32_t));
    uint64_t chunk_width_32 = (NUM_THREADS*64);
    __shared__ uint32_t chunk_32[NUM_THREADS][(8*8)+1];
    // __shared__ char chunk[NUM_THREADS][((8*8*4)+1)]; //Pad shared memory to avoid bank conflicts. 8 chars = one bit in out_bitmap, 8 bits/byte => each thread handles 8 byte
    
    unsigned long chunk_start = blockIdx.x*(chunk_width);
    unsigned long chunk_end = (blockIdx.x+1)*(chunk_width);
    unsigned long chunk_start_32 = blockIdx.x*(chunk_width_32);
    unsigned long chunk_end_32 = blockIdx.x*(chunk_width_32);

    unsigned long bitmap_start_32 = blockIdx.x*(NUM_THREADS*4);
    if (chunk_end > length)
    {
        chunk_end = length;
    }
    if (chunk_end_32 > (length/4)+1)
    {
        chunk_end_32 = (length/4)+1;
    }
    

    // uint64_t num_subchunks = (chunk_end-chunk_start)/8;    

    for (size_t i = 0; i < NUM_THREADS; i++)
    {
        for (size_t j = threadIdx.x; j < 64; j+=blockDim.x)
        {
            if (chunk_start_32 + (i*64+j) >= chunk_end_32)
            {
                break;
            }
            
            chunk_32[i][j] = bitmap_32[chunk_start_32 + (i*64 +j)]; //chunk 32 will contain 64 * 4 data points -> 4 groups to be calculated
        }
    }
    __syncthreads();

    uint8_t* chunk = (uint8_t *)(&chunk_32[threadIdx.x]); //pointer to first data point of thread's row
    uint8_t out_val[4];

    for(size_t out_idx = 0; out_idx < 4; out_idx++){
        for (size_t group = 0; group < 8; group++)
        {
            int ones = 0;
            for (size_t bit = 0; bit < 8; bit++)
            {
                ones+=(int)('1'==chunk[(out_idx*64+ group*8 + bit)]);
            }
            int result = (int)(ones > 0);
            out_val[out_idx] |= result << group;
        }
    }
    uint32_t final_result = out_val[0] + (out_val[1] << 8) + (out_val[2] <<16) + (out_val[3] << 24);

    out_bitmap_32[threadIdx.x + bitmap_start_32] = final_result;    
}

__global__ void bitprefix_gen_32(uint8_t *bitmap, int *pfix, uint64_t bitmapLength){
    uint32_t *bitmap_32 = (uint32_t *)bitmap;

    for (unsigned long tid = threadIdx.x+blockDim.x*blockIdx.x; tid < bitmapLength; tid+=blockDim.x*gridDim.x)
    {
        pfix[tid] = __popc(bitmap_32[tid]);

    }
}

__global__ void bitprefix_gen(uint8_t *bitmap, int *pfix, uint64_t bitmapLength){
    
    for (unsigned long tid = threadIdx.x+blockDim.x*blockIdx.x; tid < bitmapLength; tid+=blockDim.x*gridDim.x)
    {
        pfix[tid] = __popc(bitmap[tid]);

    }
}

__global__ void prefix_gen(uint32_t *bitmap, int *pfix, uint64_t bitmapLength){
    
    for (unsigned long tid = threadIdx.x+blockDim.x*blockIdx.x; tid < bitmapLength; tid+=blockDim.x*gridDim.x)
    {
        pfix[tid] = __popc(bitmap[tid]);
    }
}

__global__ void reorder_values(int *pfix, uint32_t *bitmap, uint64_t bitmapLength, uint64_t length, float *sig_values, float *reordered_data, uint64_t numSigValues){

    for (unsigned long tid = threadIdx.x+blockDim.x*blockIdx.x; tid < bitmapLength; tid+=blockDim.x*gridDim.x)
    {
        int pfix_ind = pfix[tid];

        uint32_t bitmap_val = bitmap[tid];


        for (int i = 0; i < 32; i++)
        {
            if (tid*32+i >= length | pfix_ind >=numSigValues)
            {
                break;
            }

            if ((bitmap_val >> i) & 1 == 1)
            {
                reordered_data[tid*32+i] = sig_values[pfix_ind];
                pfix_ind++;
            }else{
                reordered_data[tid*32+i] = 0.0;
            }
            
        }
        
    }

}

__global__ void reorder_bits_32(int *pfix, uint8_t *bitmap, uint64_t bitmapLength, uint64_t length, uint8_t *sig_values, uint8_t *reordered_data, uint64_t numSigValues){

    uint32_t *bitmap_32 = (uint32_t *)bitmap;

    for (unsigned long tid = threadIdx.x+blockDim.x*blockIdx.x; tid < bitmapLength; tid+=blockDim.x*gridDim.x)
    {
        int pfix_ind = pfix[tid];
        uint32_t bitmap_val = bitmap_32[tid];

        
        for (int i = 0; i < 32; i++)
        {
            if (tid*32+i >= length | pfix_ind >=numSigValues)
            {
                break;
            }

            if ((bitmap_val >> i) & 1 == 1)
            {
                reordered_data[tid*32+i] = sig_values[pfix_ind];
                pfix_ind++;
            }else{
                reordered_data[tid*32+i] = 0.0;
            }
            
        }
        
    }

}

__global__ void reorder_bits(int *pfix, uint8_t *bitmap, uint64_t bitmapLength, uint64_t length, uint8_t *sig_values, uint8_t *reordered_data, uint64_t numSigValues){

    for (unsigned long tid = threadIdx.x+blockDim.x*blockIdx.x; tid < bitmapLength; tid+=blockDim.x*gridDim.x)
    {
        int pfix_ind = pfix[tid];
        uint8_t bitmap_val = bitmap[tid];

        
        for (int i = 0; i < 8; i++)
        {
            if (tid*8+i >= length | pfix_ind >=numSigValues)
            {
                break;
            }

            if ((bitmap_val >> i) & 1 == 1)
            {
                reordered_data[tid*8+i] = sig_values[pfix_ind];
                pfix_ind++;
            }else{
                reordered_data[tid*8+i] = 0;
            }
            
        }
        
    }

}

void run_bitcompression_32(unsigned long dataLength, uint32_t *bitmap_final, char *d_bitmap, char *mapFilePath, char *valueFilePath){
    uint8_t *d_dat;
    uint8_t *d_map;
    uint64_t chunks = dataLength/(NUM_THREADS*64*4);
    dim3 gridDim(chunks,1,1);
    uint8_t *d_bitmap_transfer;
    FILE *bitmapFile, *bitmapMap;

    size_t map_size = (dataLength/64)+1;
    size_t max_value_size = sizeof(uint32_t)*((dataLength/32)+1);

    cudaError_t ret = cudaMalloc(&d_bitmap_transfer, max_value_size);
    cudaMemcpy(d_bitmap_transfer, bitmap_final, max_value_size, cudaMemcpyHostToDevice);

    cudaMalloc(&d_map, map_size);
    cudaMalloc(&d_dat, max_value_size);

    #ifdef TIMING
    float time_NVCOMP;
    cudaEvent_t start_2, stop_2;
    cudaEventCreate(&start_2);
    cudaEventCreate(&stop_2);
    #endif

    #ifdef TIMING
    cudaEventRecord(start_2, 0);
    #endif
    uint8_t *result_copy = thrust::copy_if(thrust::cuda::par, d_bitmap_transfer, d_bitmap_transfer + max_value_size, d_dat, is_nonzero());
    compress_bitmap_uint32<<<gridDim, NUM_THREADS>>>(d_bitmap,d_map,dataLength);
    #ifdef TIMING
    cudaEventRecord(stop_2, 0);
    cudaEventSynchronize(stop_2);
    cudaEventElapsedTime(&time_NVCOMP, start_2, stop_2);
    
    #endif
    cudaDeviceSynchronize();
    #ifdef TIMING
    printf("NVCOMP: %.3f ms\n", time_NVCOMP);
    #endif
    int num_out = result_copy-d_dat;
    printf("Number of nonzero bytes: %ld\n", num_out);
    uint8_t *resultant_map, *resultant_values;
    
    // cudaMemcpy(&num_out, &d_num_out, sizeof(int), cudaMemcpyDeviceToHost);
    resultant_map = (uint8_t *)malloc(map_size);
    resultant_values = (uint8_t *)malloc(num_out);
    cudaMemcpy(resultant_map, d_map, map_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(resultant_values, d_dat, num_out, cudaMemcpyDeviceToHost);

    bitmapFile = fopen(valueFilePath, "wb");
    fwrite(resultant_values, sizeof(uint8_t), num_out, bitmapFile);
    fclose(bitmapFile);
    free(resultant_values);

    bitmapMap = fopen(mapFilePath, "wb");
    fwrite(resultant_map, sizeof(uint8_t), (dataLength/64)+1, bitmapMap);
    fclose(bitmapMap);
    free(resultant_map);
}

void run_bitcompression(unsigned long dataLength, uint32_t *bitmap_final, char *d_bitmap, char *mapFilePath, char *valueFilePath){
    uint8_t *d_dat;
    uint8_t *d_map;
    uint64_t chunks = dataLength/(NUM_THREADS*64);
    dim3 gridDim(chunks,1,1);
    uint8_t *d_bitmap_transfer;
    FILE *bitmapFile, *bitmapMap;

    size_t map_size = (dataLength/64)+1;
    size_t max_value_size = sizeof(uint32_t)*((dataLength/32)+1);

    cudaError_t ret = cudaMalloc(&d_bitmap_transfer, max_value_size);
    cudaMemcpy(d_bitmap_transfer, bitmap_final, max_value_size, cudaMemcpyHostToDevice);

    cudaMalloc(&d_map, map_size);
    cudaMalloc(&d_dat, max_value_size);

    #ifdef TIMING
    float time_NVCOMP;
    cudaEvent_t start_2, stop_2;
    cudaEventCreate(&start_2);
    cudaEventCreate(&stop_2);
    #endif

    #ifdef TIMING
    cudaEventRecord(start_2, 0);
    #endif
    uint8_t *result_copy = thrust::copy_if(thrust::cuda::par, d_bitmap_transfer, d_bitmap_transfer + max_value_size, d_dat, is_nonzero());
    compress_bitmap<<<gridDim, NUM_THREADS>>>(d_bitmap,d_map,dataLength);
    #ifdef TIMING
    cudaEventRecord(stop_2, 0);
    cudaEventSynchronize(stop_2);
    cudaEventElapsedTime(&time_NVCOMP, start_2, stop_2);
    
    #endif
    cudaDeviceSynchronize();
    #ifdef TIMING
    printf("NVCOMP: %.3f ms\n", time_NVCOMP);
    #endif
    int num_out = result_copy-d_dat;
    printf("Number of nonzero bytes: %ld\n", num_out);
    uint8_t *resultant_map, *resultant_values;
    
    // cudaMemcpy(&num_out, &d_num_out, sizeof(int), cudaMemcpyDeviceToHost);
    resultant_map = (uint8_t *)malloc(map_size);
    resultant_values = (uint8_t *)malloc(num_out);
    cudaMemcpy(resultant_map, d_map, map_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(resultant_values, d_dat, num_out, cudaMemcpyDeviceToHost);

    bitmapFile = fopen(valueFilePath, "wb");
    fwrite(resultant_values, sizeof(uint8_t), num_out, bitmapFile);
    fclose(bitmapFile);
    free(resultant_values);

    bitmapMap = fopen(mapFilePath, "wb");
    fwrite(resultant_map, sizeof(uint8_t), (dataLength/64)+1, bitmapMap);
    fclose(bitmapMap);
    free(resultant_map);
}

void run_bitdecompress_32(unsigned long dataLength, char* inPath, uint32_t *d_bitmap){

    int *d_pfix, *pfix;
    uint8_t *value, *map;
    uint8_t *d_value, *d_map;
    size_t map_size = (dataLength/64)+1;
    size_t map_size_32 = (dataLength/(4*64))+1;
    size_t bitmapLength = sizeof(uint32_t)*((dataLength/32)+1);

    char valueFilePath[256];
    char mapFilePath[256];
    FILE *valueFile, *mapFile;
    sprintf(valueFilePath, "%s.bitmap", inPath);
    sprintf(mapFilePath, "%s.map", inPath);

    valueFile = fopen(valueFilePath, "rb");
    mapFile = fopen(mapFilePath, "rb");

    struct stat st;
    size_t value_size;
    stat(valueFilePath, &st);
    value_size = st.st_size;
    value = (uint8_t *)malloc(value_size);
    map = (uint8_t *)malloc(map_size);

    fread(value, sizeof(uint8_t), value_size, valueFile);
    fclose(valueFile);
    fread(map, sizeof(uint8_t), map_size, mapFile);
    fclose(mapFile);

    CUDA_CHECK_ERR(cudaMalloc(&d_value, sizeof(uint8_t)*value_size));
    CUDA_CHECK_ERR(cudaMemcpy(d_value, value, sizeof(uint8_t)*value_size, cudaMemcpyHostToDevice));
    CUDA_CHECK_ERR(cudaMalloc(&d_map, sizeof(uint8_t)*map_size));
    CUDA_CHECK_ERR(cudaMemcpy(d_map, map, sizeof(uint8_t)*map_size, cudaMemcpyHostToDevice));

    CUDA_CHECK_ERR(cudaMalloc(&d_pfix, sizeof(int)*map_size_32));
    CUDA_CHECK_ERR(cudaMalloc(&pfix, sizeof(int)*map_size_32));
    

    printf("Starting nvcomp\n");
    #ifdef TIMING
    float time_NVCOMP;
    cudaEvent_t start_2, stop_2;
    cudaEventCreate(&start_2);
    cudaEventCreate(&stop_2);
    #endif
    #ifdef TIMING
    cudaEventRecord(start_2, 0);
    #endif

    bitprefix_gen_32<<<80,256>>>(d_map, d_pfix, map_size_32);
    cudaDeviceSynchronize();
    CUDA_CHECK_ERR(cudaGetLastError());
    thrust::exclusive_scan(thrust::cuda::par, d_pfix, &d_pfix[map_size_32], d_pfix);
    cudaDeviceSynchronize();
    CUDA_CHECK_ERR(cudaGetLastError());
    reorder_bits_32<<<80,256>>>(d_pfix, d_map, map_size_32, bitmapLength, d_value, (uint8_t *)d_bitmap, value_size);
    cudaDeviceSynchronize();
    CUDA_CHECK_ERR(cudaGetLastError());
    #ifdef TIMING
    cudaEventRecord(stop_2, 0);
    cudaEventSynchronize(stop_2);
    cudaEventElapsedTime(&time_NVCOMP, start_2, stop_2);
    printf("NVCOMP: %.3f ms\n", time_NVCOMP);
    #endif

}

void run_bitdecompress(unsigned long dataLength, char* inPath, uint32_t *d_bitmap){

    int *d_pfix, *pfix;
    uint8_t *value, *map;
    uint8_t *d_value, *d_map;
    size_t map_size = (dataLength/64)+1;
    size_t bitmapLength = sizeof(uint32_t)*((dataLength/32)+1);

    char valueFilePath[256];
    char mapFilePath[256];
    FILE *valueFile, *mapFile;
    sprintf(valueFilePath, "%s.bitmap", inPath);
    sprintf(mapFilePath, "%s.map", inPath);

    valueFile = fopen(valueFilePath, "rb");
    mapFile = fopen(mapFilePath, "rb");

    struct stat st;
    size_t value_size;
    stat(valueFilePath, &st);
    value_size = st.st_size;
    value = (uint8_t *)malloc(value_size);
    map = (uint8_t *)malloc(map_size);

    fread(value, sizeof(uint8_t), value_size, valueFile);
    fclose(valueFile);
    fread(map, sizeof(uint8_t), map_size, mapFile);
    fclose(mapFile);

    CUDA_CHECK_ERR(cudaMalloc(&d_value, sizeof(uint8_t)*value_size));
    CUDA_CHECK_ERR(cudaMemcpy(d_value, value, sizeof(uint8_t)*value_size, cudaMemcpyHostToDevice));
    CUDA_CHECK_ERR(cudaMalloc(&d_map, sizeof(uint8_t)*map_size));
    CUDA_CHECK_ERR(cudaMemcpy(d_map, map, sizeof(uint8_t)*map_size, cudaMemcpyHostToDevice));

    CUDA_CHECK_ERR(cudaMalloc(&d_pfix, sizeof(int)*map_size));
    CUDA_CHECK_ERR(cudaMalloc(&pfix, sizeof(int)*map_size));
    

    printf("Starting nvcomp\n");
    #ifdef TIMING
    float time_NVCOMP;
    cudaEvent_t start_2, stop_2;
    cudaEventCreate(&start_2);
    cudaEventCreate(&stop_2);
    #endif
    #ifdef TIMING
    cudaEventRecord(start_2, 0);
    #endif

    bitprefix_gen<<<80,256>>>(d_map, d_pfix, map_size);
    cudaDeviceSynchronize();
    CUDA_CHECK_ERR(cudaGetLastError());
    thrust::exclusive_scan(thrust::cuda::par, d_pfix, &d_pfix[map_size], d_pfix);
    cudaDeviceSynchronize();
    CUDA_CHECK_ERR(cudaGetLastError());
    reorder_bits<<<80,256>>>(d_pfix, d_map, map_size, bitmapLength, d_value, (uint8_t *)d_bitmap, value_size);
    cudaDeviceSynchronize();
    CUDA_CHECK_ERR(cudaGetLastError());
    #ifdef TIMING
    cudaEventRecord(stop_2, 0);
    cudaEventSynchronize(stop_2);
    cudaEventElapsedTime(&time_NVCOMP, start_2, stop_2);
    printf("NVCOMP: %.3f ms\n", time_NVCOMP);
    #endif

}


// struct cub_nonzero
// {
//     // float compare;

//     // CUB_RUNTIME_FUNCTION __forceinline__
//     // cub_nonzero(float compare) : compare(compare) {}

//     CUB_RUNTIME_FUNCTION __forceinline__
//     bool operator()(const float &a) const {
//         return (a != 0.0);
//     }
// };

/**
 * @brief Helper functions for host
 * 
 */

void checkEndian(){
    int x = 1;
    char *y = (char*)&x;
    if (*y+48 =='1')
    {
        littleEndian = 1;
    }
    
}



void writeByteData(unsigned char *bytes, size_t byteLength, char *tgtFilePath)
{
	FILE *pFile = fopen(tgtFilePath, "wb");
    if (pFile == NULL)
    {
        printf("Failed to open input file. 3\n");
        return;
    }

    fwrite(bytes, 1, byteLength, pFile); //write outSize bytes
    fclose(pFile);

}

void writeFloatData_inBytes(float *data, size_t nbEle, char* tgtFilePath)
{
	size_t i = 0;

	lfloat buf;
	unsigned char* bytes = (unsigned char*)malloc(nbEle*sizeof(float));
	for(i=0;i<nbEle;i++)
	{
		buf.value = data[i];
		bytes[i*4+0] = buf.byte[0];
		bytes[i*4+1] = buf.byte[1];
		bytes[i*4+2] = buf.byte[2];
		bytes[i*4+3] = buf.byte[3];
	}

	size_t byteLength = nbEle*sizeof(float);
	writeByteData(bytes, byteLength, tgtFilePath);
	free(bytes);

}

void writeDoubleData_inBytes(double *data, size_t nbEle, char* tgtFilePath)
{
	size_t i = 0, index = 0;

	ldouble buf;
	unsigned char* bytes = (unsigned char*)malloc(nbEle*sizeof(double));
	for(i=0;i<nbEle;i++)
	{
		index = i*8;
		buf.value = data[i];
		bytes[index+0] = buf.byte[0];
		bytes[index+1] = buf.byte[1];
		bytes[index+2] = buf.byte[2];
		bytes[index+3] = buf.byte[3];
		bytes[index+4] = buf.byte[4];
		bytes[index+5] = buf.byte[5];
		bytes[index+6] = buf.byte[6];
		bytes[index+7] = buf.byte[7];
	}

	size_t byteLength = nbEle*sizeof(double);
	writeByteData(bytes, byteLength, tgtFilePath);
	free(bytes);
}

void symTransform_8bytes(unsigned char data[8])
{
	unsigned char tmp = data[0];
	data[0] = data[7];
	data[7] = tmp;

	tmp = data[1];
	data[1] = data[6];
	data[6] = tmp;

	tmp = data[2];
	data[2] = data[5];
	data[5] = tmp;

	tmp = data[3];
	data[3] = data[4];
	data[4] = tmp;
}

unsigned char *readByteData(char *srcFilePath, size_t *byteLength)
{
	FILE *pFile = fopen(srcFilePath, "rb");
    if (pFile == NULL)
    {
        printf("Failed to open input file. 1\n");
        
        return 0;
    }
	fseek(pFile, 0, SEEK_END);
    *byteLength = ftell(pFile);
    fclose(pFile);

    unsigned char *byteBuf = ( unsigned char *)malloc((*byteLength)*sizeof(unsigned char)); //sizeof(char)==1

    pFile = fopen(srcFilePath, "rb");
    if (pFile == NULL)
    {
        printf("Failed to open input file. 2\n");
        
        return 0;
    }
    fread(byteBuf, 1, *byteLength, pFile);
    fclose(pFile);
    
    return byteBuf;
}

double *readDoubleData_systemEndian(char *srcFilePath, size_t *nbEle)
{
	size_t inSize;
	FILE *pFile = fopen(srcFilePath, "rb");
    if (pFile == NULL)
    {
        printf("Failed to open input file. 1\n");
        
        return NULL;
    }
	fseek(pFile, 0, SEEK_END);
    inSize = ftell(pFile);
    *nbEle = inSize/8; //only support double in this version
    fclose(pFile);

    double *daBuf = (double *)malloc(inSize);

    pFile = fopen(srcFilePath, "rb");
    if (pFile == NULL)
    {
        printf("Failed to open input file. 2\n");
        
        return NULL;
    }
    fread(daBuf, 8, *nbEle, pFile);
    fclose(pFile);
    
    return daBuf;
}

double *readDoubleData(char *srcFilePath, size_t *nbEle)
{
	int state = 0;
	if(littleEndian)
	{
		double *daBuf = readDoubleData_systemEndian(srcFilePath, nbEle);
		
		return daBuf;
	}
	else
	{
		size_t i,j;

		size_t byteLength;
		unsigned char* bytes = readByteData(srcFilePath, &byteLength);
		
		double *daBuf = (double *)malloc(byteLength);
		*nbEle = byteLength/8;

		ldouble buf;
		for(i = 0;i<*nbEle;i++)
		{
			j = i*8;
			memcpy(buf.byte, bytes+j, 8);
			symTransform_8bytes(buf.byte);
			daBuf[i] = buf.value;
		}
		free(bytes);
		return daBuf;
	}
}


int main(int argc, char* argv[]){
    char* inPath = NULL;
    char* inPathDecomp = NULL;
    char outputFilePath[256];
    unsigned char *bytes = NULL;

    int preCompression = 0;
    int doGroup = 0;
    int castToFloat = 0;
    int useNVCOMP = 0;
    int doAlternateCompaction = 0;
    int doR2R = 0;

    float threshold = 0.0;
    unsigned long numSigValues = 0;
    unsigned long dataLength = 0;
    float r2r_value = 0.0;

    size_t nbEle;

    // cub_nonzero select_op();

    checkEndian();

    for(int i=0;i<argc;i++){
        switch (argv[i][1])
        {
        case 'z':
            preCompression = 1;
            break;
        case 'd':
            preCompression = 0;
            i++;
			inPathDecomp = argv[i];		
            break;
        case 'T':
            i++;
            threshold = atof(argv[i]);
            break;
        case 'i':
			i++;
			inPath = argv[i];		
			break;
        case 'g':
            doGroup = 1;
            i++;
            numSigValues = atoi(argv[i]);
            break;
        case 'L':
            i++;
            dataLength = atoi(argv[i]);
            break;
        case 'F':
            castToFloat = 1;
            break;
        case 'N':
            useNVCOMP = 1;
            break;
        case 'S':
            doAlternateCompaction = 1;
            break;
        case 'R':
            doR2R = 1;
            i++;
            r2r_value = atof(argv[i]);
            break;
        default:
            break;
        }
    }

    


    if (preCompression)
    {
        unsigned long dataToCopy = dataLength;
        char *h_bitmap, *d_bitmap;

        uint32_t *bitmap_final, *d_bitmap_final;

        double *data = readDoubleData(inPath, &nbEle);
        float *d_data;
        float *out_data;
        cudaMalloc(&d_data, sizeof(float)*dataLength);
        
        float *floatTmpData = (float *)malloc(dataLength*sizeof(float));

        float max_value = (float) data[0];
        float min_value = (float) data[0];

        for (size_t i = 0; i < dataLength; i++)
        {
            floatTmpData[i] = (float) data[i];
            
            if (floatTmpData[i] > max_value)
            {
                max_value = floatTmpData[i];
            }
            if (floatTmpData[i] < min_value)
            {
                min_value = floatTmpData[i];
            }
            
        }

        if (doR2R)
        {
            threshold = r2r_value * (max_value - min_value);
        }
        
        // writeFloatData_inBytes(floatTmpData, dataToCopy, outputFilePath);
        cudaMemcpy(d_data, floatTmpData, sizeof(float)*dataLength, cudaMemcpyHostToDevice);
        free(floatTmpData);

        free(data);

        

        if (doGroup)
        {
            h_bitmap = (char*)malloc(sizeof(char)*dataLength);
            cudaMalloc(&d_bitmap, sizeof(char)*dataLength);
            bitmap_final = (uint32_t*)malloc(sizeof(uint32_t)*((dataLength/32)+1));
        }
        
        unsigned long long int c = 0;

        

        // IMPLEMENTING THRESHOLD+GROUP OPTIMIZATION
        // // STEP 1: Allocate space for bitmap
        // // STEP 2: Launch kernel that applies threshold and sets bitmap values, returns number of significant values
        // // STEP 3: Alloc space for significant values and launch kernel to scan bitmap and transfer sig values
        // // STEP 4: Free old data and feed significant values to compress

        if (!doGroup)
        {
            #ifdef TIMING
            float time_weak;
            cudaEvent_t start_weak, stop_weak;
            cudaEventCreate(&start_weak);
            cudaEventCreate(&stop_weak);
            #endif

            #ifdef TIMING
            cudaEventRecord(start_weak, 0);
            #endif

            weak_threshold<<<80,256>>>(d_data, threshold, dataLength);
            #ifdef TIMING
            cudaEventRecord(stop_weak, 0);
            #endif
            cudaDeviceSynchronize();
            dataToCopy = dataLength;

            #ifdef TIMING
            // cudaEventRecord(stop_weak, 0);
            cudaEventSynchronize(stop_weak);
            cudaEventElapsedTime(&time_weak, start_weak, stop_weak);
            printf("Time to execute: %.3f ms\n", time_weak);
            #endif

            out_data = (float *)malloc(sizeof(float)*dataToCopy);

            cudaMemcpy(out_data, d_data, sizeof(float)*dataToCopy, cudaMemcpyDeviceToHost);

        }else{
            #ifdef TIMING
            float total;
            float time_thresh;
            cudaEvent_t start_thresh, stop_thresh;
            cudaEventCreate(&start_thresh);
            cudaEventCreate(&stop_thresh);
            #endif

            #ifdef TIMING
            cudaEventRecord(start_thresh, 0);
            #endif
            apply_threshold<<<80,256>>>(d_data, threshold, dataLength, d_bitmap);
            #ifdef TIMING
            cudaEventRecord(stop_thresh, 0);
            #endif
            // weak_threshold<<<80,256>>>(d_data, threshold, dataLength);
            cudaDeviceSynchronize();
            #ifdef TIMING
            // cudaEventRecord(stop_weak, 0);
            cudaEventSynchronize(stop_thresh);
            cudaEventElapsedTime(&time_thresh, start_thresh, stop_thresh);
            #endif
            
            // cudaMemcpy(h_bitmap, d_bitmap, sizeof(char)*dataLength, cudaMemcpyDeviceToHost);
            cudaMemcpyFromSymbol(&c, d_sigValues, sizeof(unsigned long long int));

            // cudaMemcpy(data, d_data, sizeof(double)*dataToCopy, cudaMemcpyDeviceToHost);
            float *d_finaldata;

            cudaMalloc(&d_finaldata, sizeof(double)*c);
            // cudaMalloc(&d_finaldata, sizeof(float)*dataLength);

            #ifdef TIMING
            float time_compact;
            cudaEvent_t start_compact, stop_compact;
            cudaEventCreate(&start_compact);
            cudaEventCreate(&stop_compact);
            #endif
            if (doAlternateCompaction)
            {
                #ifdef TIMING
                cudaEventRecord(start_compact, 0);
                #endif
                cuCompactor::compact<float>(d_data, d_finaldata, dataLength, is_nonzero(), 256);
                #ifdef TIMING
                cudaEventRecord(stop_compact, 0);
                #endif
            }else{
                // thrust::copy_if(thrust::cuda::par, d_data, d_data + dataLength, d_finaldata, is_nonzero());
                void *d_temp_storage = NULL;
                size_t temp_storage_bytes=  0;
                int *d_num_selected_out;
                cudaMalloc(&d_num_selected_out, sizeof(int));

                #ifdef TIMING
                cudaEventRecord(start_compact, 0);
                #endif
                cub::DeviceSelect::If(d_temp_storage, temp_storage_bytes, d_data, d_finaldata, d_num_selected_out, dataLength, is_nonzero());
                cudaMalloc(&d_temp_storage, temp_storage_bytes);

                cub::DeviceSelect::If(d_temp_storage, temp_storage_bytes, d_data, d_finaldata, d_num_selected_out, dataLength, is_nonzero());
                #ifdef TIMING
                cudaEventRecord(stop_compact, 0);
                #endif
            }
            

            

            #ifdef TIMING
            // cudaEventRecord(stop_compact, 0);
            cudaEventSynchronize(stop_compact);
            cudaEventElapsedTime(&time_compact, start_compact, stop_compact);
            total = time_compact + time_thresh;
            printf("Time to execute: %.3f ms\n", total);
            #endif

            cudaMemcpy(h_bitmap, d_bitmap, sizeof(char)*dataLength, cudaMemcpyDeviceToHost);

            float *tmpData = (float *)malloc(c*sizeof(float));

            cudaMemcpy(tmpData, d_finaldata, sizeof(float)*c,cudaMemcpyDeviceToHost);
            cudaFree(d_finaldata);
            printf("tmpData from thrust: %f\n", tmpData[0]);

            int sig_ind = 0;
            for (size_t i = 0; i < dataLength; i++)
            {
                if (h_bitmap[i]=='1')
                {
                    // tmpData[sig_ind] = data[i];
                    bitmap_final[i/32] = bitmap_final[i/32] | (1 << (i%32));

                    sig_ind++;
                }
                
            }

            char bitmapFilePath[256];
            char bitmapMapPath[256];
            FILE *bitmapFile, *bitmapMap;
            sprintf(bitmapFilePath, "%s.bitmap", inPath);
            sprintf(bitmapMapPath, "%s.map", inPath);
            cudaFree(d_data);	
            if (useNVCOMP)
            {
                run_bitcompression_32(dataLength, bitmap_final, d_bitmap, bitmapMapPath, bitmapFilePath);
            } else{
                bitmapFile = fopen(bitmapFilePath, "wb");
                fwrite(bitmap_final, sizeof(uint32_t), ((dataLength/32)+1), bitmapFile);
                fclose(bitmapFile);
            }
            

            // #ifdef TIMING
            // cudaEventRecord(stop, 0);
            // cudaEventSynchronize(stop);
            // cudaEventElapsedTime(&time, start, stop);
            // printf("Time to execute: %.3f ms\n", time);
            // #endif


            // free(data);
            out_data = tmpData;
            dataToCopy = c;
        }
        
        

        

        sprintf(outputFilePath, "%s.threshold", inPath);	

        // if (castToFloat)
        // {
        //     float *floatTmpData = (float *)malloc(dataToCopy*sizeof(float));
        //     for (size_t i = 0; i < dataToCopy; i++)
        //     {
        //         floatTmpData[i] = (float) data[i];
        //     }
        //     writeFloatData_inBytes(floatTmpData, dataToCopy, outputFilePath);
        //     free(floatTmpData);

        // }else{
        //     writeDoubleData_inBytes(data, dataToCopy, outputFilePath);
           
        // }
        writeFloatData_inBytes(out_data, dataToCopy, outputFilePath);
        free(out_data);
        printf("Number of significant values: %d\n", c);
        
        
    } else {

        uint64_t bitmapLength = ((dataLength/32)+1);
        uint32_t *bitmap = (uint32_t *)malloc(sizeof(uint32_t)*bitmapLength);
        uint8_t *d_comp;
        uint32_t *d_bitmap;
        
        

        int *pfix;
        cudaMalloc(&pfix, sizeof(int)*bitmapLength);
        // int *pfix = (int *)malloc(sizeof(int)*bitmapLength);
        
        double *final_data = (double *)malloc(sizeof(double)*dataLength);
        float *final_data_f =(float *)malloc(sizeof(float)*dataLength);
        float *d_final_data;
        cudaMalloc(&d_final_data, sizeof(float)*dataLength);

        CUDA_CHECK_ERR(cudaMalloc(&d_bitmap, sizeof(uint32_t)*bitmapLength));

        if (!useNVCOMP){
            char bitmapFilePath[256];
            FILE *bitmapFile;
            sprintf(bitmapFilePath, "%s.bitmap", inPath);

            bitmapFile = fopen(bitmapFilePath, "rb");
            fread(bitmap, sizeof(uint32_t), ((dataLength/32)+1), bitmapFile);
            // cudaMalloc(&d_bitmap, sizeof(uint32_t)*bitmapLength);
            fclose(bitmapFile);
        }

        float *sig_values_f = (float *)malloc(sizeof(float)*numSigValues);
        // double *sig_values = (double *)malloc(sizeof(double)*numSigValues);
        float *d_sig_values;
        cudaMalloc(&d_sig_values, sizeof(float)*numSigValues);
        
        // Need to memcpy and do a float->double cast

        FILE* sigFile;
        sigFile = fopen(inPathDecomp, "rb");

        fread(sig_values_f, sizeof(float), numSigValues, sigFile);

        fclose(sigFile);

        // for (size_t i = 0; i < numSigValues; i++)
        // {
        //     sig_values[i] = (double)sig_values_f[i];
        // }

        // free(sig_values_f);

        cudaMemcpy(d_sig_values, sig_values_f, sizeof(float)*numSigValues, cudaMemcpyHostToDevice);
        if(useNVCOMP){
            // cudaMalloc(&d_decomp, sizeof(uint8_t)*((dataLength/8)+1));
            run_bitdecompress_32(dataLength, inPath, d_bitmap);
        }else{
            cudaMemcpy(d_bitmap, bitmap, sizeof(uint32_t)*bitmapLength, cudaMemcpyHostToDevice);
        }
        #ifdef TIMING
        float time_pre, time_scan, time_reorder, total_2;
        cudaEvent_t start_pre, stop_pre, start_scan, stop_scan, start_reorder, stop_reorder;
        cudaEventCreate(&start_pre);
        cudaEventCreate(&stop_pre);
        cudaEventCreate(&start_scan);
        cudaEventCreate(&stop_scan);
        cudaEventCreate(&start_reorder);
        cudaEventCreate(&stop_reorder);
        #endif
        #ifdef TIMING
        cudaEventRecord(start_pre, 0);
        #endif

        prefix_gen<<<80,256>>>(d_bitmap, pfix, bitmapLength);
        #ifdef TIMING
        cudaEventRecord(stop_pre, 0);
        #endif
        cudaDeviceSynchronize();
        #ifdef TIMING
        cudaEventSynchronize(stop_pre);
        cudaEventElapsedTime(&time_pre, start_pre, stop_pre);
        #endif
        CUDA_CHECK_ERR(cudaGetLastError());

        #ifdef TIMING
        cudaEventRecord(start_scan, 0);
        #endif
        thrust::exclusive_scan(thrust::cuda::par, pfix, pfix+bitmapLength, pfix);
        CUDA_CHECK_ERR(cudaGetLastError());
        #ifdef TIMING
        cudaEventRecord(stop_scan, 0);
        #endif
        #ifdef TIMING
        cudaEventSynchronize(stop_scan);
        cudaEventElapsedTime(&time_scan, start_scan, stop_scan);
        #endif
        #ifdef TIMING
        cudaEventRecord(start_reorder, 0);
        #endif
        reorder_values<<<80,256>>>(pfix, d_bitmap, bitmapLength, dataLength, d_sig_values, d_final_data, numSigValues);
        #ifdef TIMING
        cudaEventRecord(stop_reorder, 0);
        #endif
        cudaDeviceSynchronize();

        #ifdef TIMING
        cudaEventSynchronize(stop_reorder);
        cudaEventElapsedTime(&time_reorder, start_reorder, stop_reorder);
        #endif

        #ifdef TIMING
        total_2 = time_pre+time_scan+time_reorder;
        printf("Time to execute: %.3f ms\n", total_2);
        #endif

        cudaMemcpy(final_data_f, d_final_data, sizeof(float)*dataLength, cudaMemcpyDeviceToHost);

        for (size_t i = 0; i < dataLength; i++)
        {
            final_data[i] = (double) final_data_f[i];
        }
        

        cudaFree(d_final_data);
        cudaFree(d_sig_values);
        cudaFree(d_bitmap);
        cudaFree(pfix);
        free(bitmap);
        free(sig_values_f);

        char finalOutPath[256];
        FILE *finalFile;
        sprintf(finalOutPath, "%s.out", inPath);

        finalFile = fopen(finalOutPath, "wb");

        fwrite(final_data, sizeof(double), dataLength, finalFile);

        fclose(finalFile);

        printf("final data [0]: %f\n", final_data[0]);
        free(final_data);
        free(final_data_f);
    }
    

}