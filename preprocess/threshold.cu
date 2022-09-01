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
int littleEndian = 0;
__device__ unsigned long long int d_sigValues = 0;

/**
 * @brief CUDA kernels
 * 
 */

__global__ void weak_threshold(double *data, float threshold, unsigned long dataLength){
    
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

__global__ void apply_threshold(double *data, float threshold, unsigned long dataLength, char *bitmap){
    
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

__global__ void prefix_gen(uint32_t *bitmap, int *pfix, uint64_t bitmapLength){
    
    for (unsigned long tid = threadIdx.x+blockDim.x*blockIdx.x; tid < bitmapLength; tid+=blockDim.x*gridDim.x)
    {
        pfix[tid] = __popc(bitmap[tid]);
    }
}

__global__ void reorder_values(int *pfix, uint32_t *bitmap, uint64_t bitmapLength, uint64_t length, double *sig_values, double *reordered_data, uint64_t numSigValues){

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


struct is_nonzero
{
    __host__ __device__
    bool operator()(const double x){
        return x != 0.0;
    }
};

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

    float threshold = 0.0;
    unsigned long numSigValues = 0;
    unsigned long dataLength = 0;

    size_t nbEle;

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
        default:
            break;
        }
    }

    #ifdef TIMING
    float time;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    #endif


    if (preCompression)
    {
        int dataToCopy = dataLength;
        char *h_bitmap, *d_bitmap;

        uint32_t *bitmap_final, *d_bitmap_final;

        double *data = readDoubleData(inPath, &nbEle);
        double *d_data;
        cudaMalloc(&d_data, sizeof(double)*dataLength);
        cudaMemcpy(d_data, data, sizeof(double)*dataLength, cudaMemcpyHostToDevice);

        if (doGroup)
        {
            h_bitmap = (char*)malloc(sizeof(char)*dataLength);
            cudaMalloc(&d_bitmap, sizeof(char)*dataLength);
            bitmap_final = (uint32_t*)malloc(sizeof(uint32_t)*((dataLength/32)+1));
        }
        
        unsigned long long int c = 0;

        #ifdef TIMING
        cudaEventRecord(start, 0);
        #endif

        // IMPLEMENTING THRESHOLD+GROUP OPTIMIZATION
        // // STEP 1: Allocate space for bitmap
        // // STEP 2: Launch kernel that applies threshold and sets bitmap values, returns number of significant values
        // // STEP 3: Alloc space for significant values and launch kernel to scan bitmap and transfer sig values
        // // STEP 4: Free old data and feed significant values to compress

        if (!doGroup)
        {
            weak_threshold<<<80,256>>>(d_data, threshold, dataLength);
            cudaDeviceSynchronize();
            dataToCopy = dataLength;

            #ifdef TIMING
            cudaEventRecord(stop, 0);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&time, start, stop);
            printf("Time to execute: %.3f ms\n", time);
            #endif

            cudaMemcpy(data, d_data, sizeof(double)*dataToCopy, cudaMemcpyDeviceToHost);

        }else{
            apply_threshold<<<80,256>>>(d_data, threshold, dataLength, d_bitmap);
            // weak_threshold<<<80,256>>>(d_data, threshold, dataLength);
            cudaDeviceSynchronize();

            // cudaMemcpy(h_bitmap, d_bitmap, sizeof(char)*dataLength, cudaMemcpyDeviceToHost);
            cudaMemcpyFromSymbol(&c, d_sigValues, sizeof(unsigned long long int));

            // cudaMemcpy(data, d_data, sizeof(double)*dataToCopy, cudaMemcpyDeviceToHost);
            double *d_finaldata;

            cudaMalloc(&d_finaldata, sizeof(double)*c);

            if (doAlternateCompaction)
            {
                cuCompactor::compact<double>(d_data, d_finaldata, dataLength, is_nonzero(), 256);
            }else{
                thrust::copy_if(thrust::cuda::par, d_data, d_data + dataLength, d_finaldata, is_nonzero());
            }
            

            

            #ifdef TIMING
            cudaEventRecord(stop, 0);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&time, start, stop);
            printf("Time to execute: %.3f ms\n", time);
            #endif

            cudaMemcpy(h_bitmap, d_bitmap, sizeof(char)*dataLength, cudaMemcpyDeviceToHost);

            double *tmpData = (double *)malloc(c*sizeof(double));

            cudaMemcpy(tmpData, d_finaldata, sizeof(double)*c,cudaMemcpyDeviceToHost);

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
            FILE *bitmapFile;
            sprintf(bitmapFilePath, "%s.bitmap", inPath);	
            if (useNVCOMP)
            {

                #ifdef TIMING
                float time_NVCOMP;
                cudaEvent_t start_2, stop_2;
                cudaEventCreate(&start_2);
                cudaEventCreate(&stop_2);
                #endif

                size_t input_buffer_len = 4*((dataLength/32)+1);
                uint8_t *device_input_ptrs;

                cudaMalloc(&device_input_ptrs, input_buffer_len);
                cudaMemcpy(device_input_ptrs, bitmap_final, input_buffer_len, cudaMemcpyHostToDevice);

                cudaStream_t stream;
                cudaStreamCreate(&stream);

                const int chunk_size = 1 << 16;
                
                
                nvcompType_t data_type = NVCOMP_TYPE_CHAR;

                LZ4Manager nvcomp_manager{chunk_size, data_type, stream};
                CompressionConfig comp_config = nvcomp_manager.configure_compression(input_buffer_len);

                uint8_t* comp_buffer;
                cudaMalloc(&comp_buffer, comp_config.max_compressed_buffer_size);

                #ifdef TIMING
                cudaEventRecord(start_2, 0);
                #endif
                nvcomp_manager.compress(device_input_ptrs, comp_buffer, comp_config);

                #ifdef TIMING
                cudaEventRecord(stop_2, 0);
                cudaEventSynchronize(stop_2);
                cudaEventElapsedTime(&time_NVCOMP, start_2, stop_2);
                printf("NVCOMP Time to execute: %.3f ms\n", time_NVCOMP);
                #endif

                printf("max size %ld final size %ld\n", comp_config.max_compressed_buffer_size, nvcomp_manager.get_compressed_output_size(comp_buffer));

                bitmapFile = fopen(bitmapFilePath, "wb");
                fwrite(comp_buffer, sizeof(uint8_t), nvcomp_manager.get_compressed_output_size(comp_buffer), bitmapFile);
                fclose(bitmapFile);
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


            free(data);
            data = tmpData;
            dataToCopy = c;
        }
        
        

        

        sprintf(outputFilePath, "%s.threshold", inPath);	

        if (castToFloat)
        {
            float *floatTmpData = (float *)malloc(dataToCopy*sizeof(float));
            for (size_t i = 0; i < dataToCopy; i++)
            {
                floatTmpData[i] = (float) data[i];
            }
            writeFloatData_inBytes(floatTmpData, dataToCopy, outputFilePath);
            free(floatTmpData);

        }else{
            writeDoubleData_inBytes(data, dataToCopy, outputFilePath);
           
        }
        
        free(data);
        printf("Number of significant values: %d\n", c);
        cudaFree(d_data);
        
    } else {

        uint64_t bitmapLength = ((dataLength/32)+1);
        uint32_t *bitmap = (uint32_t *)malloc(sizeof(uint32_t)*bitmapLength);
        // uint8_t *d_decomp;
        uint32_t *d_bitmap;
        cudaMalloc(&d_bitmap, sizeof(uint32_t)*bitmapLength);
        

        int *pfix;
        cudaMalloc(&pfix, sizeof(int)*bitmapLength);
        // int *pfix = (int *)malloc(sizeof(int)*bitmapLength);
        
        double *final_data = (double *)malloc(sizeof(double)*dataLength);
        double *d_final_data;
        cudaMalloc(&d_final_data, sizeof(double)*dataLength);

        char bitmapFilePath[256];
        FILE *bitmapFile;
        sprintf(bitmapFilePath, "%s.bitmap", inPath);

        bitmapFile = fopen(bitmapFilePath, "rb");

        if (useNVCOMP)
        {
            struct stat st;
            size_t size;
            stat(bitmapFilePath, &st);
            size = st.st_size;
            fread(bitmap, sizeof(uint8_t), size, bitmapFile);
        }else{
            fread(bitmap, sizeof(uint32_t), ((dataLength/32)+1), bitmapFile);
        }
        


        fclose(bitmapFile);



        float *sig_values_f = (float *)malloc(sizeof(float)*numSigValues);
        double *sig_values = (double *)malloc(sizeof(double)*numSigValues);
        double *d_sig_values;
        cudaMalloc(&d_sig_values, sizeof(double)*numSigValues);
        
        // Need to memcpy and do a float->double cast

        FILE* sigFile;
        sigFile = fopen(inPathDecomp, "rb");

        fread(sig_values_f, sizeof(float), numSigValues, sigFile);

        fclose(sigFile);

        for (size_t i = 0; i < numSigValues; i++)
        {
            sig_values[i] = (double)sig_values_f[i];
        }

        free(sig_values_f);

        cudaMemcpy(d_sig_values, sig_values, sizeof(double)*numSigValues, cudaMemcpyHostToDevice);
        if(useNVCOMP){
            // cudaMalloc(&d_decomp, sizeof(uint8_t)*((dataLength/8)+1));

            cudaStream_t stream;
            cudaStreamCreate(&stream);

            const int chunk_size = 1 << 16;
            
            
            nvcompType_t data_type = NVCOMP_TYPE_CHAR;

            LZ4Manager nvcomp_manager{chunk_size, data_type, stream};

            DecompressionConfig decomp_config = nvcomp_manager.configure_decompression((uint8_t *)bitmap);
            
            #ifdef TIMING
            float time_NVCOMP;
            cudaEvent_t start_2, stop_2;
            cudaEventCreate(&start_2);
            cudaEventCreate(&stop_2);
            #endif
            #ifdef TIMING
            cudaEventRecord(start_2, 0);
            #endif
            nvcomp_manager.decompress((uint8_t*)d_bitmap, (uint8_t*)bitmap, decomp_config);
            #ifdef TIMING
            cudaEventRecord(stop_2, 0);
            cudaEventSynchronize(stop_2);
            cudaEventElapsedTime(&time_NVCOMP, start_2, stop_2);
            printf("NVCOMP Time to execute: %.3f ms\n", time_NVCOMP);
            #endif
            // d_bitmap = (uint32_t *)d_decomp;
        }else{
            cudaMemcpy(d_bitmap, bitmap, sizeof(uint32_t)*bitmapLength, cudaMemcpyHostToDevice);
        }
        #ifdef TIMING
        cudaEventRecord(start, 0);
        #endif

        prefix_gen<<<80,256>>>(d_bitmap, pfix, bitmapLength);
        cudaDeviceSynchronize();

        thrust::exclusive_scan(thrust::cuda::par, pfix, pfix+bitmapLength, pfix);

        reorder_values<<<80,256>>>(pfix, d_bitmap, bitmapLength, dataLength, d_sig_values, d_final_data, numSigValues);
        cudaDeviceSynchronize();



        #ifdef TIMING
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&time, start, stop);
        printf("Time to execute: %.3f ms\n", time);
        #endif

        cudaMemcpy(final_data, d_final_data, sizeof(double)*dataLength, cudaMemcpyDeviceToHost);

        cudaFree(d_final_data);
        cudaFree(d_sig_values);
        cudaFree(d_bitmap);
        cudaFree(pfix);
        free(bitmap);
        free(sig_values);

        char finalOutPath[256];
        FILE *finalFile;
        sprintf(finalOutPath, "%s.out", inPath);

        finalFile = fopen(finalOutPath, "rb");

        fwrite(final_data, sizeof(double), dataLength, finalFile);

        fclose(finalFile);

        printf("final data [0]: %f\n", final_data[0]);
        free(final_data);
    }
    

}