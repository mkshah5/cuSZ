#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <chrono>
#include <bit>
#include <cuda.h>
#include <cuda_runtime.h>

#define TIMING


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


/**
 * @brief Globals
 * 
 */
int littleEndian = 0;
__device__ int d_sigValues = 0;

/**
 * @brief CUDA kernels
 * 
 */

__global__ void weak_threshold(double *data, float threshold, unsigned long dataLength){
    
    for (unsigned long tid = threadIdx.x+blockDim.x*blockIdx.x; tid < dataLength; tid+=blockDim.x*blockIdx.x)
    {
        if (fabs(data[tid]) <= threshold)
        {
            data[tid] = 0.0;
        } else{
            atomicAdd(&d_sigValues, 1);
        }
        
    }
    
}

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
    char outputFilePath[256];
    unsigned char *bytes = NULL;

    int preCompression = 0;
    int doGroup = 0;

    float threshold = 0.0;
    unsigned long numSigValues = 0;
    unsigned long dataLength = 0;

    size_t nbEle;

    for(int i=0;i<argc;i++){
        switch (argv[i][1])
        {
        case 'z':
            preCompression = 1;
            break;
        case 'd':
            preCompression = 0;
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
        double *data = readDoubleData(inPath, &nbEle);
        double *d_data;
        cudaMalloc(&d_data, sizeof(double)*dataLength);
        cudaMemcpy(d_data, data, sizeof(double)*dataLength, cudaMemcpyHostToDevice);

        #ifdef TIMING
        cudaEventRecord(start, 0);
        #endif

        if (!doGroup)
        {
            weak_threshold<<<80,256>>>(d_data, threshold, dataLength);
            cudaDeviceSynchronize();
        }
        
        #ifdef TIMING
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&time, start, stop);
        printf("Time to execute: %.3f ms\n", time);
        #endif

        int c;
        cudaMemcpyFromSymbol(&c, d_sigValues, sizeof(int));
        printf("Number of significant values: %d\n", c);
        cudaFree(d_data);
    }
    

}