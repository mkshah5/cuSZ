#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <chrono>
//#include <bit>
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
__device__ int d_sigValues = 0;

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

__global__ void grouping(double *data, char* bitmap, int pointsPerScan, unsigned long dataLength){

    __shared__ double loaded_data[4096];
    int starting_index = 4096*blockIdx.x;

    for (size_t i = threadIdx.x+starting_index; i < starting_index+4096; i+=blockDim.x)
    {
        if (i >= dataLength)
        {
            break;
        }
        
        loaded_data[]
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
    char outputFilePath[256];
    unsigned char *bytes = NULL;

    int preCompression = 0;
    int doGroup = 0;
    int castToFloat = 0;

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

        uint32_t *bitmap_final;

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
        
        int c = 0;

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
            cudaDeviceSynchronize();

            cudaMemcpy(h_bitmap, d_bitmap, sizeof(char)*dataLength, cudaMemcpyDeviceToHost);
            cudaMemcpyFromSymbol(&c, d_sigValues, sizeof(int));

            cudaMemcpy(data, d_data, sizeof(double)*dataToCopy, cudaMemcpyDeviceToHost);

            double *tmpData = (double *)malloc(c*sizeof(double));
            int sig_ind = 0;
            for (size_t i = 0; i < dataLength; i++)
            {
                if (h_bitmap[i]=='1')
                {
                    tmpData[sig_ind] = data[i];
                    bitmap_final[i/32] = bitmap_final[i/32] | (1 << (i%32));

                    sig_ind++;
                }
                
            }

            #ifdef TIMING
            cudaEventRecord(stop, 0);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&time, start, stop);
            printf("Time to execute: %.3f ms\n", time);
            #endif


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
        
    }
    

}
