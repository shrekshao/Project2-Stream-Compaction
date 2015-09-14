#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"

namespace StreamCompaction {
namespace Naive {
	//const int blockSize = 128;

	int* dev_odata;
	int* dev_idata;
	//int* dev_tdata;		//temp transfer one


	__global__ void kernWriteOneSum(int n,int threshold, int* odata, const int* idata)
	{
		//threshold ... 2^(d-1)
		int k = threadIdx.x + blockDim.x * blockIdx.x;
		if( k < n )
		{
			if( k >= threshold )
			{
				odata[k] = idata[k - threshold] + idata[k];
			}
			else
			{
				odata[k] = idata[k];
			}
		}
	}



	
	/**
	* Performs prefix-sum (aka scan) on idata, storing the result into odata.
	*/
	void scan(int n, int *odata, const int *idata) {
		//naive parrellel scan
		int ceil_log2n = ilog2ceil(n);


		dim3 fullBlocksPerGrid((n + blockSize - 1) / blockSize); 

		cudaMalloc((void**)&dev_idata, n * sizeof(int));
		checkCUDAError("cudaMalloc dev_idata failed");

		cudaMalloc((void**)&dev_odata, n * sizeof(int));
		checkCUDAError("cudaMalloc dev_odata failed");


		int* cur_out = dev_odata;
		int* cur_in = dev_idata;
		/*
		//make sure the last write to idata (before inclusive 2 exclusive)
		if(ceil_log2n % 2 == 0)
		{
			cur_out = dev_odata;
			cur_in = dev_idata;
		}
		else
		{
			cur_out = dev_idata;
			cur_in = dev_odata;
		}
		*/

		cudaMemcpy(cur_in,idata,n*sizeof(int),cudaMemcpyHostToDevice);
		checkCUDAError("cudaMemcpy from idata to cur_in failed");

		
		cudaDeviceSynchronize();
		


		for (int d = 1; d <= ceil_log2n ; d++)
		{
			kernWriteOneSum<<< fullBlocksPerGrid, blockSize>>> (n , 1<<(d-1) , cur_out, cur_in);

			int* tmp_p = cur_out;
			cur_out = cur_in;
			cur_in = tmp_p;

			
			cudaDeviceSynchronize();
		}

		Common::kernInclusive2Exclusive<<< fullBlocksPerGrid, blockSize>>>(n,cur_out,cur_in);

		cudaMemcpy(odata,cur_out,n*sizeof(int),cudaMemcpyDeviceToHost);
		checkCUDAError("cudaMemcpy from dev_odata to odata failed");

		cudaFree(dev_idata);
		cudaFree(dev_odata);
		
	}

}
}
