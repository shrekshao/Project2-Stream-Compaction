#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

namespace StreamCompaction {
	namespace Efficient {
		//const int blockSize = 128;

		

		__global__ void kernUpSweep(int size, int step, int * data)
		{
			//step = 2^(d+1)
			int k = threadIdx.x + blockDim.x * blockIdx.x;
			
			if(k < size)
			{
				if ( k % step == 0 )
				{
					data[k + step - 1] += data[k + (step>>1) - 1];
				}
			}
			
		}

		__global__ void kernDownSweep(int size,int step, int * data)
		{
			//step = 2^(d+1)
			int k = threadIdx.x + blockDim.x * blockIdx.x;

			if(k < size)
			{
				if ( k % step == 0 )
				{
					int left_child = data[k + (step>>1) - 1];
					data[k + (step>>1) - 1] = data[k + step - 1];
					data[k + step - 1] += left_child;
				}
			}
		}


		__global__ void kernSetRootZero(int rootId, int * data)
		{
			int k = threadIdx.x + blockDim.x * blockIdx.x;
			if(k == rootId)
			{
				data[k] = 0;
			}
		}

		/**
		* Performs prefix-sum (aka scan) on idata, storing the result into odata.
		*/
		void scan(int n, int *odata, const int *idata,bool is_dev_data) {
			//if using device data directly
			
			
			int * dev_data;

			int ceil_log2n = ilog2ceil(n);
			int size = 1 << ceil_log2n;

			dim3 fullBlocksPerGrid((size + blockSize - 1) / blockSize); 


			cudaMalloc((void**)&dev_data, size * sizeof(int));
			checkCUDAError("cudaMalloc dev_data failed");
			Common::kernZeroArray<<< fullBlocksPerGrid, blockSize>>>(size, dev_data);
			if(!is_dev_data)
			{
				//host data
				cudaMemcpy(dev_data,idata, n * sizeof(int),cudaMemcpyHostToDevice);
				checkCUDAError("cudaMemcpy from data to dev_data failed");
			}
			else
			{
				cudaMemcpy(dev_data,idata, n * sizeof(int),cudaMemcpyDeviceToDevice);
				checkCUDAError("cudaMemcpy from data to dev_data failed");
			}
			cudaDeviceSynchronize();

			//UpSweep
			for(int d = 0 ; d < ceil_log2n - 1 ; d++)
			{
				kernUpSweep<<< fullBlocksPerGrid, blockSize>>> (size, 1<<(d+1) , dev_data);
				cudaDeviceSynchronize();
			}

			kernSetRootZero<<< fullBlocksPerGrid, blockSize>>> ( size - 1 , dev_data);
			cudaDeviceSynchronize();
			
			for(int d = ceil_log2n - 1 ; d >= 0 ; d--)
			{
				kernDownSweep<<< fullBlocksPerGrid, blockSize>>> (size, 1<<(d+1) , dev_data);
				cudaDeviceSynchronize();
			}


			if(!is_dev_data)
			{
				cudaMemcpy(odata,dev_data,n * sizeof(int),cudaMemcpyDeviceToHost);
				checkCUDAError("cudaMemcpy from dev_data to odata failed");
			}
			else
			{
				cudaMemcpy(odata,dev_data,n * sizeof(int),cudaMemcpyDeviceToDevice);
				checkCUDAError("cudaMemcpy from dev_data to odata failed");
			}
			cudaFree(dev_data);
		}

		/**
		* Performs stream compaction on idata, storing the result into odata.
		* All zeroes are discarded.
		*
		* @param n      The number of elements in idata.
		* @param odata  The array into which to store elements.
		* @param idata  The array of elements to compact.
		* @returns      The number of elements remaining after compaction.
		*/
		int compact(int n, int *odata, const int *idata) {
			int hos_scans;
			int hos_bools;
			int * dev_bools;
			int * dev_scans;
			int * dev_idata;
			int * dev_odata;
			dim3 fullBlocksPerGrid((n + blockSize - 1) / blockSize); 

			cudaMalloc((void**)&dev_bools, n * sizeof(int));
			checkCUDAError("cudaMalloc dev_bools failed");
			cudaMalloc((void**)&dev_scans, n * sizeof(int));
			checkCUDAError("cudaMalloc dev_scans failed");
			cudaMalloc((void**)&dev_idata, n * sizeof(int));
			checkCUDAError("cudaMalloc dev_idata failed");
			cudaMalloc((void**)&dev_odata, n * sizeof(int));
			checkCUDAError("cudaMalloc dev_odata failed");

			cudaMemcpy(dev_idata,idata, n * sizeof(int),cudaMemcpyHostToDevice);
			checkCUDAError("cudaMemcpy from data to dev_data failed");
			cudaDeviceSynchronize();

			Common::kernMapToBoolean<<< fullBlocksPerGrid, blockSize>>> ( n , dev_bools, dev_idata );
			cudaDeviceSynchronize();

			//cudaMemcpy(hos_bools,dev_bools, n * sizeof(int),cudaMemcpyDeviceToHost);
			//checkCUDAError("cudaMemcpy from data to dev_data failed");
			//cudaDeviceSynchronize();

			scan(n,dev_scans,dev_bools,true);

			//cudaMemcpy(dev_scans,hos_scans, n * sizeof(int),cudaMemcpyHostToDevice);
			//checkCUDAError("cudaMemcpy from hos_scans to dev_scans failed");
			//cudaDeviceSynchronize();

			Common::kernScatter<<< fullBlocksPerGrid, blockSize>>>(n, dev_odata,
				dev_idata, dev_bools, dev_scans);
			cudaDeviceSynchronize();

			cudaMemcpy(odata,dev_odata,n * sizeof(int),cudaMemcpyDeviceToHost);
			checkCUDAError("cudaMemcpy from dev_odata to odata failed");
			//cudaDeviceSynchronize();

			cudaMemcpy(&hos_scans,dev_scans+n-1,sizeof(int),cudaMemcpyDeviceToHost);
			checkCUDAError("cudaMemcpy scans[n-1] failed");

			cudaMemcpy(&hos_bools,dev_bools+n-1,sizeof(int),cudaMemcpyDeviceToHost);
			checkCUDAError("cudaMemcpy bools[n-1] failed");

			cudaDeviceSynchronize();



			cudaFree(dev_idata);
			cudaFree(dev_odata);
			cudaFree(dev_bools);
			cudaFree(dev_scans);

			//int num = hos_scans[n-1] + hos_bools[n-1];
			int num = hos_scans + hos_bools;
			//delete[] hos_scans;
			//delete[] hos_bools;

			return num;
		}







		//Radix sort


		__global__ void kernGetE(int n, int * odata, const int * idata,int cur_bit)
		{
			int index = threadIdx.x + blockDim.x * blockIdx.x;
			if( index < n)
			{
				odata[index] = 1 - ( ( idata[index] & (1 << cur_bit ) ) >> cur_bit   );
			}
		}

		__global__ void kernGetK(int n, int* t, const int * f, const int totalFalses)
		{
			int index = threadIdx.x + blockDim.x * blockIdx.x;
			if( index < n)
			{
				t[index] = index - f[index] + totalFalses;
			}
		}

		__global__ void kernRadixScatter(int n, int * odata,const int * idata, const int * e, const int * t, const int * f)
		{
			int index = threadIdx.x + blockDim.x * blockIdx.x;
			if( index < n)
			{
				 odata[  (e[index]==0) ? t[index] : f[index]  ] = idata[index] ;
			}
		}



		int * dev_i;
		int * dev_o;
		int * dev_e;	// dev_e[i] = 1 - dev_idata[i].cur_bit
		int * dev_f;	// exclusive scan of dev_e, id if false
		int * dev_t;	// i ¨Cf[i] + totalFalses, id if true
		

		

		void radixSort(int n, int *dev_odata, const int *dev_idata, int cur_bit)
		{
			dim3 fullBlocksPerGrid((n + blockSize - 1) / blockSize);
			// get e
			kernGetE<<< fullBlocksPerGrid, blockSize>>>(n,dev_e,dev_idata,cur_bit);
			cudaDeviceSynchronize();

			scan(n,dev_f,dev_e,true);
			int totalFalses;
			int last_e;
			cudaMemcpy(&last_e,dev_e+n-1,sizeof(int),cudaMemcpyDeviceToHost);
			checkCUDAError("cudaMemcpy dev_e[n-1] failed");
			cudaMemcpy(&totalFalses,dev_f+n-1,sizeof(int),cudaMemcpyDeviceToHost);
			checkCUDAError("cudaMemcpy dev_f[n-1] failed");
			totalFalses += last_e;

			//get t
			kernGetK<<< fullBlocksPerGrid, blockSize>>>(n,dev_t,dev_f,totalFalses);

			//scatter
			kernRadixScatter<<< fullBlocksPerGrid, blockSize>>>(n,dev_odata,dev_idata,dev_e,dev_t,dev_f);
		}



		//wrapper
		void radixSortLauncher(int n, int *odata, const int *idata, int msb,int lsb)
		{
			//simple version
			//no split, no merge, no shared memory
			
			//split
			 

			//sort
			//for each split
			
			
			cudaMalloc((void**)&dev_i, n * sizeof(int));
			checkCUDAError("cudaMalloc dev_i failed");
			cudaMalloc((void**)&dev_o, n * sizeof(int));
			checkCUDAError("cudaMalloc dev_o failed");
			cudaMalloc((void**)&dev_e, n * sizeof(int));
			checkCUDAError("cudaMalloc dev_e failed");
			cudaMalloc((void**)&dev_f, n * sizeof(int));
			checkCUDAError("cudaMalloc dev_f failed");
			cudaMalloc((void**)&dev_t, n * sizeof(int));
			checkCUDAError("cudaMalloc dev_t failed");

			int * dev_cur_i = dev_i;
			int * dev_cur_o = dev_o;
			/*
			if( (msb - lsb) % 2 == 0)
			{
				dev_cur_i = dev_i;
				dev_cur_o = dev_o;
			}
			else
			{
				dev_cur_i = dev_o;
				dev_cur_o = dev_i;
			}
			*/

			cudaMemcpy(dev_cur_i,idata,n*sizeof(int),cudaMemcpyHostToDevice);
			checkCUDAError("cudaMemcpy from idata to dev_cur_i failed");


			for(int i = lsb; i <= msb; i++)
			{
				radixSort(n,dev_cur_o,dev_cur_i,i);

				int * tmp = dev_cur_i;
				dev_cur_i = dev_cur_o;
				dev_cur_o = tmp;
			}


			//merge


			////////
			
			cudaMemcpy(odata,dev_cur_i,n*sizeof(int),cudaMemcpyDeviceToHost);
			checkCUDAError("cudaMemcpy from dev_cur_o to odata failed");


			cudaFree(dev_i);
			cudaFree(dev_o);
			cudaFree(dev_e);
			cudaFree(dev_f);
			cudaFree(dev_t);
		}






	}
}
