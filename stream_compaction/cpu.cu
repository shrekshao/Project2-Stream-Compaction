#include <cstdio>
#include "cpu.h"

namespace StreamCompaction {
namespace CPU {

/**
 * CPU scan (prefix sum).
 */
void scan(int n, int *odata, const int *idata) {
	if(n > 0)
	{
		odata[0] = 0;
		for(int i = 1 ; i < n; i++)
		{
			odata[i] = idata[i-1] + odata[i-1];
		}
	}
}

/**
 * CPU stream compaction without using the scan function.
 *
 * @returns the number of elements remaining after compaction.
 */
int compactWithoutScan(int n, int *odata, const int *idata) {
	int r = 0;
	for(int i = 0; i < n; i++)
	{
		if(idata[i] != 0)
		{
			odata[r] = idata[i];
			r++;
		}
	}
	return r;
}

/**
 * CPU stream compaction using scan and scatter, like the parallel version.
 *
 * @returns the number of elements remaining after compaction.
 */
int compactWithScan(int n, int *odata, const int *idata) {
	int* mapped_ary = new int [n];
	int* scan_ary = new int [n];

	//map input to 0s and 1s
	for(int i = 0; i < n; i++)
	{
		mapped_ary[i] = (idata[i]!=0) ? 1 : 0;
	}

	scan(n,scan_ary,mapped_ary);

	//scatter
	for(int i = 0; i < n; i++)
	{
		if(mapped_ary[i] != 0)
		{
			odata[ scan_ary[i] ] = idata[i];
		}
	}

	int r = scan_ary[n-1] + mapped_ary[n-1];
	delete[] mapped_ary;
	delete[] scan_ary;
	return r;
}



/**
* CPU simple merge sort
*/
void merge(int left, int right, int mid,int * data)
{
	int i = left;
	int j = mid+1;
	int k = left;
	
	int * odata = new int[right+1];

	while(i <= mid && j <= right)
	{
		if(data[i] <= data[j])
		{
			odata[k] = data[i];
			i++;
			k++;
		}
		else
		{
			odata[k] = data[j];
			j++;
			k++;
		}
	}

	while( i <= mid )
	{
		odata[k] = data[i];
		i++;
		k++;
	}

	while( j <= right )
	{
		odata[k] = data[j];
		j++;
		k++;
	}

	for(int i = left; i<=right; i++)
	{
		data[i] = odata[i];
	}

	delete[] odata;
}


void mergeSort(int left, int right, int *odata)
{
	if(left < right)
	{
		int mid = (left+right)/2;
		mergeSort(left,mid,odata);
		mergeSort(mid+1,right,odata);
		merge(left,right,mid,odata);
	}
}

void mergeLauncher(int left, int right, int *odata, const int *idata)
{
	for(int i=left; i<=right; i++)
	{
		odata[i] = idata[i];
	}
	mergeSort(left,right,odata);
}






}
}
