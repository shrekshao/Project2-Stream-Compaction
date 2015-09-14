#pragma once

namespace StreamCompaction {
namespace CPU {
    void scan(int n, int *odata, const int *idata);

    int compactWithoutScan(int n, int *odata, const int *idata);

    int compactWithScan(int n, int *odata, const int *idata);

	//void merge(int left, int right, int mid,int * odata,const int * idata);

	//void mergeSort(int left, int right, int *odata);
	void mergeLauncher(int left, int right, int *odata, const int *idata);
}
}
