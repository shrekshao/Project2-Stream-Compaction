#pragma once

namespace StreamCompaction {
namespace Efficient {
    void scan(int n, int *odata, const int *idata,bool is_dev_data=false);

    int compact(int n, int *odata, const int *idata);

	void radixSortLauncher(int n, int *odata, const int *idata, int msb,int lsb);
}
}
