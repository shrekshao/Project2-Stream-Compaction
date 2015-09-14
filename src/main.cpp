/**
 * @file      main.cpp
 * @brief     Stream compaction test program
 * @authors   Kai Ninomiya
 * @date      2015
 * @copyright University of Pennsylvania
 */
#include <iostream>
#include <fstream>
#include <chrono>

#include <stdio.h>
#include <stdlib.h>

#include <ctime>

#include <cstdio>
#include <stream_compaction/common.h>
#include <stream_compaction/cpu.h>
#include <stream_compaction/naive.h>
#include <stream_compaction/efficient.h>
#include <stream_compaction/thrust.h>
#include "testing_helpers.hpp"


cudaEvent_t beginEvent;
cudaEvent_t endEvent;

std::ofstream of;

void cudaRecordEndAndPrint()
{
	cudaEventRecord(endEvent,0);
	cudaEventSynchronize( endEvent );
	float ms;
	cudaEventElapsedTime(&ms,beginEvent,endEvent);
	printf("time:%f\n",ms);

	of << "," <<ms;
}


//#define RADIX_SORT_TEST


int main(int argc, char* argv[]) {
    
	const int ppp = 20;
	const int SIZE = 1 << ppp;
    const int NPOT = SIZE - 3;
    //int a[SIZE], b[SIZE], c[SIZE];
	int * a = new int[SIZE];
	int * b = new int[SIZE];
	int * c = new int[SIZE];


	//FILE *fp;

	//char filename[50];
	//sprintf(filename,"zztt_log_2(%d)_%d.txt",ppp,blockSize);
	//if((fp=freopen(filename, "w" ,stdout))==NULL) {
	//	printf("Cannot open file.\n");
	//	exit(1);
	//}

	
	of.open("zzz_time_table.txt",std::ofstream::out | std::ofstream::app);
	of<<ppp<<','<<SIZE<<','<<blockSize;

	printf("ArraySize:2^(%d), %d\nBlockSize:%d\n",ppp,SIZE,blockSize);
	
	clock_t t1;
	clock_t t2;

	//std::chrono::system_clock::time_point start;
	//std::chrono::system_clock::time_point end;
	
	//auto start = std::chrono::steady_clock::now();
	

	
	cudaEventCreate( &beginEvent );
	cudaEventCreate( &endEvent );


#ifndef RADIX_SORT_TEST
    // Scan tests

    printf("\n");
    printf("****************\n");
    printf("** SCAN TESTS **\n");
    printf("****************\n");

    genArray(SIZE - 1, a, 50);  // Leave a 0 at the end to test that edge case
	a[SIZE-1] = 0;
    printArray(SIZE, a, true);

    zeroArray(SIZE, b);
    printDesc("cpu scan, power-of-two");
	t1 = clock();
    StreamCompaction::CPU::scan(SIZE, b, a);
	t2 = clock();
	printf("time:%f\n",((float)t2-(float)t1));
	of<<","<<((float)t2-(float)t1);
    printArray(SIZE, b, true);

	
    zeroArray(SIZE, c);
    printDesc("cpu scan, non-power-of-two");
	t1 = clock();
    StreamCompaction::CPU::scan(NPOT, c, a);
	t2 = clock();
	printf("time:%f\n",((float)t2-(float)t1));
	of<<","<<((float)t2-(float)t1);
    printArray(NPOT, c, true);
    printCmpResult(NPOT, b, c);
	
    zeroArray(SIZE, c);
    printDesc("naive scan, power-of-two");
	cudaEventRecord(beginEvent,0);
    StreamCompaction::Naive::scan(SIZE, c, a);
	cudaRecordEndAndPrint();
    //printArray(SIZE, c, true);
    printCmpResult(SIZE, b, c);

    zeroArray(SIZE, c);
    printDesc("naive scan, non-power-of-two");
	cudaEventRecord(beginEvent,0);
    StreamCompaction::Naive::scan(NPOT, c, a);
	cudaRecordEndAndPrint();
    //printArray(NPOT, c, true);
    printCmpResult(NPOT, b, c);

    zeroArray(SIZE, c);
    printDesc("work-efficient scan, power-of-two");
	cudaEventRecord(beginEvent,0);
    StreamCompaction::Efficient::scan(SIZE, c, a);
	cudaRecordEndAndPrint();
    //printArray(SIZE, c, true);
    printCmpResult(SIZE, b, c);

    zeroArray(SIZE, c);
    printDesc("work-efficient scan, non-power-of-two");
	cudaEventRecord(beginEvent,0);
    StreamCompaction::Efficient::scan(NPOT, c, a);
	cudaRecordEndAndPrint();
    //printArray(NPOT, c, true);
    printCmpResult(NPOT, b, c);
	
    zeroArray(SIZE, c);
    printDesc("thrust scan, power-of-two");
	cudaEventRecord(beginEvent,0);
    StreamCompaction::Thrust::scan(SIZE, c, a);
    cudaRecordEndAndPrint();
	//printArray(SIZE, c, true);
    printCmpResult(SIZE, b, c);

    zeroArray(SIZE, c);
    printDesc("thrust scan, non-power-of-two");
	cudaEventRecord(beginEvent,0);
    StreamCompaction::Thrust::scan(NPOT, c, a);
    cudaRecordEndAndPrint();
	//printArray(NPOT, c, true);
    printCmpResult(NPOT, b, c);
	
    printf("\n");
    printf("*****************************\n");
    printf("** STREAM COMPACTION TESTS **\n");
    printf("*****************************\n");

    // Compaction tests

    genArray(SIZE - 1, a, 4);  // Leave a 0 at the end to test that edge case
	a[SIZE-1] = 0;
    printArray(SIZE, a, true);

    int count, expectedCount, expectedNPOT;

    zeroArray(SIZE, b);
    printDesc("cpu compact without scan, power-of-two");
	t1 = clock();
    count = StreamCompaction::CPU::compactWithoutScan(SIZE, b, a);
	t2 = clock();
	printf("time:%f\n",((float)t2-(float)t1));
	of<<","<<((float)t2-(float)t1);
    expectedCount = count;
    printArray(count, b, true);
    printCmpLenResult(count, expectedCount, b, b);

    zeroArray(SIZE, c);
    printDesc("cpu compact without scan, non-power-of-two");
	t1 = clock();
    count = StreamCompaction::CPU::compactWithoutScan(NPOT, c, a);
	t2 = clock();
	printf("time:%f\n",((float)t2-(float)t1));
	of<<","<<((float)t2-(float)t1);
    expectedNPOT = count;
    printArray(count, c, true);
    printCmpLenResult(count, expectedNPOT, b, c);

    zeroArray(SIZE, c);
    printDesc("cpu compact with scan");
	t1 = clock();
    count = StreamCompaction::CPU::compactWithScan(SIZE, c, a);
	t2 = clock();
	printf("time:%f\n",((float)t2-(float)t1));
	of<<","<<((float)t2-(float)t1);
    printArray(count, c, true);
    printCmpLenResult(count, expectedCount, b, c);

    zeroArray(SIZE, c);
    printDesc("work-efficient compact, power-of-two");
	cudaEventRecord(beginEvent,0);
    count = StreamCompaction::Efficient::compact(SIZE, c, a);
	cudaRecordEndAndPrint();
    printArray(count, c, true);
    printCmpLenResult(count, expectedCount, b, c);

    zeroArray(SIZE, c);
    printDesc("work-efficient compact, non-power-of-two");
	cudaEventRecord(beginEvent,0);
    count = StreamCompaction::Efficient::compact(NPOT, c, a);
	cudaRecordEndAndPrint();
    //printArray(count, c, true);
    printCmpLenResult(count, expectedNPOT, b, c);
	
#endif




#ifdef RADIX_SORT_TEST
	//Radix sort test

	printf("\n");
    printf("*****************************\n");
    printf("** SIMPLE RADIX SORT TESTS **\n");
    printf("*****************************\n");

	genArray(SIZE - 1, a, SIZE-1);  // Leave a 0 at the end to test that edge case
	a[SIZE-1] = 0;
    printArray(SIZE, a, true);

	zeroArray(SIZE, b);
    printDesc("cpu sort, power-of-two");
	t1 = clock();
	StreamCompaction::CPU::mergeLauncher(0,SIZE-1, b, a);
	t2 = clock();
	printf("time:%f\n",((float)t2-(float)t1));
	of<<","<<((float)t2-(float)t1);
    printArray(SIZE, b, true);
    //printCmpLenResult(count, expectedCount, b, b);

	zeroArray(SIZE, c);
    printDesc("radix sort, power-of-two");
	cudaEventRecord(beginEvent,0);
	StreamCompaction::Efficient::radixSortLauncher(SIZE,c,a,ppp,0);
	cudaRecordEndAndPrint();
    printArray(SIZE, c, true);
	printCmpResult(SIZE, b, c);
#endif


	cudaEventDestroy( beginEvent );
	cudaEventDestroy( endEvent );


	delete[] a;
	delete[] b;
	delete[] c;
	
	
	//auto end = std::chrono::steady_clock::now();
	//printf("time:%d\n",std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count());
	//fclose(fp);
	of<<'\n';
	of.close();

	return 0;
}
