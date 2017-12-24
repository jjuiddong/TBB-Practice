// parallel_for break
// https://software.intel.com/en-us/blogs/2007/11/08/have-a-fish-how-break-from-a-parallel-loop-in-tbb


#include <stdio.h>
#include <windows.h>
#include <mmsystem.h>
#include <iostream>
#include <thread>
#include <vector>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "tbb/tbb.h"
#include "tbb/task_scheduler_init.h"

#ifdef _DEBUG
#pragma comment(lib, "opencv_core310d.lib")
#pragma comment(lib, "opencv_imgcodecs310d.lib")
#pragma comment(lib, "opencv_highgui310d.lib")
#pragma comment(lib, "opencv_imgproc310d.lib")
#else
#pragma comment(lib, "opencv_core310.lib")
#pragma comment(lib, "opencv_imgcodecs310.lib")
#pragma comment(lib, "opencv_highgui310.lib")
#pragma comment(lib, "opencv_imgproc310.lib")
#endif

#pragma comment(lib, "winmm.lib")

using namespace std;
using namespace cv;


// https://software.intel.com/en-us/blogs/2007/11/08/have-a-fish-how-break-from-a-parallel-loop-in-tbb

template<typename Value>
class cancelable_range {
	tbb::blocked_range<Value> my_range;
	volatile bool& my_stop;
public:
	// Constructor for client code
	/** Range becomes empty if stop==true. */
	cancelable_range(int begin, int end, int grainsize, volatile bool& stop) :
		my_range(begin, end, grainsize),
		my_stop(stop)
	{}

	//! Splitting constructor used by parallel_for
	cancelable_range(cancelable_range& r, tbb::split) :
		my_range(r.my_range, tbb::split()),
		my_stop(r.my_stop)
	{}

	//! Cancel the range.
	void cancel() const { my_stop = true; }

	//! True if range is empty.
	/** Range is empty if there is request to cancel the range. */
	bool empty() const { return my_stop || my_range.empty(); }

	//! True if range is divisible
	/** Range becomes indivisible if there is request to cancel the range. */
	bool is_divisible() const { return !my_stop && my_range.is_divisible(); }

	//! Initial value in range.
	Value begin() const { return my_range.begin(); }

	//! One past last value in range
	/** Note that end()==begin() if there is request to cancel the range.
	The value of end() may change asynchronously if another thread cancels the range. **/
	Value end() const { return my_stop ? my_range.begin() : my_range.end(); }
};


void threadfunction(Mat &input, Mat &matObj, cancelable_range<int> &r)
{
	cout << "in" << endl;
	const cv::Size csize(input.cols - matObj.cols + 1, input.rows - matObj.rows + 1);
	Point left_top;
	double min = 0, max = 0;
	Mat matResult(csize, IPL_DEPTH_32F);
	cv::matchTemplate(input, matObj, matResult, CV_TM_CCOEFF_NORMED);
	cv::minMaxLoc(matResult, &min, &max, NULL, &left_top);
	r.cancel();
	cout << "cancel" << endl;
}


void ParallelApplyFoo(Mat &input, Mat &matObj, size_t n) {
	bool stop = false;
	tbb::parallel_for( cancelable_range<int>(0, n, 1, stop), 
		[&](cancelable_range<int> &r) {
			threadfunction(input, matObj, r);
	});
}


void main()
{
	Mat input = imread("srcimg.jpg");
	Mat matObj = imread("temp.jpg");

	const int t1 = timeGetTime();
	ParallelApplyFoo(input, matObj, 100);

	const int t2 = timeGetTime();
	cout << t2 - t1 << endl;

	getchar();
}
