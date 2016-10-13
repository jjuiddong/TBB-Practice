//
// OPENCV_OPENCL_DEVICE=:GPU:1
//

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


void threadfunction(UMat &input, UMat &matObj)
{
	const cv::Size csize(input.cols - matObj.cols + 1, input.rows - matObj.rows + 1);
	Point left_top;
	double min = 0, max = 0;
	UMat matResult(csize, IPL_DEPTH_32F);
	cv::matchTemplate(input, matObj, matResult, CV_TM_CCOEFF_NORMED);
	cv::minMaxLoc(matResult, &min, &max, NULL, &left_top);
	cout << "ok" << endl;
}

void ParallelApplyFoo(Mat &input, Mat &obj, size_t n) {

	tbb::parallel_for(size_t(0), n, [&](size_t i) {
		UMat uinput, uobj;
		uinput = input.getUMat(cv::ACCESS_READ);
		uobj = obj.getUMat(cv::ACCESS_READ);
		threadfunction(uinput, uobj);
	});

}

void main()
{
	Mat input = imread("srcimg.jpg");
	Mat obj = imread("temp.jpg");

	const int t1 = timeGetTime();
	ParallelApplyFoo(input, obj, 100);

	const int t2 = timeGetTime();
	cout << t2 - t1 << endl;
}
