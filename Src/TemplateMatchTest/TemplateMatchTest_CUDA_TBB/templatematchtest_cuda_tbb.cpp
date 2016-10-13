// templateMatch + CUDA + TBB

#include <stdio.h>
#include <windows.h>
#include <mmsystem.h>
#include <iostream>
#include <thread>
#include <atomic>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/cuda/cudaimgproc.hpp>
#include "tbb/tbb.h"
#include "tbb/task_scheduler_init.h"

#ifdef _DEBUG
	#pragma comment(lib, "opencv_core310d.lib")
	#pragma comment(lib, "opencv_imgcodecs310d.lib")
	#pragma comment(lib, "opencv_highgui310d.lib")
	#pragma comment(lib, "opencv_imgproc310d.lib")
	#pragma comment(lib, "opencv_cudaimgproc310d.lib")
#else
	#pragma comment(lib, "opencv_core310.lib")
	#pragma comment(lib, "opencv_imgcodecs310.lib")
	#pragma comment(lib, "opencv_highgui310.lib")
	#pragma comment(lib, "opencv_imgproc310.lib")
	#pragma comment(lib, "opencv_cudaimgproc310.lib")
#endif

#pragma comment(lib, "winmm.lib")

using namespace std;
using namespace cv;
atomic<int> g_matchCount = 0;

void ParallelFunc(Mat &input, Mat &matObj)
{
	if (g_matchCount >= 100)
		return;

	++g_matchCount;

	const cv::Size csize(input.cols - matObj.cols + 1, input.rows - matObj.rows + 1);
	Point left_top;
	double min = 0, max = 0;
	Mat matResult(csize, IPL_DEPTH_32F);
	cv::matchTemplate(input, matObj, matResult, CV_TM_CCOEFF_NORMED);
	cv::minMaxLoc(matResult, &min, &max, NULL, &left_top);
	cout << "ok " << g_matchCount << endl;
}

void ParallelApplyFoo(Mat &input, Mat &matObj, size_t n) {
	tbb::parallel_for(size_t(0), n, [&](size_t i) {
		ParallelFunc(input, matObj);
	});
}


void threadfunction(Mat &input, Mat &matObj)
{
	Ptr<cuda::TemplateMatching> tmp = cuda::createTemplateMatching(input.type(), CV_TM_CCOEFF_NORMED);

	cv::cuda::GpuMat cinput(input);
	cv::cuda::GpuMat cobj(matObj);
	const cv::Size csize(input.cols - matObj.cols + 1, input.rows - matObj.rows + 1);
	cv::cuda::GpuMat cmatResult(csize, IPL_DEPTH_32F);
	if (cinput.empty() || cobj.empty() || cmatResult.empty())
		return;

	while (g_matchCount < 100)
	{
		g_matchCount++;

		Point left_top;
		double min = 0, max = 0;
		tmp->match(cinput, cobj, cmatResult);
		Mat matResult;
		cmatResult.download(matResult);
		cv::minMaxLoc(matResult, &min, &max, NULL, &left_top);
		cout << "ok " << g_matchCount << endl;
	}
}

void main()
{
	Mat input = imread("srcimg.jpg");
	Mat obj = imread("temp.jpg");

	const int t1 = timeGetTime();
	std::thread th0(threadfunction, input, obj);
	//ParallelApplyFoo(input, obj, 100);
	th0.join();
	const int t2 = timeGetTime();

	cout << t2 - t1 << endl;

	getchar();
}
