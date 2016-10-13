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


void threadfunction(vector<Mat*> &input, vector<Mat*> &matObj, const bool cp, const size_t idx)
{
	int i = idx % 50;

	const cv::Size csize(input.front()->cols - matObj.front()->cols + 1, input.front()->rows - matObj.front()->rows + 1);
	Point left_top;
	double min = 0, max = 0;
	Mat matResult(csize, IPL_DEPTH_32F);
	
	if (cp)
	{
		Mat scene = input[i]->clone();
		Mat obj = matObj[i]->clone();
		cv::matchTemplate(scene, obj, matResult, CV_TM_CCOEFF_NORMED);

// 		for (uint i = 0; i < input.size(); ++i)
// 		{
// 			Mat scene = input[i]->clone();
// 			Mat obj = matObj[i]->clone();
// 			cv::matchTemplate(scene, obj, matResult, CV_TM_CCOEFF_NORMED);
// 		}
	}
	else
	{
 			cv::matchTemplate(*input[i], *matObj[i], matResult, CV_TM_CCOEFF_NORMED);

// 		for (uint i = 0; i < input.size(); ++i)
// 		{
// 			cv::matchTemplate(*input[i], *matObj[i], matResult, CV_TM_CCOEFF_NORMED);
// 		}
	}

	cv::minMaxLoc(matResult, &min, &max, NULL, &left_top);
	cout << "ok" << endl;
}

void ParallelApplyFoo(vector<Mat*> &input, vector<Mat*> &matObj, size_t n, const bool cp) {
	tbb::parallel_for(size_t(0), n, [&](size_t i) {
		threadfunction(input, matObj, cp, i);
	});
}

void main()
{
	vector<Mat*> inputs, objs;
	for (int i = 0; i < 50; ++i)
	{
		Mat *input = new Mat();
		*input = imread("srcimg.jpg");
		Mat *matObj = new Mat();
		*matObj  = imread("temp.jpg");
		cvtColor(*input, *input, CV_BGR2GRAY);
		cvtColor(*matObj, *matObj, CV_BGR2GRAY);
		inputs.push_back(input);
		objs.push_back(matObj);
	}

	const int t1 = timeGetTime();
	ParallelApplyFoo(inputs, objs, 300, false);
	const int t2 = timeGetTime();
	cout << "======" << endl;
	ParallelApplyFoo(inputs, objs, 300, true);
	const int t3 = timeGetTime();

	cout << "no copy = " << t2 - t1 << endl;
	cout << "copy = " << t3 - t2 << endl;

	for each (auto x in inputs)
		delete x;
	for each (auto x in objs)
		delete x;

	getchar();
}
