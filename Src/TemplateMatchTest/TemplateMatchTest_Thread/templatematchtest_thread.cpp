
#include <stdio.h>
#include <windows.h>
#include <mmsystem.h>
#include <iostream>
#include <thread>
#include <vector>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>


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


void threadfunction(Mat &input, Mat &matObj)
{
	const cv::Size csize(input.cols - matObj.cols + 1, input.rows - matObj.rows + 1);
	Point left_top;
	double min = 0, max = 0;
	Mat matResult(csize, IPL_DEPTH_32F);
	cv::matchTemplate(input, matObj, matResult, CV_TM_CCOEFF_NORMED);
	cv::minMaxLoc(matResult, &min, &max, NULL, &left_top);
	cout << "ok" << endl;
}

void main()
{
	Mat input = imread("srcimg.jpg");
	Mat matObj = imread("temp.jpg");

	const int t1 = timeGetTime();
	for (int i = 0; i < 10; ++i)
	{
		vector<std::thread> thrs(10);
		for (int i = 0; i < 10; ++i)
			thrs[i] = thread(threadfunction, input, matObj);
		for (uint i = 0; i < thrs.size(); ++i)
			thrs[i].join();
	}

	const int t2 = timeGetTime();
	cout << t2 - t1 << endl;
}
