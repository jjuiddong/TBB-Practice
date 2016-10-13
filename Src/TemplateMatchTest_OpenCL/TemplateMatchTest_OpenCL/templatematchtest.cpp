
#include <stdio.h>
#include <windows.h>
#include <mmsystem.h>
#include <iostream>
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

void main()
{
	Mat input = imread("srcimg.jpg");
	Mat matObj = imread("temp.jpg");

	UMat uinput;
	uinput = input.getUMat(cv::ACCESS_READ);
	UMat uobj;
	uobj = matObj.getUMat(cv::ACCESS_READ);

	const cv::Size csize(input.cols - matObj.cols + 1, input.rows - matObj.rows + 1);
	Point left_top;
	double min = 0, max = 0;
	UMat matResult(csize, IPL_DEPTH_32F);

	const int t1 = timeGetTime();
	for (int i = 0; i < 100; ++i)
	{
		cv::matchTemplate(uinput, uobj, matResult, CV_TM_CCOEFF_NORMED);
		cv::minMaxLoc(matResult, &min, &max, NULL, &left_top);
		cout << "ok" << endl;
	}
	const int t2 = timeGetTime();
	cout << t2 - t1 << endl;
}
