// templateMatch + CUDA

#include <stdio.h>
#include <windows.h>
#include <mmsystem.h>
#include <iostream>
#include <string>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/cuda/cudaimgproc.hpp>

#ifdef _DEBUG
	#pragma comment(lib, "opencv_core310d.lib")
	#pragma comment(lib, "opencv_imgcodecs310d.lib")
	#pragma comment(lib, "opencv_highgui310d.lib")
	#pragma comment(lib, "opencv_cudaimgproc310d.lib")
#else
	#pragma comment(lib, "opencv_core310.lib")
	#pragma comment(lib, "opencv_imgcodecs310.lib")
	#pragma comment(lib, "opencv_highgui310.lib")
	#pragma comment(lib, "opencv_cudaimgproc310.lib")
#endif

#pragma comment(lib, "winmm.lib")


using namespace std;
using namespace cv;

void main()
{
	Mat input = imread("srcimg.jpg");
	Mat matObj = imread("temp.jpg");
	cv::cuda::GpuMat cinput(input);
	cv::cuda::GpuMat cobj(matObj);

	const cv::Size csize(input.cols - matObj.cols + 1, input.rows - matObj.rows + 1);
	Point left_top;
	double min = 0, max = 0;
	cv::cuda::GpuMat cmatResult(csize, IPL_DEPTH_32F);

	Ptr<cuda::TemplateMatching> tmp = cuda::createTemplateMatching(input.type(), CV_TM_CCOEFF_NORMED);
	const int t1 = timeGetTime();
	for (int i = 0; i < 100; ++i)
	{
		tmp->match(cinput, cobj, cmatResult);
		Mat matResult;
		cmatResult.download(matResult);
		cv::minMaxLoc(matResult, &min, &max, NULL, &left_top);
		cout << "ok" << endl;
	}
	const int t2 = timeGetTime();
	cout << t2 - t1 << endl;

	getchar();
}
