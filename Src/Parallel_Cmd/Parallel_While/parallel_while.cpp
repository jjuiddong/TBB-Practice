#include <stdio.h>
#include <windows.h>
#include <mmsystem.h>
#include <iostream>
#include <thread>
#include <vector>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <tbb/tbb.h>
#include <tbb/parallel_while.h>
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

struct sData
{
	Mat *img;
	Mat *obj;
	int cnt;
};

class CItemStream
{
public:
	CItemStream(sData p) : m_p(p) {}
	bool pop_if_present(sData &p)
	{
		m_p.cnt++;
		p = m_p;
		return (m_p.cnt < 10);
	}
public:
	sData m_p;
};

class CBody
{
public:
	void operator()(sData &p) const
	{
		cout << "in " << p.cnt << endl;

		for (int i = 0; i < 10; ++i)
		{
			const cv::Size csize(p.img->cols - p.obj->cols + 1, p.img->rows - p.obj->rows + 1);
			Point left_top;
			double min = 0, max = 0;
			Mat matResult(csize, IPL_DEPTH_32F);
			cv::matchTemplate(*p.img, *p.obj, matResult, CV_TM_CCOEFF_NORMED);
			cv::minMaxLoc(matResult, &min, &max, NULL, &left_top);
		}
 		cout << "ok " << p.cnt << endl;
	}
	typedef sData argument_type;
};

void main()
{
 	Mat img = imread("srcimg.jpg");
 	Mat obj = imread("temp.jpg");
	sData data;
	data.img = &img;
	data.obj = &obj;
	data.cnt = 0;

	const int t1 = timeGetTime();
	tbb::parallel_while<CBody> w;
	CItemStream stream_(data);
	CBody body;
	w.run(stream_, body);
	const int t2 = timeGetTime();
	
	cout << endl << t2 - t1 << endl;

	getchar();
}
