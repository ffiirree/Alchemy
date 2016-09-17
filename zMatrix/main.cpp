#include <iostream>
#include <opencv2\core.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\imgproc\imgproc.hpp>
#include <string>
#include <ctime>  

#include "zcore.h"
#include "zimgproc\zimgproc.h"
#include "zimgproc\transform.h"
#include "zgui\zgui.h"
#include "kinect\KinectSensor.h"

using namespace std;
using namespace z;


int main(int argc, char *argv[])
{
	KinectSensor kinect(FrameTypes_Color);
	cv::Mat depthImg48, depthImg24(424, 512, CV_8U), IRImg24(424, 512, CV_8U);
	char key = 1;
	while (key != ' ') {
		if (kinect.isNewFrameArrived(FrameTypes_Color) == S_OK) {
			if (SUCCEEDED(kinect.update(FrameTypes_Color))) {

				// 显示彩色图像
				imshow("color", kinect.getColorImg());

				//// 显示深度图像，像素位数为48位
				//depthImg48 = kinect.getDepthImg();
				//cout << ((USHORT *)depthImg48.data)[512 / 2 + 512 * (424 / 2)] << endl;
				//depthImg48.convertTo(depthImg24, CV_8U, 255.0f / kinect.getDepthMaxReliableDistance());
				//circle(depthImg24, cv::Point(512 / 2, 424 / 2), 10, cv::Scalar(0, 255, 0));
				//imshow("depth", depthImg24);

				//// 显示红外图像
				//kinect.getInfraImg().convertTo(IRImg24, CV_8U, 1.0 / 256.0);
				//cv::imshow("infra", IRImg24);

				key = cv::waitKey(10);
			}
		}
	}

	return 0;
}