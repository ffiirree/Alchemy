#include <iostream>
#include <opencv2\core.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\imgproc\imgproc.hpp>

#include "zcore.h"
#include "zimgproc\zimgproc.h"
#include "zimgproc\transform.h"
#include "zgui\zgui.h"
#include "kinect\KinectSensor.h"

using namespace std;
using namespace z;


int main(int argc, char *argv[])
{
	KinectSensor kinect(FrameTypes_All);
	cv::Mat depthImg16, depthImg8(424, 512, CV_8U), IRImg8(424, 512, CV_8U);
	char key = 0;

	while (key != ' ') {
		if (kinect.isNewFrameArrived(FrameTypes_All) == S_OK) {
			if (SUCCEEDED(kinect.update(FrameTypes_All))) {

				// 显示彩色图像
				imshow("color", kinect.getColorImg());

				// 显示深度图像，像素位数为16位
				depthImg16 = kinect.getDepthImg();
				cout << ((USHORT *)depthImg16.data)[512 / 2 + 512 * (424 / 2)] << endl;
				depthImg16.convertTo(depthImg8, CV_8U, 255.0f / kinect.getDepthMaxReliableDistance());
				circle(depthImg8, cv::Point(512 / 2, 424 / 2), 10, cv::Scalar(0, 255, 0));
				cv::imshow("depth", depthImg8);

				// 显示红外图像
				kinect.getInfraImg().convertTo(IRImg8, CV_8U, 1.0 / 256.0);
				cv::imshow("infra", IRImg8);

				key = cv::waitKey(10);
			}
		}
	}

	return 0;
}