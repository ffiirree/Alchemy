#include "zmatrix.h"
#include "opencv2/opencv.hpp"

int main()
{
//    z::VideoCapture camera(0);
//
//    if(camera.isOpened()) {
//        z::Matrix frame;
//
//        while(z::waitKey(10) != 'q') {
//            camera >> frame;
//
//            z::imshow("frame", frame);
//        }
//    }

    cv::Mat a(5, 5, CV_8UC3, cv::Scalar{2, 2, 5});
    std::cout << a << std::endl;

    std::cout << a.dot(a) << std::endl;

    z::Matrix b(5, 5, 2, {2, 2, 5});
    std::cout << b << std::endl;

    std::cout << b.dot(b) << std::endl;

    return 0;
}