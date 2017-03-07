#include <iostream>
#include <opencv2\opencv.hpp>
#include <fstream>

#include "zmatrix.h"

int main(int argc, char *argv[])
{
    auto test = z::imread("test.jpeg");
    z::Matrix gray;
    
    z::cvtColor(test, gray, BGR2GRAY);
    z::Matrix64f gray64 = gray;

    z::SIFT detector;
    detector.detect(gray64);
    cv::waitKey(0);
    //z::Matrix gray;
    //z::cvtColor(test, gray, BGR2GRAY);

    //z::Matrix64f gray64;

    //z::Matrix64f dst1, dst2;
    //gray64 = gray;
    //z::DoG(gray64, dst1, { 5, 5 }, 1.414 * 1, 1);

    //// 
    //cv::Mat cvt = cv::imread("test.jpeg");
    //cv::Mat cvgray;
    //cv::cvtColor(cvt, cvgray, CV_BGR2GRAY);

    //cv::Mat_<double> cvgray64 = cvgray;
    //cv::Mat cvdst1, cvdst2, cvres;
    //cv::GaussianBlur(cvgray64, cvdst1, cv::Size(5, 5),1.414, 1.414);
    //cv::GaussianBlur(cvgray64, cvdst2, cv::Size(5, 5), 1, 1);

    //cvres = cvdst1 - cvdst2;

    return 0;
}