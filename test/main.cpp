#include "zmatrix.h"

int main()
{
    z::VideoCapture camera(0);

    if(camera.isOpened()) {
        z::Matrix frame;

        while(z::waitKey(10) != 'q') {
            camera >> frame;

            z::imshow("frame", frame);
        }
    }

    return 0;
}