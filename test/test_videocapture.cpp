#include <alchemy.h>

int main()
{
    alchemy::VideoCapture camera(0);

    if(camera.isOpened()) {
        alchemy::Matrix frame;

        while(alchemy::waitKey(10) != 'q') {
            camera >> frame;

            alchemy::imshow("frame", frame);
        }
    }

    return 0;
}