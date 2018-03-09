#include <alchemy.h>

int main()
{
    alchemy::VideoCapture video(0);

    if(video.isOpened()) {
        alchemy::Matrix frame;

        do {
            video >> frame;
            alchemy::imshow("frame", frame);
        } while(alchemy::waitKey(10) != 'q');
    }

    return 0;
}