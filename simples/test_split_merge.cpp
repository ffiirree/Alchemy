#include <alchemy.h>

int main()
{
    auto image = alchemy::imread("simples.jpeg");
    alchemy::imshow("原图", image);

    // 通道分离
    std::vector<alchemy::Matrix> mv;
    alchemy::spilt(image, mv);

    alchemy::imshow("B", mv[0]);
    alchemy::imshow("G", mv[1]);
    alchemy::imshow("R", mv[2]);

    // 合成
    alchemy::Matrix merge_image;
    alchemy::merge(mv, merge_image);
    alchemy::imshow("再次合成", merge_image);

    alchemy::waitKey(0);

    return 0;
}