#include <alchemy.h>

using namespace alchemy;

int main()
{

    Tensor<float> a({ 2, 2, 3, 2});
    Tensor<float> b(a.shape());
    Filler<float>::fill(a, NORMAL);
    Filler<float>::fill(b, NORMAL);

    auto c = a + b;


    print_gpu(a.count(), a.gptr());
    print_gpu(b.count(), b.gptr());
    print_gpu(c.count(), c.gptr());
    return 0;
}
