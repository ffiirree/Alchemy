#include <alchemy.h>

using namespace alchemy;

int main()
{

    Tensor<GPU, float> a({ 2, 2, 3, 2});
    Tensor<GPU, float> b(a.shape());
    Filler<GPU, float>::fill(a, NORMAL);
    Filler<GPU, float>::fill(b, NORMAL);

    auto c = a + b;


    print_gpu(a.size(), a.gptr());
    print_gpu(b.size(), b.gptr());
    print_gpu(c.size(), c.gptr());
    return 0;
}
