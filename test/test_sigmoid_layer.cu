#include "gtest/gtest.h"
#include "nn/blob.h"
#include "defines.h"
#include "util/filler.h"
#include "nn/layers/sigmoid_layer.h"

namespace alchemy{
using TestTypes = ::testing::Types<GPU_F>;

template <typename Type>
class GPUSigmoidLayerTest : public ::testing::Test {
    using container = Blob<typename Type::TypeA, typename Type::TypeB>;
protected:
    GPUSigmoidLayerTest()
            : input_(new container({2, 3, 4, 5})), output_(new container())
    {
        inputs_.push_back(input_);
        outputs_.push_back(output_);
    }

    ~GPUSigmoidLayerTest() override { delete input_; delete output_; }

    void TestForward()
    {
        Filler<typename Type::TypeA, typename Type::TypeB>::normal_fill(this->input_->size(), this->input_->mutable_data_cptr(), 0.0, 1.0);

        LayerParameter layer_param;

        SigmoidLayer<typename Type::TypeA, typename Type::TypeB> layer(layer_param);
        layer.setup(this->inputs_, this->outputs_);
        layer.Forward(this->inputs_, this->outputs_);


        for(size_t idx = 0; idx < input_->size(); ++idx) {
            typename Type::TypeB expected_value = 1.0 / (1.0 + std::exp(-input_->data_cptr()[idx]));
            EXPECT_NEAR(expected_value, this->output_->data_cptr()[idx], 1e-5);
        }
    }
    void TestBackward()
    {

    }

private:

    container * const input_;
    container * const output_;

    vector<container *> inputs_;
    vector<container *> outputs_;
};

TYPED_TEST_CASE(GPUSigmoidLayerTest, TestTypes);

TYPED_TEST(GPUSigmoidLayerTest, GPUTestSimoidLayerForward){
    this->TestForward();
}
}

