#include "gtest/gtest.h"
#include "core/tensor.h"
#include "defines.h"

namespace alchemy {

using TestTypes = ::testing::Types<XPU_F>;

template <typename T>
class TensorTest : public ::testing::Test {
protected:
    TensorTest()
            : tensor_(new Tensor<typename T::TypeA, typename T::TypeB>()),
              tensor_preshaped_(new Tensor<typename T::TypeA, typename T::TypeB>({ 2, 3, 4, 5})) { }

    ~TensorTest() override { delete tensor_, delete tensor_preshaped_; }
    Tensor<typename T::TypeA, typename T::TypeB> * const tensor_;
    Tensor<typename T::TypeA, typename T::TypeB> * const tensor_preshaped_;
};

TYPED_TEST_CASE(TensorTest, TestTypes);

TYPED_TEST(TensorTest, TestInitialization){
    EXPECT_TRUE(this->tensor_);
    EXPECT_TRUE(this->tensor_preshaped_);
    EXPECT_EQ(this->tensor_preshaped_->shape(0), 2);
    EXPECT_EQ(this->tensor_preshaped_->shape(1), 3);
    EXPECT_EQ(this->tensor_preshaped_->shape(2), 4);
    EXPECT_EQ(this->tensor_preshaped_->shape(3), 5);
    EXPECT_EQ(this->tensor_preshaped_->size(), 120);
    EXPECT_EQ(this->tensor_->size(), 0);
}

TYPED_TEST(TensorTest, TestPointers) {
    EXPECT_TRUE(this->tensor_preshaped_->cptr());
    EXPECT_TRUE(this->tensor_preshaped_->gptr());
    EXPECT_TRUE(this->tensor_preshaped_->mutable_cptr());
    EXPECT_TRUE(this->tensor_preshaped_->mutable_gptr());
}

TYPED_TEST(TensorTest, TestReset) {
    this->tensor_->reset({  5, 4, 3 ,2 });
    EXPECT_EQ(this->tensor_->shape(0), 5);
    EXPECT_EQ(this->tensor_->shape(1), 4);
    EXPECT_EQ(this->tensor_->shape(2), 3);
    EXPECT_EQ(this->tensor_->shape(3), 2);
    EXPECT_EQ(this->tensor_->size(), 120);
}
}

