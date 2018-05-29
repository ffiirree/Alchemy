namespace alchemy {

template <typename Device, typename T>
void ReLuLayer<Device, T>::setup(const vector<container *> &input,
                         const vector<container *> &output)
{
    output[0]->reshape(input[0]->shape());
}

template <typename Device, typename T>
void ReLuLayer<Device, T>::ForwardCPU(const vector<container *> &input,
                              const vector<container *> &output)
{
    auto count = input[0]->size();
    auto input_data = input[0]->data_cptr();
    auto output_data = output[0]->mutable_data_cptr();
    auto alpha = relu_param_.alpha();

    /// max(0, z) + alpha * min(0, z)
    for(size_t i = 0; i < count; ++i) {
        output_data[i] = std::max(input_data[i], (T)0.0) + alpha * std::min(input_data[i], (T)0.0);
    }
}

template <typename Device, typename T>
void ReLuLayer<Device, T>::BackwardCPU(const vector<container *> &input,
                               const vector<container *> &output)
{
    auto count = input[0]->size();
    auto input_data = input[0]->data_cptr();
    auto input_diff = input[0]->mutable_diff_cptr();
    auto output_diff = output[0]->diff_cptr();
    auto alpha = relu_param_.alpha();

    for(size_t i = 0; i < count; ++i) {
        input_diff[i] = output_diff[i] * ((input_data[i] > 0) ? 1 : alpha);
    }
}
}