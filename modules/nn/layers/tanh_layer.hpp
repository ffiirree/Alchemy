namespace alchemy {
template <typename Device, typename T>
void TanhLayer<Device, T>::setup(const vector<container *> &input,
                         const vector<container *> &output)
{
    output[0]->reshape(input[0]->shape());
}

template <typename Device, typename T>
void TanhLayer<Device, T>::ForwardCPU(const vector<container *> &input,
                              const vector<container *> &output)
{
    auto input_data = input[0]->data_cptr();
    auto count = input[0]->size();
    auto output_data = output[0]->mutable_data_cptr();

    for(size_t i = 0; i < count; ++i) {
        output_data[i] = std::tanh(input_data[i]);
    }
}

template <typename Device, typename T>
void TanhLayer<Device, T>::BackwardCPU(const vector<container *> &input,
                               const vector<container *> &output)
{
    auto count = input[0]->size();
    auto output_data = output[0]->data_cptr();
    auto input_diff = input[0]->mutable_diff_cptr();
    auto output_diff = output[0]->diff_cptr();

    for(size_t i = 0; i < count; ++i) {
        auto tx = output_data[i];
        input_diff[i] = output_diff[i] *(1 - tx * tx);
    }
}
}