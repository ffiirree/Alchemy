namespace alchemy {

template <typename T>
__global__ void max_pooling(int size,
                            int batch_size, int channels,
                            int input_rows, int input_cols,
                            int output_rows, int output_cols,
                            uint32_t stride, size_t ksize,
                            const T* input_data, T* output_data, size_t * max_idx)
{
    for (int output_idx = blockIdx.x * blockDim.x + threadIdx.x; output_idx < size; output_idx += blockDim.x * gridDim.x) {
        auto batch_idx = output_idx / output_cols / output_rows / channels;
        auto ch_idx = (output_idx / output_cols / output_rows) % channels;
        auto row_idx = (output_idx / output_cols) % output_rows;
        auto col_idx = output_idx % output_cols;

        size_t input_row_start = row_idx * stride;
        size_t input_col_start = col_idx * stride;
        size_t input_row_end = input_row_start + ksize;
        size_t input_col_end = input_col_start + ksize;

        auto input = input_data + (batch_idx * channels + ch_idx) * input_cols * input_rows;
        for(auto row = input_row_start; row < input_row_end; ++row) {
            for(auto col = input_col_start; col < input_col_end; ++col) {
                size_t input_idx = row * input_cols + col;

                if(input[input_idx] > output_data[output_idx]) {
                    output_data[output_idx] = input[input_idx];
                    max_idx[output_idx] = input_idx;
                }
            }
        }
    }
}

template <typename T>
__global__ void max_pooling_backward(int size,
                                     int batch_size, int channels,
                                     int input_rows, int input_cols,
                                     int output_rows, int output_cols,
                                     uint32_t stride, size_t ksize,
                                     const T* output_diff, const size_t * max_idx, T* input_diff)
{
    for (int output_idx = blockIdx.x * blockDim.x + threadIdx.x; output_idx < size; output_idx += blockDim.x * gridDim.x) {
        auto batch_idx = output_idx / output_cols / output_rows / channels;
        auto ch_idx = (output_idx / output_cols / output_rows) % channels;
//        auto row_idx = (output_idx / output_cols) % output_rows;
//        auto col_idx = output_idx % output_cols;

        auto input = input_diff + (batch_idx * channels + ch_idx) * input_cols * input_rows;
        input[max_idx[output_idx]] = output_diff[output_idx];
    }
}

template <typename Device, typename T>
void PoolingLayer<Device, T>::ForwardGPU(const vector<container *> &input,
                                 const vector<container *> &output)
{
    auto count = output[0]->size();
    auto input_data = input[0]->data_gptr();
    auto output_data = output[0]->mutable_data_gptr();
    auto max_idx = max_idx_.mutable_gptr();

    vector_set_gpu(output[0]->size(), -std::numeric_limits<T>::max(), output_data);

    switch(pooling_param_.type()) {
        case MAX: max_pooling<<<CUDA_BLOCK_NUM(count), CUDA_THREAD_NUM>>>(
                            count,
                            input[0]->shape(0), input[0]->shape(1),
                            input[0]->shape(2), input[0]->shape(3),
                            output[0]->shape(2), output[0]->shape(3),
                            pooling_param_.stride(), pooling_param_.kernel_size(),
                            input_data, output_data, max_idx
            );
        break;

        case AVERAGE: LOG(FATAL) << "Not implement!"; break;
        default: LOG(FATAL) << "Unknown Pooling type!"; break;
    }
}

template <typename Device, typename T>
void PoolingLayer<Device, T>::BackwardGPU(const vector<container *> &input,
                                  const vector<container *> &output)
{
    auto count = output[0]->size();
    auto input_diff = input[0]->mutable_diff_gptr();
    auto output_diff = output[0]->mutable_diff_gptr();
    auto max_idx = max_idx_.gptr();

    vector_set_gpu(input[0]->size(), (T)0.0, input_diff);

    switch(pooling_param_.type()) {
        case MAX:
            max_pooling_backward<<<CUDA_BLOCK_NUM(count), CUDA_THREAD_NUM>>>(
                    count,
                    input[0]->shape(0), input[0]->shape(1),
                    input[0]->shape(2), input[0]->shape(3),
                    output[0]->shape(2), output[0]->shape(3),
                    pooling_param_.stride(), pooling_param_.kernel_size(),
                    output_diff, max_idx, input_diff
            );
            break;

        case AVERAGE: LOG(FATAL) << "Not implement!"; break;
        default: LOG(FATAL) << "Unknown Pooling type!"; break;
    }
}
}