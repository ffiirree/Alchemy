namespace alchemy {

template <typename Device, typename T>
void ConvolutionLayer<Device, T>::ForwardGPU(const vector<container *> &input,
                                     const vector<container *> &output)
{
    LOG(FATAL) << "Not implement!";
}


template <typename Device, typename T>
void ConvolutionLayer<Device, T>::BackwardGPU(const vector<container *> &input,
                                      const vector<container *> &output)
{
    LOG(FATAL) << "Not implement!";
}
}