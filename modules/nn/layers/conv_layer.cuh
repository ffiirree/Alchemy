namespace alchemy {

template<typename T>
void ConvolutionLayer<T>::ForwardGPU(const vector<Blob<T> *> &input,
                                     const vector<Blob<T> *> &output)
{
    LOG(FATAL) << "Not implement!";
}


template<typename T>
void ConvolutionLayer<T>::BackwardGPU(const vector<Blob<T> *> &input,
                                      const vector<Blob<T> *> &output)
{
    LOG(FATAL) << "Not implement!";
}
}