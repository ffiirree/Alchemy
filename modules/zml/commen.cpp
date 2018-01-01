#include "commen.hpp"
#include "boost/thread.hpp"

namespace z {
// 保证线程安全
static boost::thread_specific_ptr<Global> thread_instance_;

Global& Global::Instance() {
    if (!thread_instance_.get()) {
        thread_instance_.reset(new Global());
    }
    return *(thread_instance_.get());
}

Global::Global()
{

}

Global::~Global()
{
}

}