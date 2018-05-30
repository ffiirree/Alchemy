#ifndef ALCHEMY_GTEST_DEFINES_H
#define ALCHEMY_GTEST_DEFINES_H

namespace alchemy {

template <typename TA, typename TB>
struct TypeDefinitions {
    using TypeA = TA;
    using TypeB = TB;
};

#define CPU_S       TypeDefinitions<CPU, float>
#define CPU_D       TypeDefinitions<CPU, double>
#define GPU_S       TypeDefinitions<GPU, float>
#define GPU_D       TypeDefinitions<GPU, double>

#define XPU_S       CPU_S, GPU_S
#define XPU_D       CPU_D, GPU_D

#define CPU_F       CPU_S, CPU_D
#define GPU_F       GPU_S, GPU_D
#define XPU_F       XPU_S, XPU_D
}

#endif //! ALCHEMY_GTEST_DEFINES_H
