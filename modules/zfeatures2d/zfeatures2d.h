#ifndef _ZMATRIX_ZFEATURES2D_H
#define _ZMATRIX_ZFEATURES2D_H

#include "zcore\zmatrix.h"
#include "zimgproc\zimgproc.h"

namespace z {

    /**
     * \ KeyPoint
     */
    class KeyPoint {
    public:
        KeyPoint() {  }
        KeyPoint(Point2f pos, float size, float ang = -1, int oct = 0, int ci = -1)
            :pos_(pos), size_(size), angle_(ang), octave_(oct), class_id_(ci){ }

        Point2f pos_{ 0, 0 };
        float size_ = 0.0;
        float angle_ = -1;

        int octave_ = 0;
        int class_id_ = -1;
    };

    void drawKeypoints();

    /**
     * @brief DoG: difference of gaussian
     * @param[in]: src
     * @param[out]: dst
     * @param[in]: size
     * @param[in]: g1, sigma1
     * @param[in]: g2, sigma2
     * @ret None
     */
    void DoG(Matrix64f& src, Matrix64f& dst, const Size &size, double g1, double g2);

    /**
     * \ SIFT
     */
    class SIFT {
    public:
        SIFT() {  }

        std::vector<KeyPoint> detect(const Matrix64f& InputImage);

    private:
        void gaussianPyramid(const Matrix64f& InputImage);
        void octave(const std::vector<z::Matrix64f> &gp);
        void DoGOctave();

        std::vector<Matrix64f> gaussianPyramid_;
        std::vector<std::vector<Matrix64f>> octave_;
        std::vector<std::vector<Matrix64f>> DoGOctave_;
    };
};


#include "zfeatures2d.hpp"
#endif