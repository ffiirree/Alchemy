#include "zfeatures2d.h"
#include "zgui/zgui.h"

void z::DoG(Matrix64f& src, Matrix64f& dst, const Size &size, double g1, double g2)
{
    conv(src, dst, Gassion(size, g1, g1) - Gassion(size, g2, g2));
}



std::vector<z::KeyPoint> z::SIFT::detect(const Matrix64f& InputImage)
{
    gaussianPyramid(InputImage);
    octave(gaussianPyramid_);
    DoGOctave();
    return std::vector<z::KeyPoint>();
}

void z::SIFT::gaussianPyramid(const Matrix64f& InputImage)
{
    gaussianPyramid_.push_back(InputImage);

    for (int i = 1; i < 4; ++i) {
        z::Matrix64f temp;
        z::pyrDown(gaussianPyramid_.back(), temp);
        gaussianPyramid_.push_back(temp);
    }
}

void z::SIFT::octave(const std::vector<z::Matrix64f> &gp)
{
    double sigma0 = 0.3;
    double k = std::sqrt(2);
    double g = 0.5;
    for (size_t i = 0; i < gp.size(); ++i) {
        double sigma;
        if (i > 0)
            sigma = std::sqrt(std::pow(std::pow(k, i) * sigma0, 2) - std::pow(std::pow(k, i - 1) * sigma0, 2));
        else
            sigma = sigma0;
        g = sigma;
        std::vector<Matrix64f> oct;
        for (int j = 0; j < 5; ++j) {
            
            Matrix64f temp;
            conv(gp.at(i), temp, Gassion({ 5, 5 }, g, g));
            g *= k;
            oct.push_back(temp);
        }
        octave_.push_back(oct);
    }
}

void z::SIFT::DoGOctave()
{
    for (size_t i = 0; i < octave_.size(); ++i) {
        std::vector<Matrix64f> dogs;
        for (size_t j = 0; j < octave_.at(i).size() - 1; ++j) {
            auto temp = octave_.at(i).at(j + 1) - octave_.at(i).at(j);
            dogs.push_back(temp);

//            z::imshow(std::string("oct") + std::to_string(i) + std::to_string(j), dogs.back());
        }
        DoGOctave_.push_back(dogs);
    }
}