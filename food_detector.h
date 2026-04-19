#ifndef FOOD_DETECTOR_H
#define FOOD_DETECTOR_H

#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

class FoodDetector {
private:
    torch::nn::Module model;
    torch::Device device;
    std::vector<std::string> class_names;
    
public:
    FoodDetector(const torch::Device& device = torch::kCPU);
    ~FoodDetector() = default;
    
    bool loadModel(const std::string& model_path);
    std::pair<std::string, float> detectFood(const cv::Mat& image);
    std::vector<std::pair<std::string, float>> detectMultipleFoods(const cv::Mat& image);
    
    // Utility methods
    void setClassNames(const std::vector<std::string>& names);
    std::vector<std::string> getClassNames() const;
};

#endif // FOOD_DETECTOR_H
