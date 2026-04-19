#ifndef FRESHNESS_DETECTOR_H
#define FRESHNESS_DETECTOR_H

#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <map>

class FreshnessDetector {
private:
    torch::nn::Module model;
    torch::Device device;
    std::map<std::string, std::vector<float>> freshness_metrics;
    
public:
    FreshnessDetector(const torch::Device& device = torch::kCPU);
    ~FreshnessDetector() = default;
    
    bool loadModel(const std::string& model_path);
    std::map<std::string, float> detectFreshness(const cv::Mat& image);
    
    // Utility methods
    float calculateColorFreshness(const cv::Mat& image);
    float calculateTextureFreshness(const cv::Mat& image);
    float calculateShapeFreshness(const cv::Mat& image);
    std::string getFreshnessCategory(float score);
    
    // Database methods
    void initializeFreshnessDatabase();
    void addFreshnessData(const std::string& food, float score);
    float getExpectedFreshness(const std::string& food);
};

#endif // FRESHNESS_DETECTOR_H
