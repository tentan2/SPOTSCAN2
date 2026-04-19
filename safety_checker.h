#ifndef SAFETY_CHECKER_H
#define SAFETY_CHECKER_H

#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <map>

class SafetyChecker {
private:
    torch::nn::Module model;
    torch::Device device;
    std::map<std::string, std::vector<float>> safety_database;
    
public:
    SafetyChecker(const torch::Device& device = torch::kCPU);
    ~SafetyChecker() = default;
    
    bool loadModel(const std::string& model_path);
    std::map<std::string, float> checkSafety(const cv::Mat& image);
    
    // Utility methods
    float detectMold(const cv::Mat& image);
    float detectSpoilage(const cv::Mat& image);
    float detectContamination(const cv::Mat& image);
    float checkColorSafety(const cv::Mat& image);
    std::string getSafetyCategory(float score);
    
    // Database methods
    void initializeSafetyDatabase();
    void addSafetyData(const std::string& food, float score);
    float getExpectedSafety(const std::string& food);
};

#endif // SAFETY_CHECKER_H
