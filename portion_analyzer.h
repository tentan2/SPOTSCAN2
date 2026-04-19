#ifndef PORTION_ANALYZER_H
#define PORTION_ANALYZER_H

#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <map>

class PortionAnalyzer {
private:
    torch::nn::Module model;
    torch::Device device;
    std::map<std::string, float> portion_database;
    
public:
    PortionAnalyzer(const torch::Device& device = torch::kCPU);
    ~PortionAnalyzer() = default;
    
    bool loadModel(const std::string& model_path);
    std::map<std::string, float> analyzePortion(const cv::Mat& image);
    
    // Utility methods
    float estimateVolume(const cv::Mat& image);
    float estimateWeight(const cv::Mat& image, float volume);
    float calculateArea(const cv::Mat& image);
    float calculateCalories(const cv::Mat& image, float weight);
    
    // Database methods
    void initializePortionDatabase();
    void addPortionData(const std::string& food, float portion);
    float getExpectedPortion(const std::string& food);
    
    // Conversion utilities
    std::string formatPortion(float grams);
    std::string formatServings(float servings);
};

#endif // PORTION_ANALYZER_H
