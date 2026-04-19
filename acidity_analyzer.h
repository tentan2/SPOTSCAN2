#ifndef ACIDITY_ANALYZER_H
#define ACIDITY_ANALYZER_H

#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <map>

class AcidityAnalyzer {
private:
    torch::nn::Module model;
    torch::Device device;
    std::map<std::string, std::vector<float>> acidity_database;
    
public:
    AcidityAnalyzer(const torch::Device& device = torch::kCPU);
    ~AcidityAnalyzer() = default;
    
    bool loadModel(const std::string& model_path);
    std::map<std::string, float> analyzeAcidity(const cv::Mat& image);
    
    // Utility methods
    float detectAcidity(const cv::Mat& image);
    float calculateColorBasedAcidity(const cv::Mat& image);
    float measurePh(const cv::Mat& image);
    std::string getAcidityCategory(float score);
    
    // Database methods
    void initializeAcidityDatabase();
    void addAcidityData(const std::string& food, float score);
    float getExpectedAcidity(const std::string& food);
};

#endif // ACIDITY_ANALYZER_H
