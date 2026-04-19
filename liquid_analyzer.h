#ifndef LIQUID_ANALYZER_H
#define LIQUID_ANALYZER_H

#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <map>

class LiquidAnalyzer {
private:
    torch::nn::Module model;
    torch::Device device;
    std::map<std::string, std::vector<float>> liquid_database;
    
public:
    LiquidAnalyzer(const torch::Device& device = torch::kCPU);
    ~LiquidAnalyzer() = default;
    
    bool loadModel(const std::string& model_path);
    std::map<std::string, float> analyzeLiquid(const cv::Mat& image);
    
    // Utility methods
    float detectLiquidLevel(const cv::Mat& image);
    float calculateViscosity(const cv::Mat& image);
    float detectTransparency(const cv::Mat& image);
    float measureColor(const cv::Mat& image);
    std::string getLiquidCategory(float score);
    
    // Database methods
    void initializeLiquidDatabase();
    void addLiquidData(const std::string& liquid, float score);
    float getExpectedLiquid(const std::string& liquid);
};

#endif // LIQUID_ANALYZER_H
