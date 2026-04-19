#ifndef TEXTURE_ANALYZER_H
#define TEXTURE_ANALYZER_H

#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <map>

class TextureAnalyzer {
private:
    torch::nn::Module model;
    torch::Device device;
    std::map<std::string, std::vector<float>> texture_database;
    
public:
    TextureAnalyzer(const torch::Device& device = torch::kCPU);
    ~TextureAnalyzer() = default;
    
    bool loadModel(const std::string& model_path);
    std::map<std::string, float> analyzeTexture(const cv::Mat& image);
    
    // Utility methods
    float calculateSmoothness(const cv::Mat& image);
    float calculateRoughness(const cv::Mat& image);
    float calculateHardness(const cv::Mat& image);
    std::string getTextureCategory(float score);
    
    // Database methods
    void initializeTextureDatabase();
    void addTextureData(const std::string& food, float score);
    float getExpectedTexture(const std::string& food);
};

#endif // TEXTURE_ANALYZER_H
