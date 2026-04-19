#ifndef COLOR_ANALYZER_H
#define COLOR_ANALYZER_H

#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <map>

class ColorAnalyzer {
private:
    torch::nn::Module model;
    torch::Device device;
    std::map<std::string, std::vector<float>> color_database;
    
public:
    ColorAnalyzer(const torch::Device& device = torch::kCPU);
    ~ColorAnalyzer() = default;
    
    bool loadModel(const std::string& model_path);
    std::map<std::string, float> analyzeColor(const cv::Mat& image);
    
    // Utility methods
    std::vector<float> extractColorFeatures(const cv::Mat& image);
    float calculateColorIntensity(const cv::Mat& image);
    float detectColorDominance(const cv::Mat& image);
    std::string getColorCategory(float intensity);
    
    // Database methods
    void initializeColorDatabase();
    void addColorData(const std::string& food, const std::vector<float>& features);
    std::vector<float> getExpectedColor(const std::string& food);
};

#endif // COLOR_ANALYZER_H
