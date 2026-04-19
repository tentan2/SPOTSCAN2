#ifndef NUTRITION_ANALYZER_H
#define NUTRITION_ANALYZER_H

#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <map>

class NutritionAnalyzer {
private:
    torch::nn::Module model;
    torch::Device device;
    std::map<std::string, std::map<std::string, float>> nutrition_database;
    
public:
    NutritionAnalyzer(const torch::Device& device = torch::kCPU);
    ~NutritionAnalyzer() = default;
    
    bool loadModel(const std::string& model_path);
    std::map<std::string, float> analyzeNutrition(const cv::Mat& image);
    
    // Utility methods
    void initializeNutritionDatabase();
    float estimateCalories(const cv::Mat& image);
    float estimateProtein(const cv::Mat& image);
    float estimateCarbs(const cv::Mat& image);
    float estimateFat(const cv::Mat& image);
    
    // Database methods
    void addFoodToDatabase(const std::string& food, const std::map<std::string, float>& nutrition);
    std::map<std::string, float> getNutritionInfo(const std::string& food);
};

#endif // NUTRITION_ANALYZER_H
