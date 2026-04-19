#ifndef OCR_ANALYZER_H
#define OCR_ANALYZER_H

#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <map>

class OCRAnalyzer {
private:
    torch::nn::Module model;
    torch::Device device;
    std::map<std::string, std::string> ocr_database;
    
public:
    OCRAnalyzer(const torch::Device& device = torch::kCPU);
    ~OCRAnalyzer() = default;
    
    bool loadModel(const std::string& model_path);
    std::map<std::string, std::string> analyzeOCR(const cv::Mat& image);
    
    // Utility methods
    std::string extractText(const cv::Mat& image);
    std::vector<std::string> detectIngredients(const std::string& text);
    std::vector<std::string> detectNutritionInfo(const std::string& text);
    std::vector<std::string> detectAllergens(const std::string& text);
    
    // Database methods
    void initializeOCRDatabase();
    void addOCRData(const std::string& food, const std::string& info);
    std::string getExpectedOCR(const std::string& food);
    
    // Text processing utilities
    std::string cleanText(const std::string& text);
    std::vector<std::string> splitText(const std::string& text);
};

#endif // OCR_ANALYZER_H
