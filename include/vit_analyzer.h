#ifndef VIT_ANALYZER_H
#define VIT_ANALYZER_H

#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <map>

class ViTAnalyzer {
private:
    torch::nn::Module model;
    torch::Device device;
    std::string model_name;
    std::vector<std::string> food_classes;
    
    // Image preprocessing
    cv::Size target_size;
    std::vector<float> mean;
    std::vector<float> std;
    
public:
    ViTAnalyzer(const std::string& model_name = "google/vit-base-patch16-224");
    ~ViTAnalyzer() = default;
    
    bool loadModel(const std::string& model_path);
    std::map<std::string, float> analyzeImage(const cv::Mat& image, int top_k = 5);
    std::vector<std::string> getFoodClasses() const;
    
    // Utility methods
    cv::Mat preprocessImage(const cv::Mat& image);
    torch::Tensor imageToTensor(const cv::Mat& image);
    std::vector<std::string> loadFoodClasses();
    
    // Information methods
    std::string getModelInfo() const;
    void fineTuneInfo() const;
};

#endif // VIT_ANALYZER_H
