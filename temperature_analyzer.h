#ifndef TEMPERATURE_ANALYZER_H
#define TEMPERATURE_ANALYZER_H

#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <map>

class TemperatureAnalyzer {
private:
    torch::nn::Module model;
    torch::Device device;
    std::map<std::string, float> temperature_database;
    
public:
    TemperatureAnalyzer(const torch::Device& device = torch::kCPU);
    ~TemperatureAnalyzer() = default;
    
    bool loadModel(const std::string& model_path);
    std::map<std::string, float> analyzeTemperature(const cv::Mat& image);
    
    // Utility methods
    float estimateTemperature(const cv::Mat& image);
    float calculateColorTemperature(const cv::Mat& image);
    float calculateSteamTemperature(const cv::Mat& image);
    std::string getTemperatureCategory(float temp_celsius);
    
    // Database methods
    void initializeTemperatureDatabase();
    void addTemperatureData(const std::string& food, float temp);
    float getExpectedTemperature(const std::string& food);
    
    // Conversion utilities
    float celsiusToFahrenheit(float celsius);
    std::string formatTemperature(float celsius);
};

#endif // TEMPERATURE_ANALYZER_H
