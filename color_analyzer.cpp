#include "color_analyzer.h"
#include <iostream>
#include <algorithm>

ColorAnalyzer::ColorAnalyzer(const torch::Device& device) 
    : device(device) {
    
    // Initialize color database
    color_database["apple"] = {0.8f, 0.2f, 0.1f, 0.6f, 0.4f};
    color_database["banana"] = {0.9f, 0.8f, 0.1f, 0.7f, 0.3f};
    color_database["orange"] = {1.0f, 0.6f, 0.0f, 0.8f, 0.2f};
    color_database["lettuce"] = {0.2f, 0.8f, 0.1f, 0.3f, 0.7f};
    color_database["tomato"] = {0.9f, 0.2f, 0.1f, 0.8f, 0.2f};
    
    initializeColorDatabase();
    std::cout << "ColorAnalyzer initialized" << std::endl;
}

bool ColorAnalyzer::loadModel(const std::string& model_path) {
    try {
        model = torch::jit::load(model_path);
        model->to(device);
        std::cout << "Color analysis model loaded from: " << model_path << std::endl;
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Error loading color analysis model: " << e.what() << std::endl;
        return false;
    }
}

std::map<std::string, float> ColorAnalyzer::analyzeColor(const cv::Mat& image) {
    std::map<std::string, float> results;
    
    // Extract color features
    std::vector<float> features = extractColorFeatures(image);
    float intensity = calculateColorIntensity(image);
    float dominance = detectColorDominance(image);
    
    results["red_intensity"] = features[0];
    results["green_intensity"] = features[1];
    results["blue_intensity"] = features[2];
    results["color_intensity"] = intensity;
    results["color_dominance"] = dominance;
    results["color_category"] = static_cast<float>(getColorCategory(intensity) == "Bright");
    
    return results;
}

std::vector<float> ColorAnalyzer::extractColorFeatures(const cv::Mat& image) {
    // Calculate mean color values
    cv::Scalar mean_color = cv::mean(image);
    
    std::vector<float> features;
    features.push_back(mean_color[0] / 255.0f); // Red
    features.push_back(mean_color[1] / 255.0f); // Green
    features.push_back(mean_color[2] / 255.0f); // Blue
    features.push_back((mean_color[0] + mean_color[1] + mean_color[2]) / 3.0f); // Average
    features.push_back(std::max({mean_color[0], mean_color[1], mean_color[2]})); // Max
    features.push_back(std::min({mean_color[0], mean_color[1], mean_color[2]})); // Min
    
    return features;
}

float ColorAnalyzer::calculateColorIntensity(const cv::Mat& image) {
    // Convert to HSV for better intensity calculation
    cv::Mat hsv_image;
    cv::cvtColor(image, hsv_image, cv::COLOR_BGR2HSV);
    
    // Calculate intensity based on HSV values
    cv::Scalar mean_hsv = cv::mean(hsv_image);
    float hue = mean_hsv[0] / 180.0f;
    float saturation = mean_hsv[1] / 255.0f;
    float value = mean_hsv[2] / 255.0f;
    
    // Combined intensity score
    float intensity = (hue * 0.3f) + (saturation * 0.3f) + (value * 0.4f);
    
    return std::clamp(intensity, 0.0f, 1.0f);
}

float ColorAnalyzer::detectColorDominance(const cv::Mat& image) {
    // Convert to HSV for dominance detection
    cv::Mat hsv_image;
    cv::cvtColor(image, hsv_image, cv::COLOR_BGR2HSV);
    
    // Calculate histogram for each channel
    std::vector<cv::Mat> hsv_channels;
    cv::split(hsv_image, hsv_channels);
    
    int hist_size = 256;
    float hue_range[] = {0, 180};
    float sat_range[] = {0, 256};
    float val_range[] = {0, 256};
    
    cv::Mat hue_hist, sat_hist, val_hist;
    cv::calcHist(&hsv_channels[0], 1, 0, hue_hist, 1, &hist_size, &hue_range[0], true);
    cv::calcHist(&hsv_channels[1], 1, 0, sat_hist, 1, &hist_size, &sat_range[0], true);
    cv::calcHist(&hsv_channels[2], 1, 0, val_hist, 1, &hist_size, &val_range[0], true);
    
    // Find dominant values
    double min_hue, max_hue;
    cv::minMaxLoc(hue_hist, &min_hue, &max_hue);
    
    double min_sat, max_sat;
    cv::minMaxLoc(sat_hist, &min_sat, &max_sat);
    
    double min_val, max_val;
    cv::minMaxLoc(val_hist, &min_val, &max_val);
    
    // Calculate dominance score
    float dominance = (static_cast<float>(max_hue) / 180.0f) * 0.4f + 
                   (static_cast<float>(max_sat) / 256.0f) * 0.3f + 
                   (static_cast<float>(max_val) / 256.0f) * 0.3f;
    
    return std::clamp(dominance, 0.0f, 1.0f);
}

std::string ColorAnalyzer::getColorCategory(float intensity) {
    if (intensity >= 0.8f) {
        return "Very Bright";
    } else if (intensity >= 0.6f) {
        return "Bright";
    } else if (intensity >= 0.4f) {
        return "Moderately Bright";
    } else if (intensity >= 0.2f) {
        return "Dim";
    } else {
        return "Very Dim";
    }
}

void ColorAnalyzer::initializeColorDatabase() {
    // Initialize with common food color data
    std::cout << "Color database initialized" << std::endl;
}

void ColorAnalyzer::addColorData(const std::string& food, const std::vector<float>& features) {
    color_database[food] = features;
}

std::vector<float> ColorAnalyzer::getExpectedColor(const std::string& food) {
    auto it = color_database.find(food);
    if (it != color_database.end()) {
        return it->second;
    }
    return {0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f}; // Default color
}
