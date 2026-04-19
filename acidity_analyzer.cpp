#include "acidity_analyzer.h"
#include <iostream>
#include <algorithm>

AcidityAnalyzer::AcidityAnalyzer(const torch::Device& device) 
    : device(device) {
    
    // Initialize acidity database
    acidity_database["citrus"] = {3.0f, 2.5f, 2.0f, 1.5f, 1.0f};
    acidity_database["tomato"] = {4.0f, 3.5f, 3.0f, 2.5f, 2.0f};
    acidity_database["dairy"] = {6.0f, 5.5f, 5.0f, 4.5f, 4.0f};
    acidity_database["vinegar"] = {2.5f, 2.0f, 1.5f, 1.0f, 0.5f};
    acidity_database["soda"] = {2.8f, 2.3f, 1.8f, 1.3f, 0.8f};
    
    initializeAcidityDatabase();
    std::cout << "AcidityAnalyzer initialized" << std::endl;
}

bool AcidityAnalyzer::loadModel(const std::string& model_path) {
    try {
        model = torch::jit::load(model_path);
        model->to(device);
        std::cout << "Acidity analysis model loaded from: " << model_path << std::endl;
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Error loading acidity analysis model: " << e.what() << std::endl;
        return false;
    }
}

std::map<std::string, float> AcidityAnalyzer::analyzeAcidity(const cv::Mat& image) {
    std::map<std::string, float> results;
    
    // Calculate acidity metrics
    float acidity_score = detectAcidity(image);
    float color_acidity = calculateColorBasedAcidity(image);
    float ph_level = measurePh(image);
    
    // Combine scores
    float overall_acidity = (acidity_score * 0.4f) + (color_acidity * 0.3f) + (ph_level * 0.3f);
    
    results["acidity_score"] = acidity_score;
    results["color_acidity"] = color_acidity;
    results["ph_level"] = ph_level;
    results["overall_acidity"] = overall_acidity;
    results["acidity_category"] = static_cast<float>(getAcidityCategory(overall_acidity) == "Low");
    
    return results;
}

float AcidityAnalyzer::detectAcidity(const cv::Mat& image) {
    // Convert to HSV for acidity detection
    cv::Mat hsv_image;
    cv::cvtColor(image, hsv_image, cv::COLOR_BGR2HSV);
    
    // Calculate acidity based on HSV values
    cv::Scalar mean_color = cv::mean(hsv_image);
    float hue = mean_color[0];
    float saturation = mean_color[1];
    
    // Acidity estimation based on color
    float acidity_score = 0.0f;
    
    if (hue >= 0 && hue < 30) {
        // Red to yellow range (acidic)
        acidity_score += saturation / 255.0f * 0.8f;
    } else if (hue >= 120 && hue < 180) {
        // Green to cyan range (less acidic)
        acidity_score += saturation / 255.0f * 0.3f;
    }
    
    return std::clamp(acidity_score, 0.0f, 1.0f);
}

float AcidityAnalyzer::calculateColorBasedAcidity(const cv::Mat& image) {
    // Calculate color metrics for acidity
    cv::Scalar mean_color = cv::mean(image);
    float red = mean_color[0];
    float green = mean_color[1];
    float blue = mean_color[2];
    
    // Color-based acidity estimation
    float red_ratio = red / (red + green + blue + 1.0f);
    float green_ratio = green / (red + green + blue + 1.0f);
    
    float color_acidity = 0.0f;
    
    if (red_ratio > 0.5f) {
        color_acidity += 0.7f; // Red foods often acidic
    } else if (green_ratio > 0.4f) {
        color_acidity += 0.4f; // Green foods moderately acidic
    } else {
        color_acidity += 0.2f; // Blue foods less acidic
    }
    
    return std::clamp(color_acidity, 0.0f, 1.0f);
}

float AcidityAnalyzer::measurePh(const cv::Mat& image) {
    // Convert to grayscale for pH measurement
    cv::Mat gray_image;
    cv::cvtColor(image, gray_image, cv::COLOR_BGR2GRAY);
    
    // Calculate pH based on color intensity
    cv::Scalar mean_intensity = cv::mean(gray_image);
    float intensity = mean_intensity[0] / 255.0f;
    
    // pH estimation (lower intensity = more acidic)
    float ph_level = 7.0f - (intensity * 4.0f);
    
    return std::clamp(ph_level, 0.0f, 14.0f);
}

std::string AcidityAnalyzer::getAcidityCategory(float score) {
    if (score >= 0.8f) {
        return "Highly Acidic";
    } else if (score >= 0.6f) {
        return "Acidic";
    } else if (score >= 0.4f) {
        return "Moderately Acidic";
    } else if (score >= 0.2f) {
        return "Slightly Acidic";
    } else {
        return "Neutral";
    }
}

void AcidityAnalyzer::initializeAcidityDatabase() {
    // Initialize with common food acidity data
    std::cout << "Acidity database initialized" << std::endl;
}

void AcidityAnalyzer::addAcidityData(const std::string& food, float score) {
    acidity_database[food] = {score, score * 0.9f, score * 0.8f, score * 0.7f, score * 0.6f};
}

float AcidityAnalyzer::getExpectedAcidity(const std::string& food) {
    auto it = acidity_database.find(food);
    if (it != acidity_database.end()) {
        return it->second[0]; // Return first day's acidity
    }
    return 3.0f; // Default acidity
}
