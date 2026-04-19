#include "safety_checker.h"
#include <iostream>
#include <algorithm>

SafetyChecker::SafetyChecker(const torch::Device& device) 
    : device(device) {
    
    // Initialize safety database
    safety_database["mold_risk"] = {0.9f, 0.8f, 0.7f, 0.6f, 0.5f};
    safety_database["spoilage_risk"] = {0.8f, 0.7f, 0.6f, 0.4f, 0.3f};
    safety_database["contamination_risk"] = {0.7f, 0.6f, 0.5f, 0.3f, 0.2f};
    safety_database["color_safety"] = {0.9f, 0.8f, 0.7f, 0.6f, 0.5f};
    
    initializeSafetyDatabase();
    std::cout << "SafetyChecker initialized" << std::endl;
}

bool SafetyChecker::loadModel(const std::string& model_path) {
    try {
        model = torch::jit::load(model_path);
        model->to(device);
        std::cout << "Safety checking model loaded from: " << model_path << std::endl;
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Error loading safety checking model: " << e.what() << std::endl;
        return false;
    }
}

std::map<std::string, float> SafetyChecker::checkSafety(const cv::Mat& image) {
    std::map<std::string, float> results;
    
    // Calculate safety metrics
    float mold_score = detectMold(image);
    float spoilage_score = detectSpoilage(image);
    float contamination_score = detectContamination(image);
    float color_safety_score = checkColorSafety(image);
    
    // Combine scores (weighted average)
    float overall_safety = (mold_score * 0.3f) + (spoilage_score * 0.3f) + 
                         (contamination_score * 0.2f) + (color_safety_score * 0.2f);
    
    results["mold_risk"] = mold_score;
    results["spoilage_risk"] = spoilage_score;
    results["contamination_risk"] = contamination_score;
    results["color_safety"] = color_safety_score;
    results["overall_safety"] = overall_safety;
    results["safety_category"] = static_cast<float>(getSafetyCategory(overall_safety) == "Safe");
    
    return results;
}

float SafetyChecker::detectMold(const cv::Mat& image) {
    // Convert to HSV for mold detection
    cv::Mat hsv_image;
    cv::cvtColor(image, hsv_image, cv::COLOR_BGR2HSV);
    
    // Mold typically appears as green/gray patches
    cv::Mat lower_green = cv::Mat(hsv_image.size(), CV_8UC3, cv::Scalar(35, 40, 40));
    cv::Mat upper_green = cv::Mat(hsv_image.size(), CV_8UC3, cv::Scalar(85, 255, 255));
    cv::Mat green_mask;
    cv::inRange(hsv_image, lower_green, upper_green, green_mask);
    
    // Calculate mold risk (lower green pixels = higher risk)
    int green_pixels = cv::countNonZero(green_mask);
    int total_pixels = green_mask.rows * green_mask.cols;
    float green_ratio = static_cast<float>(green_pixels) / total_pixels;
    
    float mold_risk = 1.0f - green_ratio; // Higher ratio = lower risk
    return std::clamp(mold_risk, 0.0f, 1.0f);
}

float SafetyChecker::detectSpoilage(const cv::Mat& image) {
    // Convert to grayscale for spoilage detection
    cv::Mat gray_image;
    cv::cvtColor(image, gray_image, cv::COLOR_BGR2GRAY);
    
    // Calculate texture changes indicating spoilage
    cv::Mat laplacian;
    cv::Laplacian(gray_image, laplacian, CV_64F);
    
    cv::Scalar mean, stddev;
    cv::meanStdDev(laplacian, mean, stddev);
    
    // Higher standard deviation = more texture changes = possible spoilage
    float spoilage_risk = std::min(stddev[0] / 50.0f, 1.0f);
    return spoilage_risk;
}

float SafetyChecker::detectContamination(const cv::Mat& image) {
    // Convert to grayscale for contamination detection
    cv::Mat gray_image;
    cv::cvtColor(image, gray_image, cv::COLOR_BGR2GRAY);
    
    // Apply edge detection to find foreign objects
    cv::Mat edges;
    cv::Canny(gray_image, edges, 50, 150);
    
    // Count edge pixels (more edges = possible contamination)
    int edge_pixels = cv::countNonZero(edges);
    int total_pixels = edges.rows * edges.cols;
    float edge_ratio = static_cast<float>(edge_pixels) / total_pixels;
    
    float contamination_risk = std::min(edge_ratio * 2.0f, 1.0f);
    return contamination_risk;
}

float SafetyChecker::checkColorSafety(const cv::Mat& image) {
    // Calculate color metrics for safety
    cv::Scalar mean_color = cv::mean(image);
    
    // Check for unusual colors (indicating potential issues)
    float blue = mean_color[0];
    float green = mean_color[1];
    float red = mean_color[2];
    
    // Safety based on color balance
    float color_balance = std::abs(red - green) + std::abs(green - blue) + std::abs(blue - red);
    float color_safety = 1.0f - (color_balance / 765.0f); // Normalize to 0-1
    
    return std::clamp(color_safety, 0.0f, 1.0f);
}

std::string SafetyChecker::getSafetyCategory(float score) {
    if (score >= 0.8f) {
        return "Safe";
    } else if (score >= 0.6f) {
        return "Caution";
    } else if (score >= 0.4f) {
        return "Questionable";
    } else {
        return "Unsafe";
    }
}

void SafetyChecker::initializeSafetyDatabase() {
    // Initialize with common food safety data
    std::cout << "Safety database initialized" << std::endl;
}

void SafetyChecker::addSafetyData(const std::string& food, float score) {
    safety_database[food] = {score, score * 0.9f, score * 0.8f, score * 0.7f, score * 0.6f};
}

float SafetyChecker::getExpectedSafety(const std::string& food) {
    auto it = safety_database.find(food);
    if (it != safety_database.end()) {
        return it->second[0]; // Return first day's safety
    }
    return 0.7f; // Default safety
}
