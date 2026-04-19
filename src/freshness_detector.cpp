#include "freshness_detector.h"
#include <iostream>
#include <algorithm>

FreshnessDetector::FreshnessDetector(const torch::Device& device) 
    : device(device) {
    
    // Initialize freshness metrics
    freshness_metrics["apple"] = {0.9f, 0.8f, 0.7f, 0.6f, 0.5f};
    freshness_metrics["banana"] = {0.8f, 0.7f, 0.6f, 0.4f, 0.3f};
    freshness_metrics["orange"] = {0.9f, 0.8f, 0.7f, 0.6f, 0.5f};
    freshness_metrics["lettuce"] = {0.7f, 0.6f, 0.5f, 0.3f, 0.2f};
    freshness_metrics["tomato"] = {0.8f, 0.7f, 0.6f, 0.4f, 0.3f};
    
    initializeFreshnessDatabase();
    std::cout << "FreshnessDetector initialized" << std::endl;
}

bool FreshnessDetector::loadModel(const std::string& model_path) {
    try {
        model = torch::jit::load(model_path);
        model->to(device);
        std::cout << "Freshness detection model loaded from: " << model_path << std::endl;
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Error loading freshness detection model: " << e.what() << std::endl;
        return false;
    }
}

std::map<std::string, float> FreshnessDetector::detectFreshness(const cv::Mat& image) {
    std::map<std::string, float> results;
    
    // Calculate freshness metrics
    float color_score = calculateColorFreshness(image);
    float texture_score = calculateTextureFreshness(image);
    float shape_score = calculateShapeFreshness(image);
    
    // Combine scores (weighted average)
    float overall_score = (color_score * 0.4f) + (texture_score * 0.4f) + (shape_score * 0.2f);
    
    results["color_freshness"] = color_score;
    results["texture_freshness"] = texture_score;
    results["shape_freshness"] = shape_score;
    results["overall_freshness"] = overall_score;
    results["freshness_category"] = static_cast<float>(getFreshnessCategory(overall_score) == "Fresh");
    
    return results;
}

float FreshnessDetector::calculateColorFreshness(const cv::Mat& image) {
    // Convert to HSV for better color analysis
    cv::Mat hsv_image;
    cv::cvtColor(image, hsv_image, cv::COLOR_BGR2HSV);
    
    // Calculate color metrics
    cv::Scalar mean_color = cv::mean(hsv_image);
    float saturation = mean_color[1] / 255.0f;
    float brightness = mean_color[2] / 255.0f;
    
    // Freshness based on color (higher saturation and brightness = fresher)
    float freshness_score = (saturation + brightness) / 2.0f;
    return std::clamp(freshness_score, 0.0f, 1.0f);
}

float FreshnessDetector::calculateTextureFreshness(const cv::Mat& image) {
    // Convert to grayscale for texture analysis
    cv::Mat gray_image;
    cv::cvtColor(image, gray_image, cv::COLOR_BGR2GRAY);
    
    // Calculate texture using Laplacian variance
    cv::Mat laplacian;
    cv::Laplacian(gray_image, laplacian, CV_64F);
    cv::Scalar mean, stddev;
    cv::meanStdDev(laplacian, mean, stddev);
    
    // Higher variance = more texture = fresher
    float texture_score = std::min(stddev[0] / 100.0f, 1.0f);
    return texture_score;
}

float FreshnessDetector::calculateShapeFreshness(const cv::Mat& image) {
    // Find contours for shape analysis
    cv::Mat gray_image;
    cv::cvtColor(image, gray_image, cv::COLOR_BGR2GRAY);
    
    std::vector<std::vector<cv::Point>> contours;
    cv::Mat threshold_image;
    cv::threshold(gray_image, threshold_image, 127, 255, cv::THRESH_BINARY);
    cv::findContours(threshold_image, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    
    if (contours.empty()) {
        return 0.5f; // Neutral score
    }
    
    // Calculate shape metrics
    float total_area = 0;
    for (const auto& contour : contours) {
        total_area += cv::contourArea(contour);
    }
    
    // Shape freshness based on area and contour count
    float shape_score = std::min(total_area / (image.rows * image.cols), 1.0f);
    return shape_score;
}

std::string FreshnessDetector::getFreshnessCategory(float score) {
    if (score >= 0.8f) {
        return "Very Fresh";
    } else if (score >= 0.6f) {
        return "Fresh";
    } else if (score >= 0.4f) {
        return "Moderately Fresh";
    } else if (score >= 0.2f) {
        return "Slightly Stale";
    } else {
        return "Stale";
    }
}

void FreshnessDetector::initializeFreshnessDatabase() {
    // Initialize with common food freshness data
    std::cout << "Freshness database initialized" << std::endl;
}

void FreshnessDetector::addFreshnessData(const std::string& food, float score) {
    freshness_metrics[food] = {score, score * 0.9f, score * 0.8f, score * 0.7f, score * 0.6f};
}

float FreshnessDetector::getExpectedFreshness(const std::string& food) {
    auto it = freshness_metrics.find(food);
    if (it != freshness_metrics.end()) {
        return it->second[0]; // Return first day's freshness
    }
    return 0.5f; // Default freshness
}
