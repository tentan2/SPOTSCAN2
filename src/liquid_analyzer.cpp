#include "liquid_analyzer.h"
#include <iostream>
#include <algorithm>

LiquidAnalyzer::LiquidAnalyzer(const torch::Device& device) 
    : device(device) {
    
    // Initialize liquid database
    liquid_database["water"] = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f};
    liquid_database["juice"] = {0.3f, 0.4f, 0.5f, 0.6f, 0.7f};
    liquid_database["milk"] = {0.2f, 0.3f, 0.4f, 0.5f, 0.6f};
    liquid_database["oil"] = {0.8f, 0.7f, 0.6f, 0.5f, 0.4f};
    liquid_database["syrup"] = {0.9f, 0.8f, 0.7f, 0.6f, 0.5f};
    
    initializeLiquidDatabase();
    std::cout << "LiquidAnalyzer initialized" << std::endl;
}

bool LiquidAnalyzer::loadModel(const std::string& model_path) {
    try {
        model = torch::jit::load(model_path);
        model->to(device);
        std::cout << "Liquid analysis model loaded from: " << model_path << std::endl;
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Error loading liquid analysis model: " << e.what() << std::endl;
        return false;
    }
}

std::map<std::string, float> LiquidAnalyzer::analyzeLiquid(const cv::Mat& image) {
    std::map<std::string, float> results;
    
    // Calculate liquid metrics
    float liquid_level = detectLiquidLevel(image);
    float viscosity = calculateViscosity(image);
    float transparency = detectTransparency(image);
    float color = measureColor(image);
    
    // Combine scores
    float overall_score = (liquid_level * 0.3f) + (viscosity * 0.3f) + 
                        (transparency * 0.2f) + (color * 0.2f);
    
    results["liquid_level"] = liquid_level;
    results["viscosity"] = viscosity;
    results["transparency"] = transparency;
    results["color_intensity"] = color;
    results["overall_liquid"] = overall_score;
    results["liquid_category"] = static_cast<float>(getLiquidCategory(overall_score) == "Watery");
    
    return results;
}

float LiquidAnalyzer::detectLiquidLevel(const cv::Mat& image) {
    // Convert to grayscale for level detection
    cv::Mat gray_image;
    cv::cvtColor(image, gray_image, cv::COLOR_BGR2GRAY);
    
    // Find liquid surface (brightest area)
    cv::Mat threshold_image;
    cv::threshold(gray_image, threshold_image, 200, 255, cv::THRESH_BINARY);
    
    // Find contours of liquid surface
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(threshold_image, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    
    if (contours.empty()) {
        return 0.5f; // Default level
    }
    
    // Calculate liquid level based on largest contour
    float max_area = 0.0f;
    for (const auto& contour : contours) {
        float area = cv::contourArea(contour);
        max_area = std::max(max_area, area);
    }
    
    float liquid_level = max_area / (image.rows * image.cols);
    return std::clamp(liquid_level, 0.0f, 1.0f);
}

float LiquidAnalyzer::calculateViscosity(const cv::Mat& image) {
    // Convert to grayscale for viscosity analysis
    cv::Mat gray_image;
    cv::cvtColor(image, gray_image, cv::COLOR_BGR2GRAY);
    
    // Apply Sobel filter to detect flow patterns
    cv::Mat grad_x, grad_y;
    cv::Sobel(gray_image, grad_x, CV_64F, 1, 0, 3);
    cv::Sobel(gray_image, grad_y, CV_64F, 0, 1, 3);
    
    cv::Mat magnitude;
    cv::magnitude(grad_x, grad_y, magnitude);
    
    // Calculate viscosity based on gradient magnitude
    cv::Scalar mean_grad = cv::mean(magnitude);
    float viscosity = std::min(mean_grad[0] / 100.0f, 1.0f);
    
    return viscosity;
}

float LiquidAnalyzer::detectTransparency(const cv::Mat& image) {
    // Convert to grayscale for transparency detection
    cv::Mat gray_image;
    cv::cvtColor(image, gray_image, cv::COLOR_BGR2GRAY);
    
    // Calculate transparency based on brightness variance
    cv::Scalar mean, stddev;
    cv::meanStdDev(gray_image, mean, stddev);
    
    // Higher variance = less transparent
    float transparency = 1.0f - (stddev[0] / 128.0f);
    return std::clamp(transparency, 0.0f, 1.0f);
}

float LiquidAnalyzer::measureColor(const cv::Mat& image) {
    // Calculate color intensity
    cv::Scalar mean_color = cv::mean(image);
    
    // Calculate overall color intensity
    float intensity = (mean_color[0] + mean_color[1] + mean_color[2]) / 3.0f;
    float normalized_intensity = intensity / 255.0f;
    
    return normalized_intensity;
}

std::string LiquidAnalyzer::getLiquidCategory(float score) {
    if (score >= 0.8f) {
        return "Very Thick";
    } else if (score >= 0.6f) {
        return "Thick";
    } else if (score >= 0.4f) {
        return "Medium";
    } else if (score >= 0.2f) {
        return "Thin";
    } else {
        return "Very Thin";
    }
}

void LiquidAnalyzer::initializeLiquidDatabase() {
    // Initialize with common liquid data
    std::cout << "Liquid database initialized" << std::endl;
}

void LiquidAnalyzer::addLiquidData(const std::string& liquid, float score) {
    liquid_database[liquid] = {score, score * 0.9f, score * 0.8f, score * 0.7f, score * 0.6f};
}

float LiquidAnalyzer::getExpectedLiquid(const std::string& liquid) {
    auto it = liquid_database.find(liquid);
    if (it != liquid_database.end()) {
        return it->second[0]; // Return first day's liquid score
    }
    return 0.5f; // Default liquid score
}
