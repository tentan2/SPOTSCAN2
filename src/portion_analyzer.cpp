#include "portion_analyzer.h"
#include <iostream>
#include <algorithm>

PortionAnalyzer::PortionAnalyzer(const torch::Device& device) 
    : device(device) {
    
    // Initialize portion database
    portion_database["apple"] = 150.0f;
    portion_database["banana"] = 120.0f;
    portion_database["orange"] = 200.0f;
    portion_database["lettuce"] = 100.0f;
    portion_database["tomato"] = 150.0f;
    
    initializePortionDatabase();
    std::cout << "PortionAnalyzer initialized" << std::endl;
}

bool PortionAnalyzer::loadModel(const std::string& model_path) {
    try {
        model = torch::jit::load(model_path);
        model->to(device);
        std::cout << "Portion analysis model loaded from: " << model_path << std::endl;
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Error loading portion analysis model: " << e.what() << std::endl;
        return false;
    }
}

std::map<std::string, float> PortionAnalyzer::analyzePortion(const cv::Mat& image) {
    std::map<std::string, float> results;
    
    // Calculate portion metrics
    float volume = estimateVolume(image);
    float weight = estimateWeight(image, volume);
    float area = calculateArea(image);
    float calories = calculateCalories(image, weight);
    
    results["volume_cm3"] = volume;
    results["weight_grams"] = weight;
    results["area_pixels"] = area;
    results["calories"] = calories;
    results["servings"] = weight / 100.0f; // Assuming 100g per serving
    
    return results;
}

float PortionAnalyzer::estimateVolume(const cv::Mat& image) {
    // Convert to grayscale for volume estimation
    cv::Mat gray_image;
    cv::cvtColor(image, gray_image, cv::COLOR_BGR2GRAY);
    
    // Find contours for volume calculation
    std::vector<std::vector<cv::Point>> contours;
    cv::Mat threshold_image;
    cv::threshold(gray_image, threshold_image, 127, 255, cv::THRESH_BINARY);
    cv::findContours(threshold_image, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    
    if (contours.empty()) {
        return 0.0f;
    }
    
    // Calculate volume based on contour area
    float max_area = 0.0f;
    for (const auto& contour : contours) {
        float area = cv::contourArea(contour);
        max_area = std::max(max_area, area);
    }
    
    // Convert pixel area to volume (assuming 1 pixel = 1mm^2)
    float volume_cm3 = max_area * 0.001f; // Rough conversion
    return volume_cm3;
}

float PortionAnalyzer::estimateWeight(const cv::Mat& image, float volume) {
    // Estimate weight based on volume and food density
    // Default density is 1.0 g/cm^3 (water density)
    float density = 1.0f;
    
    // Adjust density based on color (darker foods are often denser)
    cv::Scalar mean_color = cv::mean(image);
    float brightness = (mean_color[0] + mean_color[1] + mean_color[2]) / 3.0f;
    
    if (brightness < 100) {
        density = 1.2f; // Darker food
    } else if (brightness > 200) {
        density = 0.8f; // Lighter food
    }
    
    float weight_grams = volume * density;
    return weight_grams;
}

float PortionAnalyzer::calculateArea(const cv::Mat& image) {
    // Convert to grayscale for area calculation
    cv::Mat gray_image;
    cv::cvtColor(image, gray_image, cv::COLOR_BGR2GRAY);
    
    // Find contours for area calculation
    std::vector<std::vector<cv::Point>> contours;
    cv::Mat threshold_image;
    cv::threshold(gray_image, threshold_image, 127, 255, cv::THRESH_BINARY);
    cv::findContours(threshold_image, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    
    if (contours.empty()) {
        return 0.0f;
    }
    
    // Calculate total area
    float total_area = 0.0f;
    for (const auto& contour : contours) {
        total_area += cv::contourArea(contour);
    }
    
    return total_area;
}

float PortionAnalyzer::calculateCalories(const cv::Mat& image, float weight) {
    // Estimate calories based on weight and food type
    // Default is 0.5 calories per gram (average)
    float calories_per_gram = 0.5f;
    
    // Adjust based on color (redder foods often have more calories)
    cv::Scalar mean_color = cv::mean(image);
    float red_ratio = mean_color[2] / 255.0f;
    
    if (red_ratio > 0.6f) {
        calories_per_gram = 0.7f; // Higher calorie foods
    } else if (red_ratio < 0.3f) {
        calories_per_gram = 0.3f; // Lower calorie foods
    }
    
    float total_calories = weight * calories_per_gram;
    return total_calories;
}

void PortionAnalyzer::initializePortionDatabase() {
    // Initialize with common food portion data
    std::cout << "Portion database initialized" << std::endl;
}

void PortionAnalyzer::addPortionData(const std::string& food, float portion) {
    portion_database[food] = portion;
}

float PortionAnalyzer::getExpectedPortion(const std::string& food) {
    auto it = portion_database.find(food);
    if (it != portion_database.end()) {
        return it->second;
    }
    return 100.0f; // Default portion
}

std::string PortionAnalyzer::formatPortion(float grams) {
    return std::to_string(static_cast<int>(grams)) + "g";
}

std::string PortionAnalyzer::formatServings(float servings) {
    return std::to_string(static_cast<float>(std::round(servings * 10.0f) / 10.0f) + " servings";
}
