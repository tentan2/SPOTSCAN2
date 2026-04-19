#include "temperature_analyzer.h"
#include <iostream>
#include <algorithm>

TemperatureAnalyzer::TemperatureAnalyzer(const torch::Device& device) 
    : device(device) {
    
    // Initialize temperature database
    temperature_database["hot_soup"] = 85.0f;
    temperature_database["warm_food"] = 65.0f;
    temperature_database["room_temp"] = 20.0f;
    temperature_database["cold_food"] = 5.0f;
    temperature_database["frozen"] = -18.0f;
    
    initializeTemperatureDatabase();
    std::cout << "TemperatureAnalyzer initialized" << std::endl;
}

bool TemperatureAnalyzer::loadModel(const std::string& model_path) {
    try {
        model = torch::jit::load(model_path);
        model->to(device);
        std::cout << "Temperature analysis model loaded from: " << model_path << std::endl;
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Error loading temperature analysis model: " << e.what() << std::endl;
        return false;
    }
}

std::map<std::string, float> TemperatureAnalyzer::analyzeTemperature(const cv::Mat& image) {
    std::map<std::string, float> results;
    
    // Calculate temperature
    float estimated_temp = estimateTemperature(image);
    float color_temp = calculateColorTemperature(image);
    float steam_temp = calculateSteamTemperature(image);
    
    // Combine estimates (weighted average)
    float overall_temp = (estimated_temp * 0.5f) + (color_temp * 0.3f) + (steam_temp * 0.2f);
    
    results["estimated_temp_celsius"] = overall_temp;
    results["estimated_temp_fahrenheit"] = celsiusToFahrenheit(overall_temp);
    results["color_temp_celsius"] = color_temp;
    results["steam_temp_celsius"] = steam_temp;
    results["temperature_category"] = static_cast<float>(getTemperatureCategory(overall_temp) == "Hot");
    
    return results;
}

float TemperatureAnalyzer::estimateTemperature(const cv::Mat& image) {
    // Convert to HSV for temperature estimation
    cv::Mat hsv_image;
    cv::cvtColor(image, hsv_image, cv::COLOR_BGR2HSV);
    
    // Calculate temperature based on color temperature
    cv::Scalar mean_color = cv::mean(hsv_image);
    float hue = mean_color[0];
    float saturation = mean_color[1];
    float value = mean_color[2];
    
    // Temperature estimation based on HSV values
    float temp_celsius = 20.0f; // Base room temperature
    
    if (saturation > 0.7f && value > 0.8f) {
        temp_celsius += 40.0f; // Hot food
    } else if (saturation > 0.5f && value > 0.6f) {
        temp_celsius += 20.0f; // Warm food
    } else if (saturation < 0.3f && value < 0.4f) {
        temp_celsius -= 15.0f; // Cold food
    }
    
    return std::clamp(temp_celsius, -20.0f, 100.0f);
}

float TemperatureAnalyzer::calculateColorTemperature(const cv::Mat& image) {
    // Calculate color temperature
    cv::Scalar mean_color = cv::mean(image);
    float blue = mean_color[0];
    float green = mean_color[1];
    float red = mean_color[2];
    
    // Color temperature estimation (red = warmer, blue = cooler)
    float color_ratio = (red - blue) / (red + blue + 1.0f);
    float color_temp = 20.0f + (color_ratio * 30.0f);
    
    return std::clamp(color_temp, 0.0f, 100.0f);
}

float TemperatureAnalyzer::calculateSteamTemperature(const cv::Mat& image) {
    // Convert to grayscale for steam detection
    cv::Mat gray_image;
    cv::cvtColor(image, gray_image, cv::COLOR_BGR2GRAY);
    
    // Apply threshold to detect steam/heat
    cv::Mat threshold_image;
    cv::threshold(gray_image, threshold_image, 200, 255, cv::THRESH_BINARY);
    
    // Count white pixels (steam/heat)
    int white_pixels = cv::countNonZero(threshold_image);
    int total_pixels = threshold_image.rows * threshold_image.cols;
    float white_ratio = static_cast<float>(white_pixels) / total_pixels;
    
    // Temperature based on steam/heat
    float steam_temp = 20.0f + (white_ratio * 60.0f);
    
    return std::clamp(steam_temp, 0.0f, 100.0f);
}

std::string TemperatureAnalyzer::getTemperatureCategory(float temp_celsius) {
    if (temp_celsius >= 60.0f) {
        return "Hot";
    } else if (temp_celsius >= 40.0f) {
        return "Warm";
    } else if (temp_celsius >= 15.0f) {
        return "Room Temperature";
    } else if (temp_celsius >= 0.0f) {
        return "Cold";
    } else {
        return "Frozen";
    }
}

void TemperatureAnalyzer::initializeTemperatureDatabase() {
    // Initialize with common food temperature data
    std::cout << "Temperature database initialized" << std::endl;
}

void TemperatureAnalyzer::addTemperatureData(const std::string& food, float temp) {
    temperature_database[food] = temp;
}

float TemperatureAnalyzer::getExpectedTemperature(const std::string& food) {
    auto it = temperature_database.find(food);
    if (it != temperature_database.end()) {
        return it->second;
    }
    return 20.0f; // Default room temperature
}

float TemperatureAnalyzer::celsiusToFahrenheit(float celsius) {
    return (celsius * 9.0f / 5.0f) + 32.0f;
}

std::string TemperatureAnalyzer::formatTemperature(float celsius) {
    return std::to_string(static_cast<int>(celsius)) + "°C (" + 
           std::to_string(static_cast<int>(celsiusToFahrenheit(celsius))) + "°F)";
}
