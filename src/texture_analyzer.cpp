#include "texture_analyzer.h"
#include <iostream>
#include <algorithm>

TextureAnalyzer::TextureAnalyzer(const torch::Device& device) 
    : device(device) {
    
    // Initialize texture database
    texture_database["apple"] = {0.7f, 0.6f, 0.5f, 0.4f, 0.3f};
    texture_database["banana"] = {0.8f, 0.7f, 0.6f, 0.5f, 0.4f};
    texture_database["orange"] = {0.6f, 0.5f, 0.4f, 0.3f, 0.2f};
    texture_database["lettuce"] = {0.9f, 0.8f, 0.7f, 0.6f, 0.5f};
    texture_database["tomato"] = {0.5f, 0.4f, 0.3f, 0.2f, 0.1f};
    
    initializeTextureDatabase();
    std::cout << "TextureAnalyzer initialized" << std::endl;
}

bool TextureAnalyzer::loadModel(const std::string& model_path) {
    try {
        model = torch::jit::load(model_path);
        model->to(device);
        std::cout << "Texture analysis model loaded from: " << model_path << std::endl;
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Error loading texture analysis model: " << e.what() << std::endl;
        return false;
    }
}

std::map<std::string, float> TextureAnalyzer::analyzeTexture(const cv::Mat& image) {
    std::map<std::string, float> results;
    
    // Calculate texture metrics
    float smoothness = calculateSmoothness(image);
    float roughness = calculateRoughness(image);
    float hardness = calculateHardness(image);
    
    // Combine scores
    float overall_score = (smoothness * 0.3f) + (roughness * 0.4f) + (hardness * 0.3f);
    
    results["smoothness"] = smoothness;
    results["roughness"] = roughness;
    results["hardness"] = hardness;
    results["overall_texture"] = overall_score;
    results["texture_category"] = static_cast<float>(getTextureCategory(overall_score) == "Smooth");
    
    return results;
}

float TextureAnalyzer::calculateSmoothness(const cv::Mat& image) {
    // Convert to grayscale
    cv::Mat gray_image;
    cv::cvtColor(image, gray_image, cv::COLOR_BGR2GRAY);
    
    // Apply Gaussian blur to measure smoothness
    cv::Mat blurred;
    cv::GaussianBlur(gray_image, blurred, cv::Size(5, 5), 0);
    
    // Calculate difference between original and blurred
    cv::Mat diff;
    cv::absdiff(gray_image, blurred, diff);
    
    // Calculate smoothness score (lower difference = smoother)
    cv::Scalar mean_diff = cv::mean(diff);
    float smoothness = 1.0f - (mean_diff[0] / 255.0f);
    
    return std::clamp(smoothness, 0.0f, 1.0f);
}

float TextureAnalyzer::calculateRoughness(const cv::Mat& image) {
    // Convert to grayscale
    cv::Mat gray_image;
    cv::cvtColor(image, gray_image, cv::COLOR_BGR2GRAY);
    
    // Calculate gradient magnitude for roughness
    cv::Mat grad_x, grad_y;
    cv::Sobel(gray_image, grad_x, CV_64F, 1, 0, 3);
    cv::Sobel(gray_image, grad_y, CV_64F, 0, 1, 3);
    
    cv::Mat magnitude;
    cv::magnitude(grad_x, grad_y, magnitude);
    
    // Calculate roughness score (higher gradient = rougher)
    cv::Scalar mean_mag = cv::mean(magnitude);
    float roughness = std::min(mean_mag[0] / 100.0f, 1.0f);
    
    return roughness;
}

float TextureAnalyzer::calculateHardness(const cv::Mat& image) {
    // Convert to grayscale
    cv::Mat gray_image;
    cv::cvtColor(image, gray_image, cv::COLOR_BGR2GRAY);
    
    // Calculate edge density for hardness
    cv::Mat edges;
    cv::Canny(gray_image, edges, 50, 150);
    
    // Count edge pixels
    int edge_pixels = cv::countNonZero(edges);
    int total_pixels = edges.rows * edges.cols;
    
    // Calculate hardness score (more edges = harder)
    float hardness = static_cast<float>(edge_pixels) / total_pixels;
    
    return std::clamp(hardness, 0.0f, 1.0f);
}

std::string TextureAnalyzer::getTextureCategory(float score) {
    if (score >= 0.8f) {
        return "Very Smooth";
    } else if (score >= 0.6f) {
        return "Smooth";
    } else if (score >= 0.4f) {
        return "Moderately Rough";
    } else if (score >= 0.2f) {
        return "Rough";
    } else {
        return "Very Rough";
    }
}

void TextureAnalyzer::initializeTextureDatabase() {
    // Initialize with common food texture data
    std::cout << "Texture database initialized" << std::endl;
}

void TextureAnalyzer::addTextureData(const std::string& food, float score) {
    texture_database[food] = {score, score * 0.9f, score * 0.8f, score * 0.7f, score * 0.6f};
}

float TextureAnalyzer::getExpectedTexture(const std::string& food) {
    auto it = texture_database.find(food);
    if (it != texture_database.end()) {
        return it->second[0]; // Return first day's texture
    }
    return 0.5f; // Default texture
}
