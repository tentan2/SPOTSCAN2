#include "image_processor.h"
#include <iostream>
#include <algorithm>

ImageProcessor::ImageProcessor(const cv::Size& target_size) 
    : target_size(target_size), mean({0.485, 0.456, 0.406}), std({0.229, 0.224, 0.225}) {
    std::cout << "ImageProcessor initialized with target size: " 
              << target_size.width << "x" << target_size.height << std::endl;
}

cv::Mat ImageProcessor::loadImage(const std::string& image_path) {
    cv::Mat image = cv::imread(image_path);
    if (image.empty()) {
        std::cerr << "Error: Could not load image from " << image_path << std::endl;
        return cv::Mat();
    }
    return image;
}

cv::Mat ImageProcessor::preprocessImage(const cv::Mat& image) {
    cv::Mat processed = image.clone();
    
    // Resize
    processed = resizeImage(processed, target_size);
    
    // Normalize
    processed = normalizeImage(processed);
    
    return processed;
}

cv::Mat ImageProcessor::resizeImage(const cv::Mat& image, const cv::Size& size) {
    cv::Mat resized;
    cv::resize(image, resized, size, 0, 0, cv::INTER_LINEAR);
    return resized;
}

cv::Mat ImageProcessor::normalizeImage(const cv::Mat& image) {
    cv::Mat normalized;
    image.convertTo(normalized, CV_32F, 1.0/255.0);
    
    // Apply ImageNet normalization
    std::vector<cv::Mat> channels;
    cv::split(normalized, channels);
    
    for (int i = 0; i < 3; i++) {
        channels[i] = (channels[i] - mean[i]) / std[i];
    }
    
    cv::merge(channels, normalized);
    return normalized;
}

cv::Mat ImageProcessor::enhanceImage(const cv::Mat& image) {
    cv::Mat enhanced;
    
    // Apply histogram equalization
    cv::Mat ycrcb;
    cv::cvtColor(image, ycrcb, cv::COLOR_BGR2YCrCb);
    std::vector<cv::Mat> channels;
    cv::split(ycrcb, channels);
    cv::equalizeHist(channels[0], channels[0]);
    cv::merge(channels, ycrcb);
    cv::cvtColor(ycrcb, enhanced, cv::COLOR_YCrCb2BGR);
    
    return enhanced;
}

cv::Mat ImageProcessor::cropImage(const cv::Mat& image, const cv::Rect& roi) {
    return image(roi).clone();
}

cv::Mat ImageProcessor::rotateImage(const cv::Mat& image, double angle) {
    cv::Mat rotated;
    cv::Point2f center(image.cols/2.0, image.rows/2.0);
    cv::Mat rotation_matrix = cv::getRotationMatrix2D(center, angle, 1.0);
    cv::warpAffine(image, rotated, rotation_matrix, image.size());
    return rotated;
}

cv::Mat ImageProcessor::flipImage(const cv::Mat& image, int flip_code) {
    cv::Mat flipped;
    cv::flip(image, flipped, flip_code);
    return flipped;
}

cv::Mat ImageProcessor::adjustBrightnessContrast(const cv::Mat& image, double alpha, double beta) {
    cv::Mat adjusted;
    image.convertTo(adjusted, -1, alpha, beta);
    return adjusted;
}

std::vector<cv::Mat> ImageProcessor::processBatch(const std::vector<std::string>& image_paths) {
    std::vector<cv::Mat> processed_images;
    
    for (const auto& path : image_paths) {
        cv::Mat image = loadImage(path);
        if (!image.empty()) {
            cv::Mat processed = preprocessImage(image);
            processed_images.push_back(processed);
        }
    }
    
    return processed_images;
}

std::vector<torch::Tensor> ImageProcessor::convertToTensor(const std::vector<cv::Mat>& images) {
    std::vector<torch::Tensor> tensors;
    
    for (const auto& image : images) {
        // Convert BGR to RGB
        cv::Mat rgb_image;
        cv::cvtColor(image, rgb_image, cv::COLOR_BGR2RGB);
        
        // Convert to tensor
        auto tensor = torch::from_blob(rgb_image.data, {1, rgb_image.rows, rgb_image.cols, 3}, torch::kByte);
        tensor = tensor.to(torch::kFloat);
        
        tensors.push_back(tensor);
    }
    
    return tensors;
}
