#ifndef IMAGE_PROCESSOR_H
#define IMAGE_PROCESSOR_H

#include <opencv2/opencv.hpp>
#include <torch/torch.h>
#include <vector>
#include <string>

class ImageProcessor {
private:
    cv::Size target_size;
    std::vector<float> mean;
    std::vector<float> std;
    
public:
    ImageProcessor(const cv::Size& target_size = cv::Size(224, 224));
    ~ImageProcessor() = default;
    
    // Core processing methods
    cv::Mat loadImage(const std::string& image_path);
    cv::Mat preprocessImage(const cv::Mat& image);
    cv::Mat resizeImage(const cv::Mat& image, const cv::Size& size);
    cv::Mat normalizeImage(const cv::Mat& image);
    cv::Mat enhanceImage(const cv::Mat& image);
    
    // Utility methods
    cv::Mat cropImage(const cv::Mat& image, const cv::Rect& roi);
    cv::Mat rotateImage(const cv::Mat& image, double angle);
    cv::Mat flipImage(const cv::Mat& image, int flip_code);
    cv::Mat adjustBrightnessContrast(const cv::Mat& image, double alpha, double beta);
    
    // Batch processing
    std::vector<cv::Mat> processBatch(const std::vector<std::string>& image_paths);
    std::vector<torch::Tensor> convertToTensor(const std::vector<cv::Mat>& images);
    
    // Getters/Setters
    void setTargetSize(const cv::Size& size) { target_size = size; }
    cv::Size getTargetSize() const { return target_size; }
};

#endif // IMAGE_PROCESSOR_H
