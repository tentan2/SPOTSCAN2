#include "model_manager.h"
#include <iostream>
#include <filesystem>
#include <torch/torch.h>
#include <torch/script.h>
#include <opencv2/opencv.hpp>
#include <vector>

namespace fs = std::filesystem;

ModelManager::ModelManager(const std::string& models_dir) 
    : models_dir(models_dir) {
    
    // Create models directory if it doesn't exist
    if (!fs::exists(models_dir)) {
        fs::create_directories(models_dir);
    }
    
    // Set device (CUDA if available, otherwise CPU)
    device = torch::kCUDA;
    if (!torch::cuda::is_available()) {
        device = torch::kCPU;
    }
    
    std::cout << "ModelManager initialized with device: " 
              << (device == torch::kCUDA ? "CUDA" : "CPU") << std::endl;
}

torch::nn::Module ModelManager::createFoodClassifier(int num_classes) {
    // Create ResNet50-based food classifier
    auto model = torch::nn::Sequential(
        torch::nn::Conv2d(torch::nn::Conv2dOptions(3, 64, 7).stride(1).padding(3)),
        torch::nn::BatchNorm2d(64),
        torch::nn::ReLU(),
        torch::nn::MaxPool2d(2),
        torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 128, 3).stride(1).padding(1)),
        torch::nn::BatchNorm2d(128),
        torch::nn::ReLU(),
        torch::nn::MaxPool2d(2),
        torch::nn::Conv2d(torch::nn::Conv2dOptions(128, 256, 3).stride(1).padding(1)),
        torch::nn::BatchNorm2d(256),
        torch::nn::ReLU(),
        torch::nn::MaxPool2d(2),
        torch::nn::Conv2d(torch::nn::Conv2dOptions(256, 512, 3).stride(1).padding(1)),
        torch::nn::BatchNorm2d(512),
        torch::nn::ReLU(),
        torch::nn::MaxPool2d(2),
        torch::nn::Flatten(),
        torch::nn::Linear(512 * 7 * 7, 4096),
        torch::nn::ReLU(),
        torch::nn::Dropout(0.5),
        torch::nn::Linear(4096, num_classes)
    );
    
    return model;
}

torch::nn::Module ModelManager::createViTClassifier(const std::string& model_name) {
    // Note: For ViT, we would typically use pre-trained models
    // This is a placeholder for ViT integration
    std::cout << "Creating ViT classifier: " << model_name << std::endl;
    
    // In practice, you would load pre-trained ViT from HuggingFace
    // For now, return a simple CNN as placeholder
    return createFoodClassifier(101);
}

bool ModelManager::loadModel(const std::string& model_name, const std::string& model_path) {
    try {
        auto module = torch::jit::load(model_path);
        models[model_name] = module;
        
        // Store metadata
        nlohmann::json metadata;
        metadata["name"] = model_name;
        metadata["path"] = model_path;
        metadata["loaded_at"] = std::time(nullptr);
        model_metadata[model_name] = metadata;
        
        std::cout << "Model loaded: " << model_name << " from " << model_path << std::endl;
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Error loading model: " << e.what() << std::endl;
        return false;
    }
}

bool ModelManager::saveModel(const torch::nn::Module& model, const std::string& model_name) {
    try {
        std::string model_path = models_dir + "/" + model_name + ".pt";
        torch::save(model, model_path);
        
        // Update metadata
        nlohmann::json metadata;
        metadata["name"] = model_name;
        metadata["path"] = model_path;
        metadata["saved_at"] = std::time(nullptr);
        model_metadata[model_name] = metadata;
        
        std::cout << "Model saved: " << model_name << " to " << model_path << std::endl;
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Error saving model: " << e.what() << std::endl;
        return false;
    }
}

torch::jit::script::Module ModelManager::getModel(const std::string& model_name) {
    auto it = models.find(model_name);
    if (it != models.end()) {
        return it->second;
    }
    return torch::jit::script::Module();
}

void ModelManager::listModels() {
    std::cout << "Available models:" << std::endl;
    for (const auto& [name, module] : models) {
        std::cout << "- " << name << std::endl;
    }
}

void ModelManager::deleteModel(const std::string& model_name) {
    auto it = models.find(model_name);
    if (it != models.end()) {
        models.erase(it);
        model_metadata.erase(model_name);
        std::cout << "Model deleted: " << model_name << std::endl;
    }
}
