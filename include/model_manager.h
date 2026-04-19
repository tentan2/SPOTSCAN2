#ifndef MODEL_MANAGER_H
#define MODEL_MANAGER_H

#include <torch/torch.h>
#include <torch/script.h>
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <map>
#include <memory>
#include <fstream>
#include <nlohmann/json.hpp>

class ModelManager {
private:
    std::string models_dir;
    torch::Device device;
    std::map<std::string, torch::jit::script::Module> models;
    std::map<std::string, nlohmann::json> model_metadata;
    
public:
    ModelManager(const std::string& models_dir = "models");
    ~ModelManager() = default;
    
    // Model creation methods
    torch::nn::Module createFoodClassifier(int num_classes = 101);
    torch::nn::Module createFreshnessDetector();
    torch::nn::Module createNutritionAnalyzer();
    torch::nn::Module createPortionAnalyzer();
    torch::nn::Module createTextureAnalyzer();
    torch::nn::Module createTemperatureAnalyzer();
    torch::nn::Module createLiquidAnalyzer();
    torch::nn::Module createAcidityAnalyzer();
    torch::nn::Module createColorAnalyzer();
    torch::nn::Module createOCRAnalyzer();
    torch::nn::Module createRipenessPredictor();
    torch::nn::Module createSafetyChecker();
    torch::nn::Module createShapeReconstructor();
    torch::nn::Module createSolidLiquidClassifier();
    torch::nn::Module createSustainabilityDetector();
    torch::nn::Module createEnhancedVisualEstimator();
    torch::nn::Module createVisualCalorieEstimator();
    torch::nn::Module createProcessedFoodClassifier();
    torch::nn::Module createArtificialNaturalClassifier();
    torch::nn::Module createViTClassifier(const std::string& model_name = "google/vit-base-patch16-224");
    
    // Model management methods
    bool loadModel(const std::string& model_name, const std::string& model_path);
    bool saveModel(const torch::nn::Module& model, const std::string& model_name);
    torch::jit::script::Module getModel(const std::string& model_name);
    void listModels();
    void deleteModel(const std::string& model_name);
    
    // Utility methods
    torch::Device getDevice() const { return device; }
    std::string getModelsDir() const { return models_dir; }
};

#endif // MODEL_MANAGER_H
