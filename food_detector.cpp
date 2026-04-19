#include "food_detector.h"
#include <iostream>
#include <algorithm>

FoodDetector::FoodDetector(const torch::Device& device) 
    : device(device) {
    
    // Initialize class names (Food-101 dataset)
    class_names = {
        "apple_pie", "baby_back_ribs", "baklava", "beef_carpaccio", "beef_tartare",
        "beet_salad", "beignets", "bibimbap", "bread_pudding", "breakfast_burrito",
        "bruschetta", "chicken_curry", "chicken_wings", "chocolate_cake", "chocolate_mousse",
        "churros", "clam_chowder", "club_sandwich", "crab_cakes", "creme_brulee",
        "croque_madame", "cup_cakes", "deviled_eggs", "donuts", "dumplings", "edamame",
        "eggs_benedict", "escargots", "falafel", "filet_mignon", "fish_and_chips",
        "foie_gras", "french_fries", "french_onion_soup", "french_toast", "fried_calamari",
        "fried_rice", "frozen_yogurt", "garlic_bread", "gnocchi", "greek_salad",
        "grilled_cheese_sandwich", "grilled_salmon", "guacamole", "gumbo", "hamburger",
        "hot_and_sour_soup", "hot_dog", "huevos_rancheros", "ice_cream", "lasagna",
        "lobster_bisque", "lobster_roll_sandwich", "macaroni_and_cheese", "macarons",
        "miso_soup", "mussels", "nachos", "omelette", "onion_rings",
        "oysters", "pad_thai", "paella", "pancakes", "panna_cotta",
        "peking_duck", "pho", "pizza", "pork_chop", "poutine",
        "prime_rib", "pulled_pork_sandwich", "ramen", "ravioli", "red_velvet_cake",
        "risotto", "samosa", "sashimi", "scallops", "seaweed_salad",
        "shrimp_scampi", "spaghetti_bolognese", "spaghetti_carbonara", "spring_rolls",
        "steak", "strawberry_shortcake", "sushi", "tacos", "takoyaki",
        "tiramisu", "tuna_tartare", "waffles", "wiener_schnitzel"
    };
    
    std::cout << "FoodDetector initialized with " << class_names.size() << " classes" << std::endl;
}

bool FoodDetector::loadModel(const std::string& model_path) {
    try {
        model = torch::jit::load(model_path);
        model->to(device);
        std::cout << "Food detection model loaded from: " << model_path << std::endl;
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Error loading food detection model: " << e.what() << std::endl;
        return false;
    }
}

std::pair<std::string, float> FoodDetector::detectFood(const cv::Mat& image) {
    if (!model) {
        return {"Error", 0.0f};
    }
    
    // Preprocess image
    cv::Mat processed_image;
    cv::resize(image, processed_image, cv::Size(224, 224));
    cv::cvtColor(processed_image, processed_image, cv::COLOR_BGR2RGB);
    
    // Convert to tensor
    auto tensor = torch::from_blob(processed_image.data, {1, 224, 224, 3}, torch::kByte);
    tensor = tensor.to(torch::kFloat);
    tensor = tensor.to(device);
    
    // Normalize
    tensor = tensor.div(255.0);
    
    // Add batch dimension
    tensor = tensor.unsqueeze(0);
    
    try {
        // Forward pass
        torch::NoGradGuard no_grad;
        auto output = model->forward({tensor});
        
        // Get predictions
        auto probabilities = torch::softmax(output, 1);
        auto [confidence, predicted_class] = torch::max(probabilities, 1);
        
        std::string predicted_class = class_names[predicted_class.item<int>()];
        float confidence_value = confidence.item<float>();
        
        return {predicted_class, confidence_value};
        
    } catch (const std::exception& e) {
        std::cerr << "Error during food detection: " << e.what() << std::endl;
        return {"Error", 0.0f};
    }
}

std::vector<std::pair<std::string, float>> FoodDetector::detectMultipleFoods(const cv::Mat& image) {
    std::vector<std::pair<std::string, float>> results;
    
    // For multiple food detection, you could implement:
    // 1. Object detection (YOLO, SSD)
    // 2. Segmentation-based detection
    // 3. Sliding window approach
    
    // For now, return single detection
    auto result = detectFood(image);
    results.push_back(result);
    
    return results;
}

void FoodDetector::setClassNames(const std::vector<std::string>& names) {
    class_names = names;
}

std::vector<std::string> FoodDetector::getClassNames() const {
    return class_names;
}
