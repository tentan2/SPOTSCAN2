#include "vit_analyzer.h"
#include <chrono>
#include <cctype>

/**
 * Constructor: Initialize ViT analyzer
 */
ViTAnalyzer::ViTAnalyzer(const std::string& model_name_param,
                         const std::optional<std::string>& model_path)
    : device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU),
      model_name(model_name_param),
      confidence_threshold(0.5),
      model_loaded(false) {
    
    try {
        food_classes = load_food_classes();
        
        std::string path = model_path.has_value() ? model_path.value() : "";
        if (!load_model(path)) {
            std::cerr << "Warning: Failed to load ViT model" << std::endl;
        }
        
        std::cout << "ViT Analyzer initialized with model: " << model_name << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error during ViT initialization: " << e.what() << std::endl;
        throw;
    }
}

/**
 * Load Food-101 class labels
 */
std::vector<std::string> ViTAnalyzer::load_food_classes() {
    return {
        "apple_pie", "baby_back_ribs", "baklava", "beef_carpaccio", "beef_tartare",
        "beet_salad", "beignets", "bibimbap", "bread_pudding", "breakfast_burrito",
        "bruschetta", "caesar_salad", "cannoli", "caprese_salad", "carrot_cake",
        "ceviche", "cheese_cake", "cheese_plate", "chicken_curry", "chicken_quesadilla",
        "chicken_wings", "chocolate_cake", "chocolate_mousse", "churros", "clam_chowder",
        "club_sandwich", "crab_cakes", "creme_brulee", "croque_madame", "cup_cakes",
        "deviled_eggs", "donuts", "dumplings", "edamame", "eggs_benedict",
        "escargots", "falafel", "filet_mignon", "fish_and_chips", "foie_gras",
        "french_fries", "french_onion_soup", "french_toast", "fried_calamari", "fried_rice",
        "frozen_yogurt", "garlic_bread", "gnocchi", "greek_salad", "grilled_cheese_sandwich",
        "grilled_salmon", "guacamole", "gyoza", "hamburger", "hot_and_sour_soup",
        "hot_dog", "huevos_rancheros", "hummus", "ice_cream", "lasagna",
        "lobster_bisque", "lobster_roll_sandwich", "macaroni_and_cheese", "macarons", "miso_soup",
        "mussels", "nachos", "omelette", "onion_rings", "oysters",
        "pad_thai", "paella", "pancakes", "panna_cotta", "peking_duck",
        "pho", "pizza", "pork_chop", "poutine", "prime_rib",
        "pulled_pork_sandwich", "ramen", "ravioli", "red_velvet_cake", "risotto",
        "samosa", "sashimi", "scallops", "seaweed_salad", "shrimp_and_grits",
        "spaghetti_bolognese", "spaghetti_carbonara", "spring_rolls", "steak", "strawberry_shortcake",
        "sushi", "tacos", "takoyaki", "tiramisu", "tuna_tartare",
        "waffles"
    };
}

/**
 * Load ViT model from disk
 */
bool ViTAnalyzer::load_model(const std::string& model_path) {
    try {
        /**
         * Note: This implementation uses torch::jit::load() to load TorchScript models.
         * To use this in production:
         * 1. Convert your HuggingFace model to TorchScript using Python:
         *    import torch
         *    from transformers import ViTForImageClassification
         *    model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224")
         *    traced_model = torch.jit.trace(model, example_input)
         *    traced_model.save("vit_model.pt")
         */
        
        std::string path;
        if (!model_path.empty() && fs::exists(model_path)) {
            // Load fine-tuned local model
            path = model_path;
        } else {
            // Load default model from current directory
            path = "vit_model.pt";
            if (!fs::exists(path)) {
                std::cerr << "Model file not found at: " << path << std::endl;
                std::cerr << "Please ensure the model has been traced and saved as 'vit_model.pt'" << std::endl;
                return false;
            }
        }
        
        model = torch::jit::load(path, device);
        model.eval();
        
        std::cout << "Model loaded successfully from: " << path << std::endl;
        model_loaded = true;
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "Error loading ViT model: " << e.what() << std::endl;
        model_loaded = false;
        return false;
    }
}

/**
 * Preprocess image for ViT
 * - Converts to RGB if necessary
 * - Resizes to 224x224
 * - Normalizes with ImageNet statistics
 * - Converts to tensor and moves to device
 */
torch::Tensor ViTAnalyzer::preprocess_image(const cv::Mat& image) {
    if (image.empty()) {
        throw std::runtime_error("Input image is empty");
    }
    
    // Convert to RGB
    cv::Mat processed;
    if (image.channels() == 3) {
        cv::cvtColor(image, processed, cv::COLOR_BGR2RGB);
    } else if (image.channels() == 1) {
        cv::cvtColor(image, processed, cv::COLOR_GRAY2RGB);
    } else if (image.channels() == 4) {
        cv::cvtColor(image, processed, cv::COLOR_BGRA2RGB);
    } else {
        processed = image.clone();
    }
    
    // Resize to 224x224 (standard ViT input size)
    cv::Mat resized;
    cv::resize(processed, resized, cv::Size(224, 224), 0, 0, cv::INTER_LINEAR);
    
    // Convert to float [0, 1]
    cv::Mat float_img;
    resized.convertTo(float_img, CV_32F, 1.0 / 255.0);
    
    // Normalize with ImageNet statistics
    // Mean: [0.485, 0.456, 0.406], Std: [0.229, 0.224, 0.225]
    std::vector<cv::Mat> channels(3);
    cv::split(float_img, channels);
    
    channels[0] = (channels[0] - 0.485) / 0.229;  // R channel
    channels[1] = (channels[1] - 0.456) / 0.224;  // G channel
    channels[2] = (channels[2] - 0.406) / 0.225;  // B channel
    
    cv::Mat normalized;
    cv::merge(channels, normalized);
    
    // Convert to tensor (H x W x C -> C x H x W)
    auto tensor = torch::from_blob(normalized.data,
                                   {1, 224, 224, 3},
                                   torch::kFloat32);
    
    // Permute to (batch, channels, height, width)
    tensor = tensor.permute({0, 3, 1, 2}).contiguous();
    
    // Move to device
    tensor = tensor.to(device);
    
    return tensor;
}

/**
 * Analyze food image using ViT
 */
AnalysisResult ViTAnalyzer::analyze(const cv::Mat& image, int top_k) {
    AnalysisResult result;
    result.model = "ViT";
    result.processing_time = 0.0;
    
    if (!model_loaded) {
        result.error = "Model not loaded. Please ensure model file exists.";
        return result;
    }
    
    try {
        auto start_time = std::chrono::high_resolution_clock::now();
        
        // Preprocess image
        torch::Tensor pixel_values = preprocess_image(image);
        
        // Run inference
        torch::NoGradGuard no_grad;
        
        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(pixel_values);
        
        torch::jit::IValue output = model.forward(inputs);
        
        // Extract logits
        torch::Tensor logits;
        
        // Handle different output formats
        if (output.isTensor()) {
            logits = output.toTensor();
        } else if (output.isTuple()) {
            auto outputs_tuple = output.toTuple();
            if (outputs_tuple->elements().size() > 0) {
                logits = outputs_tuple->elements()[0].toTensor();
            } else {
                throw std::runtime_error("Empty output tuple from model");
            }
        } else {
            throw std::runtime_error("Unexpected output type from model");
        }
        
        // Compute softmax probabilities
        torch::Tensor probabilities = torch::nn::functional::softmax(
            logits,
            torch::nn::functional::SoftmaxFuncOptions(-1)
        );
        
        // Get top k predictions
        int k_actual = std::min(top_k, static_cast<int>(food_classes.size()));
        auto [top_probs, top_indices] = torch::topk(probabilities, k_actual, 1, true);
        
        // Move tensors to CPU for safe access
        top_probs = top_probs.cpu();
        top_indices = top_indices.cpu();
        
        // Access tensor data
        auto probs_accessor = top_probs.accessor<float, 2>();
        auto indices_accessor = top_indices.accessor<long, 2>();
        
        // Prepare predictions
        for (int i = 0; i < k_actual; ++i) {
            int idx = static_cast<int>(indices_accessor[0][i]);
            float prob = probs_accessor[0][i];
            
            if (idx >= 0 && idx < static_cast<int>(food_classes.size())) {
                Prediction pred{
                    food_classes[idx],
                    static_cast<double>(prob),
                    idx
                };
                
                result.predictions.push_back(pred);
                result.confidence_scores[pred.class_name] = pred.confidence;
            }
        }
        
        // Filter by confidence threshold
        std::vector<Prediction> filtered;
        for (const auto& pred : result.predictions) {
            if (pred.confidence >= confidence_threshold) {
                filtered.push_back(pred);
            }
        }
        
        result.predictions = filtered;
        
        // Set top prediction
        if (!result.predictions.empty()) {
            result.top_prediction = result.predictions[0];
        }
        
        // Store input shape
        result.input_shape = {image.rows, image.cols, image.channels()};
        
        auto end_time = std::chrono::high_resolution_clock::now();
        result.processing_time = std::chrono::duration<double>(end_time - start_time).count();
        
    } catch (const std::exception& e) {
        std::cerr << "Error in ViT analysis: " << e.what() << std::endl;
        result.error = e.what();
    }
    
    return result;
}

/**
 * Get broader food category from specific prediction
 */
std::string ViTAnalyzer::get_food_category(const Prediction& prediction) const {
    if (prediction.class_name.empty()) {
        return "unknown";
    }
    
    // Convert to lowercase
    std::string food_class = prediction.class_name;
    std::transform(food_class.begin(), food_class.end(), food_class.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    
    // Category mapping
    const std::map<std::string, std::vector<std::string>> category_map = {
        {"desserts", {"cake", "pudding", "mousse", "creme", "tart", "pie", "cookies", "donuts", "waffles", "pancakes"}},
        {"main_dishes", {"chicken", "beef", "pork", "fish", "salmon", "steak", "ribs", "tartare", "carpaccio"}},
        {"appetizers", {"salad", "soup", "wings", "calamari", "oysters", "scallops", "mussels", "bruschetta"}},
        {"asian_cuisine", {"sushi", "pho", "ramen", "pad_thai", "bibimbap", "takoyaki", "gyoza", "dumplings"}},
        {"italian_cuisine", {"pizza", "pasta", "lasagna", "ravioli", "risotto", "gnocchi", "carbonara", "bolognese"}},
        {"mexican_cuisine", {"tacos", "quesadilla", "burrito", "nachos", "huevos", "guacamole"}},
        {"breakfast", {"eggs", "pancakes", "waffles", "french_toast", "omelette"}},
        {"sides", {"fries", "onion_rings", "garlic_bread", "macaroni", "rice", "potato"}},
        {"seafood", {"fish", "salmon", "shrimp", "crab", "lobster", "mussels", "oysters", "scallops"}}
    };
    
    for (const auto& [category, foods] : category_map) {
        for (const auto& food : foods) {
            if (food_class.find(food) != std::string::npos) {
                return category;
            }
        }
    }
    
    return "other";
}

/**
 * Get information about fine-tuning the model
 */
json ViTAnalyzer::get_fine_tune_info() const {
    json info = {
        {"current_model", model_name},
        {"num_classes", food_classes.size()},
        {"dataset_recommendation", "Food-101 dataset"},
        {"training_tips", json::array({
            "Use Food-101 dataset for food-specific fine-tuning",
            "Consider data augmentation for better generalization",
            "Start with a lower learning rate (1e-5)",
            "Use early stopping to prevent overfitting",
            "Fine-tune on a subset first, then full dataset"
        })},
        {"hyperparameters", {
            {"learning_rate", "1e-5 to 5e-5"},
            {"batch_size", "16-32"},
            {"epochs", "10-20"},
            {"weight_decay", "0.01"}
        }}
    };
    
    return info;
}
