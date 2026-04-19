#include "ocr_analyzer.h"
#include <iostream>
#include <algorithm>
#include <sstream>

OCRAnalyzer::OCRAnalyzer(const torch::Device& device) 
    : device(device) {
    
    // Initialize OCR database
    ocr_database["ingredients"] = "ingredients,ingredientes,nutrition facts,nutritional information";
    ocr_database["allergens"] = "contains,may contain,processed in facility with nuts";
    ocr_database["calories"] = "calories,energy,kcal,kilocalories";
    ocr_database["protein"] = "protein,proteins";
    ocr_database["carbs"] = "carbohydrates,carbs,total carbohydrate";
    ocr_database["fat"] = "fat,fats,total fat";
    
    initializeOCRDatabase();
    std::cout << "OCRAnalyzer initialized" << std::endl;
}

bool OCRAnalyzer::loadModel(const std::string& model_path) {
    try {
        model = torch::jit::load(model_path);
        model->to(device);
        std::cout << "OCR analysis model loaded from: " << model_path << std::endl;
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Error loading OCR analysis model: " << e.what() << std::endl;
        return false;
    }
}

std::map<std::string, std::string> OCRAnalyzer::analyzeOCR(const cv::Mat& image) {
    std::map<std::string, std::string> results;
    
    // Extract text from image
    std::string extracted_text = extractText(image);
    
    // Analyze extracted text
    std::vector<std::string> ingredients = detectIngredients(extracted_text);
    std::vector<std::string> nutrition_info = detectNutritionInfo(extracted_text);
    std::vector<std::string> allergens = detectAllergens(extracted_text);
    
    // Format results
    results["extracted_text"] = extracted_text;
    results["ingredients"] = ingredients.empty() ? "None detected" : ingredients[0];
    results["nutrition_info"] = nutrition_info.empty() ? "None detected" : nutrition_info[0];
    results["allergens"] = allergens.empty() ? "None detected" : allergens[0];
    
    return results;
}

std::string OCRAnalyzer::extractText(const cv::Mat& image) {
    // Convert to grayscale for OCR
    cv::Mat gray_image;
    cv::cvtColor(image, gray_image, cv::COLOR_BGR2GRAY);
    
    // Apply threshold for better text detection
    cv::Mat threshold_image;
    cv::threshold(gray_image, threshold_image, 127, 255, cv::THRESH_BINARY);
    
    // In a real implementation, you would use Tesseract OCR here
    // For now, simulate text extraction
    std::string simulated_text = "Ingredients: Wheat flour, sugar, eggs, milk. Nutrition Facts: Serving size 1 slice (50g). Calories 200. Protein 3g. Carbs 30g. Fat 8g.";
    
    return simulated_text;
}

std::vector<std::string> OCRAnalyzer::detectIngredients(const std::string& text) {
    std::vector<std::string> ingredients;
    
    // Convert to lowercase for case-insensitive matching
    std::string lower_text = text;
    std::transform(lower_text.begin(), lower_text.end(), lower_text.begin(), ::tolower);
    
    // Look for ingredient keywords
    std::string ingredient_keywords = ocr_database["ingredients"];
    std::vector<std::string> keywords = splitText(ingredient_keywords);
    
    for (const auto& keyword : keywords) {
        if (lower_text.find(keyword) != std::string::npos) {
            ingredients.push_back(keyword);
        }
    }
    
    return ingredients;
}

std::vector<std::string> OCRAnalyzer::detectNutritionInfo(const std::string& text) {
    std::vector<std::string> nutrition_info;
    
    // Convert to lowercase for case-insensitive matching
    std::string lower_text = text;
    std::transform(lower_text.begin(), lower_text.end(), lower_text.begin(), ::tolower);
    
    // Look for nutrition keywords
    std::string nutrition_keywords = ocr_database["calories"];
    std::vector<std::string> keywords = splitText(nutrition_keywords);
    
    for (const auto& keyword : keywords) {
        if (lower_text.find(keyword) != std::string::npos) {
            nutrition_info.push_back(keyword);
        }
    }
    
    return nutrition_info;
}

std::vector<std::string> OCRAnalyzer::detectAllergens(const std::string& text) {
    std::vector<std::string> allergens;
    
    // Convert to lowercase for case-insensitive matching
    std::string lower_text = text;
    std::transform(lower_text.begin(), lower_text.end(), lower_text.begin(), ::tolower);
    
    // Look for allergen keywords
    std::string allergen_keywords = ocr_database["allergens"];
    std::vector<std::string> keywords = splitText(allergen_keywords);
    
    for (const auto& keyword : keywords) {
        if (lower_text.find(keyword) != std::string::npos) {
            allergens.push_back(keyword);
        }
    }
    
    return allergens;
}

void OCRAnalyzer::initializeOCRDatabase() {
    // Initialize with common OCR keywords
    std::cout << "OCR database initialized" << std::endl;
}

void OCRAnalyzer::addOCRData(const std::string& food, const std::string& info) {
    ocr_database[food] = info;
}

std::string OCRAnalyzer::getExpectedOCR(const std::string& food) {
    auto it = ocr_database.find(food);
    if (it != ocr_database.end()) {
        return it->second;
    }
    return "No OCR data available";
}

std::string OCRAnalyzer::cleanText(const std::string& text) {
    std::string cleaned_text = text;
    
    // Remove extra whitespace
    cleaned_text.erase(std::remove_if(cleaned_text.begin(), cleaned_text.end(), ::isspace), cleaned_text.end());
    
    // Remove special characters
    cleaned_text.erase(std::remove_if(cleaned_text.begin(), cleaned_text.end(), 
        [](char c) { return !std::isalnum(c) && c != ' '; }), cleaned_text.end());
    
    return cleaned_text;
}

std::vector<std::string> OCRAnalyzer::splitText(const std::string& text) {
    std::vector<std::string> tokens;
    std::stringstream ss(text);
    std::string token;
    
    while (std::getline(ss, token, ',')) {
        tokens.push_back(token);
    }
    
    return tokens;
}
