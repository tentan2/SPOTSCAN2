#include "main_window.h"
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QMessageBox>
#include <QPixmap>
#include <QImage>
#include <QFileInfo>
#include <QDir>

MainWindow::MainWindow(ModelManager* model_manager, QWidget* parent)
    : QMainWindow(parent), model_manager(model_manager) {
    
    setWindowTitle("Spotscan - C++ Food Analysis");
    setMinimumSize(1200, 800);
    resize(1400, 900);
    
    // Initialize image processor
    image_processor = new ImageProcessor();
    
    setupUI();
    setupMenuBar();
    setupStatusBar();
    connectSignals();
}

MainWindow::~MainWindow() {
    delete image_processor;
}

void MainWindow::setupUI() {
    // Create central widget
    central_widget = new QWidget();
    setCentralWidget(central_widget);
    
    main_layout = new QVBoxLayout(central_widget);
    
    // Create tab widget
    tab_widget = new QTabWidget();
    main_layout->addWidget(tab_widget);
    
    // Setup image loading tab
    QWidget* image_tab = new QWidget();
    QVBoxLayout* image_layout = new QVBoxLayout(image_tab);
    
    // Image display area
    image_label = new QLabel("No image loaded");
    image_label->setAlignment(Qt::AlignCenter);
    image_label->setMinimumSize(400, 300);
    image_label->setStyleSheet("QLabel { border: 2px dashed #aaa; min-height: 300px; }");
    
    // Load image button
    load_image_button = new QPushButton("Load Image");
    load_image_button->setMaximumWidth(200);
    
    image_layout->addWidget(image_label);
    image_layout->addWidget(load_image_button);
    
    tab_widget->addTab(image_tab, "Image");
    
    // Setup analysis tab
    setupAnalysisTab();
    
    // Setup results tab
    QWidget* results_tab = new QWidget();
    QVBoxLayout* results_layout = new QVBoxLayout(results_tab);
    
    results_text = new QTextEdit();
    results_text->setReadOnly(true);
    results_text->setMaximumHeight(200);
    
    results_layout->addWidget(results_text);
    tab_widget->addTab(results_tab, "Results");
}

void MainWindow::setupAnalysisTab() {
    analysis_tab = new QWidget();
    analysis_layout = new QVBoxLayout(analysis_tab);
    
    // Analysis type selection
    analysis_type_combo = new QComboBox();
    analysis_type_combo->addItems({
        "Food Detection",
        "Nutritional Analysis", 
        "ViT Food Classification",
        "Freshness Detection",
        "Texture Analysis",
        "Temperature Analysis",
        "Portion Analysis",
        "Sustainability Detection"
    });
    
    // Analysis checkboxes
    food_detection_check = new QCheckBox("Food Detection");
    nutrition_analysis_check = new QCheckBox("Nutritional Analysis");
    vit_classification_check = new QCheckBox("ViT Food Classification");
    freshness_detection_check = new QCheckBox("Freshness Detection");
    texture_analysis_check = new QCheckBox("Texture Analysis");
    temperature_analysis_check = new QCheckBox("Temperature Analysis");
    portion_analysis_check = new QCheckBox("Portion Analysis");
    sustainability_detection_check = new QCheckBox("Sustainability Detection");
    
    // Analyze button
    analyze_button = new QPushButton("Analyze Image");
    analyze_button->setEnabled(false);
    analyze_button->setMaximumWidth(200);
    
    // Progress bar
    progress_bar = new QProgressBar();
    progress_bar->setVisible(false);
    
    // Add to layout
    analysis_layout->addWidget(new QLabel("Select Analysis Type:"));
    analysis_layout->addWidget(analysis_type_combo);
    analysis_layout->addWidget(new QLabel("Available Analyses:"));
    analysis_layout->addWidget(food_detection_check);
    analysis_layout->addWidget(nutrition_analysis_check);
    analysis_layout->addWidget(vit_classification_check);
    analysis_layout->addWidget(freshness_detection_check);
    analysis_layout->addWidget(texture_analysis_check);
    analysis_layout->addWidget(temperature_analysis_check);
    analysis_layout->addWidget(portion_analysis_check);
    analysis_layout->addWidget(sustainability_detection_check);
    analysis_layout->addWidget(analyze_button);
    analysis_layout->addWidget(progress_bar);
    
    tab_widget->addTab(analysis_tab, "Analysis");
}

void MainWindow::connectSignals() {
    connect(load_image_button, &QPushButton::clicked, this, &MainWindow::loadImage);
    connect(analyze_button, &QPushButton::clicked, this, &MainWindow::analyzeImage);
}

void MainWindow::loadImage() {
    QString file_path = QFileDialog::getOpenFileName(
        this,
        "Open Image",
        "",
        "Image Files (*.png *.jpg *.jpeg *.bmp);;All Files (*)"
    );
    
    if (!file_path.isEmpty()) {
        current_image_path = file_path;
        
        // Load and display image
        QPixmap pixmap(file_path);
        image_label->setPixmap(pixmap.scaled(400, 300, Qt::KeepAspectRatio, Qt::SmoothTransformation));
        
        // Enable analyze button
        analyze_button->setEnabled(true);
        
        statusBar()->showMessage("Image loaded: " + QFileInfo(file_path).fileName(), 3000);
    }
}

void MainWindow::analyzeImage() {
    if (current_image_path.isEmpty()) {
        QMessageBox::warning(this, "Warning", "Please load an image first.");
        return;
    }
    
    // Get selected analyses
    QStringList selected_analyses;
    if (food_detection_check->isChecked()) selected_analyses << "Food Detection";
    if (nutrition_analysis_check->isChecked()) selected_analyses << "Nutritional Analysis";
    if (vit_classification_check->isChecked()) selected_analyses << "ViT Food Classification";
    if (freshness_detection_check->isChecked()) selected_analyses << "Freshness Detection";
    if (texture_analysis_check->isChecked()) selected_analyses << "Texture Analysis";
    if (temperature_analysis_check->isChecked()) selected_analyses << "Temperature Analysis";
    if (portion_analysis_check->isChecked()) selected_analyses << "Portion Analysis";
    if (sustainability_detection_check->isChecked()) selected_analyses << "Sustainability Detection";
    
    if (selected_analyses.isEmpty()) {
        QMessageBox::warning(this, "Warning", "Please select at least one analysis type.");
        return;
    }
    
    // Show progress
    progress_bar->setVisible(true);
    progress_bar->setRange(0, 0); // Indeterminate progress
    analyze_button->setEnabled(false);
    
    // Perform analysis
    QString results = performAnalysis(current_image_path, selected_analyses);
    
    // Hide progress
    progress_bar->setVisible(false);
    analyze_button->setEnabled(true);
    
    // Display results
    onAnalysisComplete(results);
    
    statusBar()->showMessage("Analysis complete", 3000);
}

void MainWindow::onAnalysisComplete(const QString& results) {
    results_text->setPlainText(results);
    tab_widget->setCurrentIndex(2); // Switch to results tab
}

QString MainWindow::performAnalysis(const QString& image_path, const QStringList& selected_analyses) {
    QString results = "=== ANALYSIS RESULTS ===\\n\\n";
    results += "Image: " + QFileInfo(image_path).fileName() + "\\n\\n";
    
    // Load and preprocess image
    cv::Mat image = image_processor->loadImage(image_path.toStdString());
    if (image.empty()) {
        return "Error: Could not load image for analysis.";
    }
    
    cv::Mat processed_image = image_processor->preprocessImage(image);
    
    // Perform selected analyses
    for (const QString& analysis : selected_analyses) {
        results += "--- " + analysis + " ---\\n";
        
        if (analysis == "Food Detection") {
            results += "Food detected: [Placeholder - requires model]\\n";
            results += "Confidence: 85%\\n";
        } else if (analysis == "ViT Food Classification") {
            results += "ViT Classification: [Placeholder - requires model]\\n";
            results += "Top prediction: Apple\\n";
            results += "Confidence: 92%\\n";
        } else if (analysis == "Nutritional Analysis") {
            results += "Calories: 95 kcal\\n";
            results += "Protein: 1.2g\\n";
            results += "Carbs: 21g\\n";
            results += "Fat: 0.3g\\n";
        } else if (analysis == "Freshness Detection") {
            results += "Freshness: Fresh\\n";
            results += "Quality Score: 8.5/10\\n";
        } else if (analysis == "Texture Analysis") {
            results += "Texture: Crisp\\n";
            results += "Hardness: 7.2/10\\n";
        } else if (analysis == "Temperature Analysis") {
            results += "Temperature: 18°C\\n";
            results += "Optimal serving: Room temperature\\n";
        } else if (analysis == "Portion Analysis") {
            results += "Portion size: 250g\\n";
            results += "Servings: 2.5\\n";
        } else if (analysis == "Sustainability Detection") {
            results += "Organic: Yes\\n";
            results += "Fair Trade: No\\n";
            results += "Sustainability Score: 7/10\\n";
        }
        
        results += "\\n";
    }
    
    results += "\\n=== ANALYSIS COMPLETE ===\\n";
    results += "\\nNote: This is a C++ demonstration version.\\n";
    results += "Full model integration requires PyTorch C++ library setup.\\n";
    
    return results;
}
