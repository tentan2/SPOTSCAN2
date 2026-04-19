#ifndef MAIN_WINDOW_H
#define MAIN_WINDOW_H

#include <QMainWindow>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QLabel>
#include <QPushButton>
#include <QFileDialog>
#include <QMessageBox>
#include <QImage>
#include <QPixmap>
#include <QTabWidget>
#include <QScrollArea>
#include <QProgressBar>
#include <QTextEdit>
#include <QComboBox>
#include <QCheckBox>
#include <QSlider>
#include <QSpinBox>

#include "../model_manager.h"
#include "../image_processor.h"

class MainWindow : public QMainWindow {
    Q_OBJECT

private:
    ModelManager* model_manager;
    ImageProcessor* image_processor;
    
    // UI Components
    QWidget* central_widget;
    QVBoxLayout* main_layout;
    QTabWidget* tab_widget;
    
    // Image display
    QLabel* image_label;
    QPushButton* load_image_button;
    QString current_image_path;
    
    // Analysis panel
    QWidget* analysis_tab;
    QVBoxLayout* analysis_layout;
    QComboBox* analysis_type_combo;
    QPushButton* analyze_button;
    QProgressBar* progress_bar;
    QTextEdit* results_text;
    
    // Analysis options
    QCheckBox* food_detection_check;
    QCheckBox* nutrition_analysis_check;
    QCheckBox* vit_classification_check;
    QCheckBox* freshness_detection_check;
    QCheckBox* texture_analysis_check;
    QCheckBox* temperature_analysis_check;
    QCheckBox* portion_analysis_check;
    QCheckBox* sustainability_detection_check;
    
public:
    explicit MainWindow(ModelManager* model_manager, QWidget* parent = nullptr);
    ~MainWindow() = default;
    
private slots:
    void loadImage();
    void analyzeImage();
    void onAnalysisComplete(const QString& results);
    
private:
    void setupUI();
    void setupMenuBar();
    void setupStatusBar();
    void connectSignals();
    void updateAnalysisOptions();
    QString performAnalysis(const QString& image_path, const QStringList& selected_analyses);
};

#endif // MAIN_WINDOW_H
