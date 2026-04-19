#include <QApplication>
#include <QMainWindow>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QLabel>
#include <QPushButton>
#include <QFileDialog>
#include <QMessageBox>
#include <QImage>
#include <QPixmap>
#include <iostream>

#include "model_manager.h"
#include "image_processor.h"
#include "ui/main_window.h"

int main(int argc, char *argv[]) {
    QApplication app(argc, argv);
    
    // Initialize model manager
    ModelManager model_manager("models");
    
    // Create main window
    MainWindow main_window(model_manager);
    main_window.show();
    
    return app.exec();
}
