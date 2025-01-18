import sys
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QLabel, QPushButton, QFileDialog, QVBoxLayout, QWidget
)
from PyQt5.QtGui import QPixmap
import inital_notebook  # Assuming this script contains the image captioning logic

class ImageCaptionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Image Caption Generator")

        # Main widget and layout
        self.central_widget = QWidget()
        self.layout = QVBoxLayout()
        self.central_widget.setLayout(self.layout)
        self.setCentralWidget(self.central_widget)

        # Image display label
        self.image_label = QLabel("No image uploaded")
        self.image_label.setScaledContents(True)
        self.image_label.setFixedSize(400, 400)
        self.layout.addWidget(self.image_label)

        # Caption display label
        self.caption_label = QLabel("Caption will appear here")
        self.layout.addWidget(self.caption_label)

        # Upload button
        self.upload_button = QPushButton("Upload Image")
        self.upload_button.clicked.connect(self.upload_image)
        self.layout.addWidget(self.upload_button)

    def upload_image(self):
        # Open file dialog to select image
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Image File", "", "Images (*.png *.xpm *.jpg *.jpeg *.bmp)", options=options)
        if file_path:
            self.display_image(file_path)
            self.generate_caption(file_path)

    def display_image(self, file_path):
        pixmap = QPixmap(file_path)
        self.image_label.setPixmap(pixmap)

    def generate_caption(self, file_path):
        try:
            # Assuming `generate_caption` is a function in `initial_notebook.py`
            caption = inital_notebook.generate_caption(file_path)
            self.caption_label.setText(f"Caption: {caption}")
        except Exception as e:
            self.caption_label.setText(f"Error generating caption: {str(e)}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ImageCaptionApp()
    window.show()
    sys.exit(app.exec_())
