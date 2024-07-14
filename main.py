import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton, QFileDialog, QSlider, QCheckBox
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.uic import loadUi
from scipy.ndimage import prewitt

class EdgeDetectionApp(QMainWindow):
    def __init__(self):
        super(EdgeDetectionApp, self).__init__()
        loadUi('untitled.ui', self)

        # Menemukan elemen UI
        self.imgLabel = self.findChild(QLabel, 'imgLabel')
        self.hasilImage = self.findChild(QLabel, 'hasilImage')
        self.loadButton = self.findChild(QPushButton, 'loadButton')
        self.kernelSizeSlider = self.findChild(QSlider, 'kernelSizeSlider')
        self.normalizeCheckbox = self.findChild(QCheckBox, 'normalizeCheckbox')

        # Menghubungkan tombol ke fungsi
        self.loadButton.clicked.connect(self.load_image)
        self.actionSobel.triggered.connect(self.run_sobel)
        self.actionPrewitt.triggered.connect(self.run_prewitt)
        self.actionCanny.triggered.connect(self.run_canny)

        self.original_image = None

        # Mengatur slider agar hanya menerima nilai ganjil yang valid
        self.kernelSizeSlider.setMinimum(3)
        self.kernelSizeSlider.setMaximum(31)
        self.kernelSizeSlider.setSingleStep(2)

    def load_image(self):
        options = QFileDialog.Options()
        filename, _ = QFileDialog.getOpenFileName(self, "Open Image File", "", "Image Files (*.png *.jpg *.bmp)", options=options)
        if filename:
            self.original_image = cv2.imread(filename)
            self.display_image(self.original_image, self.imgLabel)

    def display_image(self, image, label):
        qformat = QImage.Format_Indexed8
        if len(image.shape) == 3:  # rows[0], cols[1], channels[2]
            if image.shape[2] == 4:
                qformat = QImage.Format_RGBA8888
            else:
                qformat = QImage.Format_RGB888

        img = QImage(image, image.shape[1], image.shape[0], image.strides[0], qformat)
        img = img.rgbSwapped()
        label.setPixmap(QPixmap.fromImage(img))
        label.setScaledContents(True)

    def run_sobel(self):
        if self.original_image is not None:
            gray = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
            ksize = self.kernelSizeSlider.value()
            if ksize % 2 == 1:  # Pastikan kernel size adalah ganjil
                sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=ksize)
                sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=ksize)
                sobel = cv2.magnitude(sobelx, sobely)
                if self.normalizeCheckbox.isChecked():
                    sobel = np.uint8(sobel / sobel.max() * 255)
                else:
                    sobel = np.uint8(sobel)
                self.display_image(sobel, self.hasilImage)
            else:
                print(f"Invalid kernel size: {ksize}")

    def run_prewitt(self):
        if self.original_image is not None:
            gray = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (3, 3), 0)  # Apply Gaussian blur to reduce noise
            prewittx = prewitt(gray, axis=0)
            prewitty = prewitt(gray, axis=1)
            prewitt_result = np.hypot(prewittx, prewitty)
            if self.normalizeCheckbox.isChecked():
                prewitt_result = np.uint8((prewitt_result / prewitt_result.max()) * 255)
            else:
                prewitt_result = np.uint8(prewitt_result)
            self.display_image(prewitt_result, self.hasilImage)

    def run_canny(self):
        if self.original_image is not None:
            gray = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
            low_threshold = self.kernelSizeSlider.value()
            high_threshold = low_threshold * 2
            edges = cv2.Canny(gray, low_threshold, high_threshold)
            if self.normalizeCheckbox.isChecked():
                edges = np.uint8(edges / edges.max() * 255)
            else:
                edges = np.uint8(edges)
            self.display_image(edges, self.hasilImage)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = EdgeDetectionApp()
    window.show()
    sys.exit(app.exec_())
