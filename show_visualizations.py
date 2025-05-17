import os
import sys
import glob
import re
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QHBoxLayout, QWidget, QLabel, QSizePolicy
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtMultimediaWidgets import QVideoWidget
from PyQt5.QtCore import Qt, QUrl, QDir
from PyQt5.QtGui import QFont, QIcon

class VideoPlayer(QMainWindow):
    def __init__(self):
        super().__init__()
        
        # Get all SNGS folders and sort them numerically
        self.base_path = "data/predictions/SoccerNet/SN-GSR-2025/test"
        folder_pattern = os.path.join(self.base_path, "SNGS-*")
        self.folders = sorted(glob.glob(folder_pattern), 
                             key=lambda x: int(re.search(r'SNGS-(\d+)', x).group(1)))
        
        if not self.folders:
            print("No video folders found!")
            sys.exit(1)
            
        self.current_index = 0
        
        # Set up the UI as a dialog
        self.setWindowTitle("Video Player")
        self.setGeometry(100, 100, 800, 900)  # Appropriate size for a dialog with vertical video
        
        # Create video widget
        self.video_widget = QVideoWidget()
        self.video_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.video_widget.setAspectRatioMode(Qt.KeepAspectRatio)  # Maintain aspect ratio
        
        # Create media player
        self.media_player = QMediaPlayer(None, QMediaPlayer.VideoSurface)
        self.media_player.setVideoOutput(self.video_widget)
        
        # Create status label
        self.status_label = QLabel("Loading...")
        self.status_label.setAlignment(Qt.AlignCenter)
        font = QFont()
        font.setPointSize(12)
        self.status_label.setFont(font)
        self.status_label.setStyleSheet("color: white; background-color: rgba(0, 0, 0, 128);")
        
        # Create navigation buttons
        self.prev_button = QPushButton("←")  # Left arrow
        self.prev_button.clicked.connect(self.play_previous)
        self.prev_button.setFixedSize(50, 200)
        self.prev_button.setFont(QFont('Arial', 20))
        
        self.next_button = QPushButton("→")  # Right arrow
        self.next_button.clicked.connect(self.play_next)
        self.next_button.setFixedSize(50, 200)
        self.next_button.setFont(QFont('Arial', 20))
        
        # Create layouts - main layout is horizontal with buttons on sides
        main_layout = QHBoxLayout()
        main_layout.addWidget(self.prev_button)
        
        # Center video and status in vertical layout
        center_layout = QVBoxLayout()
        center_layout.addWidget(self.video_widget)
        center_layout.addWidget(self.status_label)
        
        # Add center layout to main layout
        center_widget = QWidget()
        center_widget.setLayout(center_layout)
        main_layout.addWidget(center_widget)
        
        main_layout.addWidget(self.next_button)
        
        # Set stretch factors to make video take most space
        main_layout.setStretchFactor(self.prev_button, 0)
        main_layout.setStretchFactor(center_widget, 10)
        main_layout.setStretchFactor(self.next_button, 0)
        
        # Create central widget
        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)
        
        # Add keyboard shortcuts
        self.setFocusPolicy(Qt.StrongFocus)
        
        # Load first video
        self.load_video(self.current_index)
    
    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            self.close()
        elif event.key() == Qt.Key_Left:
            self.play_previous()
        elif event.key() == Qt.Key_Right:
            self.play_next()
        else:
            super().keyPressEvent(event)
    
    def load_video(self, index):
        if 0 <= index < len(self.folders):
            self.current_index = index
            folder = self.folders[index]
            folder_name = os.path.basename(folder)
            
            # Set window title
            self.setWindowTitle(folder_name)
            
            # Construct video path
            video_path = os.path.join(folder, "visualizer_yolo", "combined_view.mp4")
            
            if os.path.exists(video_path):
                # Update status and load video
                self.status_label.setText(f"{folder_name} ({index + 1}/{len(self.folders)})")
                
                # Convert to absolute path and QUrl
                abs_path = os.path.abspath(video_path)
                url = QUrl.fromLocalFile(abs_path)
                
                self.media_player.setMedia(QMediaContent(url))
                self.media_player.play()
            else:
                self.status_label.setText(f"Error: Video not found for {folder_name}")
                
            # Update button states
            self.prev_button.setEnabled(index > 0)
            self.next_button.setEnabled(index < len(self.folders) - 1)
    
    def play_next(self):
        if self.current_index < len(self.folders) - 1:
            self.load_video(self.current_index + 1)
    
    def play_previous(self):
        if self.current_index > 0:
            self.load_video(self.current_index - 1)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    player = VideoPlayer()
    player.show()
    sys.exit(app.exec_())

