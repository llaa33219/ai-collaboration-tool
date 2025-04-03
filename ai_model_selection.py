import os
import sys
import json
import uuid
import base64
import threading
import subprocess
from datetime import datetime
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QTextEdit, QLineEdit, QPushButton, QLabel, QSplitter, QToolBar,
    QStatusBar, QAction, QFileDialog, QMessageBox, QDialog, QFormLayout,
    QCheckBox, QDialogButtonBox, QGroupBox, QComboBox, QSizePolicy
)
from PyQt5.QtCore import Qt, QObject, pyqtSignal, QTimer

# 클릭 가능한 라벨 클래스
class ClickableLabel(QLabel):
    """클릭 가능한 라벨 위젯"""
    clicked = pyqtSignal()
    
    def __init__(self, text="", parent=None):
        super().__init__(text, parent)
        self.setCursor(Qt.PointingHandCursor)  # 마우스 커서를 손가락 모양으로 변경
        
    def mousePressEvent(self, event):
        """마우스 클릭 이벤트 처리"""
        self.clicked.emit()
        super().mousePressEvent(event)


class AIModelSelectionDialog(QDialog):
    """AI 모델 선택 대화 상자"""
    def __init__(self, parent=None, agent_name="", current_model=None, models=None):
        super().__init__(parent)
        self.setWindowTitle(f"{agent_name} 모델 선택")
        self.resize(400, 300)
        
        layout = QVBoxLayout(self)
        
        # 설명 라벨
        description = QLabel(f"{agent_name} 에이전트에서 사용할 AI 모델을 선택하세요:")
        layout.addWidget(description)
        
        # 모델 선택 콤보박스
        self.model_combo = QComboBox()
        self.model_combo.setMinimumWidth(350)
        
        # 모델 목록 로드
        self.load_models(agent_name, models, current_model)
        
        layout.addWidget(self.model_combo)
        
        # 버튼
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
    
    def load_models(self, agent_name, models, current_model):
        """모델 목록을 콤보박스에 로드"""
        try:
            # 에이전트 유형에 따른 모델 목록 키 결정
            model_key = ""
            if agent_name == "ChatGPT":
                model_key = "openai"
            elif agent_name == "Claude":
                model_key = "anthropic"
            elif agent_name == "Gemini":
                model_key = "google"
            elif agent_name == "DeepSeek":
                model_key = "deepseek"
            elif agent_name == "Perplexity":
                model_key = "perplexity"
            elif agent_name == "Qwen":
                model_key = "qwen"
            
            # 콤보박스 초기화
            self.model_combo.clear()
            
            # 모델 목록이 있으면 추가
            if model_key in models and models[model_key]:
                for model in models[model_key]:
                    self.model_combo.addItem(f"{model['name']} - {model['description']}", model['name'])
                
                # 현재 선택된 모델이 있으면 해당 모델 선택
                if current_model:
                    for i in range(self.model_combo.count()):
                        if self.model_combo.itemData(i) == current_model:
                            self.model_combo.setCurrentIndex(i)
                            break
        
        except Exception as e:
            print(f"Error loading models: {e}")
    
    def get_selected_model(self):
        """선택된 모델 반환"""
        return self.model_combo.currentData()
