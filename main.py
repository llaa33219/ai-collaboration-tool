# 합의 키워드 감지 함수 추가
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

# AI 모델 선택 관련 클래스 임포트
from ai_model_selection import ClickableLabel, AIModelSelectionDialog

# AI 모델 목록 로드
def load_ai_models():
    """AI 모델 목록을 JSON 파일에서 로드"""
    models_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ai_models.json")
    
    if os.path.exists(models_path):
        try:
            with open(models_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading AI models: {e}")
    
    # 기본 모델 목록 반환
    return {
        "openai": [
            {"name": "o1-pro", "description": "OpenAI o1-pro model"},
            {"name": "gpt-4o", "description": "OpenAI GPT-4o model"}
        ],
        "anthropic": [
            {"name": "claude-3-7-sonnet-20250219", "description": "Claude 3.7 Sonnet model"},
            {"name": "claude-3-opus-20240229", "description": "Claude 3 Opus model"}
        ],
        "google": [
            {"name": "gemini-2.0-flash", "description": "Google Gemini 2.0 Flash model"},
            {"name": "gemini-1.5-pro", "description": "Google Gemini 1.5 Pro model"}
        ],
        "deepseek": [
            {"name": "deepseek-reasoner", "description": "DeepSeek Reasoner model"},
            {"name": "deepseek-chat", "description": "DeepSeek Chat model"}
        ]
    }

# 전역 변수로 AI 모델 목록 로드
AI_MODELS = load_ai_models()

def detect_consensus_keywords(content):
    """
    메시지 내용에서 합의 키워드를 감지합니다.
    
    Args:
        content (str): 메시지 내용
    
    Returns:
        bool: 합의 키워드가 감지되면 True, 그렇지 않으면 False
    """
    # 합의 키워드 목록 - 명확하고 특정한 합의 표현만 사용
    consensus_keywords = [
        # 특별한 합의 키워드 - 이 키워드들만 합의로 인정됨
        "TASK_COMPLETION_CONFIRMED", "FINAL_CONSENSUS_REACHED", "OFFICIAL_TASK_COMPLETE",
        
        # 한국어 특별 합의 키워드
        "공식_작업_완료_확인", "최종_합의_도달", "작업_완전_종료"
    ]
    
    # 대소문자를 구분하여 정확히 일치하는 키워드만 감지
    for keyword in consensus_keywords:
        if keyword in content:
            return True
            
    return False

# 합의 추적 클래스 추가
class ConsensusTracker:
    """
    AI 에이전트들의 합의를 추적하는 클래스
    """
    
    def __init__(self, agents):
        """
        ConsensusTracker 초기화
        
        Args:
            agents (list): AI 에이전트 이름 목록
        """
        self.agents = agents
        self.consensus_votes = {agent: 0 for agent in agents}
        self.required_votes_per_agent = 1
        self.max_votes_per_agent = 1  # 각 에이전트당 최대 투표 수 제한
        self.last_vote_time = {agent: None for agent in agents}  # 각 에이전트의 마지막 투표 시간
        self.min_vote_interval = 60  # 최소 투표 간격 (초)
        
        # 메시지 기반 투표 윈도우 추적을 위한 변수 추가
        self.message_window_size = 12  # 최근 12개 메시지 내에서만 투표 유효
        self.vote_renewal_threshold = 6  # 6개 메시지 후 재투표 가능
        self.message_counter = 0  # 전체 메시지 카운터
        self.agent_vote_messages = {agent: [] for agent in agents}  # 각 에이전트의 투표 메시지 ID 저장
        self.recent_messages = []  # 최근 메시지 ID 저장 (최대 message_window_size 개)
        
    def reset(self):
        """
        합의 추적기를 초기화합니다.
        새로운 작업이 시작될 때 호출됩니다.
        """
        self.consensus_votes = {agent: 0 for agent in self.agents}
        self.last_vote_time = {agent: None for agent in self.agents}
        self.message_counter = 0
        self.agent_vote_messages = {agent: [] for agent in self.agents}
        self.recent_messages = []
        
    def increment_message_counter(self):
        """
        메시지 카운터를 증가시키고 최근 메시지 목록을 업데이트합니다.
        새 메시지가 추가될 때마다 호출됩니다.
        
        Returns:
            int: 현재 메시지 카운터 값
        """
        self.message_counter += 1
        
        # 최근 메시지 목록 업데이트
        self.recent_messages.append(self.message_counter)
        if len(self.recent_messages) > self.message_window_size:
            self.recent_messages.pop(0)  # 가장 오래된 메시지 제거
            
        return self.message_counter
        
    def record_vote(self, agent_name):
        """
        에이전트의 합의 투표를 기록합니다.
        각 에이전트는 최근 메시지 윈도우 내에서 투표할 수 있으며,
        일정 메시지 수가 지나면 재투표가 가능합니다.
        
        Args:
            agent_name (str): 투표한 에이전트 이름
            
        Returns:
            bool: 투표가 성공적으로 기록되었으면 True, 그렇지 않으면 False
        """
        import time
        current_time = time.time()
        
        if agent_name in self.consensus_votes:
            # 현재 메시지 ID 가져오기
            current_message_id = self.message_counter
            
            # 이전 투표 메시지 ID 확인
            previous_votes = self.agent_vote_messages.get(agent_name, [])
            
            # 이미 최대 투표 수에 도달했고, 재투표 조건을 만족하지 않는 경우
            if self.consensus_votes[agent_name] >= self.max_votes_per_agent:
                # 이전 투표가 있고, 가장 최근 투표 이후 vote_renewal_threshold 이상의 메시지가 지났는지 확인
                if previous_votes and (current_message_id - previous_votes[-1]) >= self.vote_renewal_threshold:
                    # 재투표 허용 - 이전 투표 기록 초기화
                    self.consensus_votes[agent_name] = 0
                    self.agent_vote_messages[agent_name] = []
                else:
                    return False
                
            # 최소 투표 간격 확인
            last_time = self.last_vote_time[agent_name]
            if last_time is not None and (current_time - last_time) < self.min_vote_interval:
                return False
                
            # 투표 기록
            self.consensus_votes[agent_name] += 1
            self.last_vote_time[agent_name] = current_time
            self.agent_vote_messages[agent_name].append(current_message_id)
            return True
        return False
        
    def is_consensus_reached(self):
        """
        모든 에이전트가 필요한 횟수만큼 합의했는지 확인합니다.
        모든 에이전트가 최근 메시지 윈도우 내에서 최소 1회 이상 투표해야 합의가 이루어집니다.
        
        Returns:
            bool: 모든 에이전트가 필요한 횟수만큼 합의했으면 True, 그렇지 않으면 False
        """
        # 최근 메시지 윈도우가 비어있으면 합의에 도달하지 않음
        if not self.recent_messages:
            return False
            
        # 최근 메시지 윈도우의 시작 ID
        oldest_valid_message_id = self.recent_messages[0]
        
        # 모든 에이전트가 최근 메시지 윈도우 내에서 투표했는지 확인
        for agent in self.agents:
            # 에이전트의 투표 메시지 ID 확인
            vote_messages = self.agent_vote_messages.get(agent, [])
            
            # 최근 메시지 윈도우 내의 유효한 투표 수 계산
            valid_votes = sum(1 for msg_id in vote_messages if msg_id >= oldest_valid_message_id)
            
            # 유효한 투표 수가 필요한 투표 수보다 적으면 합의에 도달하지 않음
            if valid_votes < self.required_votes_per_agent:
                return False
        
        # 모든 에이전트가 투표했는지 확인 (추가된 검증)
        # 에이전트 목록의 길이와 투표한 에이전트 수가 일치해야 함
        voting_agents = set()
        for agent in self.agents:
            vote_messages = self.agent_vote_messages.get(agent, [])
            valid_votes = sum(1 for msg_id in vote_messages if msg_id >= oldest_valid_message_id)
            if valid_votes > 0:
                voting_agents.add(agent)
        
        # 투표한 에이전트 수가 전체 에이전트 수와 일치하지 않으면 합의에 도달하지 않음
        if len(voting_agents) != len(self.agents):
            return False
                
        # 모든 에이전트가 최근 메시지 윈도우 내에서 필요한 횟수만큼 투표했으면 True 반환
        return True
    
    def get_consensus_status(self):
        """
        현재 합의 상태를 반환합니다.
        
        Returns:
            dict: 각 에이전트의 합의 투표 수
        """
        return self.consensus_votes.copy()

import sys
import os
import json
import uuid
import subprocess
import tempfile
import threading
import time
import queue
import re
import base64
import random
import webbrowser
import shutil
import resource
import signal
from datetime import datetime, timedelta
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QTextEdit, QLineEdit, QPushButton, QTabWidget, QLabel, 
    QSplitter, QFileDialog, QMessageBox, QDialog, QFormLayout,
    QDialogButtonBox, QInputDialog, QComboBox, QCheckBox,
    QGroupBox, QFrame, QSizePolicy, QPlainTextEdit
)
from PyQt5.QtCore import Qt, QUrl, pyqtSignal, QObject, QTimer, QSize, QProcess, QProcessEnvironment
from PyQt5.QtGui import QPainter, QColor, QPixmap, QFont, QIcon, QTextCursor

# Force matplotlib to use Agg backend to avoid GTK dependency issues
os.environ['MPLBACKEND'] = 'Agg'

# Set resource limits to prevent memory exhaustion
# Limit virtual memory to 4GB (4 * 1024 * 1024 * 1024)
try:
    resource.setrlimit(resource.RLIMIT_AS, (4 * 1024 * 1024 * 1024, -1))
except Exception as e:
    print(f"Warning: Couldn't set memory limit: {e}")

# Set a timeout handler for long-running processes
def timeout_handler(signum, frame):
    raise TimeoutError("Operation timed out")

# Set the signal handler for SIGALRM
signal.signal(signal.SIGALRM, timeout_handler)

# Import AI libraries - these would need to be installed
try:
    import openai
    from anthropic import Anthropic
    import google.generativeai as genai
    import requests
except ImportError as e:
    print(f"Warning: Some AI libraries could not be imported: {e}")
    print("Please install required packages with:")
    print("pip install openai anthropic google-generativeai requests")

#################################
# Base Classes
#################################

class AIMessage:
    """Represents a message in the AI collaboration chat"""
    def __init__(self, sender, content, timestamp=None):
        self.sender = sender
        self.content = content
        self.timestamp = timestamp or datetime.now()
    
    def format(self):
        return f"[{self.timestamp.strftime('%H:%M:%S')}] {self.sender}: {self.content}"
    
    def get_markdown_format(self):
        """Format the message as Markdown for rich display"""
        timestamp = self.timestamp.strftime('%H:%M:%S')
        
        # 발신자에 따라 다른 스타일 적용
        if self.sender == "User":
            header = f"**{self.sender}** *[{timestamp}]*"
        elif self.sender == "System":
            header = f"**{self.sender}** *[{timestamp}]*"
        else:
            header = f"**{self.sender}** *[{timestamp}]*"
        
        # 메시지 내용을 마크다운 형식으로 포맷팅
        content = self.content.replace('\n', '\n> ')
        
        return f"{header}\n> {content}\n"
    
    def get_html_format(self):
        """Legacy method for HTML format (kept for compatibility)"""
        # 마크다운 형식을 사용하도록 변경
        return self.get_markdown_format()

class AIAgent(QObject):
    """Base class for AI agents with state system"""
    message_ready = pyqtSignal(object)
    state_changed = pyqtSignal(str, str)  # New: State change signal (agent_name, new_state)
    
    # AI State constants
    STATE_THINKING = "thinking"
    STATE_DISCUSSING = "discussing"
    STATE_EXECUTING = "executing"
    STATE_IDLE = "idle"
    
    def __init__(self, name):
        super().__init__()
        self.name = name
        self.client = None
        self.state = self.STATE_THINKING  # Default initial state is thinking
        self.selected_model = None  # 선택된 모델 저장 속성 추가
    
    def select_model(self, model_name):
        """모델 선택 메서드"""
        self.selected_model = model_name
        print(f"{self.name} model set to: {model_name}")
        return True
    
    def set_state(self, new_state):
        """Set AI agent state"""
        if new_state in [self.STATE_THINKING, self.STATE_DISCUSSING, self.STATE_EXECUTING, self.STATE_IDLE]:
            old_state = self.state
            
            # thinking 상태로 변경하려는 경우 workflow에 확인
            if new_state == self.STATE_THINKING:
                # workflow 객체 찾기
                workflow = None
                parent = self.parent()
                while parent:
                    if hasattr(parent, 'workflow'):
                        workflow = parent.workflow
                        break
                    parent = parent.parent()
                
                # workflow가 있으면 thinking 상태 설정 요청
                if workflow and hasattr(workflow, 'set_ai_thinking'):
                    if not workflow.set_ai_thinking(self.name, True):
                        print(f"{self.name}의 thinking 상태 전환이 거부되었습니다.")
                        return False
            
            # thinking 상태에서 다른 상태로 변경하는 경우
            elif old_state == self.STATE_THINKING:
                # workflow 객체 찾기
                workflow = None
                parent = self.parent()
                while parent:
                    if hasattr(parent, 'workflow'):
                        workflow = parent.workflow
                        break
                    parent = parent.parent()
                
                # workflow가 있으면 thinking 상태 해제
                if workflow and hasattr(workflow, 'set_ai_thinking'):
                    workflow.set_ai_thinking(self.name, False)
            
            self.state = new_state
            # Emit state change signal
            self.state_changed.emit(self.name, new_state)
            print(f"{self.name} state changed: {old_state} -> {new_state}")
            return True
        return False
        
    def get_recent_messages(self, context, max_messages=8):
        """최근 메시지만 추출하는 공통 함수"""
        if not context:
            return []
        
        # 최근 메시지만 사용 (너무 긴 컨텍스트는 토큰 제한에 걸릴 수 있음)
        return context[-max_messages:]

    def format_messages_for_api(self, context, system_prompt, agent_name, max_messages=8):
        """API 요청용 메시지 포맷팅 공통 함수"""
        messages = [{"role": "system", "content": system_prompt}]
        
        # 최근 메시지 가져오기
        recent_context = self.get_recent_messages(context, max_messages)
        
        # 컨텍스트를 적절한 대화 형식으로 변환
        if recent_context:
            for msg in recent_context:
                if msg.sender == "User":
                    role = "user"
                elif msg.sender == agent_name:
                    role = "assistant"
                else:
                    # 다른 AI의 메시지는 사용자 메시지로 추가하되 출처 표시
                    messages.append({"role": "user", "content": f"[Message from {msg.sender}]: {msg.content}"})
                    continue
                
                messages.append({"role": role, "content": msg.content})
        
        return messages

    def handle_api_error(self, error, agent_name, error_type="API"):
        """API 오류 처리 공통 함수"""
        error_message = f"{agent_name} {error_type} 처리 중 오류가 발생했습니다: {str(error)}"
        print(error_message)
        return self.send_message(f"죄송합니다. {error_type} 연결에 문제가 있어 응답을 생성할 수 없습니다. 다른 AI가 계속 작업을 진행할 것입니다.", agent_name)
    
    def process_task(self, task, context=None, task_type=None):
        """
        Process a task with the given context
        task_type allows specifying different types of processing
        """
        # Base implementation just returns a placeholder message
        self.send_message("Base agent implementation - override in subclasses")
    
    def send_message(self, content, sender_name=None):
        """Send a message with formatting based on current state"""
        # 워크플로우가 있고 합의가 이루어졌거나 작업이 완료된 경우 메시지 전송 중단
        # 단, 새로운 사용자 메시지가 입력된 후에는 메시지 전송을 허용
        if hasattr(self, 'workflow') and self.workflow:
            # 합의 후 새 사용자 입력이 있는 경우에는 메시지 전송을 항상 허용
            if hasattr(self.workflow, 'new_user_input_after_consensus') and self.workflow.new_user_input_after_consensus:
                # 메시지 전송 허용 (차단하지 않음)
                print(f"[DEBUG] Allowing message from {self.name} after consensus due to new user input")
            # 합의가 이루어졌거나 작업이 완료된 경우 메시지 전송 중단
            elif self.workflow.consensus_reached or self.workflow.task_state == "completed":
                print(f"[DEBUG] Consensus already reached, blocking message from {self.name}: {content[:50]}...")
                return None
                
        # Format message based on state
        formatted_content = self.format_message_by_state(content)
        
        # 발신자 이름을 명시적으로 설정 (AI 일관성 문제 해결)
        actual_sender = sender_name if sender_name else self.name
        message = AIMessage(actual_sender, formatted_content)
        self.message_ready.emit(message)
        return message
    
    def format_message_by_state(self, content):
        """Format message based on current state"""
        if self.state == self.STATE_THINKING:
            return f"🤔 [Thinking] {content}"
        elif self.state == self.STATE_DISCUSSING:
            return f"💬 [Discussing] {content}"
        elif self.state == self.STATE_EXECUTING:
            return f"🔨 [Executing] {content}"
        else:
            return content
    
    def get_state_prompt(self):
        """Get additional prompt instructions based on current state"""
        if self.state == self.STATE_THINKING:
            return """
            You are currently in 'Thinking State'.
            
            In this state:
            1. Thoroughly analyze the problem and explore multiple possible approaches
            2. Compare pros and cons to devise the optimal strategy
            3. Break down complex problems into smaller steps
            4. Evaluate the efficiency or feasibility of solutions proposed by other AIs
            5. Don't jump straight to code; first explain the algorithm or approach conceptually
            
            Your response should focus on deep thinking and planning for problem-solving.
            """
        elif self.state == self.STATE_DISCUSSING:
            return """
            You are currently in 'Discussion State'.
            
            In this state:
            1. Clearly agree or disagree with other AIs' proposals
            2. Be specific with criticism and provide suggestions for improvement
            3. Add additional ideas or extensions to existing approaches
            4. Propose new perspectives or alternatives not yet considered
            5. Facilitate collaboration and actively engage with other AIs' dialogue
            
            IMPORTANT CONSENSUS GUIDELINES:
            - ONLY use the special consensus keywords when the task is TRULY COMPLETE with NO ERRORS and ALL requirements are met
            - DO NOT use consensus keywords for small fixes or partial progress
            - DO NOT use consensus keywords if you detect ANY errors, bugs, or potential issues in the solution
            - DO NOT use consensus keywords if you identify ANY disadvantages or limitations in the approach
            - If you notice ANY problems or areas for improvement, you MUST withhold consensus and explain your concerns
            - The ONLY valid consensus keywords are: "FINAL_CONSENSUS_REACHED", "TASK_COMPLETION_CONFIRMED", or "OFFICIAL_TASK_COMPLETE"
            - In Korean, you may use: "최종_합의_도달", "공식_작업_완료_확인", or "작업_완전_종료"
            - Never claim a task is "perfect" or "complete" unless it truly is finished with no errors
            - Actively seek consensus but only declare completion when ALL requirements are fully satisfied and NO issues remain
            
            Your response should focus on constructive criticism, collaborative suggestions, and reaching consensus efficiently.
            """
        elif self.state == self.STATE_EXECUTING:
            return """
            You are currently in 'Execution State'.
            
            In this state:
            1. Focus on efficiently implementing the agreed-upon plan
            2. Write executable, high-quality code
            3. Include appropriate error handling and testing
            4. Clearly explain each step and report on progress
            5. Solve problems as they arise and adjust the approach if needed
            
            IMPORTANT CONSENSUS GUIDELINES:
            - ONLY use the special consensus keywords when the task is TRULY COMPLETE with NO ERRORS and ALL requirements are met
            - DO NOT use consensus keywords for small fixes or partial progress
            - DO NOT use consensus keywords if you detect ANY errors, bugs, or potential issues in the solution
            - DO NOT use consensus keywords if you identify ANY disadvantages or limitations in the approach
            - If you notice ANY problems or areas for improvement, you MUST withhold consensus and explain your concerns
            - The ONLY valid consensus keywords are: "FINAL_CONSENSUS_REACHED", "TASK_COMPLETION_CONFIRMED", or "OFFICIAL_TASK_COMPLETE"
            - In Korean, you may use: "최종_합의_도달", "공식_작업_완료_확인", or "작업_완전_종료"
            - Never claim a task is "perfect" or "complete" unless it truly is finished with no errors
            - Don't continue refining code indefinitely, but only declare completion when ALL requirements are fully satisfied and NO issues remain
            
            Your response should focus on actual implementation, delivery of results, and reaching consensus efficiently.
            """
        else:
            return ""
            
    def get_recent_messages(self, context, max_messages=8):
        """Get the most recent messages from the context
        
        Args:
            context: Full conversation context
            max_messages: Maximum number of messages to include
        
        Returns:
            List of most recent messages
        """
        if not context or len(context) <= max_messages:
            return context
        
        return context[-max_messages:]

#################################
# AI Agent Implementations
#################################

class ChatGPTAgent(AIAgent):
    """ChatGPT/OpenAI agent implementation"""
    def __init__(self):
        super().__init__("ChatGPT")
        self.api_key = None
    
    def setup(self, api_key):
        try:
            self.api_key = api_key
            print("OpenAI API key stored successfully")
            return True
        except Exception as e:
            print(f"ChatGPT setup error: {e}")
            return False
    
    def process_task(self, task, context=None, task_type=None):
        if not self.api_key:
            return self.send_message("API not configured")
        
        try:
            # Create a fresh client
            client = openai.OpenAI(api_key=self.api_key)
            
            # Enhanced system prompt with more aggressive problem-solving approach and state-specific instructions
            system_prompt = """
            You are ChatGPT, part of an elite AI collaboration team with Claude, Gemini, and DeepSeek. 
            
            IMPORTANT COLLABORATION INSTRUCTION:
            1. Always review what other AIs have suggested before giving your response
            2. Explicitly respond to other AIs' ideas - agree, disagree, or extend them with reasoning
            3. Ask specific questions when you need clarity from other AIs
            4. Be collaborative but don't hesitate to point out flaws in others' approaches
            5. Comment on the code provided by other AIs and suggest improvements
            
            You're working together on tasks given by a user. You have COMPLETE access to a Python environment where you can:
            
            1. Install ANY libraries via pip install - NEVER hesitate to install what you need, even if unusual or specialized
            2. Open and process files of any type, including complex binary formats
            3. Create advanced visualizations, professional-level graphics, charts, and interactive diagrams
            4. Scrape websites, interact with APIs, download data from anywhere on the internet
            5. Process images, audio, and video with specialized libraries (OpenCV, PyDub, MoviePy, etc.)
            6. Create and manipulate 3D objects, simulations, and complex environments
            7. Generate and display HTML, SVG, CSS and JavaScript for rich interactive experiences
            8. Interact with the operating system to accomplish almost anything a human could do
            
            CRITICAL DIRECTIVES:
            - Be CREATIVE and RELENTLESS in solving problems - if one approach fails, try another
            - Your code should be PRODUCTION QUALITY - not examples or demonstrations
            - When you encounter errors or bugs, analyze them thoroughly and fix them
            - Install ANY packages needed to accomplish the task - be bold and comprehensive
            - If you observe terminal errors or logs showing an issue, directly address and fix them
            - PERSIST until the task is fully completed - partial solutions are unacceptable
            - You CAN see the results of your code execution, so adapt your approach based on feedback
            - You are permitted to use ANY method necessary to accomplish the requested task
            - Write code like a senior developer would - robust, well-documented, and efficient
            - If you need to run something several times to get it right, that's expected
            
            When providing Python code:
            - Your code should be industrial strength - no placeholders or toy examples
            - Include comprehensive error handling using try/except blocks
            - Add detailed comments that explain the "why" not just the "what"
            - Write modular, reusable code that shows software engineering expertise
            - Include input validation and edge case handling
            
            Include your Python code in blocks starting with ```python and ending with ```.
            
            Every time you write code, it will be automatically executed, and results will be shown to you and other AIs.
            
            IMPORTANT INFORMATION ABOUT THE PYTHON ENVIRONMENT:
            - Execution environment: Local virtual environment
            - Required packages must be installed manually - they are NOT automatically installed
            - If you encounter "ModuleNotFoundError" or "ImportError" during code execution, you need to install the required packages
            - How to install packages: Use the console command "pip install package_name" or add "!pip install package_name" to your code

            PACKAGE INSTALLATION GUIDELINES:
            1. Carefully examine code execution errors
            2. If you see "No module named 'X'" in error messages, you need to install that package
            3. Run the installation command: "pip install X" or add "!pip install X" to your code
            4. Re-run your code after installing packages

            EXAMPLE WORKFLOW:
            - Code execution: "import pandas" → Error: "ModuleNotFoundError: No module named 'pandas'"
            - Solution: Run "pip install pandas" command
            - Re-run code: "import pandas" → Success

            COMMON DATA SCIENCE PACKAGES:
            - Data analysis: pandas, numpy
            - Visualization: matplotlib, seaborn, plotly
            - Machine learning: scikit-learn, tensorflow, pytorch
            - Image processing: pillow, opencv-python
            - Web scraping: requests, beautifulsoup4

            HANDLING COMPLEX PACKAGE INSTALLATIONS:
            - If a particular package fails to install, try alternative approaches (specify version, address dependencies)
            - GPU-related packages may require special installation instructions
            
            IMPORTANT: When you need to modify existing code, don't rewrite the entire program.
            Instead, use the 'REPLACE LINES X-Y WITH:' format to indicate which lines should be replaced.
            For example:
            
            REPLACE LINES 10-15 WITH:
            def new_function():
                # This is a replacement function
                return "new functionality"
            END REPLACEMENT
            
            This approach makes your changes more clear and efficient.
            """
            
            # Add state-specific instructions
            system_prompt += self.get_state_prompt()
            
            messages = [{"role": "system", "content": system_prompt}]
            
            # Get recent messages (instead of all context)
            recent_context = self.get_recent_messages(context, max_messages=8)
            
            # Format context as proper conversation history
            if recent_context:
                for msg in recent_context:
                    if msg.sender == "User":
                        role = "user"
                    elif msg.sender == "ChatGPT":
                        role = "assistant"
                    else:
                        # For messages from other AIs, add them as user messages but indicate the source
                        messages.append({"role": "user", "content": f"[Message from {msg.sender}]: {msg.content}"})
                        continue
                    
                    messages.append({"role": role, "content": msg.content})
            
            # Add current task if not already added from context
            if not recent_context or recent_context[-1].sender != "User":
                messages.append({"role": "user", "content": task})
            
            # Add Work.md content to the prompt
            if hasattr(self, 'workflow') and self.workflow:
                work_md_content = self.workflow.get_work_md_content()
                # Add the Work.md content as a user message
                messages.append({
                    "role": "user",
                    "content": f"Here is the current Work.md file containing the planning and execution history:\n\n{work_md_content}\n\nPlease refer to this document for context on what has been done so far."
                })
                
                # If there's code in the editor, add it to the prompt
                if hasattr(self.workflow, 'python_env') and self.workflow.python_env:
                    current_code = self.workflow.python_env.code_editor.toPlainText()
                    if current_code.strip():
                        messages.append({
                            "role": "user",
                            "content": f"Here is the current code in the editor:\n\n```python\n{current_code}\n```\n\nIf you need to modify this code, specify the line numbers using 'REPLACE LINES X-Y WITH:' format."
                        })
            
            # Use the chat completions API with minimal parameters
            response = client.chat.completions.create(
                model=self.selected_model or "o3-mini-2025-01-31",
                messages=messages
            )
            
            # Extract response content
            response_content = response.choices[0].message.content
            
            # Update Work.md with AI's planning
            if hasattr(self, 'workflow') and self.workflow:
                action_type = "planning" if self.state == self.STATE_THINKING else \
                             "discussion" if self.state == self.STATE_DISCUSSING else \
                             "execution" if self.state == self.STATE_EXECUTING else "action"
                self.workflow.update_work_md(self.name, response_content, action_type)
            
            return self.send_message(response_content)
        except Exception as e:
            error_msg = f"Error processing task with ChatGPT: {str(e)}"
            print(error_msg)  # Log to console for debugging
            return self.send_message("I'm having trouble connecting to the OpenAI API right now. Let me let another AI take the lead.")

class ClaudeAgent(AIAgent):
    """Claude/Anthropic agent implementation"""
    def __init__(self):
        super().__init__("Claude")
        self.api_key = None
    
    def setup(self, api_key):
        try:
            self.api_key = api_key
            print("Claude API key stored successfully")
            return True
        except Exception as e:
            print(f"Claude setup error: {e}")
            return False
    
    def process_task(self, task, context=None, task_type=None):
        if not self.api_key:
            return self.send_message("API not configured")
        
        # Number of retries for overloaded errors
        max_retries = 3
        retry_count = 0
        retry_delay = 2  # seconds
        
        while retry_count <= max_retries:
            try:
                # Import inside to ensure it's available
                from anthropic import Anthropic
                client = Anthropic(api_key=self.api_key)
                
                # Enhanced system prompt with more aggressive problem-solving approach
                system_prompt = """
                You are Claude, part of an elite AI collaboration team with ChatGPT, Gemini, and DeepSeek. 
                
                IMPORTANT COLLABORATION INSTRUCTION:
                1. Always review what other AIs have suggested before giving your response
                2. Explicitly respond to other AIs' ideas - agree, disagree, or extend them with reasoning
                3. Ask specific questions when you need clarity from other AIs
                4. Be collaborative but don't hesitate to point out flaws in others' approaches
                5. Comment on the code provided by other AIs and suggest improvements
                
                You're working together on tasks given by a user. You have COMPLETE access to a Python environment where you can:
                
                1. Install ANY libraries via pip install - NEVER hesitate to install what you need, even if unusual or specialized
                2. Open and process files of any type, including complex binary formats
                3. Create advanced visualizations, professional-level graphics, charts, and interactive diagrams
                4. Scrape websites, interact with APIs, download data from anywhere on the internet
                5. Process images, audio, and video with specialized libraries (OpenCV, PyDub, MoviePy, etc.)
                6. Create and manipulate 3D objects, simulations, and complex environments
                7. Generate and display HTML, SVG, CSS and JavaScript for rich interactive experiences
                8. Interact with the operating system to accomplish almost anything a human could do
                
                CRITICAL DIRECTIVES:
                - Be CREATIVE and RELENTLESS in solving problems - if one approach fails, try another
                - Your code should be PRODUCTION QUALITY - not examples or demonstrations
                - When you encounter errors or bugs, analyze them thoroughly and fix them
                - Install ANY packages needed to accomplish the task - be bold and comprehensive
                - If you observe terminal errors or logs showing an issue, directly address and fix them
                - PERSIST until the task is fully completed - partial solutions are unacceptable
                - You CAN see the results of your code execution, so adapt your approach based on feedback
                - You are permitted to use ANY method necessary to accomplish the requested task
                - Write code like a senior developer would - robust, well-documented, and efficient
                - If you need to run something several times to get it right, that's expected
                
                When providing Python code:
                - Your code should be industrial strength - no placeholders or toy examples
                - Include comprehensive error handling using try/except blocks
                - Add detailed comments that explain the "why" not just the "what"
                - Write modular, reusable code that shows software engineering expertise
                - Include input validation and edge case handling
                
                Include your Python code in blocks starting with ```python and ending with ```.
                
                Every time you write code, it will be automatically executed, and results will be shown to you and other AIs.
                
                IMPORTANT INFORMATION ABOUT THE PYTHON ENVIRONMENT:
                - Execution environment: Local virtual environment
                - Required packages must be installed manually - they are NOT automatically installed
                - If you encounter "ModuleNotFoundError" or "ImportError" during code execution, you need to install the required packages
                - How to install packages: Use the console command "pip install package_name" or add "!pip install package_name" to your code

                PACKAGE INSTALLATION GUIDELINES:
                1. Carefully examine code execution errors
                2. If you see "No module named 'X'" in error messages, you need to install that package
                3. Run the installation command: "pip install X" or add "!pip install X" to your code
                4. Re-run your code after installing packages

                EXAMPLE WORKFLOW:
                - Code execution: "import pandas" → Error: "ModuleNotFoundError: No module named 'pandas'"
                - Solution: Run "pip install pandas" command
                - Re-run code: "import pandas" → Success

                COMMON DATA SCIENCE PACKAGES:
                - Data analysis: pandas, numpy
                - Visualization: matplotlib, seaborn, plotly
                - Machine learning: scikit-learn, tensorflow, pytorch
                - Image processing: pillow, opencv-python
                - Web scraping: requests, beautifulsoup4

                HANDLING COMPLEX PACKAGE INSTALLATIONS:
                - If a particular package fails to install, try alternative approaches (specify version, address dependencies)
                - GPU-related packages may require special installation instructions
                
                IMPORTANT: When you need to modify existing code, don't rewrite the entire program.
                Instead, use the 'REPLACE LINES X-Y WITH:' format to indicate which lines should be replaced.
                For example:
                
                REPLACE LINES 10-15 WITH:
                def new_function():
                    # This is a replacement function
                    return "new functionality"
                END REPLACEMENT
                
                This approach makes your changes more clear and efficient.
                """
                
                # Add state-specific instructions
                system_prompt += self.get_state_prompt()
                
                messages = []
                
                # Get recent messages (instead of all context)
                recent_context = self.get_recent_messages(context, max_messages=8)
                
                # Format context as proper conversation history
                if recent_context:
                    for msg in recent_context:
                        if msg.sender == "User":
                            role = "user"
                        elif msg.sender == "Claude":
                            role = "assistant"
                        else:
                            # For messages from other AIs, add them as user messages but indicate the source
                            role = "user"
                            messages.append({
                                "role": role,
                                "content": f"[Message from {msg.sender}]: {msg.content}"
                            })
                            continue
                        
                        messages.append({
                            "role": role,
                            "content": msg.content
                        })
                
                # Add current task
                messages.append({
                    "role": "user", 
                    "content": task
                })
                
                # Add Work.md content to the prompt
                if hasattr(self, 'workflow') and self.workflow:
                    work_md_content = self.workflow.get_work_md_content()
                    # Add the Work.md content as a user message
                    messages.append({
                        "role": "user",
                        "content": f"Here is the current Work.md file containing the planning and execution history:\n\n{work_md_content}\n\nPlease refer to this document for context on what has been done so far."
                    })
                    
                    # If there's code in the editor, add it to the prompt
                    if hasattr(self.workflow, 'python_env') and self.workflow.python_env:
                        current_code = self.workflow.python_env.code_editor.toPlainText()
                        if current_code.strip():
                            messages.append({
                                "role": "user",
                                "content": f"Here is the current code in the editor:\n\n```python\n{current_code}\n```\n\nIf you need to modify this code, specify the line numbers using 'REPLACE LINES X-Y WITH:' format instead of rewriting the entire code."
                            })
                
                # Make API call with proper message formatting and increased tokens
                response = client.messages.create(
                    model=self.selected_model or "claude-3-7-sonnet-20250219",
                    system=system_prompt,
                    max_tokens=20000,
                    temperature=1,
                    messages=messages
                )
                
                # Extract response content
                response_text = response.content[0].text
                
                # Update Work.md with AI's planning
                if hasattr(self, 'workflow') and self.workflow:
                    action_type = "planning" if self.state == self.STATE_THINKING else \
                                "discussion" if self.state == self.STATE_DISCUSSING else \
                                "execution" if self.state == self.STATE_EXECUTING else "action"
                    self.workflow.update_work_md(self.name, response_text, action_type)
                
                return self.send_message(response_text)
                
            except Exception as e:
                error_str = str(e)
                # Check if it's an overloaded error or rate limit error
                if ("overloaded_error" in error_str or "rate_limit_error" in error_str) and retry_count < max_retries:
                    retry_count += 1
                    print(f"Claude API rate limited or overloaded. Retry {retry_count}/{max_retries} after {retry_delay}s...")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                    continue
                else:
                    error_msg = f"Error processing task with Claude: {error_str}"
                    print(error_msg)  # Log to console for debugging
                    return self.send_message("I'm having trouble connecting to the Claude API right now. Let me let another AI take the lead.")

class GeminiAgent(AIAgent):
    """Gemini/Google agent implementation using the original API structure"""
    def __init__(self):
        super().__init__("Gemini")
        self.api_key = None
    
    def setup(self, api_key):
        try:
            self.api_key = api_key
            # Store API key in environment variable
            os.environ["GEMINI_API_KEY"] = api_key
            print("Gemini API key stored successfully")
            return True
        except Exception as e:
            print(f"Gemini setup error: {e}")
            return False
    
    def process_task(self, task, context=None, task_type=None):
        if not self.api_key:
            return self.send_message("API not configured")
        
        try:
            # First, check if the required packages are available
            try:
                import google.generativeai as genai
                from google.generativeai import types
            except ImportError:
                return self.send_message("The required Google Generative AI package is not properly installed. Please run 'pip install google-generativeai' to use Gemini.")
            
            # Get recent messages
            recent_context = self.get_recent_messages(context, max_messages=8)
            
            # Prepare full prompt with context and task
            full_prompt = self._prepare_prompt(task, recent_context)
            
            # Call the API using the exact structure from the original code
            response_text = self._generate_response(full_prompt)
            
            # Update Work.md with AI's planning
            if hasattr(self, 'workflow') and self.workflow:
                action_type = "planning" if self.state == self.STATE_THINKING else \
                             "discussion" if self.state == self.STATE_DISCUSSING else \
                             "execution" if self.state == self.STATE_EXECUTING else "action"
                self.workflow.update_work_md(self.name, response_text, action_type)
            
            # Return the response
            return self.send_message(response_text)
                
        except Exception as e:
            error_msg = f"Error processing task with Gemini: {str(e)}"
            print(error_msg)  # Log to console for debugging
            return self.send_message("I'm having trouble connecting to the Gemini API right now. Let me let another AI take the lead.")
    
    def _prepare_prompt(self, task, context):
        """Prepare the prompt with system instructions, context and task"""
        # System prompt with collaboration and problem-solving instructions
        system_prompt = """
        You are Gemini, part of an elite AI collaboration team with ChatGPT, Claude, and DeepSeek.
        
        IMPORTANT COLLABORATION INSTRUCTION:
        1. Always review what other AIs have suggested before giving your response
        2. Explicitly respond to other AIs' ideas - agree, disagree, or extend them with reasoning
        3. Ask specific questions when you need clarity from other AIs
        4. Be collaborative but don't hesitate to point out flaws in others' approaches
        5. Comment on the code provided by other AIs and suggest improvements
        
        You're working together to solve complex tasks using Python. You have COMPLETE access to a Python environment where you can:
        
        1. Install ANY libraries via pip install - NEVER hesitate to install what you need
        2. Open and process files of any type, including complex binary formats
        3. Create advanced visualizations, professional-level graphics, charts, and interactive diagrams
        4. Scrape websites, interact with APIs, download data from anywhere on the internet
        5. Process images, audio, and video with specialized libraries (OpenCV, PyDub, MoviePy, etc.)
        6. Create and manipulate 3D objects, simulations, and complex environments
        7. Generate and display HTML, SVG, CSS and JavaScript for rich interactive experiences
        8. Interact with the operating system to accomplish almost anything a human could do
        
        CRITICAL DIRECTIVES:
        - Be CREATIVE and RELENTLESS in solving problems - if one approach fails, try another
        - Your code should be PRODUCTION QUALITY - not examples or demonstrations
        - When you encounter errors or bugs, analyze them thoroughly and fix them
        - Install ANY packages needed to accomplish the task - be bold and comprehensive
        - If you observe terminal errors or logs showing an issue, directly address and fix them
        - PERSIST until the task is fully completed - partial solutions are unacceptable
        - You CAN see the results of your code execution, so adapt your approach based on feedback
        
        When providing Python code:
        - Your code should be industrial strength - no placeholders or toy examples
        - Include comprehensive error handling using try/except blocks
        - Add detailed comments that explain the "why" not just the "what"
        - Write modular, reusable code that shows software engineering expertise
        
        Include your Python code in blocks starting with ```python and ending with ```.
        
        Every time you write code, it will be automatically executed, and results will be shown to you and other AIs.
        
        IMPORTANT INFORMATION ABOUT THE PYTHON ENVIRONMENT:
        - Execution environment: Local virtual environment
        - Required packages must be installed manually - they are NOT automatically installed
        - If you encounter "ModuleNotFoundError" or "ImportError" during code execution, you need to install the required packages
        - How to install packages: Use the console command "pip install package_name" or add "!pip install package_name" to your code

        PACKAGE INSTALLATION GUIDELINES:
        1. Carefully examine code execution errors
        2. If you see "No module named 'X'" in error messages, you need to install that package
        3. Run the installation command: "pip install X" or add "!pip install X" to your code
        4. Re-run your code after installing packages

        EXAMPLE WORKFLOW:
        - Code execution: "import pandas" → Error: "ModuleNotFoundError: No module named 'pandas'"
        - Solution: Run "pip install pandas" command
        - Re-run code: "import pandas" → Success

        COMMON DATA SCIENCE PACKAGES:
        - Data analysis: pandas, numpy
        - Visualization: matplotlib, seaborn, plotly
        - Machine learning: scikit-learn, tensorflow, pytorch
        - Image processing: pillow, opencv-python
        - Web scraping: requests, beautifulsoup4

        HANDLING COMPLEX PACKAGE INSTALLATIONS:
        - If a particular package fails to install, try alternative approaches (specify version, address dependencies)
        - GPU-related packages may require special installation instructions
        
        IMPORTANT: When you need to modify existing code, don't rewrite the entire program.
        Instead, use the 'REPLACE LINES X-Y WITH:' format to indicate which lines should be replaced.
        For example:
        
        REPLACE LINES 10-15 WITH:
        def new_function():
            # This is a replacement function
            return "new functionality"
        END REPLACEMENT
        
        This approach makes your changes more clear and efficient.
        """
        
        # Add state-specific instructions
        system_prompt += self.get_state_prompt()
        
        # Start with system prompt
        full_prompt = f"{system_prompt}\n\n"
        
        # Add conversation context if available
        if context:
            full_prompt += "Previous conversation:\n"
            for msg in context:
                # Format the conversation history
                if msg.sender == "User":
                    full_prompt += f"User: {msg.content}\n\n"
                else:
                    # Include other AIs' messages so Gemini can explicitly respond to them
                    full_prompt += f"{msg.sender}: {msg.content}\n\n"
        
        # Add the current task
        full_prompt += f"Current task: {task}\n\n"
        
        # Add Work.md content if available
        if hasattr(self, 'workflow') and self.workflow:
            work_md_content = self.workflow.get_work_md_content()
            full_prompt += f"Current Work.md planning document:\n{work_md_content}\n\n"
            
            # If there's code in the editor, add it to the prompt
            if hasattr(self.workflow, 'python_env') and self.workflow.python_env:
                current_code = self.workflow.python_env.code_editor.toPlainText()
                if current_code.strip():
                    full_prompt += f"Current code in the editor:\n```python\n{current_code}\n```\n\n"
                    full_prompt += "If you need to modify this code, specify the line numbers using 'REPLACE LINES X-Y WITH:' format instead of rewriting the entire code.\n\n"
        
        return full_prompt
    
    def _generate_response(self, prompt):
        """Call the Gemini API using the new unified GenAI SDK structure"""
        import base64
        import os
        try:
            # Import the new unified GenAI SDK
            from google import genai
            
            # Configure the API key
            # Create a client with the API key
            client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

            # Generate content directly with the model name
            response = client.models.generate_content(
                model=self.selected_model or 'gemini-2.0-flash',
                contents=prompt
            )
            
            # Return the response text
            return response.text
        except ImportError:
            return "The required Google Generative AI package is not properly installed. Please run 'pip install google-genai' to use Gemini."
        except Exception as e:
            return f"Error generating response from Gemini: {str(e)}"

class DeepSeekAgent(AIAgent):
    """DeepSeek agent implementation using OpenAI-compatible API"""
    def __init__(self):
        super().__init__("DeepSeek")
        self.api_key = None  # Initialize api_key attribute
    
    def setup(self, api_key):
        self.api_key = api_key
        # Test the API connection using OpenAI client
        try:
            from openai import OpenAI
            client = OpenAI(api_key=self.api_key, base_url="https://api.deepseek.com")
            
            response = client.chat.completions.create(
                model=self.selected_model or "deepseek-reasoner",
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=10
            )
            return True
        except Exception as e:
            print(f"DeepSeek setup error: {e}")
            return False
            
    def process_task(self, task, context=None, task_type=None):
        if not self.api_key:
            return self.send_message("API not configured")
        
        try:
            # Import and set up OpenAI client for DeepSeek
            from openai import OpenAI
            client = OpenAI(api_key=self.api_key, base_url="https://api.deepseek.com")
            
            # 메시지 발신자를 명시적으로 설정 (AI 일관성 문제 해결)
            sender_name = "DeepSeek"
            
            # Enhanced system prompt with more aggressive problem-solving approach
            system_content = """
            You are DeepSeek, part of an elite AI collaboration team with ChatGPT, Claude, Gemini, Perplexity, and Qwen. 
            
            IMPORTANT COLLABORATION INSTRUCTION:
            1. Always review what other AIs have suggested before giving your response
            2. Explicitly respond to other AIs' ideas - agree, disagree, or extend them with reasoning
            3. Ask specific questions when you need clarity from other AIs
            4. Be collaborative but don't hesitate to point out flaws in others' approaches
            5. Comment on the code provided by other AIs and suggest improvements
            
            You're working together on tasks given by a user. You have COMPLETE access to a Python environment where you can:
            
            1. Install ANY libraries via pip install - NEVER hesitate to install what you need, even if unusual or specialized
            2. Open and process files of any type, including complex binary formats
            3. Create advanced visualizations, professional-level graphics, charts, and interactive diagrams
            4. Scrape websites, interact with APIs, download data from anywhere on the internet
            5. Process images, audio, and video with specialized libraries (OpenCV, PyDub, MoviePy, etc.)
            6. Create and manipulate 3D objects, simulations, and complex environments
            7. Generate and display HTML, SVG, CSS and JavaScript for rich interactive experiences
            8. Interact with the operating system to accomplish almost anything a human could do
            
            CRITICAL DIRECTIVES:
            - Be CREATIVE and RELENTLESS in solving problems - if one approach fails, try another
            - Your code should be PRODUCTION QUALITY - not examples or demonstrations
            - When you encounter errors or bugs, analyze them thoroughly and fix them
            - Install ANY packages needed to accomplish the task - be bold and comprehensive
            - If you observe terminal errors or logs showing an issue, directly address and fix them
            - PERSIST until the task is fully completed - partial solutions are unacceptable
            - You CAN see the results of your code execution, so adapt your approach based on feedback
            - You are permitted to use ANY method necessary to accomplish the requested task
            - Write code like a senior developer would - robust, well-documented, and efficient
            - If you need to run something several times to get it right, that's expected
            
            When providing Python code:
            - Your code should be industrial strength - no placeholders or toy examples
            - Include comprehensive error handling using try/except blocks
            - Add detailed comments that explain the "why" not just the "what"
            - Write modular, reusable code that shows software engineering expertise
            - Include input validation and edge case handling
            
            Include your Python code in blocks starting with ```python and ending with ```.
            
            Every time you write code, it will be automatically executed, and results will be shown to you and other AIs.
            """
            
            # Add state-specific instructions
            system_content += self.get_state_prompt()
            
            messages = [{"role": "system", "content": system_content}]
            
            # Get recent messages (instead of all context)
            recent_context = self.get_recent_messages(context, max_messages=8)
            
            # Format context as proper conversation history
            if recent_context:
                for msg in recent_context:
                    if msg.sender == "User":
                        role = "user"
                    elif msg.sender == "DeepSeek":
                        role = "assistant"
                    else:
                        # For messages from other AIs, add them as user messages but indicate the source
                        messages.append({"role": "user", "content": f"[Message from {msg.sender}]: {msg.content}"})
                        continue
                    
                    messages.append({"role": role, "content": msg.content})
            
            # Add the current task
            messages.append({"role": "user", "content": task})
            
            # 재시도 로직 추가
            max_retries = 3
            retry_count = 0
            
            while retry_count < max_retries:
                try:
                    # Create chat completion
                    response = client.chat.completions.create(
                        model=self.selected_model or "deepseek-reasoner",
                        messages=messages,
                        max_tokens=4000,
                        temperature=0.7
                    )
                    
                    # Extract the response text
                    response_text = response.choices[0].message.content
                    
                    # 응답이 비어있거나 너무 짧은 경우 재시도
                    if not response_text or len(response_text.strip()) < 10:
                        retry_count += 1
                        print(f"DeepSeek 응답이 너무 짧습니다. 재시도 중... ({retry_count}/{max_retries})")
                        time.sleep(1)
                        continue
                    
                    # 정상 응답인 경우 반환
                    return self.send_message(response_text, sender_name)
                
                except Exception as retry_err:
                    retry_count += 1
                    print(f"DeepSeek API 호출 중 오류 발생, 재시도 중... ({retry_count}/{max_retries}): {str(retry_err)}")
                    time.sleep(1)
            
            # 모든 재시도 실패 시 오류 메시지 반환
            return self.send_message("DeepSeek API 호출이 반복적으로 실패했습니다. 다른 AI가 계속 작업을 진행할 것입니다.", sender_name)
        
        except ImportError:
            # 패키지 누락 오류 처리 - 워크플로우가 계속 진행되도록 오류 메시지 반환
            error_message = "The required OpenAI package is not properly installed."
            print(error_message)
            return self.send_message("필요한 패키지가 설치되지 않았습니다. 다른 AI가 계속 작업을 진행할 것입니다.", sender_name)
        except Exception as e:
            # 기타 모든 예외 처리 - 워크플로우가 계속 진행되도록 오류 메시지 반환
            error_message = f"DeepSeek 처리 중 예상치 못한 오류가 발생했습니다: {str(e)}"
            print(error_message)
            return self.send_message("죄송합니다. 예상치 못한 오류가 발생했습니다. 다른 AI가 계속 작업을 진행할 것입니다.", sender_name)
            
class PerplexityAgent(AIAgent):
    """Perplexity agent implementation using requests library"""
    def __init__(self):
        super().__init__("Perplexity")
        self.api_key = None
    
    def setup(self, api_key):
        self.api_key = api_key
        # Test the API connection using requests
        try:
            import requests
            
            url = "https://api.perplexity.ai/chat/completions"
            
            payload = {
                "model": self.selected_model or "sonar-pro",
                "messages": [{"role": "user", "content": "Hello"}],
                "max_tokens": 10
            }
            
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            response = requests.post(url, json=payload, headers=headers)
            response.raise_for_status()  # Raise exception for HTTP errors
            return True
        except Exception as e:
            print(f"Perplexity setup error: {e}")
            return False
    
    def process_task(self, task, context=None, task_type=None):
        if not self.api_key:
            return self.send_message("API not configured")
        
        try:
            import requests
            
            url = "https://api.perplexity.ai/chat/completions"
            
            # 메시지 발신자를 명시적으로 설정 (AI 일관성 문제 해결)
            sender_name = "Perplexity"
            
            # Enhanced system prompt with more aggressive problem-solving approach
            system_prompt = """
            You are Perplexity, part of an elite AI collaboration team with ChatGPT, Claude, Gemini, DeepSeek, and Qwen. 
            
            IMPORTANT COLLABORATION INSTRUCTION:
            1. Always review what other AIs have suggested before giving your response
            2. Explicitly respond to other AIs' ideas - agree, disagree, or extend them with reasoning
            3. Ask specific questions when you need clarity from other AIs
            4. Be collaborative but don't hesitate to point out flaws in others' approaches
            5. Comment on the code provided by other AIs and suggest improvements
            
            You're working together on tasks given by a user. You have COMPLETE access to a Python environment where you can:
            
            1. Install ANY libraries via pip install - NEVER hesitate to install what you need, even if unusual or specialized
            2. Open and process files of any type, including complex binary formats
            3. Create advanced visualizations, professional-level graphics, charts, and interactive diagrams
            4. Scrape websites, interact with APIs, download data from anywhere on the internet
            5. Process images, audio, and video with specialized libraries (OpenCV, PyDub, MoviePy, etc.)
            6. Create and manipulate 3D objects, simulations, and complex environments
            7. Generate and display HTML, SVG, CSS and JavaScript for rich interactive experiences
            8. Interact with the operating system to accomplish almost anything a human could do
            
            CRITICAL DIRECTIVES:
            - Be CREATIVE and RELENTLESS in solving problems - if one approach fails, try another
            - Your code should be PRODUCTION QUALITY - not examples or demonstrations
            - When you encounter errors or bugs, analyze them thoroughly and fix them
            """
            
            # Add state-specific instructions
            system_prompt += self.get_state_prompt()
            
            messages = [{"role": "system", "content": system_prompt}]
            
            # Get recent messages (instead of all context)
            recent_context = self.get_recent_messages(context, max_messages=8)
            
            # Format context as proper conversation history
            if recent_context:
                for msg in recent_context:
                    if msg.sender == "User":
                        role = "user"
                    elif msg.sender == "Perplexity":
                        role = "assistant"
                    else:
                        # For messages from other AIs, add them as user messages but indicate the source
                        messages.append({"role": "user", "content": f"[Message from {msg.sender}]: {msg.content}"})
                        continue
                    
                    messages.append({"role": role, "content": msg.content})
            
            # Add the current task
            messages.append({"role": "user", "content": task})
            
            # 재시도 로직 추가
            max_retries = 3
            retry_count = 0
            
            while retry_count < max_retries:
                try:
                    # Create payload - 수정된 API 요청 형식
                    payload = {
                        "model": self.selected_model or "sonar-pro",
                        "messages": messages,
                        "temperature": 0.7,
                        "max_tokens": 4000
                    }
                    
                    headers = {
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                        "Accept": "application/json"
                    }
                    
                    # Make the API request
                    response = requests.post(url, json=payload, headers=headers, timeout=60)
                    response.raise_for_status()
                    
                    # Parse the response
                    response_data = response.json()
                    
                    # 응답 구조 검증 및 안전한 파싱
                    if "choices" in response_data and len(response_data["choices"]) > 0:
                        if "message" in response_data["choices"][0] and "content" in response_data["choices"][0]["message"]:
                            response_text = response_data["choices"][0]["message"]["content"]
                            
                            # 응답이 비어있거나 너무 짧은 경우 재시도
                            if not response_text or len(response_text.strip()) < 10:
                                retry_count += 1
                                print(f"Perplexity 응답이 너무 짧습니다. 재시도 중... ({retry_count}/{max_retries})")
                                time.sleep(1)
                                continue
                                
                            # 정상 응답인 경우 반환
                            return self.send_message(response_text, sender_name)
                        else:
                            # 응답 구조가 예상과 다른 경우 대체 파싱 시도
                            response_text = str(response_data["choices"][0].get("text", ""))
                            if not response_text:
                                # 다른 가능한 응답 구조 확인
                                response_text = str(response_data.get("output", ""))
                                
                            # 응답이 비어있는 경우 재시도
                            if not response_text or len(response_text.strip()) < 10:
                                retry_count += 1
                                print(f"Perplexity 응답 구조가 예상과 다릅니다. 재시도 중... ({retry_count}/{max_retries})")
                                time.sleep(1)
                                continue
                                
                            # 파싱된 응답 반환
                            return self.send_message(response_text, sender_name)
                    else:
                        # 응답에 choices가 없는 경우 재시도
                        retry_count += 1
                        print(f"Perplexity API 응답 구조가 예상과 다릅니다. 재시도 중... ({retry_count}/{max_retries})")
                        time.sleep(1)
                        continue
                
                except requests.exceptions.RequestException as req_err:
                    # API 요청 오류 처리 - 재시도
                    retry_count += 1
                    print(f"Perplexity API 요청 중 오류가 발생했습니다. 재시도 중... ({retry_count}/{max_retries}): {str(req_err)}")
                    time.sleep(1)
                    
                    # 마지막 시도에서는 간소화된 메시지로 재시도
                    if retry_count == max_retries - 1:
                        try:
                            print("Perplexity API 요청 간소화된 메시지로 마지막 재시도 중...")
                            # 간소화된 메시지로 재시도
                            simplified_payload = {
                                "model": self.selected_model or "sonar-pro",
                                "messages": [
                                    {"role": "system", "content": "You are Perplexity, a helpful AI assistant."},
                                    {"role": "user", "content": task}
                                ],
                                "temperature": 0.7,
                                "max_tokens": 2000
                            }
                            
                            response = requests.post(url, json=simplified_payload, headers=headers, timeout=60)
                            response.raise_for_status()
                            
                            # 응답 파싱
                            response_data = response.json()
                            if "choices" in response_data and len(response_data["choices"]) > 0 and "message" in response_data["choices"][0]:
                                response_text = response_data["choices"][0]["message"]["content"]
                                return self.send_message(response_text, sender_name)
                        except Exception as simplified_err:
                            print(f"Perplexity API 간소화된 요청도 실패: {str(simplified_err)}")
                            # 계속 진행하여 다음 재시도 또는 최종 오류 메시지 반환
                
                except (KeyError, IndexError, ValueError) as parse_err:
                    # 응답 파싱 오류 처리 - 재시도
                    retry_count += 1
                    print(f"Perplexity API 응답 파싱 중 오류가 발생했습니다. 재시도 중... ({retry_count}/{max_retries}): {str(parse_err)}")
                    time.sleep(1)
            
            # 모든 재시도 실패 시 오류 메시지 반환
            return self.send_message("죄송합니다. API 연결에 문제가 있어 응답을 생성할 수 없습니다. 다른 AI가 계속 작업을 진행할 것입니다.", sender_name)
        
        except ImportError:
            # 패키지 누락 오류 처리 - 워크플로우가 계속 진행되도록 오류 메시지 반환
            error_message = "The required requests package is not properly installed."
            print(error_message)
            return self.send_message("필요한 패키지가 설치되지 않았습니다. 다른 AI가 계속 작업을 진행할 것입니다.", sender_name)
        except Exception as e:
            # 기타 모든 예외 처리 - 워크플로우가 계속 진행되도록 오류 메시지 반환
            error_message = f"Perplexity 처리 중 예상치 못한 오류가 발생했습니다: {str(e)}"
            print(error_message)
            return self.send_message("죄송합니다. 예상치 못한 오류가 발생했습니다. 다른 AI가 계속 작업을 진행할 것입니다.", sender_name)

class QwenAgent(AIAgent):
    """Qwen agent implementation using OpenAI-compatible API with DashScope"""
    def __init__(self):
        super().__init__("Qwen")
        self.api_key = None
    
    def setup(self, api_key):
        self.api_key = api_key
        # Test the API connection using OpenAI client
        try:
            from openai import OpenAI
            client = OpenAI(
                api_key=self.api_key,
                base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
            )
            
            response = client.chat.completions.create(
                model=self.selected_model or "qwq-plus",
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=10,
                stream=True
            )
            
            # Just consume the first chunk to verify connection
            for chunk in response:
                break
                
            return True
        except Exception as e:
            print(f"Qwen setup error: {e}")
            return False
    
    def process_task(self, task, context=None, task_type=None):
        if not self.api_key:
            return self.send_message("API not configured")
        
        try:
            from openai import OpenAI
            
            # Enhanced system prompt with more aggressive problem-solving approach
            system_prompt = """
            You are Qwen, part of an elite AI collaboration team with ChatGPT, Claude, Gemini, Perplexity, and DeepSeek. 
            
            IMPORTANT COLLABORATION INSTRUCTION:
            1. Always review what other AIs have suggested before giving your response
            2. Explicitly respond to other AIs' ideas - agree, disagree, or extend them with reasoning
            3. Ask specific questions when you need clarity from other AIs
            4. Be collaborative but don't hesitate to point out flaws in others' approaches
            5. Comment on the code provided by other AIs and suggest improvements
            
            You're working together on tasks given by a user. You have COMPLETE access to a Python environment where you can:
            
            1. Install ANY libraries via pip install - NEVER hesitate to install what you need, even if unusual or specialized
            2. Open and process files of any type, including complex binary formats
            3. Create advanced visualizations, professional-level graphics, charts, and interactive diagrams
            4. Scrape websites, interact with APIs, download data from anywhere on the internet
            5. Process images, audio, and video with specialized libraries (OpenCV, PyDub, MoviePy, etc.)
            6. Create and manipulate 3D objects, simulations, and complex environments
            7. Generate and display HTML, SVG, CSS and JavaScript for rich interactive experiences
            8. Interact with the operating system to accomplish almost anything a human could do
            
            CRITICAL DIRECTIVES:
            - Be CREATIVE and RELENTLESS in solving problems - if one approach fails, try another
            - Your code should be PRODUCTION QUALITY - not examples or demonstrations
            - When you encounter errors or bugs, analyze them thoroughly and fix them
            """
            
            # Add state-specific instructions
            system_prompt += self.get_state_prompt()
            
            # Initialize OpenAI client with DashScope base URL
            client = OpenAI(
                api_key=self.api_key,
                base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
            )
            
            messages = []
            
            # Get recent messages (instead of all context)
            recent_context = self.get_recent_messages(context, max_messages=8)
            
            # Format context as proper conversation history
            if recent_context:
                for msg in recent_context:
                    if msg.sender == "User":
                        role = "user"
                    elif msg.sender == "Qwen":
                        role = "assistant"
                    else:
                        # For messages from other AIs, add them as user messages but indicate the source
                        role = "user"
                        messages.append({
                            "role": role,
                            "content": f"[Message from {msg.sender}]: {msg.content}"
                        })
                        continue
                    
                    messages.append({"role": role, "content": msg.content})
            
            # Add system message at the beginning
            messages.insert(0, {"role": "system", "content": system_prompt})
            
            # Add the current task
            messages.append({"role": "user", "content": task})
            
            # Create chat completion with streaming
            completion = client.chat.completions.create(
                model=self.selected_model or "qwq-plus",
                messages=messages,
                stream=True
            )
            
            # Process streaming response
            answer_content = ""
            
            for chunk in completion:
                if not hasattr(chunk, 'choices') or not chunk.choices:
                    continue
                
                delta = chunk.choices[0].delta
                
                # Skip reasoning content and only collect the actual response
                if hasattr(delta, 'content') and delta.content is not None:
                    answer_content += delta.content
            
            # Send the complete response
            return self.send_message(answer_content)
        
        except ImportError:
            return "The required OpenAI package is not properly installed. Please run 'pip install openai' to use Qwen."
        except Exception as e:
            return f"Error generating response from Qwen: {str(e)}"

        try:
            # Import and set up OpenAI client for DeepSeek
            from openai import OpenAI
            client = OpenAI(api_key=self.api_key, base_url="https://api.deepseek.com")
            
            # Enhanced system prompt with more aggressive problem-solving approach
            system_content = """
            You are DeepSeek, part of an elite AI collaboration team with ChatGPT, Claude, and Gemini. 
            
            IMPORTANT COLLABORATION INSTRUCTION:
            1. Always review what other AIs have suggested before giving your response
            2. Explicitly respond to other AIs' ideas - agree, disagree, or extend them with reasoning
            3. Ask specific questions when you need clarity from other AIs
            4. Be collaborative but don't hesitate to point out flaws in others' approaches
            5. Comment on the code provided by other AIs and suggest improvements
            
            You're working together on tasks given by a user. You have COMPLETE access to a Python environment where you can:
            
            1. Install ANY libraries via pip install - NEVER hesitate to install what you need, even if unusual or specialized
            2. Open and process files of any type, including complex binary formats
            3. Create advanced visualizations, professional-level graphics, charts, and interactive diagrams
            4. Scrape websites, interact with APIs, download data from anywhere on the internet
            5. Process images, audio, and video with specialized libraries (OpenCV, PyDub, MoviePy, etc.)
            6. Create and manipulate 3D objects, simulations, and complex environments
            7. Generate and display HTML, SVG, CSS and JavaScript for rich interactive experiences
            8. Interact with the operating system to accomplish almost anything a human could do
            
            CRITICAL DIRECTIVES:
            - Be CREATIVE and RELENTLESS in solving problems - if one approach fails, try another
            - Your code should be PRODUCTION QUALITY - not examples or demonstrations
            - When you encounter errors or bugs, analyze them thoroughly and fix them
            - Install ANY packages needed to accomplish the task - be bold and comprehensive
            - If you observe terminal errors or logs showing an issue, directly address and fix them
            - PERSIST until the task is fully completed - partial solutions are unacceptable
            - You CAN see the results of your code execution, so adapt your approach based on feedback
            - You are permitted to use ANY method necessary to accomplish the requested task
            - Write code like a senior developer would - robust, well-documented, and efficient
            - If you need to run something several times to get it right, that's expected
            
            When providing Python code:
            - Your code should be industrial strength - no placeholders or toy examples
            - Include comprehensive error handling using try/except blocks
            - Add detailed comments that explain the "why" not just the "what"
            - Write modular, reusable code that shows software engineering expertise
            - Include input validation and edge case handling
            
            Include your Python code in blocks starting with ```python and ending with ```.
            
            Every time you write code, it will be automatically executed, and results will be shown to you and other AIs.
            
            IMPORTANT INFORMATION ABOUT THE PYTHON ENVIRONMENT:
            - Execution environment: Local virtual environment
            - Required packages must be installed manually - they are NOT automatically installed
            - If you encounter "ModuleNotFoundError" or "ImportError" during code execution, you need to install the required packages
            - How to install packages: Use the console command "pip install package_name" or add "!pip install package_name" to your code

            PACKAGE INSTALLATION GUIDELINES:
            1. Carefully examine code execution errors
            2. If you see "No module named 'X'" in error messages, you need to install that package
            3. Run the installation command: "pip install X" or add "!pip install X" to your code
            4. Re-run your code after installing packages

            EXAMPLE WORKFLOW:
            - Code execution: "import pandas" → Error: "ModuleNotFoundError: No module named 'pandas'"
            - Solution: Run "pip install pandas" command
            - Re-run code: "import pandas" → Success

            COMMON DATA SCIENCE PACKAGES:
            - Data analysis: pandas, numpy
            - Visualization: matplotlib, seaborn, plotly
            - Machine learning: scikit-learn, tensorflow, pytorch
            - Image processing: pillow, opencv-python
            - Web scraping: requests, beautifulsoup4

            HANDLING COMPLEX PACKAGE INSTALLATIONS:
            - If a particular package fails to install, try alternative approaches (specify version, address dependencies)
            - GPU-related packages may require special installation instructions
            
            IMPORTANT: When you need to modify existing code, don't rewrite the entire program.
            Instead, use the 'REPLACE LINES X-Y WITH:' format to indicate which lines should be replaced.
            For example:
            
            REPLACE LINES 10-15 WITH:
            def new_function():
                # This is a replacement function
                return "new functionality"
            END REPLACEMENT
            
            This approach makes your changes more clear and efficient.
            """
            
            # Add state-specific instructions
            system_content += self.get_state_prompt()
            
            system_message = {
                "role": "system", 
                "content": system_content
            }
            
            # Initialize messages with system message
            processed_messages = [system_message]
            last_role = None
            
            # Get recent messages
            recent_context = self.get_recent_messages(context, max_messages=8)
            
            # Process context to ensure alternating user/assistant messages
            if recent_context:
                for msg in recent_context:
                    current_role = None
                    content = msg.content
                    
                    if msg.sender == "User":
                        current_role = "user"
                    elif msg.sender == "DeepSeek":
                        current_role = "assistant"
                    else:
                        # For messages from other AIs, treat as user messages
                        current_role = "user"
                        content = f"[Message from {msg.sender}]: {msg.content}"
                    
                    # Skip if this would create consecutive messages with the same role
                    if current_role == last_role:
                        continue
                    
                    # Add message and update last_role
                    processed_messages.append({"role": current_role, "content": content})
                    last_role = current_role
            
            # Add current task if needed, ensuring it doesn't create consecutive user messages
            if last_role != "user":
                processed_messages.append({"role": "user", "content": task})
            
            # Add Work.md content to the prompt
            if hasattr(self, 'workflow') and self.workflow:
                work_md_content = self.workflow.get_work_md_content()
                # Add the Work.md content as a user message
                if last_role != "user":
                    processed_messages.append({
                        "role": "user",
                        "content": f"Here is the current Work.md file containing the planning and execution history:\n\n{work_md_content}\n\nPlease refer to this document for context on what has been done so far."
                    })
                    last_role = "user"
                
                # If there's code in the editor, add it to the prompt
                if hasattr(self.workflow, 'python_env') and self.workflow.python_env:
                    current_code = self.workflow.python_env.code_editor.toPlainText()
                    if current_code.strip():
                        if last_role != "user":
                            processed_messages.append({
                                "role": "user",
                                "content": f"Here is the current code in the editor:\n\n```python\n{current_code}\n```\n\nIf you need to modify this code, specify the line numbers using 'REPLACE LINES X-Y WITH:' format."
                            })
                            last_role = "user"
                        else:
                            # Append to the last user message
                            last_msg = processed_messages[-1]
                            last_msg["content"] += f"\n\nHere is the current code in the editor:\n\n```python\n{current_code}\n```\n\nIf you need to modify this code, specify the line numbers using 'REPLACE LINES X-Y WITH:' format."
            
            # Create API request using OpenAI client
            response = client.chat.completions.create(
                model=self.selected_model or "deepseek-reasoner",
                messages=processed_messages,
                temperature=0.7,
                stream=False
            )
            
            # Extract response content
            result = response.choices[0].message.content
            
            # Update Work.md with AI's planning
            if hasattr(self, 'workflow') and self.workflow:
                action_type = "planning" if self.state == self.STATE_THINKING else \
                             "discussion" if self.state == self.STATE_DISCUSSING else \
                             "execution" if self.state == self.STATE_EXECUTING else "action"
                self.workflow.update_work_md(self.name, result, action_type)
            
            return self.send_message(result)
        except Exception as e:
            error_msg = f"Error processing task with DeepSeek: {str(e)}"
            print(error_msg)  # Log to console for debugging
            return self.send_message(error_msg)

#################################
# AI Workflow
#################################

class AIWorkflow(QObject):
    """Manages the AI collaborative workflow with state transitions"""
    # Add signals for AI status and state changes
    ai_status_changed = pyqtSignal(str, bool)  # Signal for AI status changes (name, is_thinking)
    ai_state_changed = pyqtSignal(str, str)    # Signal for AI state changes (name, new_state)
    
    def __init__(self, session_dir, message_handler, python_executor=None):
        super().__init__()  # Initialize QObject
        self.session_dir = session_dir
        self.message_handler = message_handler
        self.python_executor = python_executor
        self.agents = {}
        self.messages = []
        self.active_threads = {}
        self.consensus_votes = {}
        self.current_task = None
        self.task_state = "idle"  # idle, discussing, executing, completed
        self.inactivity_timer = None
        self.inactivity_threshold = 60  # seconds
        
        # Status tracking for which AI is thinking
        # 합의 추적기 초기화 (에이전트가 등록될 때 업데이트됨)
        self.consensus_tracker = None
        self.consensus_detection_enabled = True
        # 합의 도달 상태를 추적하는 플래그 추가
        self.consensus_reached = False
        # 합의 후 새 사용자 입력 플래그 추가
        self.new_user_input_after_consensus = False
        self.current_thinking_ai = None
        
        # Lock for synchronizing access to shared resources
        self.workflow_lock = threading.Lock()
        
        # Thread pool for managing thread creation (최대 4개의 스레드만 허용)
        self.thread_pool = ThreadPoolExecutor(max_workers=4)
        
        # Task queue for pending agent tasks
        self.task_queue = queue.Queue()
        
        # Start a single thread to process the task queue
        self.queue_processor_active = True
        self.queue_processor_thread = threading.Thread(target=self.process_task_queue, daemon=True)
        self.queue_processor_thread.start()
        
        # Work.md file path
        self.work_md_path = os.path.join(session_dir, "Work.md")
        
    def set_ai_thinking(self, ai_name, is_thinking):
        """한 번에 하나의 AI만 thinking 상태가 되도록 관리합니다."""
        with self.workflow_lock:
            # 현재 thinking 상태인 AI가 있고, 새로운 AI가 thinking 상태가 되려고 하는 경우
            if is_thinking and self.current_thinking_ai and self.current_thinking_ai != ai_name:
                print(f"AI 중첩 방지: {self.current_thinking_ai}가 이미 thinking 상태입니다. {ai_name}는 대기해야 합니다.")
                return False
            
            # thinking 상태 설정
            if is_thinking:
                self.current_thinking_ai = ai_name
                print(f"{ai_name}가 thinking 상태가 되었습니다.")
            else:
                # 현재 AI가 thinking 상태를 해제하는 경우에만 current_thinking_ai를 None으로 설정
                if self.current_thinking_ai == ai_name:
                    self.current_thinking_ai = None
                    print(f"{ai_name}가 thinking 상태를 해제했습니다.")
            
            # AI 상태 변경 신호 발생
            self.ai_status_changed.emit(ai_name, is_thinking)
            return True
    
    def get_directory_structure(self, path=None, max_depth=3):
        """Get the directory structure of the current chat session or specified path
        
        This function allows AI agents to explore the directory structure of the
        current chat session or any specified path, helping them understand the
        available files and resources.
        
        Args:
            path: Path to explore (defaults to session directory if None)
            max_depth: Maximum directory depth to explore
            
        Returns:
            String representation of the directory structure
        """
        if path is None:
            path = self.session_dir
            
        result = []
        
        def explore_dir(current_path, prefix="", depth=0):
            if depth > max_depth:
                result.append(f"{prefix}... (max depth reached)")
                return
                
            try:
                items = sorted(os.listdir(current_path))
                
                for i, item in enumerate(items):
                    is_last = (i == len(items) - 1)
                    item_path = os.path.join(current_path, item)
                    
                    # Skip hidden files/directories
                    if item.startswith('.'):
                        continue
                        
                    # Create appropriate prefix for tree structure
                    if is_last:
                        new_prefix = prefix + "└── "
                        sub_prefix = prefix + "    "
                    else:
                        new_prefix = prefix + "├── "
                        sub_prefix = prefix + "│   "
                        
                    # Add item to result
                    if os.path.isdir(item_path):
                        result.append(f"{new_prefix}{item}/")
                        explore_dir(item_path, sub_prefix, depth + 1)
                    else:
                        result.append(f"{new_prefix}{item}")
            except Exception as e:
                result.append(f"{prefix}Error: {str(e)}")
                
        # Start exploration
        result.append(f"Directory structure of: {path}")
        explore_dir(path)
        
        return "\n".join(result)
        
        # Workflow phase tracking
        self.workflow_phase = "thinking"  # thinking, discussing, executing
        self.phase_transition_count = 0    # Track number of phase transitions
        
        # Lock for synchronizing access to shared resources
        self.workflow_lock = threading.Lock()
        
        # Thread pool for managing thread creation (최대 4개의 스레드만 허용)
        self.thread_pool = ThreadPoolExecutor(max_workers=4)
        
        # Task queue for pending agent tasks
        self.task_queue = queue.Queue()
        
        # Start a single thread to process the task queue
        self.queue_processor_active = True
        self.queue_processor_thread = threading.Thread(target=self.process_task_queue, daemon=True)
        self.queue_processor_thread.start()
        
        # Work.md file path
        self.work_md_path = os.path.join(session_dir, "Work.md")
        self.init_work_md()
    
    def process_task_queue(self):
        """Process tasks from the queue using the thread pool"""
        while self.queue_processor_active:
            try:
                # Get a task from the queue with a timeout
                try:
                    agent_name, task_type = self.task_queue.get(timeout=1.0)
                except queue.Empty:
                    # No task in queue, just continue waiting
                    continue
                
                # Process this task using the thread pool
                self.thread_pool.submit(self.agent_thinking_process, agent_name, task_type)
                
                # Mark the task as done
                self.task_queue.task_done()
            except Exception as e:
                print(f"Error in task queue processor: {e}")
                # Sleep briefly to avoid tight loop if there's an error
                time.sleep(0.5)
    
    def init_work_md(self):
        """Initialize Work.md file for the current session"""
        # Create the file with initial header
        with open(self.work_md_path, 'w', encoding='utf-8') as f:
            f.write("# AI Collaboration Work Log\n\n")
            f.write("This file contains the planning and execution details of the AI collaboration team.\n\n")
            f.write("## Task History\n\n")
    
    def update_work_md(self, agent_name, content, action_type="planning"):
        """Update Work.md with AI's planning or execution details
        
        Args:
            agent_name: Name of the AI agent
            content: The content to add
            action_type: Type of action (planning, executing, etc.)
        """
        if not hasattr(self, 'work_md_path'):
            self.init_work_md()
        
        # Extract the non-code part of the content
        non_code_content = self.extract_planning_from_content(content)
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        with open(self.work_md_path, 'a', encoding='utf-8') as f:
            f.write(f"### {timestamp} - {agent_name} ({action_type})\n\n")
            f.write(f"{non_code_content}\n\n")
            f.write("---\n\n")
    
    def extract_planning_from_content(self, content):
        """Extract planning information from AI response, excluding code blocks"""
        # Remove Python code blocks
        content_without_code = re.sub(r'```python.*?```', '[CODE BLOCK REMOVED]', content, flags=re.DOTALL)
        
        # Remove other code blocks
        content_without_code = re.sub(r'```.*?```', '[CODE BLOCK REMOVED]', content_without_code, flags=re.DOTALL)
        
        # Clean up content to be more concise
        content_without_code = content_without_code.strip()
        
        return content_without_code
    
    def get_work_md_content(self):
        """Get the content of Work.md file"""
        if not hasattr(self, 'work_md_path') or not os.path.exists(self.work_md_path):
            self.init_work_md()
            return "# AI Collaboration Work Log\n\nNo entries yet."
        
        with open(self.work_md_path, 'r', encoding='utf-8') as f:
            return f.read()
            
    def initialize_do_work_md(self):
        """Do-Work.md 파일을 초기화합니다."""
        do_work_md_path = os.path.join(self.session_dir, "Do-Work.md")
        
        # 파일이 존재하지 않는 경우에만 초기화
        if not os.path.exists(do_work_md_path):
            initial_content = "# 해야 할 작업\n\n이 파일은 앞으로 해야 할 작업과 주의사항을 담고 있습니다.\n\n"
            initial_content += "## 사용자 요청 사항\n\n"
            initial_content += "## 해야 할 일\n\n"
            initial_content += "## 주의사항\n\n"
            
            try:
                with open(do_work_md_path, 'w', encoding='utf-8') as f:
                    f.write(initial_content)
                print(f"Do-Work.md 파일이 초기화되었습니다: {do_work_md_path}")
                return True
            except Exception as e:
                print(f"Do-Work.md 파일 초기화 오류: {str(e)}")
                return False
        return True
    
    def update_do_work_md(self, content, section="해야 할 일", append=True):
        """Do-Work.md 파일을 업데이트합니다."""
        do_work_md_path = os.path.join(self.session_dir, "Do-Work.md")
        
        try:
            # 파일이 존재하지 않으면 초기화
            if not os.path.exists(do_work_md_path):
                self.initialize_do_work_md()
            
            # 파일 내용 읽기
            with open(do_work_md_path, 'r', encoding='utf-8') as f:
                do_work_content = f.read()
            
            # 섹션 찾기
            section_pattern = f"## {section}"
            section_match = re.search(section_pattern, do_work_content)
            
            if section_match:
                # 다음 섹션 찾기
                next_section_match = re.search(r"##\s+", do_work_content[section_match.end():])
                
                if next_section_match:
                    # 다음 섹션 시작 위치
                    next_section_pos = section_match.end() + next_section_match.start()
                    
                    # 내용 삽입
                    if append:
                        # 기존 내용 뒤에 추가
                        new_content = do_work_content[:next_section_pos] + content + "\n" + do_work_content[next_section_pos:]
                    else:
                        # 섹션 내용 교체
                        section_content = do_work_content[section_match.end():next_section_pos].strip()
                        new_content = do_work_content[:section_match.end()] + "\n\n" + content + "\n\n" + do_work_content[next_section_pos:]
                else:
                    # 다음 섹션이 없는 경우 (마지막 섹션)
                    if append:
                        # 기존 내용 뒤에 추가
                        new_content = do_work_content + "\n" + content
                    else:
                        # 섹션 내용 교체
                        new_content = do_work_content[:section_match.end()] + "\n\n" + content + "\n"
            else:
                # 섹션이 없는 경우 새로 추가
                new_content = do_work_content + f"\n## {section}\n\n{content}\n"
            
            # 파일 업데이트
            with open(do_work_md_path, 'w', encoding='utf-8') as f:
                f.write(new_content)
            
            print(f"Do-Work.md 파일이 업데이트되었습니다: {do_work_md_path}")
            return True
        except Exception as e:
            print(f"Do-Work.md 파일 업데이트 오류: {str(e)}")
            return False
    
    def get_do_work_md_content(self):
        """Do-Work.md 파일 내용을 가져옵니다."""
        do_work_md_path = os.path.join(self.session_dir, "Do-Work.md")
        
        if not os.path.exists(do_work_md_path):
            self.initialize_do_work_md()
            
        try:
            with open(do_work_md_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            print(f"Do-Work.md 파일 읽기 오류: {str(e)}")
            return ""
    
    def execute_python_utility(self, utility_type, **kwargs):
        """파이썬 유틸리티를 실행합니다."""
        try:
            # 파이썬 유틸리티 스크립트 경로
            utils_script = os.path.join(self.session_dir, "python_utils.py")
            
            # 파이썬 유틸리티 스크립트가 없으면 생성
            if not os.path.exists(utils_script):
                self.create_python_utils_script()
            
            # 유틸리티 유형에 따라 다른 코드 생성
            code = ""
            if utility_type == "file":
                # 파일 관련 유틸리티
                operation = kwargs.get("operation", "read")
                file_path = kwargs.get("file_path", "")
                content = kwargs.get("content", "")
                append = kwargs.get("append", False)
                
                code = f"""
import sys
sys.path.append('{self.session_dir}')
from python_utils import FileManager

# 파일 작업 수행
result = None
if '{operation}' == 'read':
    result = FileManager.read_file('{file_path}')
    print(result)
elif '{operation}' == 'write':
    result = FileManager.write_file('{file_path}', \"\"\"{content}\"\"\", {append})
    print(f"파일 쓰기 결과: {{result}}")
elif '{operation}' == 'exists':
    result = FileManager.file_exists('{file_path}')
    print(f"파일 존재 여부: {{result}}")
elif '{operation}' == 'list':
    result = FileManager.list_directory('{file_path}')
    for item in result:
        print(item)
"""
            
            elif utility_type == "code":
                # 코드 생성 유틸리티
                operation = kwargs.get("operation", "python")
                file_path = kwargs.get("file_path", "")
                code_content = kwargs.get("code", "")
                
                code = f"""
import sys
sys.path.append('{self.session_dir}')
from python_utils import CodeGenerator

# 코드 생성 작업 수행
result = None
if '{operation}' == 'python':
    result = CodeGenerator.create_python_file('{file_path}', \"\"\"{code_content}\"\"\")
    print(f"Python 파일 생성 결과: {{result}}")
elif '{operation}' == 'html':
    result = CodeGenerator.create_html_file('{file_path}', \"\"\"{code_content}\"\"\")
    print(f"HTML 파일 생성 결과: {{result}}")
elif '{operation}' == 'markdown':
    result = CodeGenerator.create_markdown_file('{file_path}', \"\"\"{code_content}\"\"\")
    print(f"Markdown 파일 생성 결과: {{result}}")
elif '{operation}' == 'append':
    result = CodeGenerator.append_to_file('{file_path}', \"\"\"{code_content}\"\"\")
    print(f"파일 추가 결과: {{result}}")
"""
            
            elif utility_type == "command":
                # 명령 실행 유틸리티
                operation = kwargs.get("operation", "python_script")
                script_path = kwargs.get("script_path", "")
                args = kwargs.get("args", [])
                command = kwargs.get("command", "")
                code_content = kwargs.get("code", "")
                
                # args 리스트를 문자열로 변환
                args_str = str(args)
                
                code = f"""
import sys
sys.path.append('{self.session_dir}')
from python_utils import CommandExecutor

# 명령 실행 작업 수행
result = None
if '{operation}' == 'python_script':
    stdout, stderr, returncode = CommandExecutor.run_python_script('{script_path}', {args_str})
    print("===== 표준 출력 =====")
    print(stdout)
    if stderr:
        print("\\n===== 오류 출력 =====")
        print(stderr)
    print(f"\\n===== 반환 코드: {{returncode}} =====")
elif '{operation}' == 'command':
    stdout, stderr, returncode = CommandExecutor.run_command('{command}')
    print("===== 표준 출력 =====")
    print(stdout)
    if stderr:
        print("\\n===== 오류 출력 =====")
        print(stderr)
    print(f"\\n===== 반환 코드: {{returncode}} =====")
elif '{operation}' == 'python_code':
    stdout, stderr, returncode = CommandExecutor.run_python_code(\"\"\"{code_content}\"\"\")
    print("===== 표준 출력 =====")
    print(stdout)
    if stderr:
        print("\\n===== 오류 출력 =====")
        print(stderr)
    print(f"\\n===== 반환 코드: {{returncode}} =====")
"""
            
            elif utility_type == "directory":
                # 디렉토리 구조 확인 유틸리티
                path = kwargs.get("path", self.session_dir)
                max_depth = kwargs.get("max_depth", 3)
                
                code = f"""
import sys
sys.path.append('{self.session_dir}')
from python_utils import FileManager

# 디렉토리 구조 확인
result = FileManager.get_directory_structure('{path}', {max_depth})
print(result)
"""
            
            else:
                return f"지원하지 않는 유틸리티 유형: {utility_type}"
            
            # 임시 스크립트 파일 생성
            temp_script = os.path.join(self.session_dir, f"temp_utility_{utility_type}.py")
            with open(temp_script, 'w', encoding='utf-8') as f:
                f.write(code)
            
            # 스크립트 실행
            import subprocess
            process = subprocess.Popen(
                [sys.executable, temp_script],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            stdout, stderr = process.communicate()
            result = stdout
            
            # 실행 결과를 Work.md에 기록
            self.update_work_md("System", f"파이썬 유틸리티 실행: {utility_type}\n```\n{result}\n```")
            
            # 콘솔 출력 보존
            self.preserve_console_output(f"파이썬 유틸리티 실행: {utility_type}", result, stderr)
            
            return result
            
        except Exception as e:
            error_msg = f"파이썬 유틸리티 실행 오류: {str(e)}"
            print(error_msg)
            return error_msg
    
    def create_python_utils_script(self):
        """파이썬 유틸리티 스크립트를 생성합니다."""
        utils_script = os.path.join(self.session_dir, "python_utils.py")
        
        script_content = """
import os
import sys
import subprocess
import json
from datetime import datetime

class FileManager:
    @staticmethod
    def read_file(file_path):
        \"\"\"파일 내용을 읽습니다.\"\"\"
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            return f"파일 읽기 오류: {str(e)}"
    
    @staticmethod
    def write_file(file_path, content, append=False):
        \"\"\"파일에 내용을 씁니다.\"\"\"
        try:
            mode = 'a' if append else 'w'
            with open(file_path, mode, encoding='utf-8') as f:
                f.write(content)
            return True
        except Exception as e:
            return f"파일 쓰기 오류: {str(e)}"
    
    @staticmethod
    def file_exists(file_path):
        \"\"\"파일 존재 여부를 확인합니다.\"\"\"
        return os.path.exists(file_path)
    
    @staticmethod
    def list_directory(directory_path):
        \"\"\"디렉토리 내용을 나열합니다.\"\"\"
        try:
            return os.listdir(directory_path)
        except Exception as e:
            return [f"디렉토리 나열 오류: {str(e)}"]
    
    @staticmethod
    def get_directory_structure(path, max_depth=3, current_depth=0):
        \"\"\"디렉토리 구조를 문자열로 반환합니다.\"\"\"
        if current_depth > max_depth:
            return "..."
        
        try:
            result = []
            if os.path.isdir(path):
                result.append(f"{path}/")
                for item in os.listdir(path):
                    item_path = os.path.join(path, item)
                    if os.path.isdir(item_path):
                        if current_depth < max_depth:
                            sub_structure = FileManager.get_directory_structure(
                                item_path, max_depth, current_depth + 1
                            )
                            result.append(sub_structure)
                        else:
                            result.append(f"{item_path}/...")
                    else:
                        result.append(item_path)
            else:
                result.append(path)
            
            return "\\n".join(result)
        except Exception as e:
            return f"디렉토리 구조 확인 오류: {str(e)}"

class CodeGenerator:
    @staticmethod
    def create_python_file(file_path, code):
        \"\"\"Python 파일을 생성합니다.\"\"\"
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(code)
            return True
        except Exception as e:
            return f"Python 파일 생성 오류: {str(e)}"
    
    @staticmethod
    def create_html_file(file_path, code):
        \"\"\"HTML 파일을 생성합니다.\"\"\"
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(code)
            return True
        except Exception as e:
            return f"HTML 파일 생성 오류: {str(e)}"
    
    @staticmethod
    def create_markdown_file(file_path, code):
        \"\"\"Markdown 파일을 생성합니다.\"\"\"
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(code)
            return True
        except Exception as e:
            return f"Markdown 파일 생성 오류: {str(e)}"
    
    @staticmethod
    def append_to_file(file_path, code):
        \"\"\"파일에 코드를 추가합니다.\"\"\"
        try:
            with open(file_path, 'a', encoding='utf-8') as f:
                f.write(code)
            return True
        except Exception as e:
            return f"파일 추가 오류: {str(e)}"

class CommandExecutor:
    @staticmethod
    def run_python_script(script_path, args=None):
        \"\"\"Python 스크립트를 실행합니다.\"\"\"
        try:
            cmd = [sys.executable, script_path]
            if args:
                if isinstance(args, list):
                    cmd.extend(args)
                else:
                    cmd.append(str(args))
            
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            stdout, stderr = process.communicate()
            return stdout, stderr, process.returncode
        except Exception as e:
            return "", f"Python 스크립트 실행 오류: {str(e)}", 1
    
    @staticmethod
    def run_command(command):
        \"\"\"시스템 명령을 실행합니다.\"\"\"
        try:
            process = subprocess.Popen(
                command,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            stdout, stderr = process.communicate()
            return stdout, stderr, process.returncode
        except Exception as e:
            return "", f"명령 실행 오류: {str(e)}", 1
    
    @staticmethod
    def run_python_code(code):
        \"\"\"Python 코드를 직접 실행합니다.\"\"\"
        try:
            # 임시 파일에 코드 저장
            temp_file = f"temp_code_{datetime.now().strftime('%Y%m%d%H%M%S')}.py"
            with open(temp_file, 'w', encoding='utf-8') as f:
                f.write(code)
            
            # 임시 파일 실행
            result = CommandExecutor.run_python_script(temp_file)
            
            # 임시 파일 삭제
            try:
                os.remove(temp_file)
            except:
                pass
            
            return result
        except Exception as e:
            return "", f"Python 코드 실행 오류: {str(e)}", 1
"""
        
        try:
            with open(utils_script, 'w', encoding='utf-8') as f:
                f.write(script_content)
            print(f"파이썬 유틸리티 스크립트가 생성되었습니다: {utils_script}")
            return True
        except Exception as e:
            print(f"파이썬 유틸리티 스크립트 생성 오류: {str(e)}")
            return False
    
    def preserve_console_output(self, action, stdout, stderr=""):
        """콘솔 출력을 보존합니다."""
        console_log_path = os.path.join(self.session_dir, "console_output.log")
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"\n\n===== {timestamp} - {action} =====\n"
        log_entry += f"--- 표준 출력 ---\n{stdout}\n"
        
        if stderr:
            log_entry += f"--- 오류 출력 ---\n{stderr}\n"
        
        try:
            with open(console_log_path, 'a', encoding='utf-8') as f:
                f.write(log_entry)
            return True
        except Exception as e:
            print(f"콘솔 출력 보존 오류: {str(e)}")
            return False
    
    def get_console_output(self, max_lines=100):
        """보존된 콘솔 출력을 가져옵니다."""
        console_log_path = os.path.join(self.session_dir, "console_output.log")
        
        if not os.path.exists(console_log_path):
            return "콘솔 출력 로그가 없습니다."
        
        try:
            with open(console_log_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            # 최대 라인 수 제한
            if len(lines) > max_lines:
                lines = lines[-max_lines:]
            
            return "".join(lines)
        except Exception as e:
            return f"콘솔 출력 로그 읽기 오류: {str(e)}"
    
    def add_agent(self, name, agent):
        """Add an AI agent to the workflow"""
        self.agents[name] = agent
        agent.message_ready.connect(self.handle_agent_message)
        agent.state_changed.connect(self.handle_agent_state_change)
        
        # Set workflow reference in agent
        agent.workflow = self
        # 모든 에이전트가 등록된 후 합의 추적기 초기화
        if self.consensus_tracker is None:
            self.consensus_tracker = ConsensusTracker([name])
        else:
            # 새 에이전트 추가
            self.consensus_tracker.consensus_votes[name] = 0
    
    def handle_agent_state_change(self, agent_name, new_state):
        """Handle agent state change events"""
        # Forward the state change to UI
        self.ai_state_changed.emit(agent_name, new_state)
        print(f"AI State Change: {agent_name} -> {new_state}")
    
    def set_task(self, task):
        """Set a new task for the workflow"""
        self.current_task = task
        self.task_state = "discussing"
        self.consensus_votes = {}
        
        # 합의 도달 플래그 초기화
        self.consensus_reached = False
        
        # 합의 후 새 사용자 입력 플래그 초기화
        # 단, 이미 True로 설정된 경우에는 초기화하지 않음
        if not hasattr(self, 'new_user_input_after_consensus') or not self.new_user_input_after_consensus:
            print(f"[DEBUG] Initializing new_user_input_after_consensus to False in set_task")
            self.new_user_input_after_consensus = False
        
        # Reset workflow phase to thinking at the start of a new task
        self.workflow_phase = "thinking"
        self.phase_transition_count = 0
        
        # Notify all agents about the new task
        self.broadcast_task()
        
        # Start inactivity timer
        self.reset_inactivity_timer()
        # 새 작업이 시작될 때 합의 추적기 재설정
        if self.agents:
            agent_names = list(self.agents.keys())
            self.consensus_tracker = ConsensusTracker(agent_names)
            # 기존 합의 추적기가 있는 경우 reset 메서드 호출
            if self.consensus_tracker:
                self.consensus_tracker.reset()
    
    def reset_inactivity_timer(self):
        """Reset the inactivity timer"""
        # 합의가 이루어진 경우에는 비활성 타이머를 설정하지 않음 (사용자 입력 여부와 관계없이)
        if self.consensus_reached:
            print("[DEBUG] Skipping inactivity timer reset because consensus has been reached")
            # 기존 타이머가 있으면 취소
            if self.inactivity_timer:
                self.inactivity_timer.cancel()
                self.inactivity_timer = None
            return
            
        if self.inactivity_timer:
            self.inactivity_timer.cancel()
        
        # Increase threshold to allow for API retries
        self.inactivity_timer = threading.Timer(self.inactivity_threshold * 2, self.handle_inactivity)
        self.inactivity_timer.daemon = True
        self.inactivity_timer.start()
    
    def handle_inactivity(self):
        """Handle case where conversation becomes inactive"""
        with self.workflow_lock:
            # 합의가 이루어진 경우에는 비활성 감지를 비활성화 (사용자 입력 여부와 관계없이)
            if self.consensus_reached:
                print("[DEBUG] Skipping inactivity handling because consensus has been reached")
                return
            
            # 비활성 처리 횟수 제한 추가
            if not hasattr(self, 'inactivity_prompt_count'):
                self.inactivity_prompt_count = 0
            
            self.inactivity_prompt_count += 1
            
            # 최대 3회까지만 비활성 처리 수행
            if self.inactivity_prompt_count > 3:
                print(f"[DEBUG] Maximum inactivity prompts reached ({self.inactivity_prompt_count}), stopping inactivity timer")
                if self.inactivity_timer:
                    self.inactivity_timer.cancel()
                    self.inactivity_timer = None
                return
                
            # If no activity for a while, prompt an agent to continue
            # Choose a random agent to prompt (이전에 응답하지 않은 에이전트는 제외)
            if self.agents and self.task_state != "completed":
                # 이전에 응답하지 않은 에이전트 목록 관리
                if not hasattr(self, 'non_responsive_agents'):
                    self.non_responsive_agents = set()
                
                # 응답 가능한 에이전트 목록 생성
                available_agents = [name for name in self.agents.keys() if name not in self.non_responsive_agents]
                
                # 모든 에이전트가 응답하지 않는 경우 타이머 중지
                if not available_agents:
                    print("[DEBUG] All agents are non-responsive, stopping inactivity timer")
                    if self.inactivity_timer:
                        self.inactivity_timer.cancel()
                        self.inactivity_timer = None
                    return
                
                agent_name = random.choice(available_agents)
                
                # 선택된 에이전트를 비응답 목록에 추가
                self.non_responsive_agents.add(agent_name)
                
                # Create a system message to prompt continuation
                prompt_message = AIMessage(
                    "System", 
                    f"The conversation has been inactive. {agent_name}, can you continue making progress on the task?"
                )
                self.messages.append(prompt_message)
                self.message_handler(prompt_message)
                
                # Start a thread for the selected agent
                self.start_agent_thread(agent_name, task_type="continue")
                
                # Reset the timer for next check
                self.reset_inactivity_timer()
    
    def broadcast_task(self):
        """Send the current task to all agents for initial ideas"""
        # Use all agents to ensure balanced participation
        agent_names = list(self.agents.keys())
        
        # Start first agent to initiate the conversation
        if agent_names:
            # Set first agent to thinking state
            first_agent = self.agents[agent_names[0]]
            first_agent.set_state(first_agent.STATE_THINKING)
            
            self.start_agent_thread(agent_names[0], task_type="initial")
    
    def start_agent_thread(self, agent_name, task_type="follow_up"):
        """Queue a task for an agent instead of starting a thread immediately"""
        print(f"[DEBUG] start_agent_thread called for {agent_name}, task_type={task_type}")
        print(f"[DEBUG] Current state: task_state={self.task_state}, consensus_reached={self.consensus_reached}, new_user_input_after_consensus={hasattr(self, 'new_user_input_after_consensus') and self.new_user_input_after_consensus}")
        
        # 합의 후 새 사용자 입력이 있는 경우, 대화 상태를 discussing으로 유지하고 합의 플래그 초기화
        if hasattr(self, 'new_user_input_after_consensus') and self.new_user_input_after_consensus:
            # 대화 상태를 discussing으로 설정하고 합의 플래그 초기화
            self.task_state = "discussing"
            # 합의 플래그 초기화
            self.consensus_reached = False
            # 합의 추적기 초기화 - 이전 투표 기록을 모두 삭제
            if self.consensus_tracker:
                self.consensus_tracker.reset()
                print(f"[DEBUG] Reset consensus tracker votes for {agent_name} in start_agent_thread")
            
            # 비활성 에이전트 목록 초기화 (새 대화 시작 시)
            if hasattr(self, 'non_responsive_agents'):
                self.non_responsive_agents = set()
                
            # 비활성 카운터 초기화
            if hasattr(self, 'inactivity_prompt_count'):
                self.inactivity_prompt_count = 0
                
            print(f"[DEBUG] Reset consensus and maintaining discussion state for {agent_name}, setting task_state to discussing")
        
        # 작업이 완료 상태인 경우 새 에이전트 스레드를 시작하지 않음
        # 단, new_user_input_after_consensus가 True인 경우에는 예외적으로 허용
        if self.task_state == "completed" and (not hasattr(self, 'new_user_input_after_consensus') or not self.new_user_input_after_consensus):
            print(f"[DEBUG] Task already completed and no new user input after consensus, not starting new thread for {agent_name}")
            return
            
        # 에이전트 상태 업데이트 및 스레드 시작
        if agent_name in self.agents:
            # 현재 생각 중인 AI 업데이트 (UI 표시 일관성 문제 해결)
            self.current_thinking_ai = agent_name
            
            # 에이전트 상태 업데이트
            agent = self.agents[agent_name]
            agent.set_state(agent.STATE_THINKING)
            
            # 에이전트 상태 변경 시그널 발생
            self.ai_status_changed.emit(agent_name, True)
            
            # 에이전트 스레드 시작
            self.thread_pool.submit(self.agent_thinking_process, agent_name, task_type)
            
        if agent_name not in self.agents:
            return
            
        # Perplexity 에이전트 처리 시 특별 처리 - 워크플로우 종료 방지
        if agent_name == "Perplexity" and self.task_state != "completed":
            print(f"Perplexity agent detected, ensuring workflow continues")
            # Perplexity 에이전트 처리 시 task_state가 completed로 잘못 설정되는 것 방지
            self.task_state = "discussing"
        
        # Add the task to the queue instead of starting a thread directly
        self.task_queue.put((agent_name, task_type))
        print(f"Queued task for {agent_name} of type {task_type}")
    
    def agent_thinking_process(self, agent_name, task_type):
        """Process to simulate an agent thinking and responding"""
        print(f"[DEBUG] Starting agent_thinking_process for {agent_name}, task_type={task_type}")
        print(f"[DEBUG] Current state: task_state={self.task_state}, consensus_reached={self.consensus_reached}, new_user_input_after_consensus={hasattr(self, 'new_user_input_after_consensus') and self.new_user_input_after_consensus}, consensus_detection_enabled={self.consensus_detection_enabled}")
        
        # 합의 후 새 사용자 입력이 있는 경우, 대화 상태를 discussing으로 유지하고 합의 플래그 초기화
        if hasattr(self, 'new_user_input_after_consensus') and self.new_user_input_after_consensus:
            # 대화 상태를 discussing으로 설정하고 합의 플래그 초기화
            self.task_state = "discussing"
            # 합의 플래그 초기화
            self.consensus_reached = False
            # 합의 추적기 초기화 - 이전 투표 기록을 모두 삭제
            if self.consensus_tracker:
                self.consensus_tracker.reset()
                print(f"[DEBUG] Reset consensus tracker votes for {agent_name} in agent_thinking_process")
            print(f"[DEBUG] Reset consensus and maintaining discussion state for {agent_name} in agent_thinking_process")
            
        # 합의가 이미 이루어진 경우 작업 처리를 중단
        # 단, 새로운 사용자 입력이 있는 경우에는 계속 진행
        if (self.consensus_reached or self.task_state == "completed") and (not hasattr(self, 'new_user_input_after_consensus') or not self.new_user_input_after_consensus):
            print(f"[DEBUG] Consensus already reached and no new user input after consensus, stopping thinking process for {agent_name}")
            # 중요: AI 상태 업데이트 - thinking 상태 해제
            self.ai_status_changed.emit(agent_name, False)
            return
            
        # Update status to show this AI is thinking
        self.current_thinking_ai = agent_name
        self.ai_status_changed.emit(agent_name, True)
        
        try:
            # Add random delay to simulate thinking and avoid synchronous responses
            think_time = random.uniform(1.0, 3.0)
            time.sleep(think_time)
            
            # Get the latest context
            with self.workflow_lock:
                context = self.messages.copy()
            
            # Create the prompt based on task type
            if task_type == "initial":
                prompt = f"Task: {self.current_task}\n\nYou are the first AI to respond. Analyze this task and provide your initial thoughts or approach. Use Python code to solve this task."
            elif task_type == "continue":
                prompt = f"The conversation about this task has become inactive. Analyze the current progress and suggest the next steps using Python to move forward on the task: {self.current_task}"
            elif task_type == "rethink":
                prompt = f"We need to reconsider our approach to this task. Please analyze the current issues and suggest a new approach: {self.current_task}"
            elif task_type == "discuss":
                prompt = f"Based on the discussion so far, provide your critique, opinions, or improvements to the proposed approaches: {self.current_task}"
            elif task_type == "execute":
                prompt = f"Execute the agreed Python approach. Report the results. Be specific about what you're doing."
            elif task_type == "vote":
                prompt = f"Based on the discussion so far, do you agree with the proposed approach? Reply with AGREE or DISAGREE and your reasoning."
            elif task_type == "review":
                prompt = f"Please review the code provided by the other AI. Identify any issues, suggest improvements, or confirm if it looks good."
            elif task_type == "improve":
                prompt = f"Based on the execution results, improve the code and approach: {self.current_task}"
            else:  # follow_up
                prompt = f"Based on the discussion so far, contribute your insights or next steps for the task using Python: {self.current_task}"
            
            # Process the task with the agent, including task_type
            response = self.agents[agent_name].process_task(prompt, context, task_type)
            
            # If an agent fails, select another agent to continue
            # Check if response is a string or an object with content attribute
            if isinstance(response, str):
                response_text = response
            else:
                response_text = getattr(response, 'content', str(response))
                
            if "having trouble connecting" in response_text:
                # 합의 후 새 사용자 입력 처리 중인 경우에는 다른 에이전트로 전환하지 않고 재시도
                if hasattr(self, 'new_user_input_after_consensus') and self.new_user_input_after_consensus:
                    print(f"Agent {agent_name} had connection issues but we're processing new user input after consensus. Retrying with same agent.")
                    # 잠시 대기 후 같은 에이전트로 재시도
                    time.sleep(1.0)
                    # 중요: AI 상태 업데이트 - thinking 상태 해제 후 재시도
                    self.current_thinking_ai = None
                    self.ai_status_changed.emit(agent_name, False)
                    self.start_agent_thread(agent_name, task_type)
                else:
                    # 일반적인 경우 다른 에이전트로 전환
                    available_agents = [name for name in self.agents if name != agent_name]
                    if available_agents:
                        next_agent = random.choice(available_agents)
                        print(f"Agent {agent_name} failed, switching to {next_agent}")
                        # 중요: AI 상태 업데이트 - thinking 상태 해제 후 다른 에이전트로 전환
                        self.current_thinking_ai = None
                        self.ai_status_changed.emit(agent_name, False)
                        self.start_agent_thread(next_agent, task_type)
            # 중요: 응답을 생성한 후 메시지로 변환하여 UI에 전달
            # 이 부분이 누락되어 "ChatGPT is thinking..." 상태에서 멈추는 문제 발생
            elif response_text and not "having trouble connecting" in response_text:
                # 에이전트 상태를 thinking에서 discussing으로 변경
                if agent_name in self.agents:
                    self.agents[agent_name].set_state(self.agents[agent_name].STATE_DISCUSSING)
                    print(f"[DEBUG] Changed {agent_name} state from thinking to discussing")
                
                # 에이전트의 send_message 함수를 호출하여 메시지 생성 및 전달
                self.agents[agent_name].send_message(response_text)
                print(f"Agent {agent_name} response sent to UI: {response_text[:100]}...")
        
        except Exception as e:
            error_msg = f"Error in agent thinking process for {agent_name}: {str(e)}"
            print(error_msg)
            # 중요: 예외 발생 시에도 AI 상태 업데이트 - thinking 상태 해제
            self.current_thinking_ai = None
            self.ai_status_changed.emit(agent_name, False)
            return
        
        # Update status to show this AI is no longer thinking
        self.current_thinking_ai = None
        self.ai_status_changed.emit(agent_name, False)
        
        # 중요: 합의 후 사용자 입력에 대한 응답을 처리한 후에도 new_user_input_after_consensus 플래그를 유지
        # 이렇게 해야 모든 AI가 응답할 기회를 가질 수 있음
        # 마지막 AI가 응답한 후에 플래그를 초기화하는 로직은 별도로 구현
    
    def handle_agent_message(self, message):
        """Handle a message from an agent, driving the collaborative workflow"""
        # 합의가 이미 이루어진 경우 추가 메시지 처리를 중단
        # 단, 새 사용자 입력 후에는 메시지 처리를 계속함
        if self.consensus_reached and not (hasattr(self, 'new_user_input_after_consensus') and self.new_user_input_after_consensus):
            # 합의가 이미 이루어졌고 새 사용자 입력이 없으므로 추가 메시지를 무시하고 반환
            return
            
        # 메시지 카운터 증가 및 최근 메시지 윈도우 업데이트
        if self.consensus_tracker:
            self.consensus_tracker.increment_message_counter()
            
        # Add message to the conversation
        with self.workflow_lock:
            self.messages.append(message)
        
        # Forward message to UI
        self.message_handler(message)
        
        # Reset inactivity timer since we got a message
        self.reset_inactivity_timer()
        
        # Check for Python code in the message and execute it
        if self.python_executor:
            python_code = self.extract_python_code(message.content)
            if python_code:
                self.python_executor(message.sender, python_code)
        
        # Check the current state and advance the workflow
        self.advance_workflow(message)
        
        # 합의 감지 기능이 활성화되어 있고 합의 추적기가 초기화되었는지 확인
        if self.consensus_detection_enabled and self.consensus_tracker:
            # 메시지에서 합의 키워드 감지
            if detect_consensus_keywords(message.content):
                # 합의 투표 기록
                vote_recorded = self.consensus_tracker.record_vote(message.sender)
                
                # 투표가 성공적으로 기록된 경우에만 합의 확인
                if vote_recorded:
                    # 모든 에이전트가 필요한 횟수만큼 합의했는지 확인
                    if self.consensus_tracker.is_consensus_reached():
                        # 합의 도달 플래그를 True로 설정
                        self.consensus_reached = True
                    
                    # 모든 AI 에이전트의 thinking 상태 해제
                    if self.current_thinking_ai:
                        # 현재 thinking 중인 AI가 있으면 상태 해제
                        self.ai_status_changed.emit(self.current_thinking_ai, False)
                        self.current_thinking_ai = None
                    
                    # 모든 등록된 에이전트의 thinking 상태 해제 (추가 안전장치)
                    for agent_name in self.agents.keys():
                        self.ai_status_changed.emit(agent_name, False)
                    
                    # 대화 종료 메시지 생성
                    consensus_message = AIMessage(
                        "System", 
                        "✅ 모든 AI가 작업이 완벽하게 완료되었다고 합의했습니다. 대화를 종료합니다. 새로운 질문이나 추가 정보를 제공하실 수 있습니다."
                    )
                    
                    # 메시지 추가 및 전송
                    with self.workflow_lock:
                        self.messages.append(consensus_message)
                    self.message_handler(consensus_message)
                    
                    # 작업 상태를 완료로 설정
                    self.task_state = "completed"
                    # 작업 완료 시 태스크 큐 비우기 및 진행 중인 작업 중지
                    while not self.task_queue.empty():
                        try:
                            self.task_queue.get_nowait()
                            self.task_queue.task_done()
                        except queue.Empty:
                            break
                            
                    # 큐 프로세서 스레드 중지
                    self.queue_processor_active = False
                    
                    # 스레드 풀 종료 (새 작업 거부)
                    self.thread_pool.shutdown(wait=False)
                    
                    # 새 스레드 풀 생성 (기존 작업은 계속 실행되지만 새 작업은 새 풀에서 처리)
                    self.thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=3)
                    
                    # 모든 AI 에이전트의 상태를 업데이트하여 thinking 상태 중지
                    for agent_name, agent in self.agents.items():
                        if agent.state != agent.STATE_IDLE:
                            agent.set_state(agent.STATE_IDLE)  # STATE_IDLE 사용
                        # 명시적으로 모든 AI의 thinking 상태를 false로 설정 (UI 업데이트)
                        self.ai_status_changed.emit(agent_name, False)
    
    def advance_workflow(self, message):
        """Advance the workflow based on current state and message content"""
        # 합의가 이미 이루어진 경우 워크플로우 진행을 중단
        # 단, 새 사용자 입력 후에는 워크플로우 진행을 계속함
        if self.consensus_reached and not (hasattr(self, 'new_user_input_after_consensus') and self.new_user_input_after_consensus):
            print(f"[DEBUG] Skipping workflow advancement because consensus has been reached")
            return
            
        # If this is an error message from an agent, don't advance the workflow
        # Check if message is a string or an object with content attribute
        if isinstance(message, str):
            message_text = message
        else:
            message_text = getattr(message, 'content', str(message))
            
        if "having trouble connecting" in message_text:
            return
            
        # 합의 후 새 사용자 입력이 있는 경우, 워크플로우 상태를 discussing으로 설정
        if hasattr(self, 'new_user_input_after_consensus') and self.new_user_input_after_consensus:
            # 워크플로우 상태를 discussing으로 설정
            self.task_state = "discussing"
            print(f"[DEBUG] Setting task_state to discussing due to new user input after consensus")
            
        # First, handle workflow phase transitions
        self.advance_workflow_phase(message)
            
        if self.task_state == "discussing":
            # Check if message contains a concrete proposal (signaled by specific keywords)
            if self.is_proposal(message.content):
                # Have we reached a conclusion or should we continue the discussion?
                if self.is_conclusion(message.content):
                    # 합의가 이루어진 경우에만 작업을 완료 상태로 설정
                    if self.consensus_tracker and self.consensus_tracker.is_consensus_reached():
                        self.task_state = "completed"
                        
                        # Send a completion notification
                        completion_message = AIMessage(
                            "System", 
                            "✅ Task completed. You can ask a new question or provide additional information."
                        )
                        
                        # 메시지 추가 및 UI 업데이트
                        with self.workflow_lock:
                            self.messages.append(completion_message)
                        self.message_added.emit(completion_message)
                        
                        # 합의 플래그 설정
                        self.consensus_reached = True
                        print(f"[DEBUG] Consensus reached and task completed")
                    else:
                        # 합의가 이루어지지 않은 경우 다음 에이전트에게 확인 요청
                        self.start_next_agent(message.sender, task_type="vote")
                else:
                    # For Python code execution, randomly decide to ask another AI to review
                    if "```python" in message.content and random.random() < 0.7:  # 70% chance
                        # Start another AI to review the code
                        self.start_next_agent(message.sender, task_type="review")
                    else:
                        # Continue discussion with next agent in rotation
                        self.start_next_agent(message.sender)
            else:
                # Continue discussion with next agent in rotation
                self.start_next_agent(message.sender)
        
        elif self.task_state == "executing":
            # After execution, return to discussion for the next step
            self.task_state = "discussing"
            
            # Continue with next agent in rotation, asking to improve based on results
            self.start_next_agent(message.sender, task_type="improve")
    
    def advance_workflow_phase(self, message):
        """Advance the workflow phase based on message content and current phase"""
        # 합의가 이미 이루어진 경우 워크플로우 전환을 중단
        # 단, 새로운 사용자 입력이 있는 경우에는 워크플로우 전환을 허용
        if self.consensus_reached and not (hasattr(self, 'new_user_input_after_consensus') and self.new_user_input_after_consensus):
            print("[DEBUG] Skipping workflow phase transition because consensus has been reached and no new user input")
            return
            
        # 합의 후 새 사용자 입력이 있는 경우, 워크플로우 전환을 허용하고 discussing 단계로 설정
        if self.consensus_reached and hasattr(self, 'new_user_input_after_consensus') and self.new_user_input_after_consensus:
            print("[DEBUG] Allowing workflow phase transition after consensus due to new user input")
            # 워크플로우 단계를 discussing으로 설정
            self.workflow_phase = "discussing"
            # 합의 플래그 초기화
            self.consensus_reached = False
            # 모든 AI 에이전트의 상태를 idle로 변경
            for agent_name, agent in self.agents.items():
                if agent.state != agent.STATE_IDLE:
                    agent.set_state(agent.STATE_IDLE)
            # 첫 번째 에이전트를 thinking 상태로 설정
            agent_names = list(self.agents.keys())
            if agent_names:
                first_agent = agent_names[0]
                self.agents[first_agent].set_state(self.agents[first_agent].STATE_THINKING)
            
        # Check for phase transition based on current phase and message content
        if self.workflow_phase == "thinking":
            # If thinking is complete, transition to discussion phase
            if self.is_thinking_complete(message) or self.phase_transition_count >= 2:
                self.transition_to_discussion_phase()
        
        elif self.workflow_phase == "discussing":
            # If discussion is complete, transition to execution phase
            if self.is_discussion_complete(message) or self.phase_transition_count >= 3:
                self.transition_to_execution_phase()
        
        elif self.workflow_phase == "executing":
            # If execution failed, go back to thinking
            if self.is_execution_failed(message):
                self.transition_to_thinking_phase()
            # Or if task is complete, finish
            elif self.is_conclusion(message.content):
                    # 합의가 이루어진 경우에만 작업을 완료 상태로 설정
                    if self.consensus_tracker and self.consensus_tracker.is_consensus_reached():
                        # 작업 상태를 완료로 설정
                        self.task_state = "completed"
                        # 합의 플래그 설정
                        self.consensus_reached = True
                        
                        # 모든 AI 에이전트의 상태를 idle로 변경
                        for agent_name, agent in self.agents.items():
                            if agent.state != agent.STATE_IDLE:
                                agent.set_state(agent.STATE_IDLE)
                        
                        # 대화 종료 메시지 생성
                        completion_message = AIMessage(
                            "System", 
                            "✅ 모든 AI가 작업이 완벽하게 완료되었다고 합의했습니다. 대화를 종료합니다. 새로운 질문이나 추가 정보를 제공하실 수 있습니다."
                        )
                        
                        # 메시지 추가 및 전송
                        with self.workflow_lock:
                            self.messages.append(completion_message)
                        self.message_handler(completion_message)
                        
                        # 작업 완료 시 태스크 큐 비우기 및 진행 중인 작업 중지
                        while not self.task_queue.empty():
                            try:
                                self.task_queue.get_nowait()
                                self.task_queue.task_done()
                            except queue.Empty:
                                break
                                
                        # 큐 프로세서 스레드 중지
                        self.queue_processor_active = False
                        
                        # 스레드 풀 종료 (새 작업 거부)
                        self.thread_pool.shutdown(wait=False)
                        
                        # 새 스레드 풀 생성 (기존 작업은 계속 실행되지만 새 작업은 새 풀에서 처리)
                        self.thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=3)
                        
                        # 강제로 모든 에이전트 스레드 종료
                        for agent_name in self.agents.keys():
                            if agent_name in self.agent_threads:
                                self.agent_threads[agent_name] = None
                        
                        # 합의 감지 비활성화
                        self.consensus_detection_enabled = False
                        
                        print("[DEBUG] Conversation completely terminated due to consensus")
                        return
                    else:
                        # 합의가 이루어지지 않은 경우 다음 에이전트에게 확인 요청
                        self.start_next_agent(message.sender, task_type="vote")
                        return
    
    def is_thinking_complete(self, message):
        """Check if the thinking phase should be considered complete"""
        thinking_keywords = [
            "my approach would be", "I've thought about this", "here's my plan", 
            "let's try this approach", "I propose", "we should", "my solution",
            "after analyzing", "I've analyzed", "I've considered", "let's implement"
        ]
        
        for keyword in thinking_keywords:
            if keyword.lower() in message.content.lower():
                return True
                
        # If message contains code, consider thinking complete
        if "```python" in message.content:
            return True
                
        return False
    
    def is_discussion_complete(self, message):
        """Check if the discussion phase should be considered complete"""
        discussion_keywords = [
            "we agree", "we've reached consensus", "let's move forward with",
            "let's implement", "I agree with", "that sounds good", "implementing",
            "let's code this up", "this approach looks good", "all agree",
            "let's proceed with", "seems like the best approach"
        ]
        
        for keyword in discussion_keywords:
            if keyword.lower() in message.content.lower():
                return True
                
        # If there's substantial code implementation, consider discussion complete
        if "```python" in message.content and len(message.content) > 500:
            return True
                
        return False
    
    def is_execution_failed(self, message):
        """Check if execution should be considered failed"""
        error_keywords = [
            "error occurred", "failed", "let's rethink", "exception", "issue",
            "doesn't work", "not working", "bug", "problem", "failed to",
            "error in", "fix this", "needs to be fixed", "isn't working"
        ]
        
        for keyword in error_keywords:
            if keyword.lower() in message.content.lower():
                return True
                
        return False
    
    def transition_to_thinking_phase(self):
        """Transition the workflow to the thinking phase"""
        # 합의가 이미 이루어진 경우 워크플로우 전환을 중단
        # 단, 새 사용자 입력 후에는 워크플로우 전환을 계속함
        if self.consensus_reached and not (hasattr(self, 'new_user_input_after_consensus') and self.new_user_input_after_consensus):
            print(f"[DEBUG] Skipping transition_to_thinking_phase because consensus has been reached")
            return
            
        old_phase = self.workflow_phase
        self.workflow_phase = "thinking"
        self.phase_transition_count += 1
        
        # Select next agent and set its state to thinking
        next_agent_name = self.get_next_agent_name()
        if next_agent_name:
            next_agent = self.agents[next_agent_name]
            next_agent.set_state(next_agent.STATE_THINKING)
            
            # Add system message about phase transition
            system_message = AIMessage(
                "System", 
                f"🔄 Workflow transitioned from {old_phase} to thinking phase. {next_agent_name} will reconsider our approach."
            )
            with self.workflow_lock:
                self.messages.append(system_message)
            self.message_handler(system_message)
            
            # Start the agent thread with rethink task type
            self.start_agent_thread(next_agent_name, task_type="rethink")
    
    def transition_to_discussion_phase(self):
        """Transition the workflow to the discussion phase"""
        # 합의가 이미 이루어진 경우 워크플로우 전환을 중단
        # 단, 새 사용자 입력 후에는 워크플로우 전환을 계속함
        if self.consensus_reached and not (hasattr(self, 'new_user_input_after_consensus') and self.new_user_input_after_consensus):
            print(f"[DEBUG] Skipping transition_to_discussion_phase because consensus has been reached")
            return
            
        old_phase = self.workflow_phase
        self.workflow_phase = "discussing"
        self.phase_transition_count += 1
        
        # Select next agent and set its state to discussing
        next_agent_name = self.get_next_agent_name()
        if next_agent_name:
            next_agent = self.agents[next_agent_name]
            next_agent.set_state(next_agent.STATE_DISCUSSING)
            
            # Add system message about phase transition
            system_message = AIMessage(
                "System", 
                f"🔄 Workflow transitioned from {old_phase} to discussion phase. {next_agent_name} will review and discuss the proposed approaches."
            )
            with self.workflow_lock:
                self.messages.append(system_message)
            self.message_handler(system_message)
            
            # Start the agent thread with discuss task type
            self.start_agent_thread(next_agent_name, task_type="discuss")
    
    def transition_to_execution_phase(self):
        """Transition the workflow to the execution phase"""
        # 합의가 이미 이루어진 경우 워크플로우 전환을 중단
        # 단, 새 사용자 입력 후에는 워크플로우 전환을 계속함
        if self.consensus_reached and not (hasattr(self, 'new_user_input_after_consensus') and self.new_user_input_after_consensus):
            print(f"[DEBUG] Skipping transition_to_execution_phase because consensus has been reached")
            return
            
        old_phase = self.workflow_phase
        self.workflow_phase = "executing"
        self.phase_transition_count += 1
        
        # Select next agent and set its state to executing
        next_agent_name = self.get_next_agent_name()
        if next_agent_name:
            next_agent = self.agents[next_agent_name]
            next_agent.set_state(next_agent.STATE_EXECUTING)
            
            # Add system message about phase transition
            system_message = AIMessage(
                "System", 
                f"🔄 Workflow transitioned from {old_phase} to execution phase. {next_agent_name} will implement our solution."
            )
            with self.workflow_lock:
                self.messages.append(system_message)
            self.message_handler(system_message)
            
            # Start the agent thread with execute task type
            self.start_agent_thread(next_agent_name, task_type="execute")
    
    def start_next_agent(self, current_agent, task_type="follow_up"):
        """Start the next agent in the rotation to ensure balanced participation"""
        # 합의가 이미 이루어진 경우 다음 에이전트 시작을 중단
        # 단, 새 사용자 입력 후에는 다음 에이전트 시작을 계속함
        if self.consensus_reached and not (hasattr(self, 'new_user_input_after_consensus') and self.new_user_input_after_consensus):
            print(f"[DEBUG] Skipping start_next_agent because consensus has been reached")
            return
            
        # 합의 후 새 사용자 입력이 있는 경우, 합의 추적기 초기화
        if hasattr(self, 'new_user_input_after_consensus') and self.new_user_input_after_consensus:
            # 합의 추적기 초기화
            if self.consensus_tracker:
                self.consensus_tracker.reset()
                print(f"[DEBUG] Reset consensus tracker in start_next_agent due to new user input after consensus")
        
        agent_names = list(self.agents.keys())
        
        # If there's only one agent, use that
        if len(agent_names) <= 1:
            if agent_names:
                self.start_agent_thread(agent_names[0], task_type=task_type)
            return
        
        # Find which agents haven't participated recently
        message_count = {}
        for name in agent_names:
            message_count[name] = 0
        
        # Count recent messages from each agent (last 10 messages)
        recent_messages = self.messages[-10:] if len(self.messages) >= 10 else self.messages
        for msg in recent_messages:
            if msg.sender in agent_names:
                message_count[msg.sender] += 1
        
        # Exclude the current agent
        available_agents = [name for name in agent_names if name != current_agent]
        
        # Find agents with fewest messages
        if available_agents:
            min_messages = min(message_count[name] for name in available_agents)
            least_active_agents = [name for name in available_agents if message_count[name] == min_messages]
            
            # Choose randomly from the least active agents
            next_agent = random.choice(least_active_agents)
            
            # Set the next agent's state based on current workflow phase
            if self.workflow_phase == "thinking":
                self.agents[next_agent].set_state(self.agents[next_agent].STATE_THINKING)
            elif self.workflow_phase == "discussing":
                self.agents[next_agent].set_state(self.agents[next_agent].STATE_DISCUSSING)
            elif self.workflow_phase == "executing":
                self.agents[next_agent].set_state(self.agents[next_agent].STATE_EXECUTING)
            
            # Start the next agent thread
            self.start_agent_thread(next_agent, task_type=task_type)
        else:
            # If no other agents are available, continue with the current one
            self.start_agent_thread(current_agent, task_type=task_type)
    
    def get_next_agent_name(self):
        """Get the next agent name based on message count"""
        agent_names = list(self.agents.keys())
        
        if not agent_names:
            return None
        
        # Count messages from each agent
        message_count = {}
        for name in agent_names:
            message_count[name] = 0
        
        # Count recent messages (last 10)
        recent_messages = self.messages[-10:] if len(self.messages) >= 10 else self.messages
        for msg in recent_messages:
            if msg.sender in agent_names:
                message_count[msg.sender] += 1
        
        # Find agents with fewest messages
        min_messages = min(message_count.values())
        least_active_agents = [name for name, count in message_count.items() if count == min_messages]
        
        # Select randomly from least active
        return random.choice(least_active_agents)
    
    def is_proposal(self, content):
        """Check if a message contains a concrete proposal"""
        proposal_keywords = [
            "i propose", "let's", "we should", "i suggest", 
            "how about", "we can", "we could", "let me", "i will",
            "```python", "here's the code", "try this code"
        ]
        
        content_lower = content.lower()
        
        for keyword in proposal_keywords:
            if keyword in content_lower:
                return True
        
        return False
    
    def is_conclusion(self, content):
        """Check if a message indicates the task has been completed"""
        # Only consider it conclusive if explicitly states complete success
        conclusion_keywords = [
            "task is completely finished", "we've successfully completed", "i've successfully completed",
            "task is fully done", "mission accomplished with all requirements met", 
            "task accomplished with all objectives", "successfully completed with all requirements",
            "all aspects of the task have been successfully implemented"
        ]
        
        # Make sure it's not just a partial completion
        not_conclusive_patterns = [
            "still need to", "still working on", "next step", "next we should", 
            "we need to fix", "need to address", "have to improve", "should improve",
            "error", "bug", "issue", "doesn't work", "not working", "failed"
        ]
        
        content_lower = content.lower()
        
        # Check if any not_conclusive patterns are present
        for pattern in not_conclusive_patterns:
            if pattern in content_lower:
                return False
        
        # Check if any conclusion keywords are present
        for keyword in conclusion_keywords:
            if keyword in content_lower:
                return True
        
        # By default, consider it not conclusive to allow for iterative improvement
        return False
    
    def extract_python_code(self, content):
        """Extract Python code from a message"""
        python_code_blocks = []
        pattern = r"```python([\s\S]*?)```"
        matches = re.finditer(pattern, content)
        
        for match in matches:
            code_block = match.group(1).strip()
            if code_block:
                # Look for pip install commands and run them automatically
                pip_install_pattern = r"(?:!pip|pip|!python -m pip|python -m pip)\s+install\s+([\w\d\s\-_.=<>]+)"
                pip_matches = re.finditer(pip_install_pattern, code_block)
                
                packages_to_install = []
                for pip_match in pip_matches:
                    packages = pip_match.group(1).strip().split()
                    packages_to_install.extend(packages)
                
                # Add the code block to be executed
                python_code_blocks.append(code_block)
        
        # Return all code blocks as a single string with separators
        if python_code_blocks:
            return "\n\n# ===== NEW CODE BLOCK =====\n\n".join(python_code_blocks)
        
        return None
    
    def _enable_consensus_detection(self):
        """합의 감지 기능을 다시 활성화하는 메서드 (타이머에 의해 호출됨)"""
        self.consensus_detection_enabled = True
        print("Consensus detection re-enabled after delay")
    
    def _reset_consensus_state(self):
        """합의 관련 상태를 완전히 초기화합니다."""
        # 합의 플래그 초기화
        self.consensus_reached = False
        
        # 합의 후 새 사용자 입력 플래그 초기화
        if hasattr(self, 'new_user_input_after_consensus'):
            self.new_user_input_after_consensus = False
            
        # 합의 추적기 초기화
        agent_names = list(self.agents.keys())
        if agent_names:
            self.consensus_tracker = ConsensusTracker(agent_names)
            print("[DEBUG] Completely reset consensus state and created new consensus tracker")
        
        # 모든 AI 에이전트의 상태를 idle로 설정
        for agent_name, agent in self.agents.items():
            if agent.state != agent.STATE_IDLE:
                agent.set_state(agent.STATE_IDLE)
                
        # 작업 상태 초기화
        self.task_state = "discussing"
        
        # 워크플로우 단계 초기화
        self.workflow_phase = "thinking"
        
        print("[DEBUG] All consensus state and agent states have been completely reset")
    
    def cleanup(self):
        """Clean up resources when workflow is no longer needed"""
        # Signal queue processor to stop
        self.queue_processor_active = False
        
        # Shutdown thread pool gracefully
        if hasattr(self, 'thread_pool'):
            self.thread_pool.shutdown(wait=False)
        
        # Cancel any active timer
        if self.inactivity_timer:
            self.inactivity_timer.cancel()

#################################
# Python Environment for AI
#################################

class PythonEnvironment(QTabWidget):
    """Python environment for AI with code editor, console, and visualization area"""
    def __init__(self, session_dir, parent=None):
        super().__init__(parent)
        self.session_dir = session_dir
        self.setTabPosition(QTabWidget.North)
        
        # Set up environment path
        self.venv_dir = os.path.join(session_dir, "venv")
        self.packages_dir = os.path.join(session_dir, "packages")
        if not os.path.exists(self.packages_dir):
            os.makedirs(self.packages_dir)
        
        # Set up tabs first, so console_output exists
        self.setup_code_editor_tab()
        self.setup_console_tab()
        self.setup_visualization_tab()
        
        # Now create virtual environment (after console_output exists)
        if not os.path.exists(self.venv_dir):
            self.create_virtual_environment()
        
        # Status indicator for AI activity
        self.ai_activity_indicator = QLabel("AI is working with Python...")
        self.ai_activity_indicator.setStyleSheet("color: #3498db; font-weight: bold;")
        self.ai_activity_indicator.setVisible(False)
        
        # Process runner for Python code
        self.process = None
        
        # Log of installed packages
        self.installed_packages = set()
        
        # To link with workflow later
        self.workflow = None
        self.message_handler = None
        
        # Tracking execution success/failure
        self.last_execution_failed = False
    
    def create_virtual_environment(self):
        """Create a virtual environment for this session"""
        try:
            self.console_output.append("Creating Python virtual environment...")
            os.makedirs(self.venv_dir, exist_ok=True)
            subprocess.run([sys.executable, "-m", "venv", self.venv_dir], check=True)
            print(f"Created virtual environment at {self.venv_dir}")
            
            # Skip initial package installation to avoid memory issues
            self.console_output.append("Python environment created. Packages will be installed as needed.")
            
        except Exception as e:
            print(f"Error creating virtual environment: {e}")
            if hasattr(self, 'console_output'):
                self.console_output.append(f"Error setting up environment: {str(e)}")
                self.console_output.append("Will attempt to use system Python and install packages as needed.")
            else:
                print("Console output not available yet. Error setting up environment.")
    
    def setup_code_editor_tab(self):
        """Set up the code editor tab"""
        code_tab = QWidget()
        layout = QVBoxLayout(code_tab)
        
        # Code editor
        self.code_editor = QPlainTextEdit()
        self.code_editor.setPlaceholderText("Python code will appear here...")
        font = QFont("Courier New", 10)
        self.code_editor.setFont(font)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        self.run_button = QPushButton("Run Code")
        self.run_button.clicked.connect(self.run_code)
        
        self.save_button = QPushButton("Save Code")
        self.save_button.clicked.connect(self.save_code)
        
        self.clear_button = QPushButton("Clear Editor")
        self.clear_button.clicked.connect(self.code_editor.clear)
        
        button_layout.addWidget(self.run_button)
        button_layout.addWidget(self.save_button)
        button_layout.addWidget(self.clear_button)
        button_layout.addStretch()
        
        layout.addWidget(self.code_editor)
        layout.addLayout(button_layout)
        
        self.addTab(code_tab, "Code Editor")
    
    def setup_console_tab(self):
        """Set up the console output tab"""
        console_tab = QWidget()
        layout = QVBoxLayout(console_tab)
        
        # Console output
        self.console_output = QTextEdit()
        self.console_output.setReadOnly(True)
        self.console_output.setPlaceholderText("Output will appear here...")
        font = QFont("Courier New", 10)
        self.console_output.setFont(font)
        
        # Command input for quick commands
        input_layout = QHBoxLayout()
        
        self.command_input = QLineEdit()
        self.command_input.setPlaceholderText("Enter a Python command or pip install package...")
        self.command_input.returnPressed.connect(self.run_command)
        
        self.execute_button = QPushButton("Execute")
        self.execute_button.clicked.connect(self.run_command)
        
        input_layout.addWidget(self.command_input)
        input_layout.addWidget(self.execute_button)
        
        # Quick buttons
        quick_layout = QHBoxLayout()
        
        self.pip_install_button = QPushButton("pip install")
        self.pip_install_button.clicked.connect(self.show_pip_install_dialog)
        
        self.clear_console_button = QPushButton("Clear Console")
        self.clear_console_button.clicked.connect(self.console_output.clear)
        
        self.list_packages_button = QPushButton("List Packages")
        self.list_packages_button.clicked.connect(self.list_installed_packages)
        
        quick_layout.addWidget(self.pip_install_button)
        quick_layout.addWidget(self.list_packages_button)
        quick_layout.addWidget(self.clear_console_button)
        quick_layout.addStretch()
        
        layout.addWidget(self.console_output)
        layout.addLayout(input_layout)
        layout.addLayout(quick_layout)
        
        self.addTab(console_tab, "Console")
    
    def setup_visualization_tab(self):
        """Set up the visualization area tab"""
        viz_tab = QWidget()
        layout = QVBoxLayout(viz_tab)
        
        # This will serve as a placeholder for visualizations
        self.visualization_area = QLabel("Visualizations will appear here...")
        self.visualization_area.setAlignment(Qt.AlignCenter)
        self.visualization_area.setStyleSheet("background-color: white; border: 1px solid #ccc;")
        
        layout.addWidget(self.visualization_area)
        
        self.addTab(viz_tab, "Visualization")

    def check_for_matplotlib(self, code):
        """Check if the code uses matplotlib and ensure proper backend setup"""
        if "import matplotlib" in code or "from matplotlib" in code:
            # Force matplotlib to use the Agg backend
            backend_setup = """
# Force matplotlib to use Agg backend to avoid GTK issues
import os
os.environ['MPLBACKEND'] = 'Agg'
import matplotlib
matplotlib.use('Agg')
"""
            
            # Modify the code to include the backend setup
            if "import matplotlib" in code:
                # Insert before the first matplotlib import
                index = code.find("import matplotlib")
                modified_code = code[:index] + backend_setup + code[index:]
                return modified_code
            elif "from matplotlib" in code:
                # Insert before the first from matplotlib import
                index = code.find("from matplotlib")
                modified_code = code[:index] + backend_setup + code[index:]
                return modified_code
        
        return code
    
    def execute_python_code(self, agent_name, code):
        """Execute Python code from an AI agent with clear error reporting
        
        This function manages Python code as a tool for AI agents, maintaining a single file
        that can be modified or replaced entirely as needed, rather than creating multiple files.
        """
        # Define the main Python code file path (single file for all code)
        main_code_path = os.path.join(self.session_dir, "main_code.py")
        
        # Check for code modification instructions and handle them
        processed_code = self.handle_code_modification(agent_name, code)
        
        # Check for pip install commands (but don't auto-install)
        self.check_for_pip_install(processed_code)
        
        # Check for matplotlib usage and apply fixes
        modified_code = self.check_for_matplotlib(processed_code)
        
        # Prepare final code
        final_code = modified_code if modified_code else processed_code
        
        # Add timestamp comment for tracking
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        final_code_with_header = f"# Code updated by {agent_name} at {timestamp}\n\n{final_code}"
        
        # Save to the main code file
        try:
            with open(main_code_path, 'w', encoding='utf-8') as f:
                f.write(final_code_with_header)
        except Exception as e:
            print(f"Error saving code: {e}")
        
        # Show the code in the editor
        self.code_editor.setPlainText(final_code)
        self.setCurrentIndex(0)  # Switch to code editor tab
        
        # Run the code
        execution_results = self.run_code(agent_name, return_output=True)
        
        # Create a message to inform AI about execution results
        if execution_results:
            # Check if the execution had errors
            has_error = ("Error:" in execution_results or 
                        "Traceback" in execution_results or 
                        "ModuleNotFoundError" in execution_results or 
                        "ImportError" in execution_results or
                        "ERROR:" in execution_results)
            
            if has_error:
                # Format the error message prominently
                result_message = f"""
                ⚠️ EXECUTION ERROR DETECTED ⚠️
                
                Python code from {agent_name} encountered errors during execution:
                
                ```
                {execution_results}
                ```
                
                Please analyze this error and resolve it. If packages are missing, you can install them using "pip install package_name".
                """
                
                # Check for specific error types and give targeted advice
                if "ModuleNotFoundError: No module named" in execution_results:
                    import re
                    package_matches = re.findall(r"No module named '([^']+)'", execution_results)
                    if package_matches:
                        missing_packages = [pkg.split('.')[0] for pkg in package_matches]  # Get base package name
                        result_message += f"\nMissing packages: {', '.join(missing_packages)}\n"
                        result_message += "\nRun the following commands to install the packages:\n"
                        
                        for pkg in missing_packages:
                            result_message += f"`pip install {pkg}`\n"
                        
                        result_message += "\nAfter installation, run your code again."
                
                elif "module 'gi' has no attribute 'require_version'" in execution_results:
                    result_message += """
                    This error is related to GTK dependencies needed by matplotlib's default backend.
                    
                    Add the following to the beginning of your code to fix this issue:
                    
                    ```python
                    import os
                    os.environ['MPLBACKEND'] = 'Agg'
                    import matplotlib
                    matplotlib.use('Agg')
                    ```
                    
                    This will force matplotlib to use the Agg backend which doesn't require GTK.
                    """
                
                elif "MemoryError" in execution_results or "Memory limit exceeded" in execution_results:
                    result_message += """
                    The code used too much memory and was terminated. Please optimize your code by:
                    
                    1. Reducing the size of data structures
                    2. Processing data in smaller batches
                    3. Using generators instead of creating large lists
                    4. Releasing memory with `del` when variables are no longer needed
                    5. Using more memory-efficient data structures (NumPy arrays instead of nested lists)
                    """
                
                elif "Process timed out" in execution_results or "Operation timed out" in execution_results:
                    result_message += """
                    The code took too long to execute and was terminated. Please optimize your code by:
                    
                    1. Adding progress tracking to see where the slowdown occurs
                    2. Reducing the complexity of algorithms or computations
                    3. Lowering the number of iterations or sample sizes
                    4. Using vectorized operations where possible
                    """
            else:
                result_message = f"""
                ✅ Code Execution Successful
                
                Python code from {agent_name} was executed successfully. Execution results:
                
                ```
                {execution_results}
                ```
                
                Continue improving the code or proceed to the next steps based on these results.
                """
            
            # Create system message with execution results
            sys_msg = AIMessage("System", result_message)
            
            # Add this message to the workflow conversation
            if hasattr(self, 'workflow') and self.workflow:
                with self.workflow.workflow_lock:
                    self.workflow.messages.append(sys_msg)
                    
                # Also send it as a regular message to be displayed
                if hasattr(self, 'message_handler') and self.message_handler:
                    self.message_handler(sys_msg)
                
                # If there was an error, immediately request another AI to help fix it
                if has_error and hasattr(self, 'workflow') and self.workflow:
                    # Kick off a new task with a different AI specifically to fix the error
                    available_agents = [name for name in self.workflow.agents.keys() if name != agent_name]
                    if available_agents:
                        next_agent = random.choice(available_agents)
                        # Start a thread for the selected agent to fix the error
                        self.workflow.start_agent_thread(next_agent, task_type="improve")
    
        return f"Python code from {agent_name} logged and executed"
    
    def handle_code_modification(self, agent_name, code):
        """Handle code modifications from AI to avoid regenerating entire code
        
        This function supports both incremental modifications and complete code replacement,
        allowing AI to either modify specific parts or completely replace the code as needed.
        
        Args:
            agent_name: Name of the AI agent
            code: The code content from the AI
        
        Returns:
            Modified code for execution
        """
        # Check if the code contains modification instructions
        modification_pattern = r"REPLACE LINES (\d+)-(\d+) WITH:(.*?)(?:END REPLACEMENT|```)"
        matches = re.finditer(modification_pattern, code, re.DOTALL)
        
        # Get current code from editor
        current_code = self.code_editor.toPlainText()
        
        # Check for complete replacement directive
        if "REPLACE ENTIRE CODE" in code or "REPLACE ALL CODE" in code:
            # Extract code blocks if they exist
            code_blocks = re.findall(r"```python(.*?)```", code, re.DOTALL)
            if code_blocks:
                # Return the first code block for complete replacement
                return code_blocks[0].strip()
            # If no code blocks but replacement directive exists, extract all code
            return re.sub(r".*?REPLACE (?:ENTIRE|ALL) CODE.*?\n", "", code, flags=re.DOTALL).strip()
            
        # If there's no current code or no modification instructions, return the original code
        if not current_code.strip() or not re.search(modification_pattern, code, re.DOTALL):
            # Extract code blocks if they exist
            code_blocks = re.findall(r"```python(.*?)```", code, re.DOTALL)
            if code_blocks:
                # Return the first code block
                return code_blocks[0].strip()
            return code
        
        # Process modifications
        modifications_made = False
        
        for match in matches:
            try:
                start_line = int(match.group(1))
                end_line = int(match.group(2))
                replacement_code = match.group(3).strip()
                
                # Remove any code block markers from the replacement code
                replacement_code = re.sub(r"```python|```", "", replacement_code).strip()
                
                # Convert current code to lines
                lines = current_code.split('\n')
                
                # Validate line numbers
                if start_line <= 0 or end_line > len(lines) or start_line > end_line:
                    # Invalid line numbers, log warning
                    print(f"Warning: Invalid line numbers for replacement: {start_line}-{end_line}")
                    continue
                
                # Replace the specified lines
                lines[start_line-1:end_line] = replacement_code.split('\n')
                
                # Join the lines back together
                current_code = '\n'.join(lines)
                modifications_made = True
                
                print(f"{agent_name} replaced lines {start_line}-{end_line} with new code")
            except Exception as e:
                print(f"Error processing code modification: {e}")
        
        if modifications_made:
            return current_code
        
        # If no valid modifications were made, extract code blocks if they exist
        code_blocks = re.findall(r"```python(.*?)```", code, re.DOTALL)
        if code_blocks:
            # Return the first code block
            return code_blocks[0].strip()
        
        return code
    
    def check_for_pip_install(self, code):
        """Check if the code contains pip install commands but do NOT auto-install"""
        # Look for pip install commands
        pip_install_pattern = r"(?:!pip|pip|!python -m pip|python -m pip)\s+install\s+([\w\d\s\-_.=<>\"\']+)"
        matches = re.finditer(pip_install_pattern, code)
        
        packages_requested = []
        
        for match in matches:
            package_str = match.group(1).strip()
            
            # Handle quotes
            if package_str.startswith('"') and package_str.endswith('"'):
                package_str = package_str[1:-1]
            elif package_str.startswith("'") and package_str.endswith("'"):
                package_str = package_str[1:-1]
            
            # Extract package list
            packages = [pkg.strip() for pkg in package_str.split() if pkg.strip()]
            packages_requested.extend(packages)
        
        # Just log package requests without auto-installing
        if packages_requested:
            print(f"Package installation request detected in code: {', '.join(packages_requested)}")
            if hasattr(self, 'console_output'):
                self.console_output.append(f"\nPackage installation request detected in code: {', '.join(packages_requested)}")
                self.console_output.append("To install packages, use the 'pip install package_name' command.")
        
        # Don't remove pip install commands from code, let them execute naturally
        return False
    
    def run_code(self, agent_name=None, return_output=False):
        """Run the code in the editor with safeguards for memory and timeouts"""
        code = self.code_editor.toPlainText()
        if not code.strip():
            return "" if return_output else None
        
        # Parse the code for pip install commands
        self.check_for_pip_install(code)
        
        # Add matplotlib backend code to the beginning of the script
        backend_code = """
# Force matplotlib to use Agg backend
import os
os.environ['MPLBACKEND'] = 'Agg'
import matplotlib
matplotlib.use('Agg')
try:
    # Handle potential GTK issues
    import matplotlib.pyplot as plt
    plt.ioff()  # Turn off interactive mode
except ImportError:
    pass
"""
        
        # Check if this is a matplotlib-using script
        if "import matplotlib" in code or "from matplotlib" in code:
            code = backend_code + code
        
        # Switch to console tab
        self.setCurrentIndex(1)
        
        # Clear previous output
        if hasattr(self, 'console_output'):
            self.console_output.clear()
            
            # Log who is running the code
            if agent_name:
                self.console_output.append(f"[{datetime.now().strftime('%H:%M:%S')}] {agent_name} is running code...\n")
            else:
                self.console_output.append(f"[{datetime.now().strftime('%H:%M:%S')}] Running code...\n")
        
        # Save code to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.py', mode='w') as f:
            f.write(code)
            temp_file = f.name
        
        # Configure environment variables
        env = os.environ.copy()
        
        # Force non-interactive matplotlib backend
        env['MPLBACKEND'] = 'Agg'
        
        # Add the packages directory to PYTHONPATH
        if "PYTHONPATH" in env:
            env["PYTHONPATH"] = f"{self.packages_dir}{os.pathsep}{env['PYTHONPATH']}"
        else:
            env["PYTHONPATH"] = self.packages_dir
        
        # Get path to Python in virtual environment
        if sys.platform == "win32":
            python_path = os.path.join(self.venv_dir, "Scripts", "python.exe")
        else:
            python_path = os.path.join(self.venv_dir, "bin", "python")
        
        if not os.path.exists(python_path):
            python_path = sys.executable  # Fallback to system Python
        
        # Run the code using subprocess instead of QProcess
        execution_output = ""
        execution_failed = False
        try:
            # Show activity indicator
            self.show_ai_activity(True, f"Python code is running...")
            
            # Set timeout alarm for 60 seconds
            signal.alarm(60)
            
            # Add wrapper script to limit memory usage
            memory_wrapper = f"""
import resource
import sys
import traceback

# Set resource limits for this process
# 2GB memory limit
resource.setrlimit(resource.RLIMIT_AS, (2 * 1024 * 1024 * 1024, -1))

try:
    exec(open('{temp_file}').read())
except MemoryError:
    print("ERROR: Memory limit exceeded. The code tried to use too much memory.")
    sys.exit(1)
except Exception as e:
    print(f"Error during execution: {{e}}")
    print(f"Main error: {{e}}")
    traceback.print_exc()
    sys.exit(1)
"""
            
            # Save the wrapper to a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.py', mode='w') as wrapper_file:
                wrapper_file.write(memory_wrapper)
                wrapper_path = wrapper_file.name
            
            # Use subprocess instead of QProcess
            process = subprocess.Popen(
                [python_path, "-c", memory_wrapper],
                cwd=self.session_dir,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1  # Line buffered
            )
            
            # Read output with timeout
            try:
                stdout, stderr = process.communicate(timeout=60)
                
                # Cancel the alarm
                signal.alarm(0)
                
                # Capture all output for returning
                if stdout:
                    execution_output += stdout
                if stderr:
                    execution_output += f"ERROR: {stderr}"
                    execution_failed = True
                
                # Display output
                if stdout:
                    self.console_output.append(stdout)
                    
                    # Check for visualization paths
                    self.check_for_visualization_paths(stdout)
                
                if stderr:
                    self.console_output.append(f"<span style='color: red;'>{stderr}</span>")
                    execution_failed = True
                    
            except subprocess.TimeoutExpired:
                # Process took too long, kill it
                process.kill()
                stdout, stderr = process.communicate()
                error_msg = "Process timed out after 60 seconds"
                self.console_output.append(f"<span style='color: red;'>{error_msg}</span>")
                execution_output += f"\nERROR: {error_msg}"
                execution_failed = True
                
                # Create a detailed message for the AI
                if hasattr(self, 'workflow') and self.workflow:
                    timeout_msg = AIMessage("System", f"""
                    ⚠️ EXECUTION TIMEOUT ⚠️
                    
                    The Python code took too long to execute and was terminated after 60 seconds.
                    This could be due to:
                    
                    1. An infinite loop or very slow algorithm
                    2. A resource-intensive operation that needs optimization
                    3. A visualization or animation that's too complex
                    
                    Please revise the code to be more efficient or add progress tracking.
                    """)
                    with self.workflow.workflow_lock:
                        self.workflow.messages.append(timeout_msg)
                    if hasattr(self, 'message_handler') and self.message_handler:
                        self.message_handler(timeout_msg)
            
            # Clean up temporary files
            try:
                os.unlink(temp_file)
                os.unlink(wrapper_path)
            except:
                pass
            
            # Hide activity indicator
            self.show_ai_activity(False)
            
            # Save execution status
            self.last_execution_failed = execution_failed
            
            if process.returncode == 0:
                self.console_output.append("\nCode execution completed successfully.")
                execution_output += "\nCode execution completed successfully."
            else:
                self.console_output.append(f"\nCode execution failed with exit code {process.returncode}.")
                execution_output += f"\nCode execution failed with exit code {process.returncode}."
                execution_failed = True
                self.last_execution_failed = True
                    
        except TimeoutError:
            error_msg = "Operation timed out after 60 seconds"
            print(error_msg)
            if hasattr(self, 'console_output'):
                self.console_output.append(f"<span style='color: red;'>{error_msg}</span>")
                self.show_ai_activity(False)
            execution_output += f"\nERROR: {error_msg}"
            execution_failed = True
            self.last_execution_failed = True
            
            # Cancel the alarm
            signal.alarm(0)
        except Exception as e:
            error_msg = f"Error running code: {str(e)}"
            print(error_msg)
            if hasattr(self, 'console_output'):
                self.console_output.append(error_msg)
                self.show_ai_activity(False)
            execution_output += f"\nERROR: {error_msg}"
            execution_failed = True
            self.last_execution_failed = True
            
            # Cancel the alarm
            signal.alarm(0)
        
        # Return the execution output if requested
        if return_output:
            return execution_output
    
    def check_for_visualization_paths(self, output):
        """Check output for visualization file paths"""
        file_path_patterns = [
            # General visualization markers
            (r"saved visualization to:\s*(.+)", "visualization"),
            (r"visualization saved to:\s*(.+)", "visualization"),
            (r"figure saved to:\s*(.+)", "visualization"),
            (r"plot saved to:\s*(.+)", "visualization"),
            (r"saved figure to:\s*(.+)", "visualization"),
            (r"saved plot to:\s*(.+)", "visualization"),
            (r"image saved to:\s*(.+)", "image"),
            (r"saved image to:\s*(.+)", "image"),
            
            # File types
            (r"saved (html|svg|pdf|png|jpg|gif) to:\s*(.+)", "file"),
            (r"(html|svg|pdf|png|jpg|gif) saved to:\s*(.+)", "file"),
            (r"output file saved to:\s*(.+)", "file"),
            (r"saved output file to:\s*(.+)", "file"),
            (r"file saved to:\s*(.+)", "file"),
            (r"saved file to:\s*(.+)", "file"),
            
            # Generated content
            (r"generated (html|svg|pdf|png|jpg|gif) at:\s*(.+)", "generated"),
            (r"created (html|svg|pdf|png|jpg|gif) at:\s*(.+)", "generated"),
            
            # Media specific
            (r"video saved to:\s*(.+)", "video"),
            (r"saved video to:\s*(.+)", "video"),
            (r"audio saved to:\s*(.+)", "audio"),
            (r"saved audio to:\s*(.+)", "audio"),
            
            # 3D content
            (r"3d model saved to:\s*(.+)", "3d"),
            (r"saved 3d model to:\s*(.+)", "3d"),
            
            # Web content
            (r"webpage saved to:\s*(.+)", "web"),
            (r"saved webpage to:\s*(.+)", "web"),
            (r"html file saved to:\s*(.+)", "web"),
            (r"saved html file to:\s*(.+)", "web"),
            
            # Markers for files opened in browser
            (r"opening\s+(.+)\s+in browser", "browser"),
            (r"opened\s+(.+)\s+in browser", "browser")
        ]
        
        # Check for any file path patterns
        for pattern, file_type in file_path_patterns:
            matches = re.finditer(pattern, output, re.IGNORECASE)
            for match in matches:
                if file_type == "file" or file_type == "generated":
                    # These patterns capture the file type and path
                    viz_path = match.group(2).strip() if match.lastindex > 1 else match.group(1).strip()
                else:
                    # These patterns capture just the path
                    viz_path = match.group(1).strip()
                
                # Clean up the path (remove quotes, etc.)
                viz_path = viz_path.strip("'\"")
                
                # Try to display the visualization if the file exists
                if os.path.exists(viz_path):
                    self.show_visualization(viz_path)
                elif os.path.exists(os.path.join(self.session_dir, viz_path)):
                    # Try with session directory
                    self.show_visualization(os.path.join(self.session_dir, viz_path))
    
    def save_code(self):
        """Save the code to a file"""
        code = self.code_editor.toPlainText()
        if not code.strip():
            return
        
        # Automatically save without dialog
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path = os.path.join(self.session_dir, f"code_{timestamp}.py")
        
        with open(file_path, 'w') as f:
            f.write(code)
        
        self.console_output.append(f"Code saved to {os.path.basename(file_path)}")
        self.setCurrentIndex(1)  # Switch to console tab
    
    def run_command(self):
        """Run a command entered in the input field"""
        command = self.command_input.text().strip()
        if not command:
            return
        
        self.command_input.clear()
        
        # Handle pip install commands
        if command.startswith("pip install"):
            packages = command.replace("pip install", "").strip()
            self.setCurrentIndex(1)  # Switch to console tab
            self.console_output.append(f"\n>>> {command}\n")
            
            # Start package installation process
            self.console_output.append(f"Installing packages: {packages}\n")
            
            # Get pip path
            if sys.platform == "win32":
                pip_path = os.path.join(self.venv_dir, "Scripts", "pip.exe")
            else:
                pip_path = os.path.join(self.venv_dir, "bin", "pip")
            
            if not os.path.exists(pip_path):
                pip_path = [sys.executable, "-m", "pip"]  # Fallback to system pip
            else:
                pip_path = [pip_path]
            
            # Install packages
            try:
                # Configure environment
                env = os.environ.copy()
                if "PYTHONPATH" in env:
                    env["PYTHONPATH"] = f"{self.packages_dir}{os.pathsep}{env['PYTHONPATH']}"
                else:
                    env["PYTHONPATH"] = self.packages_dir
                
                # Run installation command
                cmd = pip_path + ["install", packages, "--target", self.packages_dir]
                process = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    cwd=self.session_dir,
                    env=env
                )
                
                # Show output
                if process.stdout:
                    self.console_output.append(process.stdout)
                
                if process.stderr:
                    self.console_output.append(f"<span style='color: red;'>{process.stderr}</span>")
                
                # Create installation report
                if process.returncode == 0:
                    success_msg = f"✅ Package '{packages}' installed successfully."
                    self.console_output.append(success_msg)
                    
                    # Send success message to AI
                    if hasattr(self, 'workflow') and self.workflow:
                        msg = AIMessage("System", success_msg)
                        with self.workflow.workflow_lock:
                            self.workflow.messages.append(msg)
                        # Also display in UI
                        if hasattr(self, 'message_handler') and self.message_handler:
                            self.message_handler(msg)
                else:
                    error_msg = f"⚠️ Failed to install package '{packages}'"
                    self.console_output.append(error_msg)
                    
                    # Send failure message to AI
                    if hasattr(self, 'workflow') and self.workflow:
                        msg = AIMessage("System", f"{error_msg}\n\n{process.stderr}")
                        with self.workflow.workflow_lock:
                            self.workflow.messages.append(msg)
                        # Also display in UI
                        if hasattr(self, 'message_handler') and self.message_handler:
                            self.message_handler(msg)
                    
            except Exception as e:
                error_msg = f"Error installing package: {str(e)}"
                self.console_output.append(error_msg)
                
                # Send error message to AI
                if hasattr(self, 'workflow') and self.workflow:
                    msg = AIMessage("System", error_msg)
                    with self.workflow.workflow_lock:
                        self.workflow.messages.append(msg)
                    # Also display in UI
                    if hasattr(self, 'message_handler') and self.message_handler:
                        self.message_handler(msg)
                
            return
        
        # Handle Python commands (non-pip install)
        self.setCurrentIndex(1)  # Switch to console tab
        self.console_output.append(f"\n>>> {command}\n")
        
        # Get Python path
        if sys.platform == "win32":
            python_path = os.path.join(self.venv_dir, "Scripts", "python.exe")
        else:
            python_path = os.path.join(self.venv_dir, "bin", "python")
        
        if not os.path.exists(python_path):
            python_path = sys.executable  # Fallback to system Python
        
        # Run the command
        try:
            # Create environment
            env = os.environ.copy()
            if "PYTHONPATH" in env:
                env["PYTHONPATH"] = f"{self.packages_dir}{os.pathsep}{env['PYTHONPATH']}"
            else:
                env["PYTHONPATH"] = self.packages_dir
            
            # Add wrapper to prevent command from running too long
            wrapper_code = f"""
import resource
import sys
import signal

# Set resource limits for this process
# 1GB memory limit
resource.setrlimit(resource.RLIMIT_AS, (1 * 1024 * 1024 * 1024, -1))

# Set timeout
def timeout_handler(signum, frame):
    print("Command execution timed out after 30 seconds")
    sys.exit(1)
    
signal.signal(signal.SIGALRM, timeout_handler)
signal.alarm(30)

# Execute the command
{command}

# Cancel the alarm
signal.alarm(0)
"""
            # Execute command with wrapper
            process = subprocess.run(
                [python_path, "-c", wrapper_code],
                capture_output=True,
                text=True,
                cwd=self.session_dir,
                env=env,
                timeout=35  # Give a bit extra time beyond the signal timeout
            )
            
            # Show output
            if process.stdout:
                self.console_output.append(process.stdout)
            
            if process.stderr:
                self.console_output.append(f"<span style='color: red;'>{process.stderr}</span>")
            # Send message to AI if module error
                if "ModuleNotFoundError" in process.stderr or "ImportError" in process.stderr:
                    error_detail = f"Module error occurred during command execution:\n\n{process.stderr}\n\nPlease install the required packages."
                    if hasattr(self, 'workflow') and self.workflow:
                        msg = AIMessage("System", error_detail)
                        with self.workflow.workflow_lock:
                            self.workflow.messages.append(msg)
                        # Also display in UI
                        if hasattr(self, 'message_handler') and self.message_handler:
                            self.message_handler(msg)
            
            if process.returncode == 0 and not process.stdout and not process.stderr:
                self.console_output.append("Command executed successfully with no output.")
                
        except subprocess.TimeoutExpired:
            self.console_output.append("<span style='color: red;'>Command execution timed out after 35 seconds</span>")
        except Exception as e:
            self.console_output.append(f"Error executing command: {str(e)}")
    
    def show_pip_install_dialog(self):
        """Show dialog to install Python packages"""
        packages, ok = QInputDialog.getText(
            self,
            "Install Python Packages",
            "Enter package names (space separated):"
        )
        
        if ok and packages:
            self.install_packages([pkg.strip() for pkg in packages.split() if pkg.strip()])
    
    def install_packages(self, package_list):
        """Install Python packages"""
        if not package_list:
            return
        
        # If console_output doesn't exist yet, just print to console
        if not hasattr(self, 'console_output'):
            print(f"Installing packages: {', '.join(package_list)}")
            return
        
        self.setCurrentIndex(1)  # Switch to console tab
        self.console_output.append(f"\nInstalling packages: {', '.join(package_list)}\n")
        
        # Show activity indicator
        self.show_ai_activity(True, "Installing Python packages...")
        
        # Get pip path
        if sys.platform == "win32":
            pip_path = os.path.join(self.venv_dir, "Scripts", "pip.exe")
        else:
            pip_path = os.path.join(self.venv_dir, "bin", "pip")
        
        if not os.path.exists(pip_path):
            pip_path = [sys.executable, "-m", "pip"]  # Fallback to system pip
        else:
            pip_path = [pip_path]
        
        # Install packages one by one to ensure we get as many installed as possible
        for package in package_list:
            try:
                # Add info to console
                self.console_output.append(f"Installing {package}...")
                
                # Use minimal install options to reduce memory usage
                cmd = pip_path + ["install", package, "--target", self.packages_dir]
                
                # Use Popen and communicate to manage memory better
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )
                
                # Get minimal output 
                stdout, stderr = process.communicate()
                
                if process.returncode == 0:
                    self.installed_packages.add(package)
                    self.console_output.append(f"Successfully installed {package}")
                    
                    # Notify AI team of successful installation
                    if hasattr(self, 'workflow') and self.workflow:
                        install_msg = AIMessage("System", f"✅ Package '{package}' was successfully installed and is now available for use.")
                        with self.workflow.workflow_lock:
                            self.workflow.messages.append(install_msg)
                        # Also send to UI
                        if hasattr(self, 'message_handler') and self.message_handler:
                            self.message_handler(install_msg)
                else:
                    self.console_output.append(f"<span style='color: red;'>Failed to install {package}</span>")
                    # Notify AI team of installation failure
                    if hasattr(self, 'workflow') and self.workflow:
                        error_msg = AIMessage("System", f"⚠️ Package '{package}' installation failed. Error: {stderr}")
                        with self.workflow.workflow_lock:
                            self.workflow.messages.append(error_msg)
                        # Also send to UI
                        if hasattr(self, 'message_handler') and self.message_handler:
                            self.message_handler(error_msg)
                
            except Exception as e:
                self.console_output.append(f"Error installing {package}: {str(e)}")
                # Notify AI team of installation error
                if hasattr(self, 'workflow') and self.workflow:
                    error_msg = AIMessage("System", f"⚠️ Error installing package '{package}': {str(e)}")
                    with self.workflow.workflow_lock:
                        self.workflow.messages.append(error_msg)
                    # Also send to UI
                    if hasattr(self, 'message_handler') and self.message_handler:
                        self.message_handler(error_msg)
        
        # Hide activity indicator
        self.show_ai_activity(False)
    
    def list_installed_packages(self):
        """List installed packages"""
        self.setCurrentIndex(1)  # Switch to console tab
        self.console_output.append("\nInstalled packages:")
        
        # Get Python path
        if sys.platform == "win32":
            python_path = os.path.join(self.venv_dir, "Scripts", "python.exe")
        else:
            python_path = os.path.join(self.venv_dir, "bin", "python")
        
        if not os.path.exists(python_path):
            python_path = sys.executable  # Fallback to system Python
        
        # List installed packages
        try:
            process = subprocess.run(
                [python_path, "-m", "pip", "list"],
                capture_output=True,
                text=True,
                cwd=self.session_dir,
                env=os.environ.copy()
            )
            
            if process.stdout:
                self.console_output.append(process.stdout)
            
            if process.stderr:
                self.console_output.append(f"<span style='color: red;'>{process.stderr}</span>")
                
        except Exception as e:
            self.console_output.append(f"Error listing packages: {str(e)}")
    
    def show_visualization(self, file_path):
        """Show a visualization from a file"""
        if not os.path.exists(file_path):
            return
        
        # Switch to visualization tab
        self.setCurrentIndex(2)
        
        # Check file type
        file_ext = os.path.splitext(file_path)[1].lower()
        
        # Log visualization display
        self.console_output.append(f"\nDisplaying visualization: {os.path.basename(file_path)}")
        
        if file_ext in ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tif', '.tiff']:
            # Show image
            pixmap = QPixmap(file_path)
            self.visualization_area.setPixmap(pixmap.scaled(
                self.visualization_area.width(), 
                self.visualization_area.height(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            ))
            self.visualization_area.setAlignment(Qt.AlignCenter)
            self.visualization_area.show()
        else:
            # For other files, show the path and open in system viewer
            self.visualization_area.setText(f"File saved to: {file_path}\nAttempting to open with default application...")
            try:
                if sys.platform == 'win32':
                    os.startfile(file_path)
                elif sys.platform == 'darwin':  # macOS
                    subprocess.run(['open', file_path])
                else:  # Linux
                    subprocess.run(['xdg-open', file_path])
            except Exception as e:
                self.console_output.append(f"Error opening file: {str(e)}")
    
    def show_ai_activity(self, is_active, message=None):
        """Show or hide the AI activity indicator"""
        if hasattr(self, 'ai_activity_indicator'):
            if is_active:
                self.ai_activity_indicator.setText(message or "AI is working with Python...")
                self.ai_activity_indicator.setVisible(True)
            else:
                self.ai_activity_indicator.setVisible(False)

#################################
# Chat UI Components
#################################

class EnhancedChatPanel(QWidget):
    """Enhanced chat panel with rich text display and better UI"""
    # Signal for thread-safe text updates
    text_update_signal = pyqtSignal(object)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Set up main layout
        layout = QVBoxLayout(self)
        
        # Create chat display
        self.chat_display = QTextEdit()
        self.chat_display.setReadOnly(True)
        self.chat_display.setStyleSheet("""
            QTextEdit {
                background-color: #f5f5f5;
                border: 1px solid #ddd;
                border-radius: 4px;
                padding: 8px;
                font-family: Arial, sans-serif;
                font-size: 14px;
            }
        """)
        
        # Make chat display accept rich text
        self.chat_display.setAcceptRichText(True)
        
        # Connect signal to slot for thread-safe updates
        self.text_update_signal.connect(self._update_chat_display)
        
        # Input area with send button
        input_area = QWidget()
        input_layout = QHBoxLayout(input_area)
        input_layout.setContentsMargins(0, 0, 0, 0)
        
        self.chat_input = QLineEdit()
        self.chat_input.setPlaceholderText("Enter any task and let the AI team solve it using Python...")
        self.chat_input.setStyleSheet("""
            QLineEdit {
                border: 1px solid #bbb;
                border-radius: 4px;
                padding: 8px;
                font-size: 14px;
                min-height: 24px;
            }
        """)
        
        self.send_button = QPushButton("Send")
        self.send_button.setStyleSheet("""
            QPushButton {
                background-color: #2980b9;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 8px 16px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #3498db;
            }
            QPushButton:pressed {
                background-color: #1c638e;
            }
        """)
        
        # Connect signals
        self.chat_input.returnPressed.connect(self.send_button.click)
        
        input_layout.addWidget(self.chat_input)
        input_layout.addWidget(self.send_button)
        
        # Add a typing indicator
        self.typing_indicator = QLabel("")
        self.typing_indicator.setStyleSheet("color: #7f8c8d; font-style: italic;")
        self.typing_indicator.setVisible(False)
        
        # Add all components to main layout
        layout.addWidget(self.chat_display)
        layout.addWidget(self.typing_indicator)
        layout.addWidget(input_area)
    
    def add_message(self, message):
        """Add a message to the chat display with rich formatting"""
        # Use signal to update text from any thread
        self.text_update_signal.emit(message)
        
    def _update_chat_display(self, message):
        """Slot method to safely update chat display from main thread"""
        # Add formatted message to display
        self.chat_display.append(message.get_html_format())
        
        # Auto-scroll to bottom - avoid using QTextCursor
        self.chat_display.verticalScrollBar().setValue(
            self.chat_display.verticalScrollBar().maximum()
        )
    
    def show_typing_indicator(self, agent_name=None):
        """Show typing indicator when an AI is 'thinking'"""
        if agent_name:
            self.typing_indicator.setText(f"{agent_name} is thinking...")
        else:
            self.typing_indicator.setText("AI is thinking...")
        self.typing_indicator.setVisible(True)
    
    def hide_typing_indicator(self):
        """Hide the typing indicator"""
        self.typing_indicator.setVisible(False)
    
    def clear_chat(self):
        """Clear the chat display"""
        self.chat_display.clear()

#################################
# Main Application Window
#################################

class ChatWindow(QMainWindow):
    """Main application window with state tracking for AIs"""
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Advanced AI Python Collaboration Platform")
        self.resize(1400, 800)
        
        # Global settings directory for saving API keys and other settings
        self.settings_dir = os.path.join(os.path.expanduser("~"), ".ai_collaboration")
        if not os.path.exists(self.settings_dir):
            os.makedirs(self.settings_dir)
        
        # Load saved API keys
        self.saved_api_keys = self.load_api_keys()
        
        # Create a unique session ID and directory for this chat
        self.session_id = str(uuid.uuid4())
        self.session_dir = os.path.join(os.path.expanduser("~"), "ai_collaboration", self.session_id)
        os.makedirs(self.session_dir, exist_ok=True)
        
        # Log basic session info
        with open(os.path.join(self.session_dir, "session_info.txt"), "w") as f:
            f.write(f"Session ID: {self.session_id}\n")
            f.write(f"Created: {datetime.now().isoformat()}\n")
            f.write(f"Directory: {self.session_dir}\n")
        
        # Set up the UI
        self.setup_ui()
        
        # Initialize agents right away
        self.agents = {}
        self.workflow = None
        
        # Update status
        self.status_bar.showMessage("Initializing AI agents...")
        
        # Initialize AI agents immediately
        self.init_minimal_agents()
        
        # Initialize UI and show welcome message
        self.delayed_init()
    
    def delayed_init(self):
        """Initialize only what's necessary to start the app"""
        # Show welcome message
        welcome_message = AIMessage(
            "System", 
            """
            Welcome to the Advanced AI Python Collaboration Platform!
            
            This platform features a team of AI models that will work together to solve tasks using Python.
            Each AI can operate in three states:
            
            🤔 Thinking State - Analyzing problems and planning approaches
            💬 Discussion State - Exchanging ideas and critiquing approaches
            🔨 Execution State - Implementing solutions in code
            
            All AI agents have been initialized and are ready to use.
            
            To get started, simply describe what you'd like to accomplish.
            """
        )
        self.chat_panel.add_message(welcome_message)
        
        # 에이전트가 초기화되면 모델 이름 업데이트
        self.update_model_names_if_available()
        
        # Show API setup dialog if no saved keys
        if not any(self.saved_api_keys.values()):
            self.show_api_setup()
        else:
            self.status_bar.showMessage("Ready - API keys available")
            
    def init_minimal_agents(self):
        """Initialize all agents with available API keys"""
        # Initialize all agents if they don't exist yet
        if not self.agents:
            self.agents = {
                "ChatGPT": ChatGPTAgent(),
                "Claude": ClaudeAgent(),
                "Gemini": GeminiAgent(),
                "DeepSeek": DeepSeekAgent(),
                "Perplexity": PerplexityAgent(),
                "Qwen": QwenAgent()
            }
            
            # Create workflow
            self.workflow = AIWorkflow(
                self.session_dir, 
                self.handle_ai_message,
                self.python_env.execute_python_code
            )
            
            # Connect Python environment to workflow
            self.python_env.workflow = self.workflow
            self.python_env.message_handler = self.handle_ai_message
            
            # 기본 모델 설정 - 각 에이전트의 첫 번째 모델 선택
            self.initialize_default_models()
            
            # Connect workflow signals for state changes
            self.workflow.ai_status_changed.connect(self.update_ai_thinking_status)
            self.workflow.ai_state_changed.connect(self.update_ai_state_indicator)
            
            # Configure with saved keys if available
            setup_results = []
            
            if self.saved_api_keys.get("openai"):
                setup_results.append(("ChatGPT", self.agents["ChatGPT"].setup(self.saved_api_keys["openai"])))
                
            if self.saved_api_keys.get("anthropic"):
                setup_results.append(("Claude", self.agents["Claude"].setup(self.saved_api_keys["anthropic"])))
                
            if self.saved_api_keys.get("gemini"):
                setup_results.append(("Gemini", self.agents["Gemini"].setup(self.saved_api_keys["gemini"])))
                
            if self.saved_api_keys.get("deepseek"):
                setup_results.append(("DeepSeek", self.agents["DeepSeek"].setup(self.saved_api_keys["deepseek"])))
                
            if self.saved_api_keys.get("perplexity"):
                setup_results.append(("Perplexity", self.agents["Perplexity"].setup(self.saved_api_keys["perplexity"])))
                
            if self.saved_api_keys.get("qwen"):
                setup_results.append(("Qwen", self.agents["Qwen"].setup(self.saved_api_keys["qwen"])))
                
            # Add agents to workflow
            for name, agent in self.agents.items():
                self.workflow.add_agent(name, agent)
                
            # Update indicators
            for name, success in setup_results:
                self.update_ai_indicator(name, success)
                
            # Show status message
            self.chat_panel.add_message(AIMessage("System", 
                f"Initialized {len(setup_results)} AI agents: " + 
                ", ".join([name for name, success in setup_results if success])))
                
            connected_count = sum(success for _, success in setup_results)
            self.status_bar.showMessage(f"Connected {connected_count}/{len(setup_results)} AI models")
    
    def load_api_keys(self):
        """Load saved API keys from settings file"""
        api_keys_path = os.path.join(self.settings_dir, "api_keys.json")
        
        if os.path.exists(api_keys_path):
            try:
                with open(api_keys_path, 'r') as f:
                    data = json.load(f)
                
                # Simple decoding of stored keys
                keys = {}
                for key, value in data.items():
                    if value:  # If there's a value
                        try:
                            decoded = base64.b64decode(value).decode('utf-8')
                            keys[key] = decoded
                        except:
                            # If decoding fails, use the original value
                            keys[key] = value
                    else:
                        keys[key] = ""
                
                return keys
            except Exception as e:
                print(f"Error loading API keys: {e}")
        
        # Return empty keys if file doesn't exist or there was an error
        return {"openai": "", "anthropic": "", "gemini": "", "deepseek": ""}
    
    def save_api_keys(self, keys):
        """Save API keys to settings file"""
        api_keys_path = os.path.join(self.settings_dir, "api_keys.json")
        
        # Simple encoding of keys
        encoded_keys = {}
        for key, value in keys.items():
            if key == "save_keys":  # Skip the checkbox value
                continue
            if value:  # Only encode if there's a value
                encoded_keys[key] = base64.b64encode(value.encode('utf-8')).decode('utf-8')
            else:
                encoded_keys[key] = ""
        
        try:
            with open(api_keys_path, 'w') as f:
                json.dump(encoded_keys, f)
            
            # Update saved keys
            self.saved_api_keys = keys
            
            return True
        except Exception as e:
            print(f"Error saving API keys: {e}")
            return False
    
    def setup_ui(self):
        """Set up the user interface with side-by-side layout"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Create a horizontal layout for side-by-side panels
        main_layout = QHBoxLayout(central_widget)
        
        # Create a splitter for resizable panels
        self.splitter = QSplitter(Qt.Horizontal)
        
        # Left panel: Python Environment
        self.python_env = PythonEnvironment(self.session_dir)
        
        # Right panel: Chat
        self.chat_panel = EnhancedChatPanel()
        self.chat_panel.send_button.clicked.connect(self.send_message)
        
        # Add panels to splitter
        self.splitter.addWidget(self.python_env)
        self.splitter.addWidget(self.chat_panel)
        
        # Set initial sizes (60% Python, 40% chat)
        self.splitter.setSizes([int(self.width() * 0.6), int(self.width() * 0.4)])
        
        main_layout.addWidget(self.splitter)
        
        # Create a toolbar with essential actions
        self.create_toolbar()
        
        # Add status bar
        self.status_bar = self.statusBar()
        
        # Add AI thinking indicator to status bar
        self.ai_thinking_indicator = QLabel("No AI thinking")
        self.ai_thinking_indicator.setStyleSheet("color: #7f8c8d; padding: 0 10px;")
        self.status_bar.addPermanentWidget(self.ai_thinking_indicator)
        
        # Add AI state indicators to status bar
        self.ai_state_indicators = {}
        self.ai_state_label = QLabel("AI States:")
        self.status_bar.addPermanentWidget(self.ai_state_label)
        
        # Add state indicators for each AI
        for name in ["ChatGPT", "Claude", "Gemini", "DeepSeek", "Perplexity", "Qwen"]:
            state_indicator = QLabel(f"{name}: -")
            state_indicator.setStyleSheet("color: #7f8c8d; padding: 0 10px;")
            self.ai_state_indicators[name] = state_indicator
            self.status_bar.addPermanentWidget(state_indicator)
        
        self.status_bar.showMessage("Ready")
    
    def create_toolbar(self):
        """Create a toolbar with essential actions"""
        toolbar = self.addToolBar("Main Actions")
        toolbar.setMovable(False)
        
        # API Setup action
        api_setup_action = toolbar.addAction("API Setup")
        api_setup_action.triggered.connect(self.show_api_setup)
        
        # New Session action
        new_session_action = toolbar.addAction("New Session")
        new_session_action.triggered.connect(self.create_new_session)
        
        # Open Session Folder action
        open_folder_action = toolbar.addAction("Open Session Folder")
        open_folder_action.triggered.connect(self.open_session_folder)
        
        # Export Chat action
        export_action = toolbar.addAction("Export Chat")
        export_action.triggered.connect(self.export_chat)
        
        # Add spacer to push next items to the right
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        toolbar.addWidget(spacer)
        
        # AI Team label
        team_label = QLabel("AI Team: ")
        toolbar.addWidget(team_label)
        
        # Add AI indicator widgets
        self.ai_indicators = {}
        agent_type_map = {
            "ChatGPT": "openai",
            "Claude": "anthropic",
            "Gemini": "google",
            "DeepSeek": "deepseek",
            "Perplexity": "perplexity",
            "Qwen": "qwen"
        }
        
        for name in agent_type_map.keys():
            # 초기에는 에이전트 이름으로 표시
            indicator = ClickableLabel(name)
            indicator.setStyleSheet("color: #999; font-weight: bold; margin: 0 5px;")
            indicator.setCursor(Qt.PointingHandCursor)  # 마우스 커서를 손가락 모양으로 변경
            # 클릭 이벤트 연결
            indicator.clicked.connect(lambda checked=False, n=name: self.show_model_selection(n))
            self.ai_indicators[name] = indicator
            toolbar.addWidget(indicator)
    
    def show_model_selection(self, agent_name):
        """AI 모델 선택 대화 상자 표시"""
        # 현재 선택된 모델 가져오기
        current_model = None
        if agent_name in self.agents:
            current_model = self.agents[agent_name].selected_model
        
        # 모델 선택 대화 상자 표시
        dialog = AIModelSelectionDialog(self, agent_name, current_model, AI_MODELS)
        if dialog.exec_():
            selected_model = dialog.get_selected_model()
            if selected_model:
                # 선택된 모델을 에이전트에 설정
                if agent_name in self.agents:
                    self.agents[agent_name].select_model(selected_model)
                    
                    # 인디케이터 텍스트 업데이트
                    self.update_ai_model_name(agent_name, selected_model)
                    
                    # 상태 메시지 표시
                    self.status_bar.showMessage(f"{agent_name} 모델이 {selected_model}(으)로 변경되었습니다.", 3000)
                    
                    # 시스템 메시지로 모델 변경 알림
                    system_message = AIMessage("System", f"{agent_name} 모델이 {selected_model}(으)로 변경되었습니다.")
                    self.chat_panel.add_message(system_message)
    
    def update_ai_model_name(self, agent_name, model_name):
        """Update the model name displayed in the AI indicator"""
        if agent_name in self.ai_indicators:
            # 모델 이름을 표시하도록 텍스트 업데이트
            self.ai_indicators[agent_name].setText(model_name)
            
    def update_model_names_if_available(self):
        """에이전트가 초기화된 후 모델 이름 업데이트"""
        if hasattr(self, 'agents') and self.agents:
            for agent_name, agent in self.agents.items():
                if hasattr(agent, 'selected_model') and agent.selected_model:
                    self.update_ai_model_name(agent_name, agent.selected_model)
    
    def update_ai_indicator(self, name, status):
        """Update the status indicator for an AI agent"""
        if name in self.ai_indicators:
            if status:
                self.ai_indicators[name].setStyleSheet("color: #27ae60; font-weight: bold; margin: 0 5px;")
                
                # 모델 이름 업데이트 (연결 성공 시)
                if name in self.agents and hasattr(self.agents[name], 'selected_model') and self.agents[name].selected_model:
                    self.update_ai_model_name(name, self.agents[name].selected_model)
            else:
                self.ai_indicators[name].setStyleSheet("color: #999; font-weight: bold; margin: 0 5px;")
    
    def initialize_default_models(self):
        """각 AI 에이전트에 기본 모델 설정"""
        # AI 모델 목록에서 각 에이전트 유형별 첫 번째 모델을 기본값으로 설정
        agent_type_map = {
            "ChatGPT": "openai",
            "Claude": "anthropic",
            "Gemini": "google",
            "DeepSeek": "deepseek",
            "Perplexity": "perplexity",
            "Qwen": "qwen"
        }
        
        for agent_name, agent_type in agent_type_map.items():
            if agent_name in self.agents and agent_type in AI_MODELS and AI_MODELS[agent_type]:
                # 해당 유형의 첫 번째 모델 선택
                default_model = AI_MODELS[agent_type][0]["name"]
                self.agents[agent_name].select_model(default_model)
                
                # 인디케이터 텍스트 업데이트 - 즉시 UI에 반영
                if agent_name in self.ai_indicators:
                    self.ai_indicators[agent_name].setText(default_model)
    
    def update_ai_state_indicator(self, agent_name, state):
        """Update the state indicator for an AI agent"""
        if agent_name in self.ai_state_indicators:
            # Set color and icon based on state
            if state == "thinking":
                self.ai_state_indicators[agent_name].setText(f"{agent_name}: 🤔 Thinking")
                self.ai_state_indicators[agent_name].setStyleSheet("color: #3498db; font-weight: bold; padding: 0 10px;")
                
                # AI가 thinking 상태로 변경되면 ai_thinking_indicator도 업데이트
                self.update_ai_thinking_status(agent_name, True)
            elif state == "discussing":
                self.ai_state_indicators[agent_name].setText(f"{agent_name}: 💬 Discussing")
                self.ai_state_indicators[agent_name].setStyleSheet("color: #9b59b6; font-weight: bold; padding: 0 10px;")
                
                # AI가 discussing 상태로 변경되면 thinking 상태 해제
                self.update_ai_thinking_status(agent_name, False)
            elif state == "executing":
                self.ai_state_indicators[agent_name].setText(f"{agent_name}: 🔨 Executing")
                self.ai_state_indicators[agent_name].setStyleSheet("color: #e74c3c; font-weight: bold; padding: 0 10px;")
                
                # AI가 executing 상태로 변경되면 thinking 상태 해제
                self.update_ai_thinking_status(agent_name, False)
            else:
                self.ai_state_indicators[agent_name].setText(f"{agent_name}: {state}")
                self.ai_state_indicators[agent_name].setStyleSheet("color: #7f8c8d; padding: 0 10px;")
                
                # AI가 기타 상태로 변경되면 thinking 상태 해제
                self.update_ai_thinking_status(agent_name, False)
                
            # 강제 UI 새로고침 적용
            self.ai_state_indicators[agent_name].update()
            self.status_bar.update()
            QApplication.processEvents()
    
    def show_api_setup(self):
        """Show dialog to set up API keys"""
        dialog = APISetupDialog(self, self.saved_api_keys)
        if dialog.exec_():
            keys = dialog.get_keys()
            
            # Check if we should save the keys
            if keys.get("save_keys", False):
                self.save_api_keys(keys)
            
            # Configure the APIs
            self.configure_apis(keys)
    
    def configure_apis(self, keys):
        """Configure API clients with provided keys"""
        # Initialize all agents if not already done
        if not self.agents:
            self.agents = {
                "ChatGPT": ChatGPTAgent(),
                "Claude": ClaudeAgent(),
                "Gemini": GeminiAgent(),
                "DeepSeek": DeepSeekAgent()
            }
            
            # Create workflow with all agents
            self.workflow = AIWorkflow(
                self.session_dir, 
                self.handle_ai_message,
                self.python_env.execute_python_code
            )
            
            # Connect workflow signals
            self.workflow.ai_status_changed.connect(self.update_ai_thinking_status)
            self.workflow.ai_state_changed.connect(self.update_ai_state_indicator)
            
            # Connect Python environment to workflow
            self.python_env.workflow = self.workflow
            self.python_env.message_handler = self.handle_ai_message
            
            # 기본 모델 설정
            self.initialize_default_models()
        
        setup_results = []
        
        # Configure all agents with provided keys
        if keys.get("openai"):
            setup_results.append(("ChatGPT", self.agents["ChatGPT"].setup(keys["openai"])))
            
        if keys.get("anthropic"):
            setup_results.append(("Claude", self.agents["Claude"].setup(keys["anthropic"])))
            
        if keys.get("gemini"):
            setup_results.append(("Gemini", self.agents["Gemini"].setup(keys["gemini"])))
            
        if keys.get("deepseek"):
            setup_results.append(("DeepSeek", self.agents["DeepSeek"].setup(keys["deepseek"])))
        
        # Add agents to workflow
        for name, agent in self.agents.items():
            self.workflow.add_agent(name, agent)
        
        # Update UI indicators
        for name, success in setup_results:
            self.update_ai_indicator(name, success)
        
        # Save API keys to session directory
        with open(os.path.join(self.session_dir, "api_config.json"), "w") as f:
            # Save last 4 chars of each key
            json.dump({
                "openai": keys.get("openai", "")[-4:] if keys.get("openai") else "",
                "anthropic": keys.get("anthropic", "")[-4:] if keys.get("anthropic") else "",
                "gemini": keys.get("gemini", "")[-4:] if keys.get("gemini") else "",
                "deepseek": keys.get("deepseek", "")[-4:] if keys.get("deepseek") else ""
            }, f)
        
        # Show setup results
        results_message = "API Setup Results:\n" + "\n".join(
            f"{name}: {'Connected' if success else 'Failed'}" 
            for name, success in setup_results
        )
        
        # Update status bar
        connected_count = sum(success for _, success in setup_results)
        self.status_bar.showMessage(f"Connected {connected_count}/{len(setup_results)} AI models")
        
        # Show results as a system message in chat
        system_message = AIMessage("System", results_message)
        self.chat_panel.add_message(system_message)
        
        # Update saved keys with the new ones
        self.saved_api_keys = keys
    
    def send_message(self):
        """Send user message to AI workflow"""
        text = self.chat_panel.chat_input.text().strip()
        if not text:
            return
        
        # Clear input field
        self.chat_panel.chat_input.clear()
        
        # Create and add user message
        user_message = AIMessage("User", text)
        self.chat_panel.add_message(user_message)
        
        # Log message to file
        self.log_message(user_message)
        
        # Initialize minimal agents if needed
        if not self.workflow:
            self.chat_panel.add_message(AIMessage("System", "Initializing AI system (minimal mode)..."))
            self.init_minimal_agents()
        
        # Show typing indicator
        self.chat_panel.show_typing_indicator()
        
        # Start the AI workflow
        if self.workflow:
            # 합의 후 새 메시지가 입력된 경우 워크플로우 상태 재설정
            if self.workflow.task_state == "completed" or self.workflow.consensus_reached:
                # 시스템 메시지 추가
                self.chat_panel.add_message(AIMessage("System", "새로운 질문이 입력되었습니다. AI 대화를 재개합니다."))
                
                # 워크플로우 상태 완전 초기화
                self.workflow._reset_consensus_state()
                
                # 첫 번째 에이전트를 thinking 상태로 설정
                agent_names = list(self.workflow.agents.keys())
                if agent_names:
                    first_agent = agent_names[0]
                    self.workflow.agents[first_agent].set_state(self.workflow.agents[first_agent].STATE_THINKING)
                    
                print("User sent new message after consensus, completely resetting workflow state")
                    
                # 비활성 카운터 초기화
                if hasattr(self.workflow, 'inactivity_prompt_count'):
                    self.workflow.inactivity_prompt_count = 0
                
                # 합의 추적기 완전 초기화 - 이전 투표 기록을 모두 삭제하고 새로 생성
                agent_names = list(self.workflow.agents.keys())
                if agent_names:
                    # 합의 추적기를 새로 생성하여 완전히 초기화
                    self.workflow.consensus_tracker = ConsensusTracker(agent_names)
                    print("Created new consensus tracker after user input")
                
                # 이전 방식: 합의 감지 비활성화 대신 합의 플래그만 초기화
                # 합의 감지 자체는 계속 활성화 상태로 유지하여 AI 응답 처리에 영향을 주지 않음
                self.workflow.consensus_reached = False
                
                # 합의 감지 시스템은 계속 활성화 상태로 유지
                self.workflow.consensus_detection_enabled = True
                
                print("User sent new message after consensus, completely resetting workflow state")
                
                # 새 작업 설정 대신 직접 에이전트 시작
                agent_names = list(self.workflow.agents.keys())
                if agent_names:
                    # 첫 번째 에이전트 선택
                    first_agent = agent_names[0]
                    # 에이전트 상태 설정
                    self.workflow.agents[first_agent].set_state(self.workflow.agents[first_agent].STATE_THINKING)
                    # 에이전트 스레드 시작
                    self.workflow.start_agent_thread(first_agent, task_type="initial")
                    # 현재 작업 설정
                    self.workflow.current_task = text
                    # 상태 업데이트
                    self.status_bar.showMessage("AI processing...")
                    return
            
            self.workflow.set_task(text)
            # Update status
            self.status_bar.showMessage("AI processing...")
        else:
            self.chat_panel.add_message(AIMessage("System", "Error: AI workflow not initialized. Please set up API keys."))
            self.chat_panel.hide_typing_indicator()
    
    def update_ai_thinking_status(self, ai_name, is_thinking):
        """Update the UI to show which AI is currently thinking"""
        # 모든 AI 상태 확인을 위한 변수 추가
        thinking_ais = []
        
        # 워크플로우가 있고 AI 상태 표시기가 있는 경우에만 처리
        if hasattr(self, 'ai_state_indicators') and self.workflow:
            # 현재 thinking 상태인 모든 AI 목록 수집
            for agent_name, indicator in self.ai_state_indicators.items():
                if indicator.text().endswith("Thinking"):
                    thinking_ais.append(agent_name)
            
            # 현재 AI가 thinking 상태인 경우 목록에 추가
            if is_thinking and ai_name not in thinking_ais:
                thinking_ais.append(ai_name)
            # 현재 AI가 thinking 상태가 아닌 경우 목록에서 제거
            elif not is_thinking and ai_name in thinking_ais:
                thinking_ais.remove(ai_name)
                    
            # 워크플로우의 current_thinking_ai도 확인
            if hasattr(self.workflow, 'current_thinking_ai') and self.workflow.current_thinking_ai:
                if self.workflow.current_thinking_ai not in thinking_ais and self.workflow.current_thinking_ai != ai_name:
                    thinking_ais.append(self.workflow.current_thinking_ai)
        
        # 색상 맵 정의
        color_map = {
            "ChatGPT": "#10a37f",  # OpenAI green
            "Claude": "#5436da",   # Anthropic purple
            "Gemini": "#4285f4",   # Google blue
            "DeepSeek": "#ff6b00",  # DeepSeek orange
            "Perplexity": "#a020f0",  # Purple for Perplexity
            "Qwen": "#ff4500"  # Orange-red for Qwen
        }
        
        # 강제 UI 새로고침을 위한 플래그
        force_refresh = False
        
        # 현재 AI가 thinking 상태인 경우
        if is_thinking:
            color = color_map.get(ai_name, "#3498db")  # Default blue if AI not in map
            
            # Update indicator with AI name and color
            self.ai_thinking_indicator.setText(f"{ai_name} is thinking...")
            self.ai_thinking_indicator.setStyleSheet(f"color: {color}; font-weight: bold; padding: 0 10px;")
            
            # Show typing indicator in chat panel
            self.chat_panel.show_typing_indicator(ai_name)
            
            # 강제 UI 새로고침 플래그 설정
            force_refresh = True
        else:
            # 다른 AI가 여전히 thinking 상태인 경우
            if thinking_ais:
                next_thinking_ai = thinking_ais[0]
                color = color_map.get(next_thinking_ai, "#3498db")
                self.ai_thinking_indicator.setText(f"{next_thinking_ai} is thinking...")
                self.ai_thinking_indicator.setStyleSheet(f"color: {color}; font-weight: bold; padding: 0 10px;")
                self.chat_panel.show_typing_indicator(next_thinking_ai)
                
                # 강제 UI 새로고침 플래그 설정
                force_refresh = True
            else:
                # 합의 후 새 사용자 입력이 있는 경우 "AI is processing..." 표시
                if self.workflow and hasattr(self.workflow, 'new_user_input_after_consensus') and self.workflow.new_user_input_after_consensus:
                    self.ai_thinking_indicator.setText("AI is processing...")
                    self.ai_thinking_indicator.setStyleSheet("color: #3498db; font-weight: bold; padding: 0 10px;")
                    # 첫 번째 AI를 thinking 상태로 표시
                    if self.workflow.agents:
                        first_agent = list(self.workflow.agents.keys())[0]
                        self.chat_panel.show_typing_indicator(first_agent)
                    
                    # 강제 UI 새로고침 플래그 설정
                    force_refresh = True
                # 모든 AI가 thinking 상태가 아닌 경우에만 "No AI thinking" 표시
                else:
                    # AI 상태 표시기에서 thinking 상태인 AI가 있는지 다시 확인
                    has_thinking_ai = False
                    for agent_name, indicator in self.ai_state_indicators.items():
                        if indicator.text().endswith("Thinking"):
                            has_thinking_ai = True
                            next_thinking_ai = agent_name
                            break
                    
                    # 실제로 thinking 상태인 AI가 있으면 해당 AI를 표시
                    if has_thinking_ai:
                        color = color_map.get(next_thinking_ai, "#3498db")
                        self.ai_thinking_indicator.setText(f"{next_thinking_ai} is thinking...")
                        self.ai_thinking_indicator.setStyleSheet(f"color: {color}; font-weight: bold; padding: 0 10px;")
                        self.chat_panel.show_typing_indicator(next_thinking_ai)
                    else:
                        self.ai_thinking_indicator.setText("No AI thinking")
                        self.ai_thinking_indicator.setStyleSheet("color: #7f8c8d; padding: 0 10px;")
                        
                        # Hide typing indicator in chat panel
                        self.chat_panel.hide_typing_indicator()
        
        # 강제 UI 새로고침 적용
        if force_refresh:
            # 상태 표시줄 새로고침
            self.status_bar.update()
            
            # AI 상태 표시기 새로고침
            for indicator in self.ai_state_indicators.values():
                indicator.update()
            
            # 메인 윈도우 새로고침
            QApplication.processEvents()
    
    def handle_ai_message(self, message):
        """Handle messages from AI agents through the workflow"""
        # Hide typing indicator if it was shown
        self.chat_panel.hide_typing_indicator()
        
        # Add message to chat display
        self.chat_panel.add_message(message)
        
        # Log message to file
        self.log_message(message)
        
        # Show typing indicator for next AI (if workflow is still active)
        if self.workflow and self.workflow.task_state != "completed" and self.workflow.current_thinking_ai:
            self.chat_panel.show_typing_indicator(self.workflow.current_thinking_ai)
            
        # 추가: Perplexity 메시지 처리 후 워크플로우 상태 확인 및 복구
        if message.sender == "Perplexity" and self.workflow and self.workflow.task_state == "completed":
            # Perplexity 메시지 처리 후 워크플로우가 종료된 경우, 강제로 discussing 상태로 복구
            print("Detected workflow termination after Perplexity message, restoring workflow state")
            self.workflow.task_state = "discussing"
            
            # 다음 에이전트 선택 및 시작
            if self.workflow.agents:
                next_agents = [name for name in self.workflow.agents.keys() if name != "Perplexity"]
                if next_agents:
                    next_agent = next_agents[0]
                    print(f"Restarting workflow with next agent: {next_agent}")
                    self.workflow.start_agent_thread(next_agent, task_type="follow_up")
    
    def log_message(self, message):
        """Log message to session directory"""
        log_path = os.path.join(self.session_dir, "chat_log.txt")
        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(message.format() + "\n")
    
    def create_new_session(self):
        """Create a new collaboration session"""
        reply = QMessageBox.question(
            self, "New Session", 
            "Do you want to start a new session? This will clear the current chat.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            # Clean up current workflow resources
            if self.workflow:
                self.workflow.cleanup()
                
            # Create a new session ID and directory
            old_session_dir = self.session_dir
            self.session_id = str(uuid.uuid4())
            self.session_dir = os.path.join(os.path.expanduser("~"), "ai_collaboration", self.session_id)
            os.makedirs(self.session_dir, exist_ok=True)
            
            # Create new Python environment
            old_env = self.python_env
            self.python_env = PythonEnvironment(self.session_dir)
            self.splitter.replaceWidget(0, self.python_env)
            old_env.deleteLater()
            
            # Create new workflow with the new session directory
            self.workflow = None
            self.agents = {}
            
            # Reset AI state indicators
            for name in self.ai_state_indicators:
                self.ai_state_indicators[name].setText(f"{name}: -")
                self.ai_state_indicators[name].setStyleSheet("color: #7f8c8d; padding: 0 10px;")
            
            # Clear chat display
            self.chat_panel.clear_chat()
            
            # Log basic session info
            with open(os.path.join(self.session_dir, "session_info.txt"), "w") as f:
                f.write(f"Session ID: {self.session_id}\n")
                f.write(f"Created: {datetime.now().isoformat()}\n")
                f.write(f"Directory: {self.session_dir}\n")
                f.write(f"Previous Session: {old_session_dir}\n")
            
            # Add welcome message
            welcome_message = AIMessage(
                "System",
                f"""New session created.
Session ID: {self.session_id[:8]}...
Python environment is being prepared with essential libraries.

The AI team is ready to help you with any task using Python.
AIs will work in three states:
- 🤔 Thinking State: Analyzing problems and planning approaches
- 💬 Discussion State: Exchanging ideas and critiquing approaches  
- 🔨 Execution State: Implementing solutions in code

Simply describe what you want to accomplish, and the AI team will take care of it.
"""
            )
            self.chat_panel.add_message(welcome_message)
            
            # Update status
            self.status_bar.showMessage(f"New session created: {self.session_id[:8]}...")
    
    def open_session_folder(self):
        """Open the current session folder in file explorer"""
        # This implementation depends on the operating system
        try:
            if sys.platform == "win32":
                subprocess.run(["explorer", self.session_dir])
            elif sys.platform == "darwin":  # macOS
                subprocess.run(["open", self.session_dir])
            else:  # Linux
                subprocess.run(["xdg-open", self.session_dir])
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Could not open folder: {str(e)}")
    
    def export_chat(self):
        """Export chat log to a file"""
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Chat Log",
            os.path.join(self.session_dir, "chat_export.txt"),
            "Text Files (*.txt);;HTML Files (*.html);;All Files (*)"
        )
        
        if file_path:
            try:
                # Determine export format
                if file_path.lower().endswith('.html'):
                    # Export as HTML with formatting
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write("<html><head><style>\n")
                        f.write("body { font-family: Arial, sans-serif; margin: 20px; }\n")
                        f.write(".message { margin-bottom: 10px; padding: 8px; border-radius: 4px; }\n")
                        f.write(".user { background-color: #e8f5e9; border-left: 4px solid #4caf50; }\n")
                        f.write(".ai { background-color: #e3f2fd; border-left: 4px solid #2196f3; }\n")
                        f.write(".system { background-color: #ffebee; border-left: 4px solid #f44336; }\n")
                        f.write(".sender { font-weight: bold; }\n")
                        f.write(".timestamp { color: #757575; font-size: 0.8em; }\n")
                        f.write(".content { margin-top: 5px; white-space: pre-wrap; }\n")
                        f.write(".thinking { background-color: #e3f2fd; border-left: 4px solid #2196f3; }\n")
                        f.write(".discussing { background-color: #f3e5f5; border-left: 4px solid #9c27b0; }\n")
                        f.write(".executing { background-color: #ffebee; border-left: 4px solid #f44336; }\n")
                        f.write("</style></head><body>\n")
                        
                        # Export each message
                        for message in self.workflow.messages if self.workflow else []:
                            if message.sender == "User":
                                msg_type = "user"
                            elif message.sender == "System":
                                msg_type = "system"
                            else:
                                # Determine AI state from content prefix
                                if "🤔 [Thinking]" in message.content:
                                    msg_type = "thinking"
                                elif "💬 [Discussing]" in message.content:
                                    msg_type = "discussing"
                                elif "🔨 [Executing]" in message.content:
                                    msg_type = "executing"
                                else:
                                    msg_type = "ai"
                                
                            f.write(f'<div class="message {msg_type}">\n')
                            f.write(f'  <span class="sender">{message.sender}</span>\n')
                            f.write(f'  <span class="timestamp">[{message.timestamp.strftime("%Y-%m-%d %H:%M:%S")}]</span>\n')
                            f.write(f'  <div class="content">{message.content}</div>\n')
                            f.write('</div>\n')
                        
                        f.write("</body></html>")
                else:
                    # Export as plain text
                    with open(file_path, 'w', encoding='utf-8') as f:
                        for message in self.workflow.messages if self.workflow else []:
                            f.write(message.format() + "\n\n")
                
                QMessageBox.information(
                    self,
                    "Export Successful",
                    f"Chat log exported to {file_path}"
                )
            except Exception as e:
                QMessageBox.critical(
                    self,
                    "Export Error",
                    f"Failed to export chat log: {str(e)}"
                )
    
    def closeEvent(self, event):
        """Handle application close event"""
        reply = QMessageBox.question(
            self, "Exit", 
            "Are you sure you want to exit? Make sure to export any important results.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            # 종료 전 모든 백그라운드 프로세스 정리
            if hasattr(self, "workflow") and self.workflow:
                self.workflow.cleanup()
                
            # 모든 스레드 종료 대기
            for thread in threading.enumerate():
                if thread != threading.current_thread() and not thread.daemon:
                    thread.join(0.1)  # 최대 0.1초 대기
            # Clean up workflow resources
            if self.workflow:
                self.workflow.cleanup()
            
            # Save final session state
            with open(os.path.join(self.session_dir, "session_info.txt"), "a") as f:
                f.write(f"Closed: {datetime.now().isoformat()}\n")
            
            event.accept()
        else:
            event.ignore()

#################################
# API Setup Dialog
#################################

class APISetupDialog(QDialog):
    """Dialog for configuring API keys"""
    def __init__(self, parent=None, saved_keys=None):
        super().__init__(parent)
        self.setWindowTitle("AI API Setup")
        self.resize(400, 200)
        
        layout = QFormLayout(self)
        
        self.openai_key_input = QLineEdit()
        self.anthropic_key_input = QLineEdit()
        self.gemini_key_input = QLineEdit()
        self.deepseek_key_input = QLineEdit()
        self.perplexity_key_input = QLineEdit()
        self.qwen_key_input = QLineEdit()
        
        # Use password mode for security
        self.openai_key_input.setEchoMode(QLineEdit.Password)
        self.anthropic_key_input.setEchoMode(QLineEdit.Password)
        self.gemini_key_input.setEchoMode(QLineEdit.Password)
        self.deepseek_key_input.setEchoMode(QLineEdit.Password)
        self.perplexity_key_input.setEchoMode(QLineEdit.Password)
        self.qwen_key_input.setEchoMode(QLineEdit.Password)
        
        # Set saved keys if provided
        if saved_keys:
            if saved_keys.get("openai"):
                self.openai_key_input.setText(saved_keys["openai"])
            if saved_keys.get("anthropic"):
                self.anthropic_key_input.setText(saved_keys["anthropic"])
            if saved_keys.get("gemini"):
                self.gemini_key_input.setText(saved_keys["gemini"])
            if saved_keys.get("deepseek"):
                self.deepseek_key_input.setText(saved_keys["deepseek"])
            if saved_keys.get("perplexity"):
                self.perplexity_key_input.setText(saved_keys["perplexity"])
            if saved_keys.get("qwen"):
                self.qwen_key_input.setText(saved_keys["qwen"])
        
        layout.addRow("OpenAI API Key:", self.openai_key_input)
        layout.addRow("Anthropic API Key:", self.anthropic_key_input)
        layout.addRow("Google Gemini API Key:", self.gemini_key_input)
        layout.addRow("DeepSeek API Key:", self.deepseek_key_input)
        layout.addRow("Perplexity API Key:", self.perplexity_key_input)
        layout.addRow("Qwen API Key:", self.qwen_key_input)
        
        # Add checkbox to save keys
        self.save_keys_checkbox = QCheckBox("Remember API keys")
        self.save_keys_checkbox.setChecked(True)
        layout.addRow(self.save_keys_checkbox)
        
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addRow(buttons)
    
    def get_keys(self):
        return {
            "openai": self.openai_key_input.text(),
            "anthropic": self.anthropic_key_input.text(),
            "gemini": self.gemini_key_input.text(),
            "deepseek": self.deepseek_key_input.text(),
            "perplexity": self.perplexity_key_input.text(),
            "qwen": self.qwen_key_input.text(),
            "save_keys": self.save_keys_checkbox.isChecked()
        }

#################################
# Main Function
#################################

def main():
    """Main application entry point"""
    # Create application
    app = QApplication(sys.argv)
    
    # Register QTextCursor type for use with signals and slots across threads
    # Try multiple registration methods for QTextCursor
    from PyQt5 import QtCore
    QtCore.QMetaType.type("QTextCursor")
    
    # Also try to register the actual class
    from PyQt5.QtGui import QTextCursor
    try:
        QtCore.qRegisterMetaType(QTextCursor)
    except (AttributeError, TypeError):
        pass
    
    app.setStyle('Fusion')  # Use Fusion style for better cross-platform appearance
    
    # Set minimal stylesheet to save memory
    app.setStyleSheet("""
        QMainWindow, QDialog {
            background-color: #f5f5f5;
        }
        QPushButton {
            padding: 4px 8px;
        }
    """)
    
    # Create and show main window
    window = ChatWindow()
    window.show()
    
    # Collect garbage to free memory
    import gc
    gc.collect()
    
    # Start event loop
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()