"""
音频适配器接口契约

定义所有音频适配器必须实现的抽象接口，确保不同实现的互换性。

重要参考：
- migration/core/providers/vad/silero.py - SileroVAD 实现细节
- migration/core/providers/asr/fun_local.py - FunASR 实现细节
- migration/core/providers/tts/edge.py - EdgeTTS 实现细节
- migration/core/websocket_server.py - WebSocket 服务器架构
- migration/config.yaml - 完整配置结构

禁止修改：migration/ 目录中的所有代码仅作为参考，不应被修改。
"""

from abc import ABC, abstractmethod
from typing import AsyncGenerator, Optional, List
from dataclasses import dataclass
from datetime import datetime


# ============================================================================
# 核心接口定义
# ============================================================================

class BaseAudioAdapter(ABC):
    """
    音频适配器基类
    
    所有音频适配器实现必须继承此类并实现所有抽象方法。
    适配器负责音频流的完整生命周期管理。
    """
    
    @abstractmethod
    async def start(self) -> None:
        """
        启动适配器服务
        
        启动 WebSocket 服务器、初始化音频处理队列、加载 VAD/ASR/TTS 模型。
        
        Raises:
            AdapterStartError: 启动失败时抛出
        """
        pass
    
    @abstractmethod
    async def stop(self) -> None:
        """
        停止适配器服务
        
        关闭所有会话、释放资源、保存状态。
        """
        pass
    
    @abstractmethod
    async def create_session(self, device_id: str) -> str:
        """
        创建新的音频会话
        
        Args:
            device_id: 设备唯一标识符
            
        Returns:
            session_id: 新创建的会话 ID
            
        Raises:
            MaxSessionsExceededError: 超过最大并发会话数
        """
        pass
    
    @abstractmethod
    async def close_session(self, session_id: str) -> None:
        """
        关闭指定会话
        
        Args:
            session_id: 要关闭的会话 ID
            
        Raises:
            SessionNotFoundError: 会话不存在
        """
        pass
    
    @abstractmethod
    async def process_audio(
        self, 
        session_id: str, 
        audio_data: bytes
    ) -> Optional[str]:
        """
        处理音频流并返回识别的文本
        
        完整流程：VAD 检测 → ASR 识别 → 声纹验证（可选）
        
        Args:
            session_id: 会话 ID
            audio_data: 原始音频数据（PCM 16kHz 16-bit）
            
        Returns:
            识别的文本（如果语音结束），否则返回 None
            
        Raises:
            SessionNotFoundError: 会话不存在
            AudioProcessingError: 处理失败
        """
        pass
    
    @abstractmethod
    async def synthesize_speech(
        self, 
        session_id: str, 
        text: str
    ) -> AsyncGenerator[bytes, None]:
        """
        合成语音（流式输出）
        
        Args:
            session_id: 会话 ID
            text: 要合成的文本
            
        Yields:
            音频块（PCM 格式）
            
        Raises:
            SessionNotFoundError: 会话不存在
            TTSError: 合成失败
        """
        pass
    
    @abstractmethod
    async def verify_speaker(
        self, 
        session_id: str, 
        audio_data: bytes
    ) -> Optional[str]:
        """
        验证说话人身份
        
        Args:
            session_id: 会话 ID
            audio_data: 音频数据（用于声纹提取）
            
        Returns:
            speaker_id（验证成功）或 None（验证失败/未启用）
            
        Raises:
            VoiceprintError: 验证过程出错
        """
        pass
    
    @abstractmethod
    async def register_voiceprint(
        self, 
        device_id: str, 
        display_name: str,
        audio_samples: List[bytes]
    ) -> str:
        """
        注册新的声纹
        
        Args:
            device_id: 设备 ID
            display_name: 用户显示名称
            audio_samples: 3-5 个音频样本（各 2-3 秒）
            
        Returns:
            speaker_id: 新注册的说话人 ID
            
        Raises:
            VoiceprintRegistrationError: 注册失败
        """
        pass


class BaseVAD(ABC):
    """
    语音活动检测（VAD）接口
    
    检测音频流中的语音活动段落。
    """
    
    @abstractmethod
    async def detect(
        self, 
        audio_data: bytes, 
        sample_rate: int = 16000
    ) -> List[dict]:
        """
        检测语音活动段落
        
        Args:
            audio_data: 音频数据（PCM）
            sample_rate: 采样率（默认 16kHz）
            
        Returns:
            语音段落列表，每个段落包含 {"start": 0.0, "end": 2.5}（秒）
            
        Example:
            >>> vad = SileroVAD()
            >>> segments = await vad.detect(audio_data)
            >>> print(segments)
            [{"start": 0.5, "end": 3.2}, {"start": 5.0, "end": 7.8}]
        """
        pass


class BaseASR(ABC):
    """
    语音识别（ASR）接口
    
    将音频转换为文本。
    """
    
    @abstractmethod
    async def transcribe(
        self, 
        audio_data: bytes, 
        language: str = "zh"
    ) -> str:
        """
        识别音频中的文本
        
        Args:
            audio_data: 音频数据（PCM 16kHz）
            language: 语言代码（zh=中文, en=英文）
            
        Returns:
            识别的文本
            
        Raises:
            ASRError: 识别失败
            
        Example:
            >>> asr = FunASR()
            >>> text = await asr.transcribe(audio_data)
            >>> print(text)
            "今天天气怎么样"
        """
        pass


class BaseTTS(ABC):
    """
    语音合成（TTS）接口
    
    将文本转换为语音音频。
    """
    
    @abstractmethod
    async def synthesize(
        self, 
        text: str, 
        voice: str = "default"
    ) -> AsyncGenerator[bytes, None]:
        """
        合成语音（流式输出）
        
        Args:
            text: 要合成的文本
            voice: 音色 ID（provider-specific）
            
        Yields:
            音频块（PCM 16kHz 16-bit）
            
        Example:
            >>> tts = EdgeTTS()
            >>> async for chunk in tts.synthesize("你好"):
            ...     await play_audio(chunk)
        """
        pass


class BaseVoiceprint(ABC):
    """
    声纹验证接口
    
    提取声纹特征并进行身份验证。
    """
    
    @abstractmethod
    async def extract_features(self, audio_data: bytes) -> bytes:
        """
        提取声纹特征向量
        
        Args:
            audio_data: 音频数据（2-3 秒）
            
        Returns:
            特征向量（序列化为 bytes）
        """
        pass
    
    @abstractmethod
    async def verify(
        self, 
        audio_data: bytes, 
        reference_features: bytes
    ) -> float:
        """
        验证音频与参考声纹的相似度
        
        Args:
            audio_data: 待验证的音频
            reference_features: 参考声纹特征
            
        Returns:
            相似度分数（0.0-1.0），> 0.8 视为匹配
        """
        pass


# ============================================================================
# 数据传输对象（DTO）
# ============================================================================

@dataclass
class AudioFrame:
    """单个音频帧"""
    data: bytes  # PCM 数据
    timestamp: datetime
    sample_rate: int = 16000
    channels: int = 1


@dataclass
class SpeechSegment:
    """检测到的语音段落"""
    start_time: float  # 秒
    end_time: float
    audio_data: bytes
    confidence: float = 1.0


@dataclass
class TranscriptionResult:
    """ASR 识别结果"""
    text: str
    confidence: float
    language: str
    duration_ms: float


@dataclass
class VoiceprintMatch:
    """声纹匹配结果"""
    speaker_id: str
    similarity_score: float
    is_match: bool  # score > threshold
    threshold: float = 0.8


# ============================================================================
# 异常定义
# ============================================================================

class AudioAdapterError(Exception):
    """适配器基础异常"""
    pass


class AdapterStartError(AudioAdapterError):
    """适配器启动失败"""
    pass


class SessionNotFoundError(AudioAdapterError):
    """会话不存在"""
    pass


class MaxSessionsExceededError(AudioAdapterError):
    """超过最大会话数"""
    pass


class AudioProcessingError(AudioAdapterError):
    """音频处理失败"""
    pass


class ASRError(AudioAdapterError):
    """ASR 识别错误"""
    pass


class TTSError(AudioAdapterError):
    """TTS 合成错误"""
    pass


class VoiceprintError(AudioAdapterError):
    """声纹验证错误"""
    pass


class VoiceprintRegistrationError(VoiceprintError):
    """声纹注册失败"""
    pass
