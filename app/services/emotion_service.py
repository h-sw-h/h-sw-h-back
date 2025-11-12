# services/emotion_service.py
from typing import List, Dict, Optional
import os
from pathlib import Path
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import json
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# KOTE 43 레이블 정의
KOTE_LABELS = [
    '불평/불만', '환영/호의', '감동/감탄', '지긋지긋', '고마움', '슬픔', '화남/분노', '존경',
    '기대감', '우쭐댐/무시함', '안타까움/실망', '비장함', '의심/불신', '뿌듯함', '편안/쾌적', '신기함/관심',
    '아껴주는', '부끄러움', '공포/무서움', '절망', '한심함', '역겨움/징그러움', '짜증', '어이없음', '없음',
    '패배/자기혐오', '귀찮음', '힘듦/지침', '즐거움/신남', '깨달음', '죄책감', '증오/혐오',
    '흐뭇함(귀여움/예쁨)', '당황/난처', '경악', '부담/안_내킴', '서러움', '재미없음',
    '불쌍함/연민', '놀람', '행복', '불안/걱정', '기쁨', '안심/신뢰'
]


class EmotionService:
    """
    감정 분석 서비스 (KOTE 43 멀티라벨 감정 분류 모델)
    - 사용자 대화 텍스트를 입력받아 상위 5개 감정을 분석
    - kote-bert-ml 모델 사용
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Args:
            model_path: 모델 디렉토리 경로 (선택). 없으면 기본 경로 사용
        """
        if model_path:
            self.model_dir = model_path
        else:
            # 기본 경로: 프로젝트 내부 ml_models/kote-bert-ml
            base_dir = Path(__file__).parent.parent.parent
            self.model_dir = str(base_dir / "ml_models" / "kote-bert-ml")
        
        self.device = None
        self.tokenizer = None
        self.model = None
        self.thresholds = None
        self._load_model()
        self._load_thresholds()
        
        # LLM 초기화 (부정 감정 필터링용)
        self._init_llm()
    
    def _load_model(self):
        """
        모델 및 토크나이저 로드
        """
        if not os.path.exists(self.model_dir):
            print(f"⚠️  모델 디렉토리를 찾을 수 없습니다: {self.model_dir}")
            self.model = None
            return
        
        try:
            print(f"📦 감정 분석 모델 로딩 중... ({self.model_dir})")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_dir)
            self.model.eval()  # 평가 모드
            
            # GPU 사용 가능 여부 확인
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
            
            self.model.to(self.device)
            print(f"✅ 감정 분석 모델 로드 완료 (Device: {self.device})")
        except Exception as e:
            print(f"⚠️  모델 로드 실패: {e}")
            self.model = None
    
    def _load_thresholds(self):
        """
        Threshold 파일 로드
        """
        threshold_path = os.path.join(self.model_dir, "label_thresholds.json")
        if not os.path.exists(threshold_path):
            print(f"⚠️  Threshold 파일을 찾을 수 없습니다: {threshold_path}")
            self.thresholds = {}
            return
        
        try:
            with open(threshold_path, "r", encoding="utf-8") as f:
                self.thresholds = json.load(f)
            print("✅ Threshold 로드 완료")
        except Exception as e:
            print(f"⚠️  Threshold 로드 실패: {e}")
            self.thresholds = {}
    
    def _init_llm(self):
        """
        LLM 초기화 (부정 감정 필터링용)
        """
        try:
            from app.config import get_settings
            settings = get_settings()
            
            self.llm = ChatOpenAI(
                model="gpt-4o-mini",
                api_key=settings.openai_api_key,
                temperature=0
            )
            
            # 부정 감정 필터링 프롬프트
            negative_filter_prompt_template = """당신은 감정 분석 전문가입니다.
주어진 감정 리스트에서 부정적인 감정만 필터링해주세요.

[감정 리스트]
{emotions_list}

[규칙]
- 부정적인 감정만 선택하세요 (슬픔, 분노, 불안, 절망, 죄책감, 증오, 공포 등)
- 긍정적인 감정(기쁨, 행복, 즐거움, 고마움 등)은 제외하세요
- 중립적인 감정(없음, 놀람 등)은 제외하세요
- 각 감정의 emotion과 score를 그대로 유지하세요

[출력 형식 (JSON 배열)]
[
  {{"emotion": "슬픔", "score": 0.9873, "threshold": 0.73, "is_active": true}},
  {{"emotion": "불안/걱정", "score": 0.7234, "threshold": 0.58, "is_active": true}}
]

부정적인 감정만 JSON 배열로 반환하세요. 다른 설명은 하지 마세요."""
            
            self.negative_filter_prompt = ChatPromptTemplate.from_template(negative_filter_prompt_template)
            self.output_parser = StrOutputParser()
            self.filter_chain = self.negative_filter_prompt | self.llm | self.output_parser
            
        except Exception as e:
            print(f"⚠️  LLM 초기화 실패: {e}")
            self.llm = None
            self.filter_chain = None
    
    def analyze_emotions(self, text: str, top_k: int = 5) -> List[Dict[str, float]]:
        """
        텍스트에서 상위 K개 감정 분석
        
        Args:
            text: 분석할 텍스트 (대화 내용을 한 줄로 합친 것)
            top_k: 반환할 감정 개수 (기본 5개)
        
        Returns:
            감정 리스트 [{"emotion": "기쁨", "score": 0.85}, ...]
            점수 기준 내림차순 정렬
        """
        if not text or not text.strip():
            return []
        
        # 모델이 있으면 모델 사용
        if self.model is not None:
            return self._predict_with_model(text, top_k)
        else:
            # 모델이 없으면 빈 리스트 반환
            print("⚠️  모델이 로드되지 않아 감정 분석을 수행할 수 없습니다.")
            return []
    
    def _predict_with_model(self, text: str, top_k: int) -> List[Dict[str, float]]:
        """
        실제 모델을 사용한 감정 예측
        """
        try:
            # 토크나이징
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=256,
                padding=True
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # 추론
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits[0]
                probs = torch.sigmoid(logits).cpu().numpy()
            
            # 상위 K개 감정 추출
            top_indices = np.argsort(-probs)[:top_k]
            results = []
            
            for idx in top_indices:
                emotion_label = KOTE_LABELS[int(idx)]
                score = float(probs[idx])
                threshold = self.thresholds.get(emotion_label, {}).get("thr", 0.5)
                
                results.append({
                    "emotion": emotion_label,
                    "score": round(score, 4),
                    "threshold": round(threshold, 4),
                    "is_active": score >= threshold
                })
            
            return results
            
        except Exception as e:
            print(f"⚠️  감정 예측 실패: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def combine_conversation_text(self, messages: List[Dict]) -> str:
        """
        대화 메시지들을 한 줄 텍스트로 합치기
        
        Args:
            messages: 메시지 리스트 [{"role": "user", "content": "...", ...}, ...]
        
        Returns:
            사용자 메시지만 필터링하여 공백으로 연결한 텍스트
        """
        user_messages = []
        for msg in messages:
            if msg.get("role") == "user":
                content = msg.get("content", "").strip()
                if content:
                    user_messages.append(content)
        
        # 공백으로 연결
        combined_text = " ".join(user_messages)
        return combined_text
    
    def filter_negative_emotions(self, emotions: List[Dict]) -> List[Dict]:
        """
        프롬프트를 이용하여 부정적인 감정만 필터링
        
        Args:
            emotions: 감정 리스트 [{"emotion": "...", "score": 0.85, ...}, ...]
        
        Returns:
            부정적인 감정만 필터링된 리스트
        """
        if not emotions:
            return []
        
        # LLM이 없으면 원본 반환
        if self.filter_chain is None:
            print("⚠️  LLM이 초기화되지 않아 부정 감정 필터링을 건너뜁니다.")
            return emotions
        
        try:
            # 감정 리스트를 문자열로 변환
            emotions_str = "\n".join([
                f"- {item['emotion']}: {item['score']:.4f} (threshold: {item['threshold']:.4f}, active: {item['is_active']})"
                for item in emotions
            ])
            
            # LLM으로 부정 감정 필터링
            result_str = self.filter_chain.invoke({"emotions_list": emotions_str})
            
            # JSON 파싱
            # LLM 응답에서 JSON 부분만 추출
            result_str = result_str.strip()
            
            # JSON 배열 부분만 추출 (마크다운 코드 블록 제거)
            if "```json" in result_str:
                result_str = result_str.split("```json")[1].split("```")[0].strip()
            elif "```" in result_str:
                result_str = result_str.split("```")[1].split("```")[0].strip()
            
            filtered_emotions = json.loads(result_str)
            
            # 원본 emotions에서 필터링된 것만 찾아서 반환 (원본 구조 유지)
            filtered_emotion_names = {item["emotion"] for item in filtered_emotions}
            result = [
                emotion for emotion in emotions
                if emotion["emotion"] in filtered_emotion_names
            ]
            
            return result
            
        except json.JSONDecodeError as e:
            print(f"⚠️  JSON 파싱 실패: {e}")
            print(f"   LLM 응답: {result_str[:200]}...")
            # 파싱 실패 시 원본 반환
            return emotions
        except Exception as e:
            print(f"⚠️  부정 감정 필터링 실패: {e}")
            import traceback
            traceback.print_exc()
            # 오류 발생 시 원본 반환
            return emotions


# 전역 감정 분석 서비스 인스턴스
_emotion_service: Optional[EmotionService] = None

def get_emotion_service(model_path: Optional[str] = None) -> EmotionService:
    """
    감정 분석 서비스 의존성 주입 (싱글톤)
    """
    global _emotion_service
    if _emotion_service is None:
        _emotion_service = EmotionService(model_path=model_path)
    return _emotion_service

