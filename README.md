# 우울감을 갖는 분들을 위한 회복 지원 챗봇 API

우울감을 갖는 분들을 위한 회복 지원 RAG 기반 챗봇 API

## 설치 방법

```bash
# 가상환경 생성
python -m venv venv

# 가상환경 활성화 (macOS/Linux)
source venv/bin/activate

# 가상환경 활성화 (Windows)
# venv\Scripts\activate

# 의존성 설치
pip install -r requirements.txt
```

## 환경 설정

```bash
# .env 파일 생성
cp .env.example .env

# .env 파일에 OpenAI API 키 입력
# OPENAI_API_KEY=your_api_key_here
```

## PDF 파일 준비

학습할 PDF 파일들을 `data/` 폴더에 배치하세요.

```bash
# data 폴더 생성 (없는 경우)
mkdir -p data

# PDF 파일 복사 (예시)
cp ~/Downloads/support_guide.pdf data/
cp ~/Downloads/program_info.pdf data/
```

**주의**: 초기화 시 `data/` 폴더의 **모든 PDF 파일**이 자동으로 로드됩니다.

## 실행 방법

```bash
# 서버 실행
python run.py
```

또는

```bash
uvicorn app.main:app --reload
```

## API 사용법

### 1. 챗봇 초기화 (필수)

서버 시작 후 가장 먼저 챗봇을 초기화해야 합니다. `data/` 폴더의 모든 PDF가 자동으로 로드됩니다.

```bash
curl -X POST http://localhost:8000/api/chatbot/initialize
```

응답 예시:
```json
{
  "success": true,
  "message": "챗봇 초기화 완료! 2개의 PDF 파일을 로드했습니다.",
  "data": {
    "document_count": 245,
    "vector_db_path": "./chroma_db",
    "loaded_files": ["support_guide.pdf", "program_info.pdf"]
  }
}
```

### 2. 챗봇 상태 확인

```bash
curl http://localhost:8000/api/chatbot/status
```

### 3. 질문하기

```bash
curl -X POST http://localhost:8000/api/chatbot/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "우울감을 해소하기 위한 프로그램에는 어떤 것들이 있나요?"}'
```

예쁘게 출력 (jq 사용):
```bash
curl -X POST http://localhost:8000/api/chatbot/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "우울감을 해소하기 위한 지원 프로그램에는 어떤 것들이 있나요?"}' | jq
```

응답 예시:
```json
{
  "success": true,
  "message": "답변이 생성되었습니다.",
  "data": {
    "question": "우울감을 해소하기 위한 지원 프로그램에는 어떤 것들이 있나요?",
    "answer": "우울감을 해소하기 위한 사회복귀 지원 프로그램은...",
    "sources": ["지원 프로그램 안내...", "상담 서비스 정보..."]
  }
}
```

## API 엔드포인트

### Health Check
- `GET /api/health` - 서버 상태 확인

### Chatbot
- `POST /api/chatbot/initialize` - 챗봇 초기화 (data/ 폴더의 모든 PDF를 자동 로드하여 벡터 스토어 생성)
- `GET /api/chatbot/status` - 챗봇 초기화 상태 확인
- `POST /api/chatbot/chat` - 질문하기 (RAG 기반 답변 생성)

## API 문서

서버 실행 후 다음 주소에서 API 문서를 확인할 수 있습니다:

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## 프로젝트 구조

```
h-sw-h-back/
├── app/
│   ├── main.py              # FastAPI 애플리케이션
│   ├── config.py            # 설정
│   ├── routers/             # API 라우터
│   │   ├── health.py        # 헬스체크 엔드포인트
│   │   └── chatbot.py       # 챗봇 엔드포인트 (초기화, 대화, 상태)
│   ├── services/            # 비즈니스 로직
│   │   └── vector_store.py  # RAG 벡터 스토어 서비스
│   ├── utils/               # 유틸리티
│   │   └── pdf_loader.py    # PDF 로드 및 청크 분할
│   └── schemas/             # Pydantic 스키마
│       └── chatbot.py       # 요청/응답 스키마
├── data/                    # 학습할 PDF 파일들
│   ├── support_guide.pdf    # (예시) 지원 가이드
│   └── program_info.pdf     # (예시) 프로그램 정보
├── chroma_db/               # 벡터 스토어 (자동 생성됨)
├── .env                     # 환경 변수 (gitignore)
├── .env.example             # 환경 변수 템플릿
├── .gitignore
├── requirements.txt         # Python 의존성
├── run.py                   # 서버 실행 스크립트
└── README.md
```

## 기술 스택

- **FastAPI** 0.115.0: 웹 프레임워크
- **LangChain** 1.0.3: RAG 파이프라인 구현
- **OpenAI API**:
  - `gpt-4o-mini`: 답변 생성
  - `text-embedding-3-small`: 텍스트 임베딩
- **ChromaDB** 0.6.3: 벡터 스토어
- **PyPDF** 3.17.4: PDF 문서 파싱

## 작동 원리 (RAG Pipeline)

1. **문서 로드**: `data/` 폴더의 모든 PDF 파일 로드
2. **청크 분할**: 문서를 500자 단위로 분할 (100자 overlap)
3. **임베딩 생성**: OpenAI `text-embedding-3-small`로 각 청크를 벡터화
4. **배치 저장**: 100개씩 배치 단위로 ChromaDB에 저장 (토큰 제한 방지)
5. **질의 처리**:
   - 사용자 질문을 임베딩으로 변환
   - 유사도가 높은 상위 3개 청크 검색
   - `gpt-4o-mini`로 컨텍스트 기반 답변 생성
6. **답변 반환**: 답변 + 참조 문서 출처 반환

## 주요 특징

- **다중 PDF 지원**: data 폴더의 모든 PDF를 자동으로 학습
- **배치 처리**: 대용량 문서도 안정적으로 처리
- **통일된 응답 형식**: 모든 API가 일관된 JSON 구조 반환
- **친절한 한국어**: 사용자 친화적인 답변 생성
- **출처 제공**: 답변의 근거가 된 문서 청크 제공

## 환경 변수

`.env` 파일에 다음 값을 설정하세요:

```bash
# 필수
OPENAI_API_KEY=sk-...

# 선택 (기본값 사용 가능)
APP_NAME=FastAPI Application
DEBUG=True
HOST=0.0.0.0
PORT=8000
```
