# 1. 라이브러리 설치 (터미널에서 먼저 실행해야 합니다)
# (venv) C:\...> pip install fastapi uvicorn openai langchain langchain_core pydantic redis

# 2. 라이브러리 Import
from fastapi import FastAPI
from pydantic import BaseModel
import json
import os
import redis
import re

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser

# --- 1. API 키 및 모델 설정 ---
llm_mini = ChatOpenAI(
    model="gpt-4o-mini",
    api_key=os.getenv("OPENAI_API_KEY") 
)

# --- [추가] Redis 서버 연결 ---
try:
    redis_client = redis.Redis(host='127.0.0.1', port=6379, db=0, decode_responses=False)
    redis_client.ping()
    print("--- [INFO] Redis 서버에 성공적으로 연결되었습니다. ---")
except redis.exceptions.ConnectionError as e:
    print(f"--- [ERROR] Redis 서버에 연결할 수 없습니다: {e} ---")
    redis_client = None


# --- 2. 프롬프트 템플릿 정의 (동일) ---
cbt_extract_prompt_template = """
당신은 CBT(인지행동치료) 전문가입니다.
[대화 전사]를 읽고, [상황, 생각, 감정, 행동] 4가지 요소를 JSON으로 추출하세요.
[규칙]...
[대화 전사]
{transcript}
[출력 형식 (JSON)]
{{
  "situation": "...",
  "thoughts": [...],
  "emotions": [...],
  "behaviors": [...]
}}
"""
cbt_extract_prompt = ChatPromptTemplate.from_template(cbt_extract_prompt_template)

alt_perspective_prompt_template = """
당신은 친절한 CBT 코치입니다.
[자동적 사고]를 완화할 '다른 관점'을 1~2문장의 조언으로 작성해 주세요.
[자동적 사고]
{thoughts_text}
[생성할 '다른 관점']
"""
alt_perspective_prompt = ChatPromptTemplate.from_template(alt_perspective_prompt_template)

diary_generation_prompt_template = """
당신은 '일기 작성가'입니다.
주어진 [CBT 분석 데이터 (S-T-E-B)]를 바탕으로, 1인칭 '간단한 하루 일기'를 작성해 주세요.
조언은 포함하지 말고, 오직 사용자의 경험(S-T-E-B)만 서술하세요.
[CBT 분석 데이터]
{cbt_json_data}
[작성할 일기]
"""
diary_generation_prompt = ChatPromptTemplate.from_template(diary_generation_prompt_template)


# --- 3. 파서 및 LangChain 체인 구성 ---
json_parser = JsonOutputParser()
string_parser = StrOutputParser()

chain_extract_cbt = cbt_extract_prompt | llm_mini | string_parser
chain_gen_perspective = alt_perspective_prompt | llm_mini | string_parser
chain_create_diary = diary_generation_prompt | llm_mini | string_parser


# --- 4. FastAPI 앱(서버) 정의 ---
app = FastAPI()

# 4-1. 챗봇 앱으로부터 받을 데이터 형식(Schema) 정의
class ChatLogRequest(BaseModel):
    user_id: str

# 4-2. 챗봇 앱에게 돌려줄 데이터 형식(Schema) 정의 (동일)
class DiaryResponse(BaseModel):
    diary_text: str
    alternative_perspective: str

# --- [추가] AI가 보낸 ```json 껍데기 벗기는 함수 ---
def extract_json_from_markdown(text):
    """
    AI가 반환한 마크다운(```json ... ```) 텍스트에서
    순수한 JSON 문자열({ ... })만 추출합니다.
    """
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        return match.group(0)
    else:
        if text.strip().startswith("{"):
            return text
    return None

# --- 5. API 엔드포인트(창구) 생성 ---
@app.post("/create-diary", response_model=DiaryResponse)
async def create_cbt_diary(chat_log: ChatLogRequest): # <--- 'async def' (비동기 함수)

    if redis_client is None:
        return DiaryResponse(
            diary_text="일기 생성 중 오류가 발생했습니다.",
            alternative_perspective="서버 오류: Redis 서버에 연결할 수 없습니다."
        )
    
    try:
        redis_key = f"chat:{chat_log.user_id}"
        transcript_bytes = redis_client.get(redis_key)

        if transcript_bytes is None:
            return DiaryResponse(
                diary_text="저장된 대화 내용을 찾을 수 없습니다.",
                alternative_perspective=f"서버 오류: Redis에 '{redis_key}' 키가 없습니다."
            )
        
        try:
            transcript = transcript_bytes.decode('cp949')
        except UnicodeDecodeError:
            transcript = transcript_bytes.decode('utf-8')
        
        # 1. 4요소 추출 (S-T-E-B) (JSON '문자열'로 받음)
        cbt_data_str = await chain_extract_cbt.ainvoke({
            "transcript": transcript 
        })

        # 2. AI가 보낸 껍데기(```json) 벗겨내기
        pure_json_str = extract_json_from_markdown(cbt_data_str)
        
        if pure_json_str is None:
             return DiaryResponse(
                diary_text="일기 생성 중 오류가 발생했습니다.",
                alternative_perspective=f"서버 오류: AI가 생성한 응답에서 JSON을 찾지 못했습니다. (내용: {cbt_data_str})"
            )

        # 3. '문자열'을 '딕셔너리'로 수동 변환
        try:
            cbt_data = json.loads(pure_json_str)
        except json.JSONDecodeError:
            return DiaryResponse(
                diary_text="일기 생성 중 오류가 발생했습니다.",
                alternative_perspective=f"서버 오류: AI가 생성한 JSON을 파싱하는 데 실패했습니다. (알맹이: {pure_json_str})"
            )

        # [수정] 4. '다른 관점(A)' 생성 (AI가 'thoughts'를 문자열 리스트/객체 리스트 어떤 것으로 줘도 처리)
        thoughts_list = cbt_data.get('thoughts', [])
        thought_texts = []
        for t in thoughts_list:
            if isinstance(t, dict):
                thought_texts.append(t.get('text', '')) # 't'가 딕셔너리(객체)인 경우
            elif isinstance(t, str):
                thought_texts.append(t) # 't'가 그냥 문자열인 경우
        
        # (이제 thought_texts는 ["생각1", "생각2"] 같은 순수 문자열 리스트가 됩니다)
        
        final_alternative_perspective = ""
        if thought_texts:
            final_alternative_perspective = await chain_gen_perspective.ainvoke({
                "thoughts_text": "\n- ".join(thought_texts)
            })

        # 5. '1인칭 일기' 생성
        final_diary_text = await chain_create_diary.ainvoke({
            "cbt_json_data": json.dumps(cbt_data, ensure_ascii=False)
        })

        # 6. 챗봇 앱에게 최종 결과물을 JSON으로 돌려줌
        return DiaryResponse(
            diary_text=final_diary_text,
            alternative_perspective=final_alternative_perspective
        )

    except Exception as e:
        # 오류 발생 시
        return DiaryResponse(
            diary_text="일기 생성 중 오류가 발생했습니다.",
            alternative_perspective=f"서버 오류: {str(e)}"
        )

# --- 6. 서버 실행 (터미널에서 실행) ---
# (venv) C:\...> set OPENAI_API_KEY="sk-..."
# (venv) C:\...> uvicorn main_with_redis:app --reload --port 8001