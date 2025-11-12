# utils/pdf_to_vector.py
"""
PDF 파일을 PostgreSQL pgvector에 벡터로 저장하는 유틸리티
OpenAI text-embedding-3-small 모델을 사용하여 임베딩 생성 (프로젝트 표준)
"""

import os
import sys
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가 (어디서 실행하든 작동하도록)
_current_file = Path(__file__).resolve()
_project_root = _current_file.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import fitz  # PyMuPDF
import psycopg2
import psycopg2.extras as extras
import time
from typing import List, Dict, Any, Tuple, Optional
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.config import get_settings
from openai import RateLimitError

# 기본 설정 - 프로젝트와 동일한 모델 사용
EMBED_MODEL_NAME = "text-embedding-3-small"  # 1536차원 (프로젝트 표준)


def get_pdf_text_chunks(pdf_path: str) -> List[Dict[str, Any]]:
    """
    PDF 파일에서 텍스트를 추출하고 청크로 분할
    
    Args:
        pdf_path: PDF 파일 경로
    
    Returns:
        청크 데이터 리스트
    """
    if not os.path.exists(pdf_path):
        print(f"오류: '{pdf_path}' 파일을 찾을 수 없습니다.")
        return []

    try:
        doc = fitz.open(pdf_path)
        full_text = ""
        for i in range(doc.page_count):
            full_text += doc.load_page(i).get_text() + "\n\n"
        doc.close()
    except Exception as e:
        print(f"오류: PDF 텍스트 추출 실패. {e}")
        return []

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=150,
        separators=["\n\n", "\n", " ", ""]
    )

    chunks = text_splitter.split_text(full_text)
    chunk_data = []
    pdf_file_id = os.path.basename(pdf_path)

    for i, chunk in enumerate(chunks):
        chunk_data.append({
            "id": f"{pdf_file_id}_chunk_{i}",
            "content": chunk,
            "metadata": {"source": pdf_file_id, "chunk_index": i}
        })

    print(f"📄 PDF에서 총 {len(chunks)}개의 텍스트 청크를 분할했습니다.")
    return chunk_data


def load_embedder(openai_api_key: Optional[str] = None) -> Tuple[OpenAIEmbeddings, int]:
    """
    임베딩 모델 로드 (프로젝트 표준 모델 사용)
    
    Args:
        openai_api_key: OpenAI API 키 (선택). 없으면 설정에서 가져옴
    
    Returns:
        (모델, 차원) 튜플
    """
    if not openai_api_key:
        try:
            settings = get_settings()
            openai_api_key = settings.openai_api_key
        except Exception:
            # 설정 로드 실패 시 환경변수에서 직접 가져오기
            openai_api_key = os.getenv("OPENAI_API_KEY")
    
    if not openai_api_key:
        raise ValueError("OpenAI API 키가 설정되지 않았습니다. .env 파일에 OPENAI_API_KEY를 설정하거나 환경변수로 설정하세요.")
    
    model = OpenAIEmbeddings(
        model=EMBED_MODEL_NAME,
        api_key=openai_api_key
    )
    dim = 1536  # text-embedding-3-small의 차원
    print(f"🧠 임베딩 모델 로드: {EMBED_MODEL_NAME} (차원={dim})")
    return model, dim


def get_pg_conn(database_url: Optional[str] = None):
    """
    PostgreSQL 연결
    
    Args:
        database_url: 데이터베이스 URL (선택). 없으면 설정 또는 환경변수에서 가져옴
    
    Returns:
        PostgreSQL 연결 객체
    """
    if database_url:
        conn = psycopg2.connect(database_url)
    else:
        # 설정에서 가져오기 시도
        try:
            settings = get_settings()
            database_url = settings.database_url
        except Exception:
            # 설정 로드 실패 시 환경변수에서 직접 가져오기
            database_url = os.getenv("DATABASE_URL")
            if not database_url:
                # 기본값 사용
                database_url = os.getenv("PGDATABASE", "postgresql://postgres:password@localhost:5432/hi79_db")
        
        conn = psycopg2.connect(database_url)
    
    conn.autocommit = False
    return conn


def ensure_schema(conn, table_name: str, index_name: str, dim: int):
    """
    PostgreSQL 스키마 준비 (테이블 및 인덱스 생성)
    
    Args:
        conn: PostgreSQL 연결
        table_name: 테이블 이름
        index_name: 인덱스 이름
        dim: 벡터 차원
    """
    with conn.cursor() as cur:
        # 확장자 보장
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")

        # 테이블 생성
        cur.execute(f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            id TEXT PRIMARY KEY,
            content TEXT NOT NULL,
            source TEXT,
            chunk_index INT,
            embedding vector({dim})
        );
        """)

        # 벡터 인덱스(코사인) 생성
        cur.execute(f"""
        DO $$
        BEGIN
            IF NOT EXISTS (
                SELECT 1 FROM pg_class c
                JOIN pg_namespace n ON n.oid = c.relnamespace
                WHERE c.relname = '{index_name}' AND n.nspname = 'public'
            ) THEN
                EXECUTE 'CREATE INDEX {index_name}
                         ON {table_name}
                         USING ivfflat (embedding vector_cosine_ops)
                         WITH (lists = 100);';
            END IF;
        END$$;
        """)

    conn.commit()

    # ANALYZE (통계 수집)
    with conn.cursor() as cur:
        cur.execute(f"ANALYZE {table_name};")
    conn.commit()

    print(f"✅ 스키마 준비 완료 (테이블={table_name}, 인덱스={index_name}, 차원={dim})")


def upsert_chunks(conn, table_name: str, records: List[Dict[str, Any]], embeddings: List[List[float]]):
    """
    청크 데이터를 PostgreSQL에 업서트
    
    Args:
        conn: PostgreSQL 연결
        table_name: 테이블 이름
        records: 청크 데이터 리스트
        embeddings: 임베딩 리스트
    """
    rows = []
    for item, emb in zip(records, embeddings):
        vec_literal = "[" + ",".join(f"{x:.6f}" for x in emb) + "]"
        rows.append((
            item["id"],
            item["content"],
            item["metadata"]["source"],
            item["metadata"]["chunk_index"],
            vec_literal
        ))

    sql = f"""
    INSERT INTO {table_name} (id, content, source, chunk_index, embedding)
    VALUES %s
    ON CONFLICT (id) DO UPDATE
      SET content = EXCLUDED.content,
          source = EXCLUDED.source,
          chunk_index = EXCLUDED.chunk_index,
          embedding = EXCLUDED.embedding;
    """

    template = "(" + ",".join(["%s", "%s", "%s", "%s", "%s::vector"]) + ")"

    with conn.cursor() as cur:
        extras.execute_values(cur, sql, rows, template=template)

    conn.commit()
    print(f"💾 Postgres에 {len(rows)}개 청크 업서트 완료.")


def search_similar(conn, table_name: str, query_text: str, model: OpenAIEmbeddings, top_k: int = 3):
    """
    유사한 청크 검색
    
    Args:
        conn: PostgreSQL 연결
        table_name: 테이블 이름
        query_text: 검색 쿼리 텍스트
        model: 임베딩 모델
        top_k: 반환할 상위 K개
    
    Returns:
        검색 결과 리스트
    """
    q_emb = model.embed_query(query_text)
    vec_literal = "[" + ",".join(f"{x:.6f}" for x in q_emb) + "]"

    sql = f"""
    SELECT id, content, source, chunk_index
    FROM {table_name}
    ORDER BY embedding <=> %s::vector
    LIMIT %s;
    """

    with conn.cursor() as cur:
        cur.execute(sql, (vec_literal, top_k))
        rows = cur.fetchall()

    print("🔎 검색 결과 예시:")
    for rid, content, src, idx in rows:
        preview = content[:80].replace("\n", " ")
        print(f" - {rid} (src={src}, idx={idx}) | {preview}...")

    return rows


def process_pdf_to_vector(
    pdf_path: str,
    table_name: str = "thesis_chunks",
    index_name: str = "thesis_chunks_embedding_idx",
    database_url: Optional[str] = None
) -> bool:
    """
    PDF 파일을 PostgreSQL에 벡터로 저장하는 메인 함수
    
    Args:
        pdf_path: PDF 파일 경로
        table_name: 저장할 테이블 이름
        index_name: 벡터 인덱스 이름
        database_url: 데이터베이스 URL (선택)
    
    Returns:
        성공 여부
    """
    try:
        # 1) PDF → 청크
        chunk_data = get_pdf_text_chunks(pdf_path)
        if not chunk_data:
            print("❌ PDF 청크 분할 실패")
            return False

        # 2) 임베더
        embedder, dim = load_embedder()

        # 3) Postgres 스키마 준비
        conn = get_pg_conn(database_url)
        ensure_schema(conn, table_name, index_name, dim)

        # 4) 임베딩 생성 후 업서트 (작은 배치로 처리)
        documents = [c["content"] for c in chunk_data]
        total_chunks = len(documents)
        batch_size = 10  # 작은 배치 크기 (할당량 고려)
        
        print(f"📊 총 {total_chunks}개 청크를 {batch_size}개씩 배치로 처리합니다...")
        
        all_embeddings = []
        for i in range(0, total_chunks, batch_size):
            batch_docs = documents[i:i + batch_size]
            batch_num = i // batch_size + 1
            total_batches = (total_chunks - 1) // batch_size + 1
            
            print(f"🔄 배치 {batch_num}/{total_batches} 처리 중... ({len(batch_docs)}개 청크)")
            
            try:
                batch_embeddings = embedder.embed_documents(batch_docs)
                all_embeddings.extend(batch_embeddings)
                print(f"✅ 배치 {batch_num} 완료!")
            except RateLimitError as e:
                error_msg = str(e)
                # insufficient_quota는 재시도해도 해결되지 않음
                if 'insufficient_quota' in error_msg.lower():
                    print(f"\n❌ OpenAI API 할당량이 부족합니다!")
                    print(f"   계정의 할당량을 확인하고 결제 정보를 업데이트하세요.")
                    print(f"   https://platform.openai.com/account/billing")
                    print(f"   에러 상세: {error_msg}\n")
                    conn.close()
                    raise ValueError(
                        "OpenAI API 할당량이 부족합니다. "
                        "계정의 할당량을 확인하고 결제 정보를 업데이트한 후 다시 시도하세요."
                    )
                # 일반 rate limit은 재시도 가능하지만 여기서는 즉시 실패
                print(f"❌ 배치 {batch_num} 오류: {e}")
                conn.close()
                raise
            except Exception as e:
                print(f"❌ 배치 {batch_num} 오류: {e}")
                conn.close()
                raise
            
            # 배치 간 짧은 대기 (API 부하 분산)
            if i + batch_size < total_chunks:
                time.sleep(2)  # 2초 대기
        
        # 모든 임베딩이 준비되면 한 번에 업서트
        print(f"💾 총 {len(all_embeddings)}개 임베딩을 데이터베이스에 저장 중...")
        upsert_chunks(conn, table_name, chunk_data, all_embeddings)

        conn.close()
        print("✅ PDF 벡터 저장 완료!")
        return True

    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return False


# 메인 실행 (스크립트로 직접 실행할 때)
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("사용법:")
        print("  방법 1: python -m app.utils.pdf_to_vector <pdf_file_path> [table_name] [index_name]")
        print("  방법 2: python app/utils/pdf_to_vector.py <pdf_file_path> [table_name] [index_name]")
        print("\n예시:")
        print("  python -m app.utils.pdf_to_vector data/000000228819_20251113005310.pdf")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    table_name = sys.argv[2] if len(sys.argv) > 2 else "thesis_chunks"
    index_name = sys.argv[3] if len(sys.argv) > 3 else "thesis_chunks_embedding_idx"
    
    # 상대 경로를 절대 경로로 변환 (프로젝트 루트 기준)
    if not os.path.isabs(pdf_path):
        # 프로젝트 루트 기준으로 경로 생성
        pdf_path = str(_project_root / pdf_path.lstrip('/'))
    
    success = process_pdf_to_vector(pdf_path, table_name, index_name)
    
    if success:
        print("\n✅ 처리 완료!")
    else:
        print("\n❌ 처리 실패!")
        sys.exit(1)

