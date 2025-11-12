# utils/txt_to_vector.py
"""
TXT 파일을 PostgreSQL pgvector에 벡터로 저장하는 유틸리티
여러 txt 파일을 배치로 처리
"""

import os
import sys
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
_current_file = Path(__file__).resolve()
_project_root = _current_file.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import psycopg2
import psycopg2.extras as extras
import time
from typing import List, Dict, Any, Optional
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.config import get_settings
from app.utils.pdf_to_vector import get_pg_conn, load_embedder, ensure_schema, upsert_chunks
from openai import RateLimitError

# 기본 설정
EMBED_MODEL_NAME = "text-embedding-3-small"  # 1536차원


def get_txt_chunks(txt_path: str) -> List[Dict[str, Any]]:
    """
    TXT 파일에서 텍스트를 추출하고 청크로 분할
    
    Args:
        txt_path: TXT 파일 경로
    
    Returns:
        청크 데이터 리스트
    """
    if not os.path.exists(txt_path):
        print(f"❌ 오류: '{txt_path}' 파일을 찾을 수 없습니다.")
        return []
    
    try:
        with open(txt_path, "r", encoding="utf-8") as f:
            full_text = f.read()
    except Exception as e:
        print(f"❌ 오류: TXT 파일 읽기 실패. {e}")
        return []
    
    if not full_text.strip():
        print(f"⚠️  파일이 비어있습니다: {txt_path}")
        return []
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=150,
        separators=["\n\n", "\n", " ", ""]
    )
    
    chunks = text_splitter.split_text(full_text)
    chunk_data = []
    txt_file_id = os.path.basename(txt_path)
    
    for i, chunk in enumerate(chunks):
        chunk_data.append({
            "id": f"{txt_file_id}_chunk_{i}",
            "content": chunk,
            "metadata": {"source": txt_file_id, "chunk_index": i}
        })
    
    print(f"📄 {txt_file_id}에서 총 {len(chunks)}개의 텍스트 청크를 분할했습니다.")
    return chunk_data


def process_txt_to_vector(
    txt_path: str,
    table_name: str = "thesis_chunks",
    index_name: str = "thesis_chunks_embedding_idx",
    database_url: Optional[str] = None
) -> bool:
    """
    TXT 파일을 PostgreSQL에 벡터로 저장하는 메인 함수
    
    Args:
        txt_path: TXT 파일 경로
        table_name: 저장할 테이블 이름
        index_name: 벡터 인덱스 이름
        database_url: 데이터베이스 URL (선택)
    
    Returns:
        성공 여부
    """
    try:
        # 1) TXT → 청크
        chunk_data = get_txt_chunks(txt_path)
        if not chunk_data:
            print(f"❌ TXT 청크 분할 실패: {txt_path}")
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
        print(f"✅ {os.path.basename(txt_path)} 벡터 저장 완료!")
        return True
        
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return False


def process_all_txt_files(
    data_dir: str = "data",
    table_name: str = "thesis_chunks",
    database_url: Optional[str] = None
) -> Dict[str, bool]:
    """
    data 폴더의 모든 txt 파일을 벡터로 변환하여 저장
    
    Args:
        data_dir: 데이터 폴더 경로
        table_name: 저장할 테이블 이름
        database_url: 데이터베이스 URL (선택)
    
    Returns:
        {파일명: 성공여부} 딕셔너리
    """
    data_path = Path(_project_root) / data_dir
    if not data_path.exists():
        print(f"❌ 오류: '{data_dir}' 폴더를 찾을 수 없습니다.")
        return {}
    
    # 모든 txt 파일 찾기
    txt_files = sorted([f for f in data_path.glob("*.txt")])
    
    if not txt_files:
        print(f"❌ '{data_dir}' 폴더에 txt 파일이 없습니다.")
        return {}
    
    print(f"📁 총 {len(txt_files)}개의 txt 파일을 찾았습니다.\n")
    
    results = {}
    index_name = f"{table_name}_embedding_idx"
    
    for i, txt_file in enumerate(txt_files, 1):
        print(f"\n{'='*60}")
        print(f"[{i}/{len(txt_files)}] 처리 중: {txt_file.name}")
        print(f"{'='*60}\n")
        
        success = process_txt_to_vector(
            str(txt_file),
            table_name=table_name,
            index_name=index_name,
            database_url=database_url
        )
        
        results[txt_file.name] = success
        
        # 파일 간 대기 (API 부하 분산)
        if i < len(txt_files):
            print(f"\n⏳ 다음 파일 처리 전 3초 대기...\n")
            time.sleep(3)
    
    # 결과 요약
    print(f"\n{'='*60}")
    print("📊 처리 결과 요약")
    print(f"{'='*60}")
    success_count = sum(1 for v in results.values() if v)
    fail_count = len(results) - success_count
    
    for filename, success in results.items():
        status = "✅ 성공" if success else "❌ 실패"
        print(f"  {status}: {filename}")
    
    print(f"\n총 {len(results)}개 파일 중 {success_count}개 성공, {fail_count}개 실패")
    print(f"{'='*60}\n")
    
    return results


# 메인 실행
if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--all":
        # 모든 txt 파일 처리
        data_dir = sys.argv[2] if len(sys.argv) > 2 else "data"
        results = process_all_txt_files(data_dir)
        
        if all(results.values()):
            print("\n✅ 모든 파일 처리 완료!")
            sys.exit(0)
        else:
            print("\n⚠️  일부 파일 처리 실패")
            sys.exit(1)
    elif len(sys.argv) > 1:
        # 단일 파일 처리
        txt_path = sys.argv[1]
        table_name = sys.argv[2] if len(sys.argv) > 2 else "thesis_chunks"
        index_name = sys.argv[3] if len(sys.argv) > 3 else "thesis_chunks_embedding_idx"
        
        # 상대 경로를 절대 경로로 변환
        if not os.path.isabs(txt_path):
            txt_path = str(_project_root / txt_path.lstrip('/'))
        
        success = process_txt_to_vector(txt_path, table_name, index_name)
        
        if success:
            print("\n✅ 처리 완료!")
            sys.exit(0)
        else:
            print("\n❌ 처리 실패!")
            sys.exit(1)
    else:
        print("사용법:")
        print("  단일 파일: python app/utils/txt_to_vector.py <txt_file_path>")
        print("  모든 파일: python app/utils/txt_to_vector.py --all [data_dir]")
        print("\n예시:")
        print("  python app/utils/txt_to_vector.py --all data")
        print("  python app/utils/txt_to_vector.py data/수반성 관리.txt")
        sys.exit(1)

