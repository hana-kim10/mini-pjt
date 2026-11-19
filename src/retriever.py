"""RAG 파이프라인 (키워드 검색 단계).

커밋된 경량 parquet(data/search/portal_search.parquet)만으로 동작하는 키워드 검색을 제공합니다.
벡터 인덱스(CHROMA_DIR, 기본 ./chroma_db)가 없어도 오류 없이 키워드 검색으로 동작합니다.
반환 형식은 API 의 contexts 와 같습니다: [{"doc_id": ..., "text": ...}, ...]
"""

import math
import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SEARCH_PARQUET = ROOT / "data" / "search" / "portal_search.parquet"
SUMMARY_MAX = 150  # 경량화를 위한 설명 축약 길이
STOPWORDS = {"데이터", "자료", "정보", "관련", "활용", "위한", "하고", "싶어", "싶다", "찾아", "알려", "추천"}
COLUMNS = ["id", "title", "list_type", "org_nm", "keywords", "summary", "page_url", "download_cnt", "ext",
           "api_endpoint", "api_format", "api_approval", "api_ops"]


def build_search_parquet(data_dir: str | None = None, out_path: str | Path | None = None) -> Path:
    """DuckDB 원본(portal·portal_api)에서 필요한 컬럼만 뽑아 경량 parquet 을 만듭니다.

    설명은 SUMMARY_MAX 자로 줄이고, 오픈API 는 첫 오퍼레이션의 엔드포인트·형식·승인 방식을 붙입니다.
    DATA_DIR 환경변수(또는 인자)로 data 폴더 위치를 바꿀 수 있습니다.
    """
    import duckdb

    base = Path(data_dir or os.getenv("DATA_DIR") or ROOT / "data")
    out = Path(out_path or SEARCH_PARQUET)
    portal = sorted((base / "db").glob("portal__*.duckdb"))[-1]
    api = sorted((base / "db").glob("portal_api__*.duckdb"))[-1]
    out.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(":memory:")
    con.execute(f"ATTACH '{portal.as_posix()}' AS p (READ_ONLY)")
    con.execute(f"ATTACH '{api.as_posix()}' AS a (READ_ONLY)")
    con.execute(f"""
        COPY (
          SELECT p.id, p.title, p.list_type, p.org_nm,
                 coalesce(p.keywords, '') AS keywords,
                 left(coalesce(p.summary, ''), {SUMMARY_MAX}) AS summary,
                 p.page_url, coalesce(p.download_cnt, 0) AS download_cnt, p.ext,
                 x.api_endpoint, x.api_format, x.api_approval, coalesce(x.api_ops, 0) AS api_ops
          FROM p.data p
          LEFT JOIN (
            SELECT list_id,
                   max(operation_url) AS api_endpoint,
                   any_value(data_format) AS api_format,
                   any_value(is_confirmed_for_dev_nm) AS api_approval,
                   count(*) AS api_ops
            FROM a.data GROUP BY list_id
          ) x ON x.list_id = p.id
          ORDER BY p.download_cnt DESC NULLS LAST
        ) TO '{out.as_posix()}' (FORMAT PARQUET, COMPRESSION ZSTD)
    """)
    return out


@lru_cache(maxsize=1)
def _load_search_df() -> pd.DataFrame:
    """경량 parquet 을 읽고 검색용 소문자 컬럼을 붙여 캐시합니다."""
    path = Path(os.getenv("SEARCH_PARQUET") or SEARCH_PARQUET)
    df = pd.read_parquet(path).fillna({"keywords": "", "summary": "", "title": ""})
    df["_t"] = df["title"].str.lower()
    df["_k"] = df["keywords"].str.lower()
    df["_s"] = (df["summary"] + " " + df["org_nm"].fillna("")).str.lower()
    return df


def tokenize(query: str) -> list[str]:
    """질의를 검색어 목록으로 나눕니다 (2자 이상, 불용어 제외, 중복 제거)."""
    toks = re.findall(r"[0-9A-Za-z가-힣]+", query.lower())
    seen: list[str] = []
    for t in toks:
        if len(t) >= 2 and t not in STOPWORDS and t not in seen:
            seen.append(t)
    return seen


def keyword_search(terms: list[str] | str, k: int = 20, list_type: str | None = None) -> list[dict]:
    """parquet 에서 제목(3점)·키워드(2점)·설명/기관(1점) 가중치로 키워드 검색합니다.

    여러 검색어에 걸쳐 많이 맞을수록, 같으면 다운로드 수가 많을수록 앞에 옵니다.
    반환은 레코드 dict 목록(parquet 컬럼 + score)이며 맞는 것이 없으면 빈 리스트입니다.
    """
    if isinstance(terms, str):
        terms = tokenize(terms)
    terms = [t.lower() for t in terms if t and t.strip()]
    if not terms:
        return []
    df = _load_search_df()
    score = pd.Series(0.0, index=df.index)
    for t in terms:
        score += 3 * df["_t"].str.contains(t, regex=False)
        score += 2 * df["_k"].str.contains(t, regex=False)
        score += 1 * df["_s"].str.contains(t, regex=False)
    mask = score > 0
    if list_type:
        mask &= df["list_type"] == list_type
    hit = df[mask].assign(score=score[mask] + df.loc[mask, "download_cnt"].clip(lower=0).map(math.log1p) / 100)
    top = hit.nlargest(k, "score")
    return top[COLUMNS + ["score"]].to_dict("records")


REGION_PREFIXES = ("서울", "부산", "대구", "인천", "광주", "대전", "울산", "세종", "경기", "강원", "충북", "충남", "충청",
                   "전북", "전남", "전라", "경북", "경남", "경상", "제주")


def _is_local_org(org: str) -> bool:
    """기관명이 지자체(광역·기초 자치단체와 그 산하 기관)인지 판별합니다."""
    return str(org or "").startswith(REGION_PREFIXES)


def _type_name(title: str, org: str) -> str:
    """제목에서 기관명(지자체명) 접두를 떼어 데이터 유형 이름을 만듭니다. 예: '부산광역시_공영주차장 정보' -> '공영주차장 정보'."""
    t = str(title or "").strip()
    if "_" in t:
        return t.split("_", 1)[1].strip() or t
    org = str(org or "")
    if org and t.startswith(org):
        return t[len(org):].strip() or t
    return t


def _type_key(title: str, org: str, list_type: str) -> str:
    """같은 데이터 유형의 지자체 사본을 묶는 키(제목 정규화 + 목록 유형)를 만듭니다."""
    name = _type_name(title, org)
    # 이름 앞의 지역 접두('서울시 ', '제주 ' 등)는 같은 유형으로 묶기 위해 뗍니다. 뒤에 공백이 있을 때만 뗍니다.
    name = re.sub(rf"^(?:{'|'.join(REGION_PREFIXES)})(?:특별자치시|특별자치도|특별시|광역시|도|시)?\s+", "", name) or name
    name = re.sub(r"open[\s-]*api|오픈\s*api|api|서비스|조회|\d+$", "", name.lower())
    return f"{list_type}|{re.sub(r'[^0-9a-z가-힣]', '', name)}"


@lru_cache(maxsize=1)
def _group_index() -> dict[str, list[int]]:
    """전체 데이터의 유형 키별 행 위치 목록을 만들어 캐시합니다."""
    df = _load_search_df()
    keys = [_type_key(t, o, lt) for t, o, lt in zip(df["title"], df["org_nm"].fillna(""), df["list_type"])]
    idx: dict[str, list[int]] = {}
    for i, k in enumerate(keys):
        idx.setdefault(k, []).append(i)
    return idx


def extract_regions(question: str) -> list[str]:
    """질문에서 지역명(광역 접두 또는 ○○시·군·구)을 뽑습니다. 없으면 빈 리스트."""
    found = [r for r in REGION_PREFIXES if r in question]
    found += [t for t in re.findall(r"[가-힣]{2,}(?:시|군|구)(?![가-힣])", question) if t not in found]
    return found


def group_by_type(rows: list[dict], question: str = "", limit: int = 10) -> list[dict]:
    """검색 결과를 데이터 유형별로 묶어 유형당 대표 하나만 남깁니다.

    지자체 사본은 '○○ (지자체 N곳)' 한 줄로 표시합니다(N은 사본을 가진 지자체 수, 2곳 이상일 때).
    대표 선택: 질문의 지역과 맞는 사본 → 전국 단위(지자체가 아닌 기관) 사본 → 다운로드 수 최다 사본.
    결과는 입력 순서(검색 점수순)를 유지하며 유형이 서로 다른 항목만 limit 개까지 담습니다.
    각 항목은 대표 레코드 dict 에 group_name·copies·display 가 더해진 형태입니다.
    """
    df = _load_search_df()
    idx = _group_index()
    regions = extract_regions(question)
    out: list[dict] = []
    seen: set[str] = set()
    for r in rows:
        key = _type_key(r["title"], r.get("org_nm", ""), r["list_type"])
        if key in seen:
            continue
        seen.add(key)
        members = df.iloc[idx.get(key, [])]
        if members.empty:
            members = pd.DataFrame([r])
        local = members["org_nm"].map(_is_local_org)
        n_local = members.loc[local, "org_nm"].nunique()
        pick = None
        if regions:
            m = members[members.apply(lambda x: any(g in str(x["org_nm"]) or g in str(x["title"]) for g in regions), axis=1)]
            if not m.empty:
                pick = m
        if pick is None:
            nat = members[~local | members["title"].str.contains("전국", na=False)]
            pick = nat if not nat.empty else members
        rep = pick.sort_values("download_cnt", ascending=False).iloc[0].to_dict()
        name = _type_name(rep["title"], rep.get("org_nm", ""))
        rep.update(group_name=name, copies=int(n_local),
                   display=f"{name} (지자체 {n_local}곳)" if n_local >= 2 else rep["title"])
        for c in ("score",):
            if c in r:
                rep[c] = r[c]
        out.append(rep)
        if len(out) >= limit:
            break
    return out


# ---------------------------------------------------------------- 벡터 검색 (Titan v2 + Chroma)

EMBED_MODEL = "amazon.titan-embed-text-v2:0"
COLLECTION = "portal"
EMBED_TEXT_MAX = 500  # 임베딩 입력 글자 수 상한 (제목+키워드+요약)
KEYWORD_WEIGHT, VECTOR_WEIGHT = 0.3, 0.7


def _chroma_dir() -> Path:
    """벡터 인덱스 경로(CHROMA_DIR, 기본 ./chroma_db)를 돌려줍니다."""
    return Path(os.getenv("CHROMA_DIR") or "./chroma_db")


def _embeddings() -> Any:
    """Titan Embed v2 임베딩 객체를 함수 안에서 만듭니다 (리전은 BEDROCK_REGION, 없으면 us-east-1)."""
    from botocore.config import Config
    from langchain_aws import BedrockEmbeddings

    # 쓰로틀링(Too many requests)에 대비해 적응형 재시도를 넉넉히 둡니다.
    return BedrockEmbeddings(model_id=EMBED_MODEL, region_name=os.getenv("BEDROCK_REGION") or "us-east-1",
                             config=Config(retries={"max_attempts": 12, "mode": "adaptive"}))


def _embed_text(row: Any) -> str:
    """한 레코드의 임베딩 입력 텍스트(제목+키워드+요약)를 만듭니다."""
    return f"{row['title']} {row['keywords']} {row['summary']}".strip()[:EMBED_TEXT_MAX]


def build_index(limit: int | None = None, workers: int = 16, chunk: int = 500, embeddings: Any = None) -> int:
    """포털 전체 데이터를 Titan v2 로 임베딩해 Chroma(CHROMA_DIR)에 저장합니다. 중단 후 다시 실행하면 이어서 합니다.

    limit 로 앞쪽 N건만 만들 수 있고(점검용), 이미 들어 있는 id 는 건너뜁니다. 저장된 총 건수를 돌려줍니다.
    """
    import chromadb
    from concurrent.futures import ThreadPoolExecutor

    emb = embeddings or _embeddings()
    df = _load_search_df()
    if limit:
        df = df.head(limit)
    col = chromadb.PersistentClient(path=str(_chroma_dir())).get_or_create_collection(
        COLLECTION, metadata={"hnsw:space": "cosine"})
    done = set(col.get(include=[])["ids"])
    todo = df[~df["id"].astype(str).isin(done)]
    with ThreadPoolExecutor(workers) as ex:
        for i in range(0, len(todo), chunk):
            part = todo.iloc[i:i + chunk]
            vecs = list(ex.map(emb.embed_query, [_embed_text(r) for _, r in part.iterrows()]))
            col.add(ids=part["id"].astype(str).tolist(), embeddings=vecs,
                    metadatas=[{"list_type": str(r["list_type"]), "org_nm": str(r["org_nm"])} for _, r in part.iterrows()])
            print(f"build_index {col.count()}/{len(df)}", flush=True)
    return col.count()


@lru_cache(maxsize=4)
def _collection(path: str) -> Any:
    """경로의 Chroma 컬렉션을 한 번만 열어 재사용합니다."""
    import chromadb

    return chromadb.PersistentClient(path=path).get_collection(COLLECTION)


@lru_cache(maxsize=1)
def _query_embedder() -> Any:
    """질의 임베딩 객체를 한 번만 만들어 재사용합니다."""
    return _embeddings()


@lru_cache(maxsize=512)
def _embed_query(query: str) -> tuple:
    """같은 질의는 임베딩 호출을 반복하지 않도록 결과를 캐시합니다."""
    return tuple(_query_embedder().embed_query(query))


def vector_search(query: str, k: int = 100, embeddings: Any = None, list_type: str | None = None) -> list[dict]:
    """질의를 임베딩해 Chroma 에서 가까운 레코드를 찾습니다. 각 결과는 parquet 레코드 + vscore(0~1 유사도).

    list_type 이 있으면 해당 유형('OPEN API' 등)만 찾습니다.
    인덱스가 없거나 임베딩 호출이 실패하면 빈 리스트를 돌려줍니다(키워드 검색만으로 동작).
    """
    if search_mode() != "hybrid":
        return []
    try:
        col = _collection(str(_chroma_dir()))
        vec = list(embeddings.embed_query(query)) if embeddings is not None else list(_embed_query(query))
        where = {"list_type": list_type} if list_type else None
        res = col.query(query_embeddings=[vec], n_results=k, where=where)
    except Exception:
        return []
    df = _load_search_df().set_index(_load_search_df()["id"].astype(str), drop=False)
    out = []
    for i, d in zip(res["ids"][0], res["distances"][0]):
        if i in df.index:
            out.append({**df.loc[i, COLUMNS].to_dict(), "vscore": max(0.0, 1.0 - float(d))})
    return out


def hybrid_search(query: str, k: int = 100, embeddings: Any = None, list_type: str | None = None) -> list[dict]:
    """키워드(0.3)와 벡터(0.7) 점수를 각각 0~1로 맞춰 합산한 순위로 레코드를 돌려줍니다.

    list_type 이 있으면 해당 유형만 찾습니다. 벡터 인덱스가 없으면 키워드 검색 결과를 그대로 돌려줍니다.
    """
    kw = keyword_search(query, k=k, list_type=list_type)
    vec = vector_search(query, k=k, embeddings=embeddings, list_type=list_type)
    if not vec:
        return kw
    kmax = max((r["score"] for r in kw), default=0.0) or 1.0
    merged: dict[str, dict] = {}
    for r in kw:
        merged[str(r["id"])] = {**r, "kscore": r["score"] / kmax, "vscore": 0.0}
    for r in vec:
        m = merged.setdefault(str(r["id"]), {**r, "kscore": 0.0})
        m["vscore"] = r["vscore"]
    for m in merged.values():
        m["score"] = KEYWORD_WEIGHT * m.get("kscore", 0.0) + VECTOR_WEIGHT * m.get("vscore", 0.0)
    return sorted(merged.values(), key=lambda x: x["score"], reverse=True)[:k]


def search_mode() -> str:
    """벡터 인덱스(CHROMA_DIR)가 있으면 'hybrid', 없으면 'keyword_only' 를 돌려줍니다."""
    chroma = Path(os.getenv("CHROMA_DIR") or "./chroma_db")
    return "hybrid" if chroma.is_dir() and any(chroma.iterdir()) else "keyword_only"


def retrieve(query: str, embeddings: Any = None, vectorstore: Any = None) -> list[dict]:
    """질의와 관련된 데이터셋을 찾아 [{"doc_id", "text"}, ...] 로 돌려줍니다.

    하이브리드 검색(인덱스가 없으면 키워드만) 후 유형별 묶기(group_by_type)를 거치며, parquet 이 없거나
    근거가 없으면 빈 리스트를 돌려주고 상위(agent.py)에서 no_answer 처리합니다.
    """
    try:
        rows = group_by_type(hybrid_search(query, k=100), query, limit=10)
    except (FileNotFoundError, OSError):
        return []
    return [
        {"doc_id": r["id"],
         "text": f"{r['title']} [{r['list_type']}] {r['org_nm']} | {r['keywords']} | {r['summary']} | {r['page_url']}"}
        for r in rows
    ]
