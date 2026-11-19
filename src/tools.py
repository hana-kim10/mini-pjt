"""도메인 도구 모음: 포털 검색 · API 상세 조회.

도구 / 기능 / 입력 / 출력 / 데이터
  search_portal   / 검색어로 포털 전체에서 오픈API·파일데이터 후보를 찾아 유형별로 묶음 / query, region / JSON 문자열 / 실제 포털 데이터(parquet)
  get_api_detail  / 데이터셋 ID 의 오픈API 엔드포인트·형식·승인 방식 조회 / dataset_id / JSON 문자열 / 실제 포털 데이터(parquet)
오류는 예외 대신 문자열로 돌려줍니다.
"""

import json
from typing import Any

import pandas as pd
from langchain_core.tools import tool

from src.retriever import _load_search_df, group_by_type, hybrid_search, tokenize


def to_item(r: dict) -> dict:
    """검색 레코드(dict)를 추천 항목 형식으로 바꿉니다. 오픈API 는 엔드포인트 정보(api)를 포함합니다."""
    def clean(v: Any) -> Any:
        return None if v is None or (isinstance(v, float) and pd.isna(v)) else v

    item = {
        "id": str(r["id"]), "title": r["title"], "display": r.get("display") or r["title"],
        "list_type": r["list_type"], "org_nm": clean(r.get("org_nm")) or "",
        "keywords": clean(r.get("keywords")) or "", "summary": clean(r.get("summary")) or "",
        "page_url": r["page_url"], "download_cnt": int(clean(r.get("download_cnt")) or 0),
        "copies": int(r.get("copies") or 0),
    }
    if r["list_type"] == "OPEN API":
        item["api"] = {"end_point_url": clean(r.get("api_endpoint")) or "", "format": clean(r.get("api_format")) or "",
                       "approval": clean(r.get("api_approval")) or "", "operations": int(clean(r.get("api_ops")) or 0)}
    return item


def search_candidates(query: str, question: str = "", per_kind: int = 6) -> list[dict]:
    """검색어로 오픈API 와 파일데이터를 각각 찾아 유형별로 묶은 추천 항목 목록을 돌려줍니다 (오류 시 빈 리스트)."""
    terms = tokenize(query)
    if not terms:
        return []
    items: list[dict] = []
    for api_only in (True, False):
        rows = hybrid_search(query, 80, list_type="OPEN API") if api_only else [
            r for r in hybrid_search(query, 160) if r["list_type"] != "OPEN API"]
        items += [to_item(r) for r in group_by_type(rows, question or query, per_kind)]
    return items


@tool
def search_portal(query: str, region: str = "") -> str:
    """공공데이터포털 전체(약 10만 건)에서 검색어로 오픈API·파일데이터 후보를 찾습니다.

    query: 공백으로 구분한 핵심 검색어 2~4개 (예: '상가 업종 상권'). region: 질문에 지역이 있으면 지역명.
    지자체별 사본은 데이터 유형 하나로 묶여 나옵니다. 결과는 JSON 문자열(후보 목록)이며,
    찾지 못했거나 오류가 나면 '결과 없음' 또는 '오류: ...' 문자열을 돌려줍니다.
    """
    try:
        items = search_candidates(f"{query} {region}".strip(), f"{region} {query}")
    except Exception as e:  # 도구는 예외 대신 문자열로 오류를 알린다
        return f"오류: {e}"
    return json.dumps(items, ensure_ascii=False) if items else "결과 없음"


@tool
def get_api_detail(dataset_id: str) -> str:
    """오픈API 데이터셋 ID 로 엔드포인트 주소·응답 형식·활용 승인 방식·오퍼레이션 수를 조회합니다.

    dataset_id: 포털 데이터셋 ID(숫자 문자열). 결과는 JSON 문자열이며, 없는 ID 이거나 오류면 '오류: ...' 문자열입니다.
    """
    try:
        df = _load_search_df()
        row = df[df["id"].astype(str) == str(dataset_id).strip()]
        if row.empty:
            return f"오류: 포털에 없는 ID 입니다 ({dataset_id})"
        item = to_item(row.iloc[0].to_dict())
        if "api" not in item:
            return f"오류: 오픈API 가 아닌 파일데이터입니다 ({dataset_id})"
        return json.dumps(item, ensure_ascii=False)
    except Exception as e:
        return f"오류: {e}"


def get_domain_tools(llm: Any = None) -> list:
    """에이전트에 등록할 도구 목록(포털 검색 · API 상세 조회)을 돌려줍니다."""
    return [search_portal, get_api_detail]
