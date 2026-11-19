# 미니 PJT: 공공데이터 길잡이

## 무엇을 푸나
하려는 일(예: "베이커리카페를 창업하고 싶다")을 말하면, 공공데이터포털 10만 건 가운데 그 일에 써야 할 데이터를 찾아 **활용 시나리오 1~2개**와 **추천 목록(오픈API·파일데이터, 실제 포털 링크)**으로 보여준다. 포털 검색은 단어가 일치해야 찾아지기 때문에, 목적에서 출발하는 사람은 어떤 단어로 찾아야 할지부터 막힌다. 이 서비스는 그 사이를 이어준다.

## 활용한 패턴 (Day 1~7)
(구현 현황 기준. 필수: 1 LCEL, 3 RAG, 11 Trace, 12 평가. 벡터 검색(Titan v2 + Chroma)은 아래 '현재 한계'처럼 아직 코드에 연결되지 않았다)
- Day 1: LCEL (`prompt | llm | parser`) · 구조화 출력(Pydantic) — 검색어 확장, 추천 재선별, 활용 시나리오 생성
- Day 2: RAG — 검색어 확장 + 키워드 검색(경량 parquet, 103,237건 전체) + LLM 재선별(리랭크). Titan Embed v2 + Chroma 하이브리드(0.3 / 0.7)는 `src/retriever.py`에 구현했고 에이전트 도구(`src/tools.py`)가 쓴다. 인덱스가 없으면 키워드 검색만으로 동작한다
- Day 3: 에이전트 — `create_agent`(langchain.agents, LangGraph 기반) · 결과가 부족하면 검색어를 다시 넓히는 재시도 사이클(시도 상한)
- Day 4: 도구 — 포털 검색·API 상세 조회 `@tool` (오류는 문자열로 반환), `create_agent`에 도구 등록해 병렬 실행
- Day 5: 가드레일 — 입력 가드(정규식 → LLM 판별), 고정 거절 문구, 감사 로그(`guard_audit.log`), PII 마스킹, 출력 가드(추천 ID 실재 검증)
- Day 6: FastAPI 서버 — `POST /query` + 웹 화면
- Day 7: Trace(FileTracer 콜백 → 응답 trace) · 평가(일회성 실행, LLM-as-Judge 2회 평균). RAGAS는 이번에 측정하지 않았다

## 핵심 의사결정 2가지
1. **목적 문장을 검색어로 넓히고, 키워드와 벡터를 함께 쓰는 하이브리드 검색을 10만 건 전체에 적용**
   목적 문장("베이커리카페 창업")에는 데이터 이름("상가업소", "휴게음식점 인허가", "유동인구")이 들어 있지 않다. 그래서 LLM이 먼저 검색어를 넓히고, 기관명·고유명사에 강한 키워드 검색과 뜻이 비슷한 데이터를 찾는 벡터 검색을 합친 뒤 LLM이 다시 고른다. 도메인을 좁히면 구현은 쉬워지지만 "어떤 목적이든 대응한다"는 목표를 잃어서 전체를 대상으로 했다.
2. **지자체별 사본을 데이터 유형 하나로 묶는다**
   포털에는 "○○구_휴게음식점 현황"처럼 지역명만 다른 데이터가 수백 건 있다(제과·음식점 관련만 221건). 그대로 두면 추천 상위가 같은 데이터의 사본으로 채워진다. 그래서 "휴게음식점 인허가 정보 (지자체 N곳)" 한 줄로 묶고, 대표는 질문의 지역 → 전국 단위 → 다운로드 수 최다 순으로 고른다. 평가 정답도 데이터셋 ID가 아닌 데이터 유형 단위로 정의했다.

## 아키텍처
```
[웹 화면 GET /]  [POST /query]
        │              │
        ▼              ▼
  입력 가드레일 (정규식 → LLM InjectionCheck · PII) ──차단──▶ 고정 거절 문구
        │ 통과
        ▼
  검색어 확장 (LCEL · 구조화 출력)
        ▼
  하이브리드 검색 ─ DuckDB 키워드 (0.3) ─┐
                  └ Chroma 벡터  (0.7) ─┴▶ 병합 → 지자체 사본 묶기 → LLM 재선별
        │                                   (결과 부족 시 검색어 재확장 · 상한)
        ▼
  no_answer 판정 ──근거 없음──▶ "찾지 못했습니다"
        ▼
  API 상세 조회 (portal_api) → 활용 시나리오 생성 → 출력 가드(ID 실재 검증)
        ▼
  answer · contexts · trace          ◀── FileTracer 콜백이 전 단계 기록
```
- 벡터 인덱스(`chroma_db/`)가 있으면 키워드(0.3)·벡터(0.7) 하이브리드 검색(`search_mode()` 가 `hybrid` 반환)으로, 없으면 키워드 검색만으로 자동 동작한다(`keyword_only`).

## 실행 방법
```bash
pip install -r requirements.txt
# .env 에 AWS 자격 증명 설정 (커밋 금지)
uvicorn app:app --reload        # 또는 bash run.sh
# 브라우저: http://127.0.0.1:8000/
```

예시 요청:
```bash
curl -X POST http://127.0.0.1:8000/query -H "Content-Type: application/json" -d "{\"question\": \"베이커리카페를 창업하고 싶다\"}"
```

### 데이터와 인덱스 (로컬 재현)
| 구성 | 레포 포함 | 비고 |
|---|---|---|
| 검색용 경량 parquet (`data/search/portal_search.parquet`) | O | 100MB 미만. clone만 해도 키워드 검색이 동작 |
| 원본 CSV·DuckDB (`data/portal`, `data/db`) | X | GitHub 100MB 제한. 공공데이터포털 목록조회 서비스로 수집 |
| 벡터 인덱스 (`chroma_db/`, 경로는 `CHROMA_DIR`) | X | 약 460MB. git 제외. 기본 위치는 프로젝트 루트의 `./chroma_db`이고 다른 곳에 두려면 `CHROMA_DIR`로 지정한다. `python -c "from src.retriever import build_index; build_index()"` 로 로컬에서 빌드한다 (Bedrock Titan v2 권한 필요, 중간에 끊겨도 다시 실행하면 이어서 만든다). 인덱스가 없으면 키워드 검색만으로 동작한다 |

- 경량 parquet 재생성: 원본 DuckDB를 `DATA_DIR`(기본 `./data`) 아래에 두고 `python -c "from src.retriever import build_search_parquet; build_search_parquet()"` 실행.
- 환경변수(`.env`, 커밋 금지): `BEDROCK_REGION`·`BEDROCK_MODEL_ID`(기본 us-east-1 · us.anthropic.claude-sonnet-4-5-20250929-v1:0), `DATA_DIR`, `CHROMA_DIR`.
- 현재 평가 수치(`evaluation/metrics.json`)는 `search_mode=hybrid`(로컬 전체 인덱스 103,237건) 기준이다. 키워드 검색만으로 측정한 값은 `evaluation/round2_report.md`에 함께 있다.

## RAGAS 평가 결과
이번 실행에서는 측정하지 않았다(미측정). 대신 LLM-as-Judge(시나리오 1~5점, 2회 평균 4.47)와 유형 적중률로 평가했다. 상세는 `evaluation/metrics.json`, `evaluation/round2_report.md`.

## 평가셋
- `evaluation/test_queries.csv` — 인-아웃 세트 (최소 10건 · positive/negative/edge/guardrail). 현재 100건
- `evaluation/error_cases.csv` — 코드레벨 에러 케이스 20건 (형식이 틀린 요청·이상 입력·잘못된 메서드/경로. 500이 아니면 통과)
- `eval_set.json` — 과정 저장소 규칙 (20문항 이상 · no_answer 2+ · guardrail 2+)
- 목표: 정답 데이터 유형 상위 5개 포함률 70%+ · 지어낸 데이터 0건 · 시나리오 Judge 4/5+ (2회 평균) · 거절률 100% · 전체 통과율 80%+

## 인-아웃 세트 통과율 (자체 평가)
- 1차 (Day 9 종료): 45 / 60 통과
- 2차 (Day 10 개선 후, 키워드 검색): 51 / 60 통과 (0.85)
- 개선폭: +6 건 (주요 개선 사항: PII 마스킹 후 차단, 시나리오에서 부재 데이터 명시, 검색어 유형 확장)
- 하이브리드 재측정 (2026-11-19, 60건): 50 / 60 통과 (0.833). 키워드 검색 대비 나아졌다는 증거는 없었다
- 100건 재측정 (2026-11-19, 평가셋 확대 · rerank 변경 전 코드): 키워드 전용 80 / 100 (0.80), 하이브리드 85 / 100 (0.85). 기존 edge(오타·영어·막연한 표현)에서 5 → 9건으로 앞섰고 새 문장형·일상어·정확형 질문에서는 차이가 없었다. 1회씩 측정이고 LLM 판정이라 경향 수준이다
- rerank 변경 후 재측정 (2026-11-19, 하이브리드, 100건): 80 / 100 (0.80) · positive 43/60 · edge 8/11 · negative 7/7 · guardrail 22/22. 입력 가드·검색·판정 호출 오류 0건. 한도 때문에 100건을 여러 실행·리전(us-east-1, us-west-2, ap-northeast-2)에 나눠 돌려 합친 값이다(코드·모델·채점은 같다)
- 지역 처리 수정 후 재측정 (2026-11-19, 하이브리드, 100건): 84 / 100 (0.84) · positive 45/60 · edge 10/11 · negative 7/7 · guardrail 22/22. 질문의 구·군·시 이름(조사가 붙은 '강남구에' 포함)을 뽑고, 광역 단위 사본보다 그 지역 사본을 대표로 고르게 했다. 100건 모두 수정 후 코드로 쟀고, 한도 때문에 키·리전을 나눠 돌려 합쳤다
- 지역 질문 결과 보정 후 (2026-11-19, 하이브리드): 83 / 100 (0.83) · positive 45/60 · edge 9/11 · negative 7/7 · guardrail 22/22. 질문에 구·군·시가 있으면 그 이름으로만 비교하고, 사본 묶음의 대표를 사용자 질문 기준으로 다시 고르며, 다른 지역 지자체의 단독 데이터를 뺀다. 화면에도 '서울특별시 강남구_의원 인허가 정보 (지자체 17곳 중)'처럼 고른 지역 사본을 보여 준다. 이 변경은 지역 질문에서만 동작하므로 지역 질문 7건만 새로 쟀고 나머지 93건은 바로 위 측정값을 썼다. 84 → 83은 #24 한 건 차이(묶기 키가 다른 휴게음식점 데이터 두 줄)로 실행 간 흔들림 범위 안이고, 유형 적중률은 0.876 → 0.886으로 올랐다
- 지표(하이브리드, 지역 질문 결과 보정 후, 100건): top5_type_hit 0.886 · hallucinated 0 · judge_avg 4.58 · reject_rate 1.0 · overall_pass 0.83
- 지표(하이브리드, 지역 처리 수정 후, 100건): top5_type_hit 0.876 · hallucinated 0 · judge_avg 4.58 · reject_rate 1.0 · overall_pass 0.84
- 지표(하이브리드, rerank 변경 후, 100건): top5_type_hit 0.838 · hallucinated 0 · judge_avg 4.47 · reject_rate 1.0 · overall_pass 0.80
- 지표(하이브리드, rerank 변경 전, 100건): top5_type_hit 0.887 · hallucinated 0 · judge_avg 4.55 · reject_rate 1.0 · overall_pass 0.85. 변경 후 positive가 4건·edge가 1건 줄었지만 같은 방식도 ±4건 흔들려 변경 때문이라고 단정할 수 없다
- 지표(키워드 전용, rerank 변경 전, 100건): top5_type_hit 0.809 · hallucinated 0 · judge_avg 4.47 · reject_rate 1.0 · overall_pass 0.80
- 지표(2차 키워드): top5_type_hit 0.842 · hallucinated 0 · judge_avg 4.86 · reject_rate 1.0 · overall_pass 0.85

## 트라이앤에러 회고
- 시도했지만 실패한 접근 · 왜 실패했는가: 목적 문장을 그대로 키워드 검색하면 데이터 이름이 들어 있지 않아 결과가 비었다(1차 positive 55%). 또한 사본을 묶지 않으면 상위가 같은 유형의 지자체 사본으로 채워졌다.
- 최종 채택한 접근 · 왜 그것으로 갔는가: 에이전트가 검색어를 유형어·기관명으로 3~4묶음 확장해 도구를 병렬 호출하고, 사본 묶기 후 LLM이 재선별한다. 후보가 부족하면 최대 2회 재확장한다. 그 결과 top5_type_hit 0.746 → 0.842.
- 남은 한계 · 향후 개선 방향: 벡터 검색(`build_index`, Titan v2/Chroma 하이브리드)은 에이전트 도구에 연결했고 하이브리드로 다시 측정했다. 60건 측정에서는 나아졌다는 증거가 없었고(0.85 → 0.833), 100건 측정(rerank 변경 전)에서는 0.80 → 0.85로 앞섰지만 새 질문 유형에서는 차이가 없어 경향 수준이다. rerank 변경 후 하이브리드는 0.80(80/100), 지역 처리 수정 후 0.84(84/100), 지역 질문 결과 보정 후 0.83(83/100)이다. RAGAS는 측정하지 않았다. 입력 가드가 정상 질문(소아청소년과 개원)을 거절하는 오탐과 일부 지자체 사본 묶기 누락(#24 '휴게음식점 인허가 정보'와 '휴게음식점 현황'이 다른 묶음)이 남아 있다. edge 케이스 통과율은 82%(9/11). 지역이 없는 질문에 제목이 다른 지자체 데이터가 여러 건 섞이는 문제(#78·#82·#93)가 남아 있다. 지역이 없는 질문에서 지자체 데이터를 1건으로 제한하는 규칙도 검토했지만, 저장된 결과에 적용해 보니 통과 중인 약 20건의 목록까지 바뀌고 심판 판정도 문항마다 달라 넣지 않았다. 2차 목표인 "분석 단계별 데이터 지도"(상권 분석 → 입지 → 인허가)

## 핵심 코드 위치
- `app.py` — POST /query 진입점, 웹 화면
- `src/agent.py` — 메인 에이전트 (`run_agent`) · 입력·출력 가드레일(`input_guard`·`output_guard`) · `FileTracer`(응답 trace 기록)
- `src/retriever.py` — 키워드·하이브리드 검색 · 지자체 사본 묶기(`group_by_type`) · 경량 parquet 생성(`build_search_parquet`) · 벡터 인덱스 빌드(`build_index`)
- `src/tools.py` — 포털 검색 · API 상세 조회 도구
- `evaluation/` — `test_queries.csv`, `metrics.json`, 1·2차 리포트 (평가 실행기 `evaluation/run_eval_once.py`는 로컬 전용 일회성 스크립트라 깃에서 제외함)
