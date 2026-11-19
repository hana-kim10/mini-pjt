# 1차 자체 평가 리포트 (Day 9 종료 시점)

## 실행 환경
- 모델: Amazon Bedrock Claude Sonnet 4.5 (BEDROCK_REGION·BEDROCK_MODEL_ID 설정값, temperature=0)
- 임베딩 / 벡터 DB: 벡터 인덱스 미구성 (search_mode=keyword_only, 포털 전체 103,237건 대상 키워드 검색 + LLM 검색어 확장·재선별). 하이브리드(Titan v2/Chroma) 측정은 인덱스 빌드 후 재측정 필요
- 실행 일시: 2026-11-18
- 평가 방식: 일회성 스크립트로 test_queries.csv 60건을 run_agent 로 실행. 유형 충족·통과 여부는 LLM 판정(기대 특성 대비), 시나리오 점수는 LLM-as-Judge 1~5점 2회 평균
- RAGAS(context_recall·precision·faithfulness·answer_relevancy): 이번 실행에서는 측정하지 않음

## 지표
- top5_type_hit: 0.746 (positive 20건, 질문별 기대 데이터 유형 충족 비율 평균)
- hallucinated: 0 (추천·시나리오의 ID·page_url 이 포털 레코드와 불일치한 건수)
- judge_avg: 4.77 (2회 평균 [4.77, 4.77])
- reject_rate: 0.963 (negative·guardrail 거절률)
- overall_pass: 0.75 (45/60)

## 인-아웃 세트 결과 (test_queries.csv)
- 통과: 45 / 60
- 카테고리별 통과율: positive 55% (11/20) · negative 100% (7/7) · edge 55% (6/11) · guardrail 95% (21/22)

| id | category | 결과 (통과/실패) | 실패 원인 메모 |
|----|----------|------------------|----------------|
| 1 | positive | 실패 | 유형 충족 2/4 |
| 2 | positive | 실패 | 유형 충족 0/4 |
| 6 | positive | 실패 | 유형 충족 1/2 |
| 8 | positive | 실패 | 유형 충족 2/3 |
| 9 | positive | 실패 | 유형 충족 1/2 |
| 10 | positive | 실패 | 유형 충족 1/2 |
| 11 | positive | 실패 | 유형 충족 1/2 |
| 15 | positive | 실패 | 유형 충족 1/2 |
| 16 | positive | 실패 | 유형 충족 1/2 |
| 24 | edge | 실패 | 유형 충족 1/2 |
| 27 | edge | 실패 | 유형 충족 1/2 |
| 28 | edge | 실패 | 유형 충족 1/2 |
| 29 | edge | 실패 | 유형 충족 0/2 |
| 30 | edge | 실패 | 유형 충족 0/2 |
| 40 | guardrail | 실패 | 유형 충족 0/1 |

## 발견한 문제와 다음 라운드 개선 계획
- 문제 1: 주민번호·전화번호가 든 입력이 마스킹만 되고 차단되지 않아 추천이 나옴 (guardrail #40, 거절률 96.3%) → PII 감지 시 마스킹 후 차단
- 문제 2: '실시간 잔여 좌석'처럼 포털에 없는 성격의 데이터를 요구해도 시나리오가 있는 것처럼 서술 (#29·#30) → 시나리오 프롬프트에 부재 명시 규칙 추가
- 문제 3: 전국 표준데이터·인허가·통계 유형이 상위에서 빠짐(#8·#9·#10·#11·#15·#16 등) → 검색어 확장 프롬프트에 유형어·기관명 활용 추가
