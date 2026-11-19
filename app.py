"""Mini PJT 진입점 (FastAPI).

산출물 규약(regulation.md §3-2)의 API 스펙을 따릅니다.
  POST /query
  Body:     {"question": "사용자 질의"}
  Response: {"answer": "...", "contexts": [{"doc_id": "...", "text": "..."}], "trace": [{"step": "...", "input": "...", "output": "..."}]}

주의: 모듈 최상단에서 실제 모델을 만들지 않습니다. 모델 생성은 함수 안에서 합니다.
API 키는 코드에 쓰지 않고 .env 로 읽습니다. (공개 레포이므로 커밋 금지)
"""

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from src.agent import run_agent

# .env 의 API 키 등을 환경변수로 읽습니다. (모델 생성이 아니므로 모듈 최상단에서 해도 됩니다)
load_dotenv()

app = FastAPI(title="Mini PJT")


class QueryRequest(BaseModel):
    question: str


class Context(BaseModel):
    doc_id: str
    text: str


class TraceStep(BaseModel):
    step: str
    input: str
    output: str


class QueryResponse(BaseModel):
    answer: str
    contexts: list[Context]
    trace: list[TraceStep]


INDEX_HTML = r"""<!doctype html>
<html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>공공데이터 길잡이</title>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/gh/orioncactus/pretendard@v1.3.9/dist/web/variable/pretendardvariable-dynamic-subset.min.css">
<style>
:root{--bg:#f7f6f4;--panel:#fff;--text:#1f1b16;--sub:#3a342d;--muted:#6b635a;--line:#e6e2dc;
--accent:#0047ab;--accent-weak:#e8eef9;--on-accent:#fff;--brown:#8b5a2b;--brown-weak:#f6efe7;--warn:#b45309;--err:#b91c1c}
@media (prefers-color-scheme:dark){:root{--bg:#151310;--panel:#1e1b17;--text:#f1ede8;--sub:#d9d2c9;--muted:#a89f94;--line:#36302a;
--accent:#7fa6ff;--accent-weak:#1c2740;--on-accent:#0b1530;--brown:#d4a373;--brown-weak:#2e2419;--warn:#f0b04a;--err:#f87171}}
*{box-sizing:border-box}
[hidden]{display:none!important}
body{margin:0;background:var(--bg);color:var(--text);font:15px/1.6 "Pretendard Variable","Pretendard",-apple-system,"Apple SD Gothic Neo","Malgun Gothic",system-ui,sans-serif;letter-spacing:-.01em;word-break:keep-all;-webkit-font-smoothing:antialiased}
a{color:var(--accent);text-decoration:none}
h1{font-size:24px;margin:0 0 16px;letter-spacing:-.02em}
h2{font-size:18px;margin:32px 0 12px;letter-spacing:-.02em}
.muted{color:var(--muted)}
.topbar{display:flex;align-items:center;gap:28px;padding:0 24px;height:58px;background:var(--panel);border-bottom:1px solid var(--line);position:sticky;top:0;z-index:5}
.brand{font-weight:800;color:var(--accent);font-size:18px;letter-spacing:-.03em;cursor:pointer;background:none;border:0;padding:0;font-family:inherit}
main{max-width:1440px;margin:0 auto;padding:28px 24px 80px}
input,button{font:inherit;color:var(--text);background:var(--panel);border:1px solid var(--line);border-radius:6px;padding:7px 10px}
button{cursor:pointer}button:hover{border-color:var(--accent)}button:disabled{opacity:.4;cursor:default}
button.primary{background:var(--accent);border-color:var(--accent);color:var(--on-accent);font-weight:600}
.narrow{max-width:920px;margin:0 auto}
/* 홈 화면 */
.hero{max-width:760px;margin:56px auto 48px;text-align:center}
.hero .homehead h1{font-size:34px;margin:0 0 8px;letter-spacing:-.03em}
.hero .homehead p{margin:0 0 24px;font-size:16px}.hero .homehead p b{color:var(--brown)}
.askhead{display:none;align-items:baseline;gap:12px;flex-wrap:wrap;text-align:left}
.searchbar{display:flex;gap:8px;margin-bottom:16px}
.searchbar #q{flex:1;min-width:0;font-size:17px;padding:14px 18px;border-radius:12px;border-width:2px}
.searchbar #q:focus{outline:none;border-color:var(--accent)}
.searchbar .primary{padding:0 24px;font-size:16px;font-weight:700;border-radius:12px}
.chk{display:none;align-items:center;gap:4px;white-space:nowrap;color:var(--muted)}
.chips{display:flex;flex-wrap:wrap;gap:6px;justify-content:center;margin-bottom:16px}
.chip{font-size:13px;padding:5px 12px;border-radius:999px;border:1px solid var(--line);background:var(--panel);color:var(--text)}
.chip:hover{border-color:var(--accent);color:var(--accent)}
.recent{font-size:13px;color:var(--muted);display:flex;flex-wrap:wrap;gap:4px 12px;justify-content:center}
.recent a{cursor:pointer}.recent a:hover{text-decoration:underline}
.head{display:flex;align-items:baseline;gap:12px}.head h2{font-size:20px;margin:0 0 12px}
.catmini{display:grid;grid-template-columns:repeat(auto-fill,minmax(200px,1fr));gap:10px}
.catcard{display:flex;flex-direction:column;gap:2px;text-align:left;background:var(--panel);border:1px solid var(--line);border-top:3px solid var(--brown);border-radius:12px;padding:14px 16px;color:var(--text)}
.catcard b{font-size:16px}.catcard span{font-size:13px;color:var(--muted)}
.catcard:hover{border-color:var(--accent)}.catcard:hover b{color:var(--accent)}
/* 검색 후 화면 (API 추천 페이지와 같은 구성) */
body.has-results .hero{max-width:920px;margin:0 auto;text-align:left}
body.has-results .homehead,body.has-results .cats{display:none}
body.has-results .askhead{display:flex}
body.has-results .searchbar #q{font-size:15px;padding:7px 10px;border-width:1px;border-radius:6px}
body.has-results .searchbar .primary{font-size:15px;font-weight:600;padding:7px 12px;border-radius:6px}
body.has-results .chk{display:flex}
body.has-results .chips,body.has-results .recent{justify-content:flex-start}
.askstatus{margin:8px 0 20px;padding:16px 20px;border-radius:12px;background:var(--accent-weak);color:var(--accent)}
.askstatus.err{background:none;border:1px solid var(--err);color:var(--err)}
.summary{background:var(--brown-weak);border:1px solid transparent;border-radius:12px;padding:18px 22px;font-size:16px;line-height:1.75;color:var(--sub);margin-bottom:10px}
.summary b{display:block;font-size:13px;color:var(--brown);margin-bottom:4px}
.summary p{margin:0}.summary b.eff{margin-top:14px}.summary .meta{margin-top:8px;font-size:13px;line-height:1.5}
.need{margin-top:40px}
.need h2{display:flex;align-items:center;gap:10px;margin:0 0 4px;font-size:20px}
.need>p.muted:first-of-type{color:var(--brown);margin:0 0 14px 38px}
.idx{display:inline-grid;place-items:center;width:28px;height:28px;border-radius:50%;background:var(--accent);color:var(--on-accent);font-size:14px;flex:none}
.pending{margin:0;padding:14px 20px;border:1px dashed var(--line);border-radius:12px;color:var(--muted);animation:pulse 1.6s ease-in-out infinite}
.summary.pending{border:1px dashed var(--line);padding:18px 22px}
@keyframes pulse{50%{opacity:.55}}
@media (prefers-reduced-motion:reduce){.pending{animation:none}}
.apicard{background:var(--panel);border:1px solid var(--line);border-left:4px solid var(--brown);border-radius:12px;padding:16px 20px;margin-bottom:10px}
.apihead{display:flex;gap:8px;align-items:center;font-size:17px;font-weight:700;line-height:1.4}
.apihead a{color:var(--text)}.apihead a:hover{color:var(--accent);text-decoration:underline}
.apicard .reason{margin:8px 0 0;font-size:15px;line-height:1.7;color:var(--sub)}
.apicard .meta{margin:8px 0 0;font-size:13px;color:var(--muted)}
.badge{font-size:12px;font-weight:700;padding:2px 8px;border-radius:6px;white-space:nowrap;background:var(--accent-weak);color:var(--accent)}
.badge.k-file{background:var(--line);color:var(--muted)}
.apicard details{margin-top:10px;font-size:14px;color:var(--sub);border-top:1px solid var(--line);padding-top:8px}
.apicard summary{cursor:pointer;color:var(--muted);font-size:13px}.apicard summary:hover{color:var(--accent)}
.ops{display:flex;flex-wrap:wrap;gap:4px;list-style:none;margin:10px 0 0;padding:0}
.ops li{font-size:12px;background:var(--bg);border:1px solid var(--line);border-radius:6px;padding:1px 8px}
p.gap{color:var(--brown);background:var(--brown-weak);border:1px dashed var(--brown);border-radius:12px;padding:14px 18px;margin:0}
.foot{margin-top:24px;font-size:12px;color:var(--muted)}
@media (max-width:640px){.topbar{padding:0 12px;gap:12px}.hero{margin-top:16px}.hero .homehead h1{font-size:26px}.searchbar{flex-wrap:wrap}.searchbar #q{flex:1 0 100%}main{padding:20px 12px 60px}}
</style></head><body>
<header class="topbar"><button class="brand" id="home" type="button">공공데이터 길잡이</button></header>
<main>
<div class="narrow">
<section class="hero">
  <div class="homehead">
    <h1>하려는 일을 적어 주세요</h1>
    <p class="muted">공공데이터포털 <b>103,237건</b> 중에서 그 일에 필요한 데이터를 항목별로 찾아 드립니다.</p>
  </div>
  <div class="askhead"><h1>어떤 API를 쓰면 될까요?</h1><span class="muted">하려는 일을 적으면 공공데이터포털 103,237건 중에서 참조할 데이터를 골라 드립니다.</span></div>
  <form class="searchbar" id="f" autocomplete="off"><input id="q" type="text" maxlength="500" placeholder="예: 베이커리카페를 창업하고 싶어" required autofocus><label class="chk"><input type="checkbox" id="apiOnly"> 오픈API만</label><button type="submit" class="primary" id="btn">추천받기</button></form>
  <div class="chips" id="chips"></div>
  <div class="recent" id="recent"></div>
</section>
<div class="cats">
  <div class="head"><h2>분야에서 찾아보기</h2></div>
  <div class="catmini" id="cats"></div>
</div>
<div id="status" class="askstatus" hidden></div>
<div id="answer"><div id="scenarios"></div><div id="recs"></div></div>
</div>
</main>
<script>
const KEY="recent_queries";
const SAMPLES=["베이커리카페를 창업하고 싶어","아파트 실거래가 비교 앱을 만들고 싶어","반려동물 동반 여행 코스 서비스","미세먼지와 날씨에 따른 야외활동 추천","전기차 충전소 지도 서비스"];
const CATS=["공공행정","문화관광","산업고용","교통물류","환경기상","농축수산","국토관리","교육","재정금융","사회복지","보건의료","재난안전","과학기술","식품건강","통일외교안보","법률"];
const KIND={"OPEN API":"오픈API",STANDARD:"표준데이터"};
const $=id=>document.getElementById(id);
let last=null,timer=null;
function el(tag,cls,text){const e=document.createElement(tag);if(cls)e.className=cls;if(text!==undefined)e.textContent=text;return e}
function recent(){try{return JSON.parse(localStorage.getItem(KEY))||[]}catch(e){return[]}}
function renderRecent(){const box=$("recent");box.textContent="";const r=recent();if(!r.length)return;box.append("최근 검색어");
  r.forEach(t=>{const a=el("a",null,t);a.onclick=()=>go(t);box.append(a)})}
function saveRecent(t){try{const r=[t,...recent().filter(x=>x!==t)].slice(0,5);localStorage.setItem(KEY,JSON.stringify(r))}catch(e){}renderRecent()}
function safeUrl(u){return /^https?:\/\//.test(u||"")?u:null}
function card(x){
  const isApi=x.list_type==="OPEN API",c=el("article","apicard"),h=el("div","apihead");
  h.append(el("span","badge"+(isApi?"":" k-file"),KIND[x.list_type]||"파일데이터"));
  const u=safeUrl(x.page_url);if(u){const a=el("a",null,x.display);a.href=u;a.target="_blank";a.rel="noopener";h.append(a)}else h.append(x.display);
  c.append(h,el("p","reason",x.summary),el("p","meta",[x.org_nm,"활용 "+Number(x.download_cnt||0).toLocaleString()+"건","ID "+x.id].join(" · ")));
  if(isApi&&x.api){const d=el("details");d.append(el("summary",null,"엔드포인트 · 형식 · 승인 보기"));const ops=el("ul","ops");
    [["엔드포인트",x.api.end_point_url||"정보 없음"],["형식",x.api.format||"-"],["승인",x.api.approval||"-"]].forEach(p=>ops.append(el("li",null,p[0]+" "+p[1])));d.append(ops);c.append(d)}
  return c}
function render(r){
  const sc=$("scenarios"),rc=$("recs");sc.textContent="";rc.textContent="";
  const titles={};r.recommendations.forEach(x=>titles[x.id]=x.display);
  const s1=el("section","need"),h1=el("h2");h1.append(el("span","idx","1"),"활용 시나리오");s1.append(h1);
  if(!r.scenarios.length)s1.append(el("p","gap",r.scenario_notice||"시나리오를 만들지 못했습니다. 아래 추천 목록을 확인하세요."));
  r.scenarios.forEach(s=>{const c=el("div","summary");c.append(el("b",null,s.title),el("p",null,s.description));
    if(s.effect)c.append(el("b","eff","기대 효과"),el("p",null,s.effect));
    c.append(el("p","meta",s.data_ids.map(i=>titles[i]?titles[i]+"("+i+")":i).join(", ")));s1.append(c)});
  sc.append(s1);
  const list=$("apiOnly").checked?r.recommendations.filter(x=>x.list_type==="OPEN API"):r.recommendations;
  const s2=el("section","need"),h2=el("h2");h2.append(el("span","idx","2"),"추천 목록 ("+list.length+"건)");s2.append(h2);
  if(list.length)list.forEach(x=>s2.append(card(x)));else s2.append(el("p","gap","해당 조건의 데이터가 없습니다."));
  rc.append(s2,el("p","foot","검색 방식 "+r.search_mode+" · 후보 중 상위 "+r.recommendations.length+"건"))}
function show(data){
  $("scenarios").textContent="";$("recs").textContent="";
  let r;try{r=JSON.parse(data.answer)}catch(e){r=null}
  if(!r||!r.recommendations){last=null;$("scenarios").append(el("p","gap",data.answer));return}
  last=r;render(r)}
async function search(t){
  document.body.classList.add("has-results");$("btn").disabled=true;
  const st=$("status");st.hidden=false;st.className="askstatus";const t0=Date.now();
  const tick=()=>{st.textContent="검색어를 넓혀 후보를 찾고 추천을 고르는 중… "+Math.round((Date.now()-t0)/1000)+"초"};tick();timer=setInterval(tick,1000);
  $("scenarios").textContent="";$("recs").textContent="";
  const p=el("div","summary pending");p.append(el("b",null,"요약"),"추천을 고른 뒤 시나리오를 씁니다.");$("scenarios").append(p);
  try{const res=await fetch("/query",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({question:t})});
    if(!res.ok)throw new Error("오류 "+res.status);show(await res.json());saveRecent(t);st.hidden=true;
    try{history.replaceState(null,"","?q="+encodeURIComponent(t))}catch(e){}}
  catch(e){$("scenarios").textContent="";st.className="askstatus err";st.textContent=e.message}
  finally{clearInterval(timer);$("btn").disabled=false}}
function go(t){$("q").value=t;search(t)}
SAMPLES.forEach(t=>{const b=el("button","chip",t);b.type="button";b.onclick=()=>go(t);$("chips").append(b)});
CATS.forEach(n=>{const b=el("button","catcard");b.type="button";b.append(el("b",null,n),el("span",null,"이 분야 데이터 찾기"));b.onclick=()=>go(n+" 분야에서 쓸 만한 공공데이터");$("cats").append(b)});
$("apiOnly").onchange=()=>{if(last)render(last)};
$("home").onclick=()=>{document.body.classList.remove("has-results");$("scenarios").textContent="";$("recs").textContent="";$("status").hidden=true;$("q").value="";last=null;try{history.replaceState(null,"",location.pathname)}catch(e){}$("q").focus()};
$("f").onsubmit=e=>{e.preventDefault();const t=$("q").value.trim();if(t)search(t)};
renderRecent();
const q0=new URLSearchParams(location.search).get("q");if(q0)go(q0);
</script></body></html>"""


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    """검색창·활용 시나리오 카드·추천 목록을 보여주는 웹 화면(HTML 한 장)을 돌려줍니다."""
    return INDEX_HTML


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest) -> QueryResponse:
    # 실제 처리(guardrail → 검색/도구 → no_answer 판정 → 답변)는 src/agent.py 에 있습니다.
    # 여기서는 API 규약에 맞게 결과를 감싸기만 합니다.
    result = run_agent(req.question)
    return QueryResponse(**result)
