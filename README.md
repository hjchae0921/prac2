# LLM-guided kernel design for Bayesian optimisation — 모의실험 (pilot)

**질문**: 가우시안 프로세스 기반 BO에서, 커널을 고정(RBF-ARD)하는 대신
LLM에게 *설계변수 ↔ 목적함수 관계에 대한 정성적 지식*을 주고 커널 함수를 제안·조합하게 하면
최적화 성능이 좋아지는가?

이 저장소는 그 질문을 본격적으로 연구하기 전에 파이프라인 전체를 검증하는 **소규모 모의실험**입니다.

## 구성

```
llmbo/
  kernels.py     조합 가능한 커널 + 표현식 문법 파서   ("PER(0)*RBF(0) + LIN(1)")
  gp.py          GP 회귀 (numpy/scipy, 주변우도 최대화, BIC)
  bo.py          BO 루프 (LHS 초기설계 → EI 획득함수), 커널 전략 플러그인 인터페이스
  problems.py    벤치마크 함수 6개 + 자연어 설명(LLM이 받는 '도메인 지식')
  strategies.py  비교 대상 커널 선택 전략 4종
  llm_client.py  Claude 호출, 프롬프트 생성, JSON 파싱, 디스크 캐시
run_experiment.py  실험 실행 / 요약표 / 그래프
tests/             커널 파서·GP 단위 테스트 (pytest)
results/           실험 결과 json + png
```

### 비교하는 전략

| 이름 | 설명 |
|---|---|
| `fixed_rbf` | 고정 RBF-ARD 커널. 일반적인 GP-BO 기준선 |
| `fixed_mat52` | 고정 Matern-5/2 ARD 커널 (BoTorch 기본값에 해당) |
| `greedy` | 언어 지식 없이 `+`, `*`로 커널을 확장하며 BIC로 고르는 알고리즘적 탐색 (Automatic Statistician 방식) |
| `mock_llm` | **오프라인 대체용**. 문제의 구조 태그(주기성, 가법성 등)를 규칙으로 커널에 대응. "정확한 도메인 지식을 커널에 넣으면 도움이 되는가"만 분리해서 보는 상한선이며, LLM 성능 측정이 **아님** |
| `llm` | 실제 LLM 호출 (`--backend claude` 또는 `openai`). 문제 설명 + 관측 데이터 + 이전에 시도한 커널의 BIC를 보고 커널 3~5개 제안, 10회 평가마다 재제안 |

모든 전략은 같은 루프를 씁니다. 제안된 후보 커널들을 현재 데이터에 각각 적합한 뒤 **BIC가 가장 낮은 커널**을 채택하고, EI로 다음 점을 고릅니다.

### 커널 문법

```
expr   := term ('+' term)*          '+' = 가법 효과
term   := factor ('*' factor)*      '*' = 상호작용 / 변조
factor := BASE '(' dims ')' | '(' expr ')'
BASE   := RBF | MAT32 | MAT52 | PER | LIN | RQ
```
예: `RBF(0)*RBF(1)` = 2차원 ARD, `PER(0)*RBF(0)` = 국소 주기, `PER(0) + RBF(1)` = 가법 모델.
곱 안에서는 첫 인자만 분산 파라미터를 가져서 과대모수화를 막습니다.

## 실행

```bash
pip install -r requirements.txt
python -m pytest tests            # 단위 테스트

# 1) 오프라인 (API 키 불필요): fixed_rbf, fixed_mat52, greedy, mock_llm
python run_experiment.py --seeds 5 --iters 40 --tag offline_pilot

# 2) LLM이 실제로 받을 프롬프트 확인
python run_experiment.py --dry-run-prompt gear3d

# 3) 실제 LLM 포함 (ANTHROPIC_API_KEY 필요, 기본 모델 claude-opus-5)
export ANTHROPIC_API_KEY=...
python run_experiment.py --strategies fixed_rbf greedy llm --problems sinlin2d gear3d additive4d \
    --seeds 3 --iters 40 --tag llm_pilot
# LLM 응답은 results/llm_cache/ 에 캐시되어 재실행 시 비용이 들지 않습니다.
# 프롬프트/응답 전문은 results/llm_transcript.jsonl 에 남습니다.

# 3b) OpenAI 백엔드 (OPENAI_API_KEY 필요). 모델 id는 계정에서 실제 제공되는 이름으로 지정
export OPENAI_API_KEY=...
python run_experiment.py --strategies fixed_rbf greedy llm --backend openai --model gpt-5.4-mini \
    --seeds 3 --iters 40 --tag openai_pilot

# 결과 json으로 표/그래프만 다시 만들기
python run_experiment.py --plot-only results/results_offline_pilot.json
```

호출 횟수: 문제당 시드당 `iters / refresh` 회 (기본 40/10 = 4회). 위 3번 예시는 3문제 × 3시드 × 4회 = 36회 호출입니다.

## 벤치마크 문제

| 이름 | d | 구조 | 의도 |
|---|---|---|---|
| `sinlin2d` | 2 | x1 주기 + x2 2차식, 가법 | 가법·주기 구조가 명확한 쉬운 케이스 |
| `branin` | 2 | 부드러움, x1에 cos 항 | 표준 벤치마크 |
| `ackley2d` | 2 | 그릇 모양 + 격자형 국소최소 | 주기 커널이 진짜 도움이 되는지 |
| `hartmann6` | 6 | 부드러움, 비등방 | 구조 지식이 없을 때 ARD 기준선과 비기는지 (손해 보지 않는지) |
| `additive4d` | 4 | 주기+2차+선형+좁은 골, 가법 | 변수별로 다른 커널이 필요한 케이스 |
| `gear3d` | 3 | 각도 주기 × 하중 상호작용 | 공학 문제 흉내(곱 구조) |

## 해석 시 주의

- `mock_llm`은 "정답 구조를 아는 전문가"의 상한선입니다. 실제 LLM이 그 지식을 프롬프트에서 뽑아내는지는 `llm` 전략으로 따로 확인해야 합니다.
- 이 모의실험의 문제 설명(`domain_knowledge`)은 정답 구조를 꽤 직접적으로 알려줍니다. 본 연구에서는 힌트의 양을 단계별로 줄여가며(구조 명시 → 변수 의미만 → 없음) 민감도를 보는 것이 필요합니다.
- GP는 수치 미분 기반 L-BFGS로 적합하므로 정확도보다 단순함을 택했습니다. 규모를 키우면 BoTorch/GPyTorch로 교체하는 것이 맞습니다.
