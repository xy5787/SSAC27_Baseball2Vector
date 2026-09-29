# Production projection increment 및 신규 export 재현 — 사전 명세 초안

상태: **APPROVED — 사용자 확인 완료. 사용자의 후속 지시에 따라 커밋·태그는 생략한다. 승인된 명세·입력 해시로 실행을 고정한다.**

작성일: 2026-09-28. 이 명세는 데이터 점검 후, 신규 export의 모델 결과를 보기 전에 작성했다. 기존 v5 결과와 원본 간 차이·결측·표본 수는 이미 알려져 있다. 원 연구 전체의 사전등록이 아니라 신규 데이터 버전 재현과 외부 프로젝션 증분 실험의 사전 명세다.

## 1. 승인된 범위와 진행 순서

사용자는 신규 `BattingStats_2018_2026.csv`를 공통 실적 소스로 사용하여 (A) 2019–2025 기존 핵심 예측 실험 재현과 (B) 2018–2025 production projection 증분 실험을 병행하되 원 지시서의 단계 순서를 유지하도록 승인했다.

1. 이 초안을 먼저 보여주고 확인을 받는다.
2. 사용자 후속 지시로 커밋·태그를 생략한다. 확인된 명세와 입력·구현 해시를 `prespec_inputs.json`에 고정하고 실행 시 검사한다. 원본 CSV가 추적·스테이징되지 않았는지 확인한다.
3. 명세 승인 및 해시 고정 이후 A와 B의 ZiPS 작업을 병행한다. A의 결과로 B의 모델·표본·α 탐색·층화를 변경하지 않는다.
4. ZiPS Task 2 완료 시 결과를 보고하고 멈춘다. 사용자가 Task 2를 확인한 뒤에만 Steamer Task 3을 계산한다. Steamer의 점검과 프로토콜은 지금 함께 고정한다.
5. 기존 결과와 초록 파일은 보존한다. 신규 결과는 별도 폴더의 갱신 결과로 제공한다. 초록 수정은 이번 범위에 포함하지 않는다.

모델 구현은 이 문서 승인 후 진행한다. 커밋·태그 생략은 사용자의 명시적 변경이며 방법·표본·판정 기준 변경이 아니다. 이후 방법 변경은 이유·변경 시점·결과 열람 여부와 함께 `SSAC27/results/provenance/decisions_log.md`에 남긴다. 결과에 맞춘 설정 선택은 금지한다.

## 2. 입력과 출처 고정

- 유일한 신규 실적 원본: `SSAC27/data/raw/BattingStats_2018_2026.csv`.
- SHA-256: `2130d2bc07ad64f3429c60a85628a1da9938b9dc4777ddc98277855bd1289e82`.
- 12,934행, 465컬럼. 사용자 설명: 전체 batting 컬럼, season split, PA minimum 없음. 2026은 파일 점검 외에는 사용하지 않는다.
- A는 2019–2025만 사용하며, 이력 풀에서도 2018을 제외한다. B는 2018–2025만 사용한다. 이전 export의 수치와 혼합하지 않는다.
- projection: `ProjectionDataset/fangraphs-leaderboard-projections_{YEAR}_{z|s}.csv`; z=ZiPS, s=Steamer. 14개 파일의 해시는 `prespec_inputs.json`에 고정한다.
- 출처: FanGraphs Projections → Batters → Historical Preseason Projections (Members Exclusive) → Dashboard → Data Export. 멤버십이 필요하다. projection의 YEAR는 프리시즌 예측 대상 시즌이다.
- export 날짜는 명시적으로 제공되지 않아 미상으로 기록한다. 파일 mtime으로 대신하지 않는다. 정확한 프리시즌 스냅샷 시점은 제공된 획득 설명상 명시되어 있지 않으며, CSV 자체로 인증할 수 없다. 날짜 추가는 출처 메타데이터 보완으로 기록한다.
- 조인은 `PlayerId`에서 만든 `fg:<PlayerId>`와 시즌으로만 한다. 이름은 표시용이다. 이름 매칭과 MLBAMID 대체 조인은 사용하지 않는다.
- projection 원본 값은 수정·보정·재계산하지 않는다. 원본 wRC+를 그대로 사용하는 Z0/S0와, 학습 창 내부 Ridge 보정을 구분한다. 보조 타깃의 WAR/600 산출만 명시된 비율 계산으로 허용한다.
- source의 `Name.1`, `Team.1`, `Season.1`은 원 컬럼과 일치 여부를 검사 후 제거한다. 다른 `.1` 컬럼은 의미가 다를 수 있으므로 일괄 제거하지 않는다.

기존 frozen input과 신규 export는 공통 10,083개 선수×시즌 및 기본 타격 기록이 일치하지만 WAR, wRC+, BsR, Def, Fld 등의 값이 일부 다르다. 원인 점검은 `SSAC27/results_sd10/production_increment/export_difference_audit/REPORT.md`에 보존한다. 정확한 provider 갱신 시점은 확인되지 않았으며, 최신 export가 어느 모델에 유리한지에 따라 소스를 다시 선택하지 않는다.

## 3. 공통 B2V·Ridge 프로토콜

### 3.1 v5 기준과 스케일

**`B2V_SCALE=zscore`를 명시적으로 사용한다.** 사용자 지시서의 MAE 19.00과 Marcel 증분 0.40은 `SSAC27/results/task8/`의 z-score 결과다. 현재 코드의 기본값 sd10 결과는 B2V MAE 19.059123으로, 같은 기준이 아니다. sd10 clipping이나 다른 점수 변환은 이번 분석에 도입하지 않는다. 기존 점검 폴더가 `results_sd10/` 아래 있는 것은 당시 기본 출력 경로 때문이며, 그 점검에는 점수·모델 결과 계산이 없었다.

재사용할 기존 구현:

- `scripts/task8_submission_strengthening.py`: `cohort_from_raw`, `transitions`, `transform`, `fit_fast`, `summarize_block`.
- `scripts/data.py`: `extended_rolling_folds`.
- `scripts/common.py`: temporal split, Ridge pipeline, metrics, player-cluster bootstrap.
- `src/baseball2vec/marcel.py`: `project`, 기존 metric 설정과 age adjustment.

선수-시즌 PA≥100 전체 코호트에서 입력 시즌별 stabilization → constituent 표준화 → 방향 보정 → 정해진 그룹 내 동일 가중 평균 → 그룹 점수의 시즌 표준화를 기존 구현 그대로 사용한다. 프로젝션 매칭이나 다음 시즌 복귀 여부로 코호트를 먼저 줄이지 않는다.

| 차원 | constituent |
|---|---|
| Contact | Contact%, K%, AVG, xBA, SwStr% |
| Power | ISO, SLG, HardHit%, Barrel%, maxEV, EV, HR/FB |
| Speed | Spd, BsR, UBR, wSB |
| Defense | Def, Fld |
| Discipline | BB%, O-Swing%, Swing%, BB/K |

22-stat 모델은 위 direction-adjusted, stabilized constituent 컬럼을 그대로 사용한다. 새 shrinkage 상수나 feature group을 선택하지 않는다. `season_zscore`의 기존 결측 처리(표준화 후 NaN→0)를 유지한다. 2018 PA≥100 코호트의 Fld 결측 2건도 이 기존 처리에 포함된다. 이는 프로젝션 결측 대치와 구분한다.

입력 시즌 내부의 프로필 정의는 해당 시즌 종료 시점에서 관측 가능한 전체 코호트를 사용한다. Ridge용 imputer·StandardScaler·α 선택은 각 outer 학습 창 및 각 inner 학습 fold에서만 적합한다. test의 다음 시즌 결과 또는 미래 시즌으로 프로필·전처리를 적합하지 않는다. 다만 원본이 후일 갱신된 역사적 실적이므로, 엄밀한 당시 데이터 vintage의 실시간 백테스트라는 주장은 하지 않는다.

### 3.2 모델 적합

- outer: expanding window, test outcome보다 이른 outcome만 학습. 원 코드의 입력 연도 기준 split과 동일하다.
- inner: outer 학습 창 안에서 `GroupKFold(n_splits=5)` on player_id. inner는 선수별 CV이며 별도 시간순 CV로 변경하지 않는다.
- α: `np.logspace(-3, 4, 50)`, 모델·타깃·origin·window별로 독립 선택. 평균 inner MAE 최소값, 동률은 작은 α. 모든 모델에 동일 후보와 절차를 적용한다.
- Ridge intercept 포함. 학습 창 중앙값 대치와 StandardScaler를 기존 pipeline대로 사용한다. projection feature는 적합 전에 유한값을 강제 검사하므로 이 imputer가 projection을 대치하지 못하게 한다.
- seed=42, bootstrap=2,000. 라이브러리 버전과 코드를 기록한다. 가능하면 기존 환경(Python 3.14.6, NumPy 2.5.1, pandas 3.0.3, SciPy 1.17.1, sklearn 1.8.0)을 맞춘다. 환경 차이가 있으면 frozen input의 기존 핵심 예측 재현을 먼저 검증하고 차이를 보고한다. 검증되지 않은 환경 차이를 데이터 효과로 해석하지 않는다.
- R1–R3 및 A 모델은 기존 구현을 재사용한다. 새 Marcel 공식이나 별도 Ridge 최적화 구현을 만들지 않는다. 계수 추출을 위해 선택된 α에서 기존 pipeline을 적합하는 것은 동일한 모델 적합이다.

## 4. 작업 A — 2019–2025 데이터 버전 재현

목적은 지시서에 열거된 v5 핵심 예측 결과가 새 원본에서 어떻게 달라지는지 확인하는 것이다. 기존 설정을 고정한 재계산이며, 신규 방법 탐색이 아니다.

- raw 10,083행, PA≥100 코호트 3,070행, 인접 transition 1,981건.
- 두 시즌 모두 PA≥100. wRC+ 주 분석에서는 기존처럼 2020 outcome과 2020 input을 포함한다.
- 주 타깃 next-season wRC+; 보조 next-season WAR/600. 기존처럼 rate 타깃은 전체 창과 양쪽 endpoint에서 2020을 제거한 창을 계산한다. Marcel history에는 관측된 2020을 남긴다.
- 2018은 history를 포함해 전부 제외한다. 따라서 이 작업의 이력 구간은 기존 정의·관측 범위를 그대로 재현한다.

| Outcome origin | 주 분석 학습 n | 평가 n |
|---|---:|---:|
| 2021 | 275 | 279 |
| 2022 | 554 | 354 |
| 2023 | 908 | 353 |
| 2024 | 1,261 | 358 |
| 2025 | 1,619 | 362 |
| Pooled evaluation | — | 1,706 |

no2020 재현은 기존대로 2023–2025의 3개 origin, 학습 n=354/707/1,065, 평가 n=353/358/362, pooled 1,073건이다.

| 재현 모델 | 기존 Task 8 모델명 |
|---|---|
| calibrated current target | scalar |
| B2V 5개 차원 | B2V |
| 22 constituents | stats22 |
| recalibrated Marcel | history |
| Marcel+B2V | history+B2V |
| Marcel+22 constituents | history+stats22 |

주 보고 대조는 MAE(scalar)−MAE(B2V), MAE(B2V)−MAE(stats22), MAE(history)−MAE(history+B2V), MAE(history+B2V)−MAE(history+stats22)이다. 각 모델 MAE/RMSE, origin별·pooled 대조 및 CI를 산출한다. 기존 결과/새 결과/차이를 분리한 표를 만든다. 서로 다른 버전의 outcome으로 얻은 MAE 차이는 순수한 모델 개선으로 해석하지 않는다.

기존 이력 층화의 재현 대조는 MAE(history)−MAE(B2V)이며, production 증분의 Z1−Z2와 혼동하지 않는다. 1/2/3 관측 시즌 구간 n=160/493/1,053을 예상한다. 이는 원 연구의 post hoc 결과에 대한 데이터 버전 재현이며 독립 표본의 확인 연구라고 표현하지 않는다.

이번 A의 재현 범위는 위 6개 핵심 예측 모델과 두 rate 타깃·기존 no2020 민감도이다. 추가 PCA·대표변수 선택·검색·사용자 연구·counting WAR 실험은 원 지시서의 핵심 수치 재현과 분리하며 자동 확대하지 않는다.

## 5. 작업 B — 공통 production 표본과 origin

- 새 원본의 2018–2025 raw 11,462행, PA≥100 코호트 3,518행, 인접 transition 2,328건.
- 입력 t와 outcome t+1에서 모두 PA≥100. 프로젝션은 YEAR=t+1 파일에서 가져온다.
- outcome 2020의 275건을 학습·평가에서 모두 제외한다. 해당 projection은 점검만 하고 사용하지 않는다.
- 2018→2019의 347건은 학습 전용으로 사용한다. 2019 파일은 이제 유효한 학습 입력이다. 2017/2018 projection은 필요하지 않다.
- 2020→2021은 주 분석에 포함한다. 2026은 입력·outcome·history 어디에도 사용하지 않는다.
- 필터 후 학습/평가 후보는 2,053건, 실제 평가 1,706건. 주 평가 origin은 2021–2025다.
- 각 시스템 내 동일 transition 집합을 Z0–Z3, R1–R3에 사용한다. 이름/ID 없는 원본 행은 조인하지 않으며, 중복된 비결측 PlayerId가 발생하면 임의 해결하지 않고 중단한다.
- 주 타깃에서는 ID 매칭과 유한한 projection wRC+가 필요하다. 보조 타깃에서는 유한한 projection WAR와 유한한 PA>0도 필요하다. 결측/무효 projection은 해당 타깃의 모든 모델에서 함께 제외하고, 제외 건수·PA·나이·이력 구간을 보고한다. 현재 점검에서는 두 타깃 모두 100% 매칭되고 유효하다.
- A와 B는 평가 ID 집합이 같아도 학습 표본과 이력이 다르다. A의 예측·계수·성능을 B의 참조 결과로 재사용하지 않는다.

| Outcome origin | 학습 outcome 연도 | 학습 n | 평가 n |
|---|---|---:|---:|
| 2021 | 2019 | 347 | 279 |
| 2022 | 2019, 2021 | 626 | 354 |
| 2023 | 2019, 2021, 2022 | 980 | 353 |
| 2024 | 2019, 2021, 2022, 2023 | 1,333 | 358 |
| 2025 | 2019, 2021, 2022, 2023, 2024 | 1,691 | 362 |
| Pooled evaluation | — | — | 1,706 |

## 6. ZiPS 모델과 대조 — Task 2

| 모델 | 피처 / 예측 |
|---|---|
| Z0 | ZiPS 원본 wRC+를 그대로 예측값으로 사용; 적합 없음 |
| Z1 | ZiPS wRC+ 단일 피처 Ridge |
| Z2 | ZiPS wRC+ + B2V 5개 차원 |
| Z3 | ZiPS wRC+ + 22 constituents |
| R1 | recalibrated Marcel, 기존 구현 |
| R2 | Marcel + B2V 5개 차원, 기존 구현 |
| R3 | B2V 단독, 기존 구현 |

| 대조 | 정의 | 양수의 의미 |
|---|---|---|
| Z1−Z2 | MAE(Z1)−MAE(Z2) | ZiPS 위 B2V 증분; 핵심 대조 |
| Z2−Z3 | MAE(Z2)−MAE(Z3) | 22 constituents 대비 B2V 압축 비용 |
| Z0−Z1 | MAE(Z0)−MAE(Z1) | calibration 자체의 개선; 해석 보조 |

R1−R2는 이번 production 표본에서 Marcel 위 B2V 증분을 나타내는 참조 대조다. R3 성능도 같은 표본에서 보고한다.

주 타깃은 next-season wRC+. 보조 타깃은 actual WAR_t1 / PA_t1 ×600이다. 보조 Z1/Z2/Z3은 projection WAR / projection PA ×600을 projection feature로 사용한다. R1/R2에는 기존 Marcel의 `WAR_per_600` 구현을 사용하며 R3도 동일 타깃으로 적합한다. 보조 주 대조는 Z1−Z2와 Z2−Z3이며, 원 지시서가 요구하지 않은 별도 raw WAR/600 Z0 비교를 추가하지 않는다.

보고: origin×모델별 MAE·RMSE와 pooled MAE·RMSE, 대조별 pooled 차이·CI·origin별 부호, 표본·고유 선수 수, 선택 α와 튜닝 기록. pooled 값은 모든 test transition의 오차 평균이며 연도 MAE의 단순 평균이 아니다.

Z2는 origin별 모델용 StandardScaler 이후 좌표의 Ridge 계수(절편·선택 α 포함)를 기록한다. ZiPS와 각 차원의 부호·크기를 함께 제시한다. 상관된 feature의 조건부 계수이며 인과적 기여나 원래 스탯 단위의 가중치로 해석하지 않는다.

## 7. 이력 층화, 불확실성, 민감도

### 이력 정의

기존 `marcel_seasons_used` 정의를 사용한다: outcome t+1 예측 시 t, t−1, t−2 중 PA>0으로 관측된 시즌 수. t를 포함하고, PA<100 시즌도 세며, 최대 3이다. 표시는 1 / 2 / 3+로 하되 3+는 코드상 3개 관측 lookback 시즌임을 명시한다. 통산 MLB 경력이나 신인 여부가 아니다.

B의 history pool은 2018부터다. 2018 이전은 복원하지 않는다. 2021 평가 279건 중 기존 대비 225건의 구간이 바뀐다(1→2: 1건, 2→3: 224건). B 주 평가의 구간 n은 159 / 270 / 1,277이다. 알고리즘은 기존과 같고 관측 범위는 확장되었다고 명시한다.

각 구간에서 n, 고유 선수 수, Z1−Z2의 pooled MAE 차이와 95% CI를 보고한다. 이는 이미 관찰한 이력 효과를 더 강한 외부 기준선에서 확인하는 사전 명세 대조이며, 기존 연구와 독립된 선수·시즌 표본의 재현이라고 주장하지 않는다. 구간별 CI의 유의/비유의 차이만으로 구간 간 차이가 유의하다고 주장하지 않는다. 효과 집중 여부는 세 구간의 크기와 불확실성을 함께 기술하며 별도의 사후 interaction 검정을 추가하지 않는다.

### Bootstrap와 판정

2,000회 player-cluster paired bootstrap, seed=42. 고유 player_id를 복원 추출하고 해당 선수의 모든 평가 transition을 함께 가져온다. 한 대조의 두 모델은 같은 resample을 사용한다. 구간별 분석은 해당 구간에 속하는 행을 기준으로 같은 방식을 적용한다. 양측 percentile 95% CI는 2.5/97.5 percentile이다. bootstrap마다 모델을 재적합하지 않는다.

판정은 사전 명세된 MAE 차이 CI가 0을 배제하는지에 한정한다. α grid·표본·검정 수준을 결과에 맞춰 변경하지 않는다. 명목 CI이며 다중 비교나 공통 시즌 충격을 보정하지 않고, 적합된 모델 조건부 불확실성이라는 한계를 기록한다. RMSE는 보조 기술 통계다. CI가 0을 포함하면 결론 불확실로 표현하고 동등성의 증명으로 취급하지 않는다.

### 2020 입력 제외

B에서 outcome 2020은 이미 제외되어 있다. 민감도는 input 2020 transition도 학습·평가에서 제거한 뒤 전처리·α·모델을 다시 적합한다. Marcel의 다른 연도 예측에 필요한 2020 실적 history는 기존 no2020 관례대로 남긴다. 따라서 이 분석을 모든 2020 정보의 완전 제거라고 표현하지 않는다.

| Outcome origin | 민감도 학습 n | 평가 n |
|---|---:|---:|
| 2022 | 347 | 354 |
| 2023 | 701 | 353 |
| 2024 | 1,054 | 358 |
| 2025 | 1,412 | 362 |
| Pooled evaluation | — | 1,427 |

wRC+의 세 주 대조 및 보조 WAR/600의 두 대조를 같은 방식으로 보고한다. 2021 평가가 없어지는 표본 차이를 명시하고 full-window pooled 수치와 무조건 직접 비교하지 않는다.

## 8. 해석과 무결성 확인

- Z1−Z2 CI가 양수이면 B2V가 calibrated ZiPS에 추가 정보를 제공하는 결과로 기술한다. 효과가 이력 1 구간에서 더 큰지 세 구간 결과를 함께 보여준다.
- Z2−Z3 CI도 양수이면 constituent 모델의 우위와 압축 비용을 함께 기술한다. `(Z1−Z2)/(Z1−Z3)`는 분모가 양수일 때만 기술적 비율로 제시하고 분모가 작으면 불안정함을 명시한다. 분모≤0이면 해석 불가로 보고하며 비율을 0–1로 잘라내지 않는다.
- Z1−Z2 CI가 0을 포함하면 추가 개선을 입증하지 못했다고 보고한다. ZiPS가 해당 정보를 이미 반영했을 가능성과 일관되지만 이를 증명한 것은 아니다.
- 주장은 외부 프로젝션 위 정보 추가 여부다. 서술은 “adds to”, “is consistent with” 등 증분에 맞는 표현을 사용하며 B2V 단독의 외부 프로젝션에 대한 우월성 주장으로 바꾸지 않는다.
- 사용자 지시서의 과거 Marcel 증분 0.40은 이상 징후 점검 기준일 뿐, 새 production 표본의 비교값으로 섞지 않는다. Z1−Z2가 0.40보다 크면 결과를 보고하기 전에 ID·input year·projection filename year·outcome year를 원본 행과 직접 대조한다.
- 이러한 alignment 검사는 효과와 무관하게 outcome 연도별 player_id 정렬 첫 3건을 고정 선택해 수행한다. 0.40 초과 시 같은 확인을 다시 검토하고 train/test year·history cutoff를 추가 점검한다. 이상이 발견되어 수정이 필요하면 수정 전후와 사유를 기록하며 유리한 버전을 선택하지 않는다.
- 모든 모델의 test key·target·origin·표본 일치를 강제 검사한다. 중복/누락/비유한 projection, 미래 feature 또는 해시 불일치가 발생하면 해당 계산을 중단하고 보고한다.

## 9. Steamer 반복 — Task 3

Task 2 결과 확인 이후, 동일 규칙에서 ZiPS를 Steamer로 바꾼 S0–S3과 동일 참조 모델을 계산한다. 추가 feature·α grid·층화·표본 기준을 도입하지 않는다. 주 대조는 S1−S2, S2−S3, S0−S1이다. WAR/600도 같은 구조다.

최종 Marcel → Steamer → ZiPS 표에는 MAE(R1/S1/Z1), 증분 R1−R2 / S1−S2 / Z1−Z2와 CI를 나란히 놓는다. raw S0/Z0는 별도로 표시한다. 현재는 두 시스템의 평가·학습 표본이 완전히 동일하다. 향후 검증에서 차이가 발견되면 시스템 간 표에는 공통 교집합의 학습·평가 표본으로 재적합한 값만 사용하고 제외를 기록한다. 시스템별 고유 표본 결과는 별도 유지한다. 강도 순서가 예상과 달라도 그대로 보고한다.

## 10. 산출물과 재현 경로

기준 폴더는 `SSAC27/results/production_increment/`이다.

- `PRESPEC.md`, `prespec_inputs.json`: 승인된 명세와 해시. 사용자의 후속 지시로 커밋·태그 생략.
- `replication_2019_2025/`: A의 새 버전 결과와 기존 대비 표.
- `zips/`: Task 2의 성능·대조·층화·계수·튜닝·표본 흐름·민감도 CSV.
- `steamer/`: Task 3 확인 이후에만 생성하는 같은 표.
- `results.md`: 단계별 표와 각 표 아래 생성 스크립트·실행 명령. 모델 결과가 나오기 전에는 만들지 않는다.
- 각 실행에 input/code/protocol hash, 환경·seed·α와 실행 명령을 남긴다. 원본 및 선수별 projection 값이 포함된 진단 자료는 ignored `data/raw/` 내부에 둔다. 저장소용 표는 집계 결과로 제한한다.
- `data/README.md`와 `results/provenance/decisions_log.md`에 획득 방법·export 날짜 미상·파일별 행 수/해시·실행 및 변경을 기록한다.

구현할 진입점은 `SSAC27/scripts/production_increment.py`이며, 아래는 **구현 후 사용할 명령의 명세**다. 현재 실행 가능한 명령이라고 표시하지 않는다.

```bash
B2V_SCALE=zscore OPENBLAS_NUM_THREADS=1 python SSAC27/scripts/production_increment.py --stage replication-and-zips
```

이 명령은 `prespec_inputs.json`의 승인 상태 및 명세/해시와 현재 입력의 일치를 검사한 뒤 A와 ZiPS만 실행해야 한다. Steamer는 포함하지 않는다. 사용자 Task 2 확인 후 별도 `--stage steamer`로 실행한다. 단계별 보고 후 다음 단계 확인이라는 원 지시서를 자동화가 건너뛰지 않게 한다.
