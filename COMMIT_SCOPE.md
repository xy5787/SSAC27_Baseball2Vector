# 이번 커밋 범위

대상 저장소는 이 폴더(`release/ssac27/`) 안의 독립 Git 저장소입니다. 부모 연구 저장소에서 `git add .`를 실행하지 않습니다.

포함할 파일은 `RELEASE_MANIFEST.json`의 분석 허용 목록과 manifest 자체뿐입니다. 최신 원자료 취득 recipe·해시·스키마, 독립 재현 코드, 분석 프로토콜, 집계 결과, 검증 코드가 포함됩니다. 원자료·선수별 코호트/예측/관측값, 최신 abstract/paper와 그림·표, 내부 검토 문서는 포함하지 않습니다. 원격에서 삭제한 과거 `abstract/` 폴더는 복원하지 않습니다. 최신 원고도 추가하지 않습니다. Git ignore는 기존 추적 파일을 보호하지 않으므로 아래 allowlist 검사를 사용합니다.

이 저장소에서 실행합니다:

```bash
python scripts/check_release.py
python -m pytest -q tests
python scripts/check_release.py --stage
git diff --cached --check
git diff --cached --stat
git diff --cached --name-only
```

`--stage`는 허용된 파일만 stage합니다. 다른 파일이 이미 stage되어 있으면 자동으로 해제하지 않고 중단합니다. commit과 push는 수행하지 않습니다. 이후 사용자가 검토하고 커밋할 때 사용할 메시지 예: `Add reproducible ZiPS and Steamer increment analyses`.

재현 실행: `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python scripts/reproduce.py`. 새 실험 설계를 추가한 것이 아니라 기존 결과와 같은 수치 루틴을 독립 경로에 배치한 것입니다. 원자료가 없는 환경에서는 데이터 의존 테스트가 명시적으로 skip됩니다. 모든 결과 검증을 실행하려면 recipe에 따라 정확한 로컬 파일을 준비하고 전체 재현을 먼저 실행합니다.

배포 파일만 별도 경로로 옮길 경우 manifest의 파일 목록과 manifest 자체를 사용합니다. 원고나 부모 저장소 이력은 필요하지 않습니다. `scripts/build_submission_assets.py`, `scripts/task9_full_pca.py`는 과거 진입점으로, 이번 분석 allowlist와 재현 경로에 포함되지 않습니다. `abstract/`는 원격에서 삭제된 상태를 유지합니다.
