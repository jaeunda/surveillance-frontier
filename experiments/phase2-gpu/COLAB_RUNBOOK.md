# Phase 2-1 Google Colab 실행 절차

2026-10-12, 측정 전. Phase 2-1은 [프로토콜 Part I](README.md#part-i-pilot-on-an-available-machine-p0p2)의
P0–P2 파일럿이다. **로컬 회귀 검증 완료와 실제 Colab/CUDA 검증을 구별한다.** 이 준비 작업에서는 실제 Colab
할당·CUDA gate·GPU 성능을 측정하지 않았다. 첫 할당에서 아래 P0를 통과한 경로만 측정한다.
T1/T2의 셀·trial·정확도, stream, held-out 보호, H3/H4 판정 조건은 그대로 유지한다.

## 1. 실행할 코드와 노트북 고정

[`pilot_colab.ipynb`](pilot_colab.ipynb)는 [`pilot_colab.py`](pilot_colab.py)에서 생성한다. 같은 source revision의
노트북과 실행기를 사용한다. 로컬 미커밋 변경은 원격 기본 브랜치를 clone해도 포함되지 않는다.

다음 중 한 가지를 준비한다.

- **원격 commit**: 검증한 변경을 commit·push한 뒤 전체 40자리 SHA를 `REPO_REF`에 넣는다.
  `REPO_URL`은 접근 가능한 저장소 주소, `SOURCE_BUNDLE`은 빈 문자열이다.
- **배포용 git bundle**: 준비된 `phase2-1-source.bundle`과 같은 묶음의 노트북을 업로드한다.
  Colab 파일 창에 bundle을 올리고 `SOURCE_BUNDLE = "/content/phase2-1-source.bundle"`,
  `REPO_REF = SOURCE_COMMIT.txt에 적힌 40자리 SHA`로 설정한다. bundle에는 추적된 코드·Phase 1 기준자료와
  이번 준비 변경이 들어 있으며, 내부 검토 문서·인증정보·로컬 측정 결과는 포함하지 않는다.

새 계획은 notebook이 **로컬 git commit으로 저장**하고 `_colab/plan.bundle`로 Drive에 보관한다.
그 commit은 source commit에 `pilot-plan.json`만 추가한 것이다. 재개는 이 계획 commit을 복구한다.
GitHub 인증이 필요한 경우 notebook 실행 전에 인증된 방법으로 clone하며 토큰을 셀에 적지 않는다.

## 2. 설정 및 환경

Colab에서 GPU runtime을 선택하고 notebook을 연다. GPU 모델을 미리 가정하지 않는다.
최초 `RUN_MEASUREMENTS = False`를 유지한다. Drive 연결은 Colab UI에서 사람이 승인한다.

| 설정 | 기본값 및 의미 |
|---|---|
| `LABEL` | 새 실험마다 고유한 이름. 같은 실험의 두 세션은 같은 label |
| `SESSION`, `PLANNED_SESSIONS` | 첫 실행 1, 계획 2. 두 번째 세션은 다른 시간대에 실행 |
| `MODE` | `full`; `short`는 1세션, T-ref build 제외라는 사전 축소 |
| `AVAILABLE_HOURS`, `SESSION_HOURS` | 전체 16 h, 세션당 8 h. 완료 예상 시간이 아니라 검사·단계의 계산 한도 |
| `STEP_HOURS` | P0 `check=1`, CPU baseline 1.5, GPU tasks 2, parity 1.5 h |
| `RUN_CAP_MINUTES` | 고정 작업당 15분. 초과하면 검열; trial 수를 줄이지 않음 |
| `CUT_REASON` | cap·단계 몫·세션 축소 이유. 실행 전에 계획에 기록 |
| `USE_DRIVE`, `DRIVE_FOLDER` | 측정에는 `True` 필수. 기본 `MyDrive/phase2-pilot` |
| `RETRY_STEPS`, `RERUN_CHECKS` | 기본 `[]`, `False`. 원인 확인 후 명시적으로 재시도 |
| `AMEND` | 기본 `None`. 이미 만든 계획은 덮어쓰지 않고 변경 이유와 값을 추가 기록 |

노트북 2–3절은 다음을 확인하거나 보존한다.

- **Python ≥3.12**, 격리된 `.venv`, [고정 의존성](requirements-colab.lock)의 wheel 설치, `pip check` 및
  실제 `pip freeze`/설치 report. 직접 의존성 버전은 [기존 requirements](requirements.txt)와 같다.
  호환 wheel이 없으면 멈춘다. 버전을 임의로 내리거나 Colab kernel의 사전 설치 패키지를 대신 쓰지 않는다.
  Python 하한은 [NumPy](https://pypi.org/project/numpy/2.5.3/)와
  [SciPy](https://pypi.org/project/scipy/1.18.1/) 메타데이터에 따른다.
- `nvidia-smi`, `nvcc`, `cmake`, C++/OpenMP compiler. GPU의 실제 compute capability 하나를 대상으로
  Release, `SF_CUDA=ON`, SASS 빌드를 한다. toolkit/compiler 불일치는 빌드 실패로 처리한다.
  `--use_fast_math`나 호환성 검사 무시 옵션을 추가하지 않는다.
- 1분 quarter·1초 week 입력 다운로드, 124개 task 재생성 대조, Python 회귀 검사.
  입력 바이트, 실제 `sf_*` 바이너리, `CMakeCache.txt`, lock 파일을 `_colab/reproduction/<실행 ID>/`에 복사한다.
  재할당 후에는 이 입력 사본을 plan의 SHA-256과 대조해 복원하므로 다시 내려받은 데이터로 바뀌지 않는다.
- CPU affinity·quota·GPU·RAM·driver·compiler·Python·binary/input/code hash를 기록한다.
  setup 명령은 각각 최대 30분이며 실패·timeout 로그는 `_operations/<LABEL>/`에도 남긴다.

설치나 빌드가 실패하면 해당 로그를 보존한 채 원인을 해결한다. P0 gate를 생략하여 우회하지 않는다.
동일 GPU 이름이어도 **VM boot ID가 바뀌면 새 세션**이다. 같은 VM에서 허용 CPU 집합·quota가 바뀌어도 새 세션으로 다룬다.

## 3. P0 사전 검증 후 진행 판단

첫 실행은 1–5절을 실행한다. `Run all`을 눌러도 기본 설정에서는 측정 없이 P0·요약·아카이브만 만든다.
4절의 계획 commit과 Drive checksum 확인이 끝난 뒤 P0를 시작한다.

`session-K/checks/attempt-N/`에서 다음을 확인한다.

1. `checks.json`의 `cpu_pass == true`. CPU regression, stream, intervals, 합성 harness 검증,
   CPU gate, Phase 1의 18개 원본 hit 파일 재현이 통과해야 한다.
2. GPU 경로별 `execution == "device"`, `pass == true`, `validated_paths` 포함 여부.
   `gate_cuda_inc.json`, `gate_cuda_hyb.json`과 `checks.log`를 직접 확인한다. 빌드 성공이나 host emulation은
   CUDA 증거가 아니다. 한 경로만 통과하면 해당 경로만 측정하고 나머지 실패를 함께 보고한다.
3. `items.jsonl`의 시작·종료·timeout, `env.json`, `cpu-setup.json`, 입력·코드·binary hash가 현재 세션과 일치한다.
4. `colab_ops.py audit` 결과 `ok: true`, Drive의 `_colab/checkpoints/`에 새 검증 영수증이 있다.

P0의 1 h는 안전한 최초 한도이며 통과 시간 보장이 아니다. 소진되면 원기록을 남기고 중단한다.
필요할 경우 **다음 검사 전에** 예산을 다시 검토하고, 첫 셀의 `STEP_HOURS["check"]`와
`AMEND = {"reason": "P0 실측 근거 ...", "set": {"step_hours.check": 새_시간}}`를 함께 설정한다.
총 16 h 및 세션 8 h는 자동으로 늘지 않는다. CPU 실패는 측정 금지, GPU gate 실패는 해당 경로 측정 금지다.

P0가 준비되면 `RUN_MEASUREMENTS = True`로 설정하고 **6절부터** 실행한다. 같은 VM에서 1–5절 전체를 다시
실행해도 통과한 P0는 하드웨어·코드·binary가 같고 산출물이 온전할 때 재사용한다.

## 4. 측정 순서 및 해석

실행 순서는 CPU baseline → GPU rates → GPU tasks → threads → microbatch → rates → diag → T-ref rates →
deadline → parity → T-ref build이다. 각 단계 사이 Drive 복사·검증은 측정 E2E 밖에서 수행한다.

- CPU/GPU fixed tasks는 초기 설정에서 cold 3 + priming 1 + warm 3을 시도한다. 같은 조건이 cap을 넘으면
  남은 반복도 `not started (run cap)`으로 남는다. 단계 `complete`는 계획된 run이 모두 **기록되었다**는 뜻이다.
  성능 비교의 성공 여부는 개별 run과 조건별 completion을 확인한다.
- CPU 단계의 1.5 h 몫이 소진되어도 세션·전체 시간이 남으면 GPU 단계로 진행한다.
  세션·전체 한도가 소진되면 이후 단계는 실행하지 않는다. 실패를 완료로 바꾸거나 기록을 삭제하지 않는다.
- `rates`는 fixed-task 관측과 같은 설정도 측정한다. 같은 세션·설정의 uncensored warm 관측이 없으면
  projection은 `unverified`다. T-ref build는 관측 검증과 보수적 소요시간 조건이 충족될 때만 시작한다.
  늦은 parity 채택으로 앞선 T-ref rate 설정을 바꾸지 않는다. oracle 실패는 build 성공이 아니다.
- 검열 조건의 성공한 run만으로 E2E·speedup·best를 산출하지 않는다. GPU 전체 CPU hash가 없으면
  256-trial spot-only 탐색 측정이다. 정확성 불일치는 GPU 경로를 제외하며 CPU 불일치는 세션을 중단한다.
- full-week T-ref를 못 끝내도 유효한 파일럿 부분 결과다. prefix deadline 결과는 H4가 아니며,
  파일럿 반복은 정식 독립 launch·freeze·H3/H4를 대체하지 않는다.

설치·빌드·다운로드·백업 비용과 idle은 `pilot-summary`의 검사·단계 시간과 분리한다.
`_colab/operations/`의 명령별 경과시간·로그, backup 이벤트, `occupancy.json`의 VM uptime을 함께 보고한다.
uptime은 VM 나이이며 과금 청구서가 아니다. 미종결 attempt의 기존 heartbeat 기반 시간은 장애 구간의 실제
소모량을 확정하지 못한다. 무료/유료 여부, 실제 CU·요금·시작/종료 화면은 운영자가 별도로 보존한다.
유료 전환은 사전 비용 기록과 프로젝트 cap 검토가 필요한 별도 계획 변경이다.

## 5. 중단과 재개

**사람이 중단할 때:** Colab의 실행 중지 버튼을 한 번 누르고 프로세스 종료와 `finally` 백업을 기다린다.
로그를 확인한 뒤 7절 요약과 8절 아카이브를 실행한다. 백업·아카이브 checksum을 확인하기 전 VM을 삭제하지 않는다.
요약이 실패해도 8절은 별도로 실행해 원자료를 보존할 수 있다.

**같은 VM에서 kernel만 재시작했을 때:** 같은 `LABEL`, `SESSION`, 동일한 effective plan 설정으로 1절부터 실행한다.
정상 완료 단계는 원본·검사 manifest와 fixed-task 반복 기록을 검증한 뒤 건너뛴다.
실패·미종결 단계는 자동 재시도하지 않는다. 로그 확인 후 `RETRY_STEPS = ["단계명"]`으로 지정한다.
재시도는 **해당 단계 전체를 새 attempt로 실행**하며 이전 실패·시간을 유지한다. 시행 중 run의 trial checkpoint 재개가 아니다.
이미 단계 몫을 소진했다면 retry만으로 시간이 늘지 않는다. 변경이 필요하면 사전 amendment를 남긴다.

**VM 소실·재할당:** 같은 label의 다음 `SESSION`을 사용한다. Drive의 계획 bundle로 코드와 기록을 복구한다.
P0·CPU 설정·선택·각 단계를 새 머신에서 다시 실행한다. 세 번째 할당이면 첫 셀을 다음처럼 설정한다.

```python
SESSION = 3
PLANNED_SESSIONS = 3
AMEND = {"reason": "VM 재할당: session 3, 기존 전체 시간 한도 유지", "set": {"sessions": 3}}
RUN_MEASUREMENTS = False
```

완료 출력이 없거나 hash가 다르면 `artifact_missing`, terminal record가 없는 시작은 `lost`다.
손상된 JSONL을 잘라내거나 `ok`를 수동으로 만들지 않는다. 복구 검사가 멈추면 원본을 복제해 조사하고,
일치하는 백업/아카이브에서 복원한다. 복원할 수 없는 원자료는 누락으로 보고하며 성공 지표에 포함하지 않는다.

## 6. 영구 보존과 독립 복구 확인

실행 중 JSONL과 완료 run 출력은 Drive로 mirror된다. 단계 사이 전체 백업은 임시 파일로 복사하고
SHA-256 readback이 일치해야 교체한다. 실패하면 다음 측정을 시작하지 않는다.
진행 중이던 작업의 마지막 stdout이나 Drive에 아직 전달되지 않은 tail까지 보존한다고 보장할 수는 없다.

종료 시 8절을 실행한다. Drive `_archives/<LABEL>/`에 새로운 timestamp의 `.tar.gz`와 `.sha256`가 생긴다.
기존 아카이브는 덮어쓰지 않는다. `DOWNLOAD_RESULTS=True`로 두 파일을 다른 저장장치에도 보관한다.
Drive의 quota·I/O 오류·세션 제한은 실제로 발생할 수 있으므로 readback 성공을 영구 내구성 보장으로 해석하지 않는다.
[Colab 공식 안내](https://research.google.com/colaboratory/faq.html#resource-limits)를 참조한다.

아카이브에는 계획·amendments·source/plan bundle, 환경·입력 바이트·바이너리·의존성 lock/freeze,
P0/gate/spot/oracle 출력, 실행 순서, 성공·실패·중단 원본 및 요약·그림이 포함된다. setup 단계에서 중단됐다면
Drive `_operations/<LABEL>/`의 로그를 별도로 보존한다.

다운로드한 뒤 저장소에서 **빈 복구 디렉터리**로 추출하여 확인한다(실제 파일명으로 대체).

```bash
python experiments/phase2-gpu/colab_ops.py verify /archive/LABEL-TIMESTAMP.tar.gz
mkdir /tmp/phase2-restore
tar -xzf /archive/LABEL-TIMESTAMP.tar.gz -C /tmp/phase2-restore
python experiments/phase2-gpu/colab_ops.py audit /tmp/phase2-restore/LABEL
git bundle verify /tmp/phase2-restore/LABEL/_colab/plan.bundle
```

`verify`는 archive 전체 SHA-256을, `audit`는 raw 기록과 gate/run 산출물을 검사한다. 별도 머신에서 bundle로 clone하여
환경을 다시 만들고 원래 입력을 복원할 수 있다. 보존한 실행 바이너리는 해당 CPU/GPU 아키텍처용 증거이며,
다른 머신에서는 그 머신에 맞게 다시 빌드하고 새 세션 P0를 수행한다.

## 7. 성공·실패 판정과 인계

| 판정 | 필요한 증거와 다음 행동 |
|---|---|
| 환경 준비 성공 | 고정 source, 의존성 설치/검사, CUDA 빌드, 입력·124 task 대조, 계획 bundle과 검증 백업 |
| 측정 진입 가능 | CPU P0 통과, 사용 GPU path의 실제 device gate 통과, 같은 환경·코드·binary, artifact audit 통과 |
| 고정 조건의 관측 E2E 유효 | 계획된 모든 반복 `ok`, 원본 있음, spot/hash 근거 명시, 같은 session/config/cold kind. spot-only는 exploratory |
| 정상적인 부분 종료 | cap/time box, 경로 unavailable, projection unverified 또는 시간 부족. 실패/미시작 분모와 이유를 포함해 보존 |
| 중단 후 조사 필요 | CPU/gate/spot/identity 불일치, 예외, JSONL 손상, checksum/Drive 실패, artifact 누락. 원본 보존 후 원인 해결 |
| 세션 인계 완료 | 요약과 원기록 대조, 누락·실패·재시도 포함, archive 및 checksum의 독립 복사/검증, 다음 세션·P3 미확정 사항 기록 |

인계 기록에는 label, source/plan commit, session과 GPU/CPU, P0 경로별 결과, 단계별 종료 이유,
planned/started/ok/censored/lost/not-started, 정확성 근거, 검증된/미검증 projection,
계산 시간·VM 점유 관측·실제 비용, Drive archive 경로와 SHA-256을 적는다.
"GPU가 빨랐다", "CPU 증설이 필요하다", H3/H4 성공은 이 운영 절차의 성공 기준이 아니다.
