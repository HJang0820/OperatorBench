# OperatorBench

DeepONet과 Fourier Neural Operator(FNO)를 비교하기 위한
PDE operator-learning benchmark 프로젝트

먼저 2D Darcy 문제를 대상으로 데이터 생성, 전처리,
모델 학습 및 평가 과정을 구성하고, 이후 Burgers 문제까지 커버

## 현재 상태

기존 Jupyter 노트북에서 2D Darcy 문제에 대한 DeepONet과 FNO의
예비 실험을 수행했습니다.

현재는 기존 코드를 이 저장소로 이전하고,
각 기능을 재사용 가능한 Python 모듈로 분리하기 위한 초기 단계입니다.
아래 디렉터리 구조와 공통 평가 절차는 구현 계획이며,
아직 모두 구현된 상태는 아닙니다.

## 첫 번째 대상 문제: 2D Darcy

- 영역: 단위 정사각형
- 방정식: -div(a grad u) = 1
- 압력 경계조건: 전체 경계에서 u = 0
- 입력: a(x)
- 출력: u(x)
- 기존 계수장 생성: Neumann cosine basis 기반 GRF를 3/12로 이진화
- 기존 기준해 계산: 85×85 격자에서 flux-form 5-point FDM
- 비교 모델: DeepONet, FNO

기존 자체 생성 데이터와 공개 FNO 데이터의 완전한 동일성은
검증하지 않았습니다.

향후 기준해 계산 격자와 모델의 관측 격자를 분리하고,
421×421 기준해 생성 및 격자 세분화 검증을 진행할 계획입니다.

## 모듈화 설계

아래는 목표 구조입니다. 필요한 모듈부터 순차적으로 구현합니다.

```text
OperatorBench/
├── configs/                    # 데이터·모델·학습 설정
├── operatorbench/
│   ├── data_generation/
│   │   ├── fields.py           # 계수장 생성
│   │   └── darcy_fd.py         # Darcy 수치 solver
│   ├── data/
│   │   ├── datasets.py         # 저장 데이터 로딩
│   │   └── preprocessing.py    # 정규화·해상도 변환
│   ├── models/
│   │   ├── deeponet.py
│   │   └── fno.py
│   ├── training.py             # 학습·validation·checkpoint
│   └── evaluation.py           # 평가 지표·예측 결과 저장
├── scripts/                    # 데이터 생성·학습·평가 실행
├── notebooks/                  # 설명·실험 확인·시각화
├── docs/                       # 문헌조사·실험 기록·결과 그림
└── tests/                      # 수치해법·입출력·모듈 검증
```

## 설계 원칙

1. 데이터 생성과 모델 학습을 분리합니다.
   학습할 때마다 PDE를 다시 풀지 않고 저장된 데이터를 불러옵니다.

2. 원래 계수장, 해, 좌표와 생성 설정을 보존합니다.
   모델별 입력 형태로의 변환은 로딩 이후에 수행합니다.

3. 두 모델에 같은 데이터 분할과 물리적 입력 정보를 제공합니다.
   정규화 통계는 training data에서만 계산합니다.

4. Validation으로 checkpoint를 선택하고,
   test data는 최종 평가에 사용하도록 공통 절차를 구성합니다.

5. 기존 코드를 옮기는 작업과 수치해법·모델을 변경하는 작업을
   분리하여 결과 차이를 추적합니다.

## 구현 순서

- [x] 기존 노트북과 실험 설정 정리
- [ ] 계수장 생성 함수 분리
- [ ] Darcy 수치 solver 분리
- [ ] 모듈 분리 전후 결과 일치 확인
- [ ] 데이터 저장·로딩·전처리 분리
- [ ] DeepONet·FNO 모델 분리
- [ ] 공통 학습·평가 절차 구성
- [ ] 기준해 정확도 및 고해상도 데이터 생성 검증
- [ ] Burgers 문제 추가

## 참고

- Fourier Neural Operator:
  https://arxiv.org/abs/2010.08895
- PDEBench:
  https://github.com/pdebench/PDEBench