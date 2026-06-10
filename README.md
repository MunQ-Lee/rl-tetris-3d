# 3D Tetris — Reinforcement Learning (PPO)

강화학습(PPO) 에이전트가 **3차원 테트리스**를 스스로 플레이하도록 학습하고,
그 모습을 웹 브라우저(three.js)에서 실시간 3D로 확인할 수 있는 프로젝트입니다.

- **State**: 현재 보드에 쌓인 블록들 + 지금 떨어지는 블록(+ 다음 블록)
- **Action**: 블록 이동(±x, ±y) · 회전(x/y/z축) · 대기 · 하드드롭 (총 9개)
- **Reward**: 게임 스코어(레이어를 채우면 점수 획득) + 생존/형태 보조 보상

```
보드: 5 (x) × 5 (y) × 12 (z, 높이),  조각: 8종 tetracube
```

## 구성

| 파일 | 설명 |
|------|------|
| `tetris3d/pieces.py` | 3D 조각(tetracube) 정의 + 90° 회전 |
| `tetris3d/env.py`    | 3D 테트리스 환경 (충돌/중력/레이어 클리어/스코어) |
| `tetris3d/ppo.py`    | PPO(clipped objective + GAE) + Actor-Critic MLP |
| `train.py`           | 학습 루프, 체크포인트 저장 |
| `server.py`          | Flask 서버 — 에이전트가 플레이하는 상태를 스트리밍 |
| `web/index.html`     | three.js 실시간 3D 뷰어 |

## 설치

```bash
pip install -r requirements.txt
```

## 1) 학습

```bash
python train.py --updates 1000 --rollout 2048
```

- `checkpoints/latest.pt` (주기적), `checkpoints/best.pt` (최고 평균 스코어)가 저장됩니다.
- 학습 로그는 `checkpoints/train_log.jsonl`에 기록됩니다.

## 2) GUI로 플레이 보기

```bash
python server.py --ckpt checkpoints/latest.pt --speed 8
```

브라우저에서 **http://localhost:8000** 접속.
- 마우스 드래그로 시점 회전, 휠로 줌.
- 체크포인트가 없으면 랜덤 정책으로 플레이하며, 학습 중에는 체크포인트를
  자동으로 핫리로드하므로 **학습하면서 실시간으로 실력이 느는 모습**을 볼 수 있습니다.

> WSL 사용 시 Windows 브라우저에서 `http://localhost:8000`으로 바로 접속됩니다.

## 작동 방식

매 스텝마다 에이전트가 이동/회전/드롭 중 하나를 선택하고, 그 뒤 중력이 블록을
한 칸 끌어내립니다. 블록이 더 내려갈 수 없으면 고정되고, 가득 찬 수평 레이어가
사라지며 점수를 얻습니다. 보드가 가득 차 새 블록이 못 나오면 게임 오버입니다.

보상은 게임 스코어를 주축으로 하며, CPU에서도 학습이 수렴하도록 누적 높이·구멍·
표면 거칠기에 대한 잠재함수 기반(potential-based) 보조 보상을 더했습니다
(`tetris3d/env.py`의 `W_HEIGHT`, `W_HOLES`, `W_BUMPY`로 조절).
