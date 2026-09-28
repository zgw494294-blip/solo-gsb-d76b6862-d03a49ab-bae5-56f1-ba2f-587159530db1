# 潮汐调和拟合 API

基于 Python + FastAPI + NumPy + SciPy 的本地潮汐调和拟合服务。以首条观测为时间原点，对模型

```
h(t) = c + Σ_i [ a_i cos(2π t / P_i) + b_i sin(2π t / P_i) ]
```

使用全部观测最小化 Huber 损失（不预先剔除异常点），返回基准潮位 `c`、各周期系数 `a_i/b_i`、逐点残差和预测潮位。

## 环境要求

- Docker + Docker Compose（推荐，一键启动）
- 或 Python 3.11+

## 一键启动（Docker Compose）

```bash
docker compose up --build
```

服务监听 **8000** 端口。后台运行：

```bash
docker compose up -d --build
```

停止：

```bash
docker compose down
```

## 本地运行（不使用 Docker）

```bash
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## 接口

`POST /v1/tide/fit`

### 请求体（JSON）

| 字段 | 类型 | 约束 |
| --- | --- | --- |
| `observations` | 数组 | 20～500 条；每条含整数分钟 `t`（严格递增）与有限潮位 `h` |
| `periods` | 数组 | 1～4 个互异的正数周期 P（分钟） |
| `delta` | 数 | 正数 Huber 阈值 |
| `predictions` | 整数数组 | 可选，整数分钟预测时刻 |

### 请求示例

```bash
curl -s -X POST http://localhost:8000/v1/tide/fit \
  -H "Content-Type: application/json" \
  -d '{
    "observations": [
      {"t": 0, "h": 1.2}, {"t": 10, "h": 1.8}, {"t": 20, "h": 2.3},
      {"t": 30, "h": 2.1}, {"t": 40, "h": 1.5}, {"t": 50, "h": 0.8},
      {"t": 60, "h": 0.2}, {"t": 70, "h": -0.1}, {"t": 80, "h": 0.3},
      {"t": 90, "h": 1.0}, {"t": 100, "h": 1.7}, {"t": 110, "h": 2.2},
      {"t": 120, "h": 2.4}, {"t": 130, "h": 2.0}, {"t": 140, "h": 1.3},
      {"t": 150, "h": 0.6}, {"t": 160, "h": 0.0}, {"t": 170, "h": -0.2},
      {"t": 180, "h": 0.4}, {"t": 190, "h": 1.1}
    ],
    "periods": [120, 60],
    "delta": 1.0,
    "predictions": [200, 210, 220]
  }'
```

### 响应示例

```json
{
  "baseline": 1.12,
  "coefficients": [
    {"period": 120.0, "a": 0.95, "b": 0.31},
    {"period": 60.0, "a": 0.12, "b": -0.08}
  ],
  "residuals": [0.02, -0.01, 0.03, "..."],
  "predictions": [
    {"t": 200, "h": 1.85},
    {"t": 210, "h": 2.10},
    {"t": 220, "h": 2.25}
  ]
}
```

### 错误码

- `422 Unprocessable Entity`：输入非法（观测数量、`t` 非严格递增、`h` 非有限、周期非互异正数、`delta` 非正等），或设计矩阵不满列秩。

## 说明

- 以首条观测的时间 `t[0]` 为时间原点，预测时刻使用与观测相同的绝对时间坐标。
- Huber 损失通过 SciPy `least_squares(loss="huber", f_scale=delta)` 实现，`delta` 即 Huber 阈值；全部观测参与拟合，不剔除异常点。
- 设计矩阵列数为 `1 + 2×周期数`，当 `np.linalg.matrix_rank` 小于列数时返回 422。
