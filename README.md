# AFD E2E

通过标准 `vllm serve` 启动服务、从外部进行请求测试和观测的 NPU 工具。
可将 `afd-e2e` 和 `afd-plugin` 放在同级目录，使用已有推理环境执行；
也可直接测试环境中已安装的插件，不要求存在源码 checkout。

```text
work/
├── afd-plugin/
└── afd-e2e/
    ├── run.py
    └── afd_e2e/
```

## 环境与安装

使用 Python 3.10+，以及已经安装 CANN、torch-npu、vLLM、vLLM-Ascend 和 `afd-plugin` 的推理环境；
所选部署需要的 CAM / Layered 算子和自定义算子路径须由推理环境准备。
工具不导入插件内部模块、不修改源码、不注入 worker extension 或 monkey patch。
原生 `--baseline` 只需要现有 vLLM-Ascend 推理环境，不要求安装 `afd-plugin` 或 CAM 算子。
只查看启动方案无需安装推理或评测依赖：

```bash
cd afd-e2e
python run.py plan --device 0,1,2,3
```

运行 smoke 或性能测试需要 `httpx`（`python -m pip install httpx`）；完整评测安装：

```bash
python -m pip install -e '.[eval]'
```

评测依赖采用已有 E2E 的 `lm_eval==0.4.12` 和官方 GSM8K 任务。
安装在已验证的推理环境中，确认依赖解析不会替换固定版本的 torch / vLLM。
`python run.py`、`python -m afd_e2e` 和安装后的 `afd-e2e` 是同一个入口。
从任意工作目录调用源目录中的 `run.py`，默认仍寻找与工具同级的插件。
`--repo` 可指定待测 checkout 的顶层 Python 路径和 Git 元数据；该目录不是运行前置条件。
示例中的 `V4_MODEL` 和 `GSM8K_DATASET` 环境变量分别表示 V4 模型标识或权重位置、离线数据目录，
使用对应示例前设置。

## 命令与版本依赖边界

所有测试共用外部接口，工具不依赖 AFD runner、connector、ubatching 或算子的 Python 接口。
部署模板生成的 serve 参数仍需匹配目标环境，`plan` 成功不代表目标版本支持这些参数。

| 命令 | 是否需要上游推理环境 | 依赖边界 |
| --- | --- | --- |
| `--help` / `<command> --help`、`plan`、`status`、`pack` | 否 | Python 标准库；生成命令或读取、打包报告 |
| `python -m afd_e2e.performance --config ...` | 客户端无需安装 | 已有服务的 HTTP/SSE 协议及 `httpx` |
| `python -m afd_e2e.workloads --config ... --kind smoke --output ...` | 客户端无需安装 | 已有服务的 completions/chat API 及 `httpx` |
| `run --baseline` | 服务端需要，不需要 afd-plugin | 原生 `vllm serve` 参数、HTTP、日志和进程生命周期 |
| AFD `run --test smoke` / `run --test perf` | 服务端需要，包含 afd-plugin | A/F 的 `vllm serve` 启动参数与环境；测试端使用 HTTP、日志及可选 `/metrics` |
| AFD `run --test gsm8k` / `run --test all` | 服务端需要，包含 afd-plugin | 上述部署与外部接口，加固定 lm-eval/GSM8K 评测依赖 |

HTTP 客户端可在另一 Python 环境运行，但服务端仍有自身的版本要求。
性能接口需要 `/tokenize`、`/v1/completions`、完整 SSE、准确 token usage 及所用
请求参数支持；原生多 DP 路由还要求服务端接受 `X-data-parallel-rank`。
因此客户端不依赖源码版本，仍需服务端满足这些 HTTP 契约。

`run` 不硬编码允许的 vLLM/Ascend/AFD commit 列表，也不会自动安装或升级它们。
同级或 `--repo` 指定的 checkout 的 Git commit/工作树状态写入
`manifest.json` 的 `afd_plugin`；缺少 checkout 时记为 null。
目标 `--python` 环境中已安装的包版本通过 `importlib.metadata` 写入
`readiness.versions`，缺包记为 null，不导入推理包，也不核对内部源码位置。
Git 元数据描述所选 checkout，包版本描述所选 Python 环境，二者不构成实际导入来源证明。

适配不同版本时调整部署模板的 serve 参数和运行环境，或使用下述附加参数覆盖；
HTTP 测试、性能计算和验收规则无需跟随私有类名、字段或算子接口变化。
MRV、DBO、Graph 和 Layered 的实际支持情况由目标服务验证。
这些功能的配置会保存，但 HTTP 成功不能证明某个内部 executor 或 kernel 实际执行。

直接 HTTP 发压复用已生成的 `resolved-config.json`，要求目标服务已启动且模型名、
地址及域映射与配置一致。性能客户端在配置所在目录创建新结果，不覆盖已有
`performance/`，不执行 `run` 的预检、服务启动、观测快照或服务清理。

## 设备管理与启动

支持 `npu run` 申请设备后启动，以及 `--device` 使用已分配设备两种方式。
AFD 设备列表按顺序切分给 Attention 和 FFN：V2 的 2A1F 需要 3 张，2A2F 需要
4 张；V4 AFD 需要 16 张。原生 baseline 按实例顺序切分，每实例使用 DP × TP
张设备，合计为所有实例之和。超出拓扑所需的设备不使用。

### 使用 npu run

适用于通过 `npu`（`canhazgpu`）管理设备的环境。先查看设备状态和队列：

```bash
npu status
npu queue
npu schedule
```

在工具目录中先核对启动方案，再申请设备执行测试：

```bash
# Inspect the plan without reserving devices or starting services.
python run.py plan --topology 2a2f --test smoke

# Reserve four devices and run a smoke test.
npu run --gpus 4 --nonblock --note "AFD V2 2A2F smoke" -- \
  python run.py run --topology 2a2f --test smoke
```

`npu run` 先申请设备，再设置 `ASCEND_RT_VISIBLE_DEVICES` 并运行命令；
工具自动继承该列表，无需传入 `--device`。

`plan --output DIR` 保存 `DIR/plan.json`，可重新生成计划，随后用同一个
`run --output DIR` 执行。仅含普通 `plan.json` 文件的目录可复用；已有运行结果、
其他文件或符号链接计划均拒绝覆盖。`run` 使用本次命令参数，不自动从计划恢复参数。

```bash
python run.py plan --device 0,1,2,3 --test smoke --output runs/smoke-check
python run.py run  --device 0,1,2,3 --test smoke --output runs/smoke-check
```

- `--gpus N` 自动分配 N 张设备，按用户最近使用记录选卡；申请拓扑所需的最少数量。
  只有需要固定物理设备时才使用 `--gpu-ids`，并先核对当前状态；任一指定设备不可用时，整组无法申请。
- `--nonblock` 在设备不可用时立即失败；需要排队时可改为 `--wait 5m`。
  不指定两者时默认无限等待；申请失败后重新查看 `npu status` 和 `npu queue`。
- `--wait` 限制排队时间，`--timeout 30m` 限制执行时间；`--note` 标记任务用途。
  `--` 必须保留，用来分隔设备管理选项和测试命令。
- 默认连续 30 分钟未检测到持有者使用设备时自动释放；确需长时间空闲持有时使用
  `--idle-timeout <duration>`，`--idle-timeout 0` 关闭空闲释放。

`npu run` 在命令运行期间维护租约心跳，命令退出后释放设备。测试及其所有设备工作进程
必须留在该命令生命周期内，不要分离到后台，也不要手工覆盖分配的可见设备范围。
`plan` 直接运行即可；没有实际设备列表时，展示占位编号并标记
`device_assignment_pending=true`，不代表已分配设备。

### 使用 --device

适用于已获得设备使用权的环境。通过 `--device` / `--devices` 指定设备列表；
若同时继承了 `ASCEND_RT_VISIBLE_DEVICES`，显式编号必须在该可见范围内。
指定编号仅选择设备，不负责申请或释放租约。以下编号为示例，须替换为实际分配的设备：

```bash
python run.py plan --device 0,1,2,3 --topology 2a2f --test smoke
python run.py run --device 0,1,2,3 --topology 2a2f --test smoke
```

## 测试配置

### 通用参数

| 参数 | 选择 / 默认 |
| --- | --- |
| `--repo` | 待测 `afd-plugin` 路径，默认优先寻找工具同级目录 |
| `--comm` | `sync` / `async`，默认 sync |
| `--model` | `v2-lite` / `v4-flash`，默认 V2；V4 必须指定权重路径 |
| `--topology` | AFD：V2 `2a1f` / `2a2f`；V4 `dp2tp4-ep8`。baseline：`dpN` 或 `dpNtpM`，多个实例用逗号分隔 |
| `--mrv` | `mrv1` / `mrv2`，默认 mrv1，应用于双侧 |
| `--ffn-mrv` | 可单独覆盖 FFN 的 MRV 选择 |
| `--dbo` | 原生 vLLM DBO：`on` / `off`，默认 off |
| `--layered-gmm` | `on` / `off`，默认 off |
| `--moe-ubatching` | Async CAM 两段 token 拆分：`on` / `off`，异步默认 on |
| `--execution` | `eager` / `graph`，默认 eager |
| `--limit` | GSM8K 条数 `1..1319`，默认 30；`all` 为 1319 |
| `--test` | `smoke` / `gsm8k` / `all` / `perf`，可用逗号组合，如 `all,perf`；AFD 默认 all（smoke + GSM8K），baseline 默认 perf |
| `--device` / `--devices` | 逗号分隔的设备编号；也可继承 `ASCEND_RT_VISIBLE_DEVICES` |

### 功能配置与验收边界

MRV、DBO、Graph、Layered GMM 和 Async CAM 的组合通过 serve 参数及启动环境表达，
工具不复制某个插件版本的内部功能限制。目标版本拒绝配置、服务退出或日志报错时，测试失败。
`plan` 中的这些配置表示请求的部署模板，不是已验证的内部执行路径。

`--moe-ubatching` 仅配置异步 CAM 模板，`--dbo` 对应原生 DBO，原生 baseline 也可使用。
比较 DBO on / off 时应保持其他启动条件一致；例如明确设置 `--moe-ubatching off`。
工具不为 DBO 额外添加覆盖负载，性能请求数和 GSM8K `--limit` 保持用户指定值。
`--baseline` 使用原生 vLLM-Ascend，支持下面的 DP 实例组合，默认进行性能测试。

### 原生 DP 性能基线

复用当前环境和依赖版本，以 `--baseline` 启用；服务不注册 AFD 插件，不传 AFD
角色配置或 worker extension。支持一个 DP-N 实例和多个不同大小的 DP 实例。
OMP 默认 8 线程，也可继承调用方的 `OMP_NUM_THREADS`。
`dpN` 的默认 TP 为 V2 Lite 的 1、V4 Flash 的 4；也可用 `dpNtpM` 显式指定。
默认拓扑分别为 `dp4tp1` 和 `dp2tp4`。原生 MoE 服务沿用 `--enable-expert-parallel`。

```bash
# Single DP4 instance, TP1: four devices.
python run.py plan --baseline --topology dp4

# Two DP2/TP4 instances: sixteen devices, matching the two-EP8 HTTP layout.
python run.py plan --baseline --model v4-flash --model-path "$V4_MODEL" \
  --topology dp2,dp2

# Four DP1/TP4 instances: sixteen devices, matching the four-EP4 HTTP layout.
python run.py plan --baseline --model v4-flash --model-path "$V4_MODEL" \
  --topology dp1,dp1,dp1,dp1

# Mixed DP1/TP4 and DP2/TP4 instances: twelve devices.
python run.py plan --baseline --model v4-flash --model-path "$V4_MODEL" \
  --topology dp1tp4,dp2tp4

# Reserve the devices for the complete performance run.
npu run --gpus 16 --nonblock -- \
  python run.py run --baseline --model v4-flash --model-path "$V4_MODEL" \
  --topology dp2,dp2 --perf-concurrencies 1,8,16
```

各实例使用同一模型、批量设置和性能负载。第 i 个实例监听 `api-port + i`；
DP RPC 端口从 `api-port + 实例数` 起分配。启动检查等待全部实例就绪。
请求按序号轮询全局 DP 域列表，选择实例 URL，并设置实例内的
`X-data-parallel-rank`；当前性能负载是独立单轮请求。全局并发限制沿用性能测试设置。

例如 `dp1,dp3` 的四个域映射为：

| 域 | HTTP 端口 | 实例内 DP rank |
| --- | --- | --- |
| 0 | api-port | 0 |
| 1 | api-port + 1 | 0 |
| 2 | api-port + 1 | 1 |
| 3 | api-port + 1 | 2 |

`plan.json` 和运行 manifest 保存域映射，逐请求性能记录保存实际选择的 route。
预热数大于 0 时，实际预热请求数至少覆盖全部 DP 域，并记录在每个性能点的
`warmup_requests`；设置 `--perf-warmup 0` 可关闭预热。
基线可显式运行 `--test smoke`；本次没有扩展 GSM8K 的多实例路由，baseline 不接受
`--test gsm8k` 或 `--test all`。

### 网络、内存与批量校准

网卡与通信用 IP 默认沿用环境或从默认路由检测；
有多网卡时使用 `--nic eth0 --afd-host 10.0.0.1`。API / AFD 端口可分别通过
`--api-port`、`--afd-port` 修改。已有 CANN 的 `PYTHONPATH` 和自定义算子库路径保留。

内存与批量配置可用 `--max-model-len`、`--max-batched-tokens`、`--max-seqs`、
`--memory-utilization` 校准；graph 的 `--capture-sizes` 默认 `1,2,4,8,16`。
其他 vLLM 参数按单个 token 传入，例如 `--vllm-arg=--block-size --vllm-arg=128`，
也可只给 Attention / FFN 传 `--attention-arg` / `--ffn-arg`。
同名附加参数替换模板参数及其值，布尔选项同时移除相反的 `--no-...` / `--...`；
角色参数优先于公共参数。`--additional-config` 整体替换 JSON，须包含该角色所需的完整配置。
`--omit-vllm-arg=--flag` 移除两侧模板中的选项及其值，适用于目标版本没有该参数的情况。
随后显式传入的参数仍会加入命令。
插件入口等启动环境可用 `--serve-env NAME=VALUE` 覆盖模板；其他环境变量沿用调用方。
设备可见性、进程归属标记和工具 Python 路径由工具管理，不能覆盖。
HTTP host/port、served model、DP/TP、DP RPC 端口及负载长度/内存配置仍由专用参数管理，
保证设备分配和请求配置一致；支持情况交给目标 `vllm serve` 判定。

```bash
# Replace a version-specific compilation preset and remove an unavailable CLI option.
python run.py plan --execution graph \
  --vllm-arg=--compilation-config '--vllm-arg={"mode":0}' \
  --omit-vllm-arg=--shutdown-timeout

# Supply complete role-specific AFD JSON for the version being tested.
python run.py plan \
  --attention-arg=--additional-config --attention-arg="$ATTENTION_CONFIG" \
  --ffn-arg=--additional-config --ffn-arg="$FFN_CONFIG"
```

`ATTENTION_CONFIG` / `FFN_CONFIG` 为目标版本接受的完整 JSON 字符串。
`manifest.services` 和 `commands.json` 保存实际生效的命令；功能开关字段描述原始模板，
存在覆盖时以实际命令为准。自定义算子路径沿用调用方环境，不猜测插件源码内的 vendor 路径。

## 测试用例

### 正确性测试

默认 `--test all` 先运行 smoke，再运行 GSM8K；也可分别选择 `smoke` 或 `gsm8k`。
`--test all,perf` 在同一次服务启动中依次运行 smoke、GSM8K 和性能测试，结束后统一关闭。
GSM8K 仅因精度门槛失败时继续测性能，最终仍返回失败；请求、服务、评测执行错误或取消会中止后续测试。
baseline 可组合 `--test smoke,perf`，仍不支持 GSM8K 多实例路由。
以下示例通过 `npu run` 继承实际分配的设备。使用 `--device` 时，去掉外层
`npu run ... --`，为工具传入实际设备列表；其他测试参数保持一致：

```bash
# V2 Lite: sync, 2A1F, MRV1, graph, native DBO, 30 GSM8K samples.
npu run --gpus 3 --nonblock -- python run.py run \
  --model v2-lite --model-path deepseek-ai/DeepSeek-V2-Lite \
  --topology 2a1f --comm sync --mrv mrv1 \
  --execution graph --dbo on --limit 30

# V2 Lite: async, 2A2F, MRV2, eager.
npu run --gpus 4 --nonblock -- python run.py run \
  --model-path deepseek-ai/DeepSeek-V2-Lite \
  --topology 2a2f --comm async --mrv mrv2 --limit 30

# V4 Flash: 16 devices, Attention DP2/TP4 + FFN DP8/TP1/EP8, Layered GMM.
npu run --gpus 16 --nonblock -- python run.py run \
  --model v4-flash --model-path "$V4_MODEL" \
  --comm async --mrv mrv2 --layered-gmm on --limit all
```

要先核对配置，去掉外层 `npu run ... --`，将 `run` 改成 `plan`；
`plan` 不申请设备、不启动服务。异步模式中 `async` 指 AFD 的 CAM 通信方式。

默认 8-shot、512 生成 token、12 并发、strict-match 阈值 0.27；可分别调整
`--fewshot`、`--max-gen-tokens`、`--concurrency`、`--threshold`。
V4 的同步 / 异步 smoke 都检查算术答案与正常生成结束，不能仅凭 HTTP 200 通过。

### 离线评测

离线评测使用：

```bash
npu run --gpus 4 --nonblock -- python run.py run \
  --model-path deepseek-ai/DeepSeek-V2-Lite \
  --offline --dataset-path "$GSM8K_DATASET" --limit 30
```

本地数据支持根目录或 `main/` 中的 `train-00000-of-00001.parquet`（7473 行）和
`test-00000-of-00001.parquet`（1319 行）。仅替换官方任务的数据加载路径，保留评分规则。

### 性能测试

使用 `--test perf`，复用原有模型、通信、MRV、DBO、Graph、设备和清理配置。
默认依次测试各输入长度，再测试短 / 长请求混合；每种负载扫描并发上限和 RPS。
`plan` 输出完整的 `performance_points`，可以先核对测试点数量。
每点的 `planned_mix` 用与实际请求相同的固定种子抽样，显示每种输入 / 输出长度的
正式请求数和比例。若正权重的长度没有抽到请求，`warnings` 会提示增加请求数或调整比例；
随机抽样不保证小样本覆盖所有长度。运行前也显示提示，最终报告对比计划和实际长度计数。

```bash
# Run correctness and performance once; 16 requests cover both input lengths with seed 42.
python run.py run --model v4-flash --model-path "$V4_MODEL" \
  --device 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 \
  --comm async --mrv mrv2 --ffn-mrv mrv1 --moe-ubatching off \
  --test all,perf --perf-mode mixed --perf-input-lens 128,1024 \
  --perf-weights 3,1 --perf-output-lens 32 --perf-requests 16 \
  --perf-concurrencies 4 --rps 2 --perf-arrival burst
```

通过 `--perf-phase` 分别测试 prefill 和 decode，默认 `end-to-end` 保持原有输入 / 输出负载。
prefill 每个请求固定输出 1 token，可扫描输入长度；decode 每个请求固定输入 1 token，
可扫描大于 1 的输出长度。两者均可调整并发、RPS、到达方式、请求数、预热数及原有服务参数。

```bash
# Prefill: vary input length; each request generates exactly one token.
python run.py plan --test perf --perf-phase prefill --perf-mode matrix \
  --perf-input-lens 128,512,2048 --perf-concurrencies 1,8,32 --rps 2,8,inf

# Decode: each request inputs exactly one token and generates multiple tokens.
python run.py plan --test perf --perf-phase decode --perf-mode matrix \
  --perf-output-lens 128,512 --perf-concurrencies 1,8,32 --rps 2,8,inf
```

核对计划后，将 `plan` 改成 `run`，按前述 `npu run` 用法运行完整测试。
显式指定与阶段冲突的长度会在设备分配前报错：prefill 仅接受 `--perf-output-lens 1`，
decode 仅接受 `--perf-input-lens 1`，且每个输出长度必须大于 1。

```bash
# Fixed-length matrix: input x output x concurrency x RPS.
npu run --gpus 4 --nonblock -- python run.py run \
  --model-path deepseek-ai/DeepSeek-V2-Lite \
  --test perf --perf-mode matrix --perf-input-lens 128,512,2048 \
  --perf-output-lens 32,128 --perf-concurrencies 1,8,32 --rps 2,8,inf \
  --perf-requests 100 --perf-warmup 3

# V4 async CAM: 16 devices, mixed inputs (8:1:1) and burst arrivals.
npu run --gpus 16 --nonblock -- python run.py run \
  --model v4-flash --model-path "$V4_MODEL" \
  --comm async --mrv mrv2 --execution eager --test perf \
  --perf-mode mixed --perf-input-lens 128,512,4096 --perf-weights 8,1,1 \
  --perf-output-lens 32,256 --perf-concurrencies 8,32 --rps 4,16 \
  --perf-arrival burst --perf-requests 200
```

第二条命令需要 16 张设备，改成 `--comm sync` 即可测试同步 CAM。
两条命令也可按“使用 --device”中的方式直接指定已分配设备。

| 参数 | 选择 / 默认 |
| --- | --- |
| `--perf-phase` | `end-to-end` / `prefill` / `decode`，默认 end-to-end |
| `--perf-input-lens` | 输入 token 数，逗号分隔；decode 默认且固定 `1`，其他阶段默认 `128,512,2048` |
| `--perf-output-lens` | 输出 token 数，逗号分隔；prefill 默认且固定 `1`，其他阶段默认 `128` |
| `--perf-concurrencies` | 客户端同时在途请求上限，逗号分隔；默认 `1,8,32` |
| `--rps` / `--perf-rps` | 到达速率，支持逗号分隔扫描；默认 `inf`，持续填满并发上限 |
| `--perf-mode` | `matrix` 固定长度矩阵 / `mixed` 混合 / `both` 两者，默认 both |
| `--perf-weights` | 混合输入长度权重，与输入长度逐项对应；默认等概率 |
| `--perf-arrival` | `constant` 等间隔 / `poisson` 随机间隔 / `burst` 突发，默认 constant |
| `--perf-requests` | 每个测试点的正式请求数，默认 100 |
| `--perf-warmup` | 每个测试点的预热请求数，默认 3，不计入正式指标 |

混合模式中输出长度等概率抽样，随机种子固定为 42，便于不同配置使用相同负载。
实际各长度的请求数记录在 `by_length`；小样本混合不能保证每种长度都出现，
完整长度覆盖使用 matrix。最大输入长度加最大输出长度须不超过 `--max-model-len`。

有限 RPS 按到达时间生成请求，超过并发上限的请求在客户端排队；不自动降低目标 RPS。
burst 每次最多同时到达一个并发上限数量的请求，两批间隔为 `并发上限 / RPS` 秒；
poisson 使用固定种子的指数间隔。`inf` 不限到达速率，三种模式均持续填满并发上限。
RPS 过高时，队列和测试总耗时会增长；可用 `--eval-timeout` 限制整个性能阶段，
`--request-timeout` 限制每次实际发送后的完整流式请求。

工具从服务的 `/tokenize` 获取合法 token ID，再构造精确长度的合成输入，通过
`/v1/completions` 流式请求；不依赖本地 Transformers tokenizer，V2 / V4 使用同一条流程。
`ignore_eos` 固定输出长度；响应须正常结束并返回真实 token usage，输入 / 输出计数
必须与请求相符。缺少 usage、流中报错、提前截断或超时均记为失败，整体返回失败。

`performance.json` 按测试点及输入 / 输出长度分组记录：

- TTFT、TPOT、E2E、客户端排队时间及发送调度延迟的有效样本数、平均值、最小 / 最大值、
  P50 / P90 / P95 / P99；分位数按排序后的样本线性插值。
- 成功 / 失败 / 取消 / 未提交数量、预热结果和实际最大并发。
- `dispatch_rps` 实际发送速率、`achieved_rps` 成功请求数除以正式阶段总耗时，
  以及输入 / 输出 token/s；总耗时包含排队和最后一批请求完成。

TTFT 从实际发送到收到首个生成内容；E2E 从发送到收到最后生成内容；
TPOT 为 `(E2E - TTFT) / (实际输出 token 数 - 1)`，再对各请求统计均值和分位数，
采用 [vLLM 的客户端指标口径](https://docs.vllm.ai/en/stable/benchmarking/cli/)。
排队时间单独记录，不计入 TTFT；空控制消息和后到的 usage 消息不计入生成耗时。
流式分块可能包含多个 token，不能把分块数当成 token 数。仅输出一个 token 时 TPOT
为 null，不纳入 TPOT 的样本数；失败请求不纳入生成延迟统计，仍保留错误及排队信息。

prefill 主要查看 TTFT、成功请求速率和 `input_tokens_per_second`；decode 主要查看
TPOT 和 `output_tokens_per_second`。decode 请求仍包含单 token 输入的 prefill 和首 token
生成时间；TPOT 排除首 token 等待，吞吐仍按完整正式阶段耗时计算。

逐请求原始数据在 `performance/requests.jsonl`，包含预热 / 正式标记、长度、随机种子、
到达和发送时间、延迟、usage、分块数及失败原因。
配置及每个测试点的 `phase` 记录性能阶段；逐请求 `performance_phase` 记录相同阶段，
原有 `phase` 继续区分 `warmup` 和 `measured`。
取消或超时也保留已产生的数据，并沿用原有进程清理、设备释放流程。
`report.txt` 提供简短指标，`pack` 会收录这些结果。
这是合成 token 负载的客户端性能测量，包含 HTTP 和流式缓冲的影响；精度仍用 GSM8K 测试。
逐请求数据和精确分位数在内存中保存，超大请求数应分批运行。

## 结果与诊断

### 结果文件

结果默认写入 `runs/<唯一运行编号>/`，或指定新的 / 仅含 `plan.json` 的 `--output` 目录：

- `resolved-config.json`、`commands.json`：配置、启动命令与环境。
- `manifest.json`：工具源文件哈希、可选插件 Git 状态、实际命令、域映射、分配设备、模型元数据和包版本。
- `preflight.log`、`attention.log`、`ffn.log`、评测日志：原始输出。
- `smoke.json`：请求、响应及错误。
- `gsm8k/`：lm-eval 原始结果和逐条样本；精确校验数量和 strict-match。
  `summary.gsm8k` 附最多 20 条错题的标准值、提取值、输出片段与数值等价提示；终端展示前 5 条。
  `26.00` 与 `26` 的提示只用于解释失败，不改变官方 strict-match 或验收门槛。
  样本日志缺失或格式不兼容时明确标注，不影响官方精度判定。
- `performance.json`、`performance/requests.jsonl`：性能汇总及逐请求原始测量。
- `metrics/before-*.prom`、`metrics/after-*.prom`：各 HTTP 实例在负载前后的 `/metrics` 原始快照。
  `summary.metrics` 记录每次采集的状态、路径或不可用原因；指标名称按服务端版本原样保留。
- `report.txt`：简短结果，包含精度、条数、阈值、耗时及失败原因，结束时也输出到终端。
- `diagnostics.json`：Attention / FFN 按推理和退出阶段记录错误，附原始日志文件名与行号；
  每个角色、阶段保留最早 10 条和最新 10 条，额外保留第一个具体异常（最多 21 条）；
  失败摘要优先显示具体异常，完整栈仍保存在原始日志和 compact 证据中。
- `advisories.json`：profiler 要求离线解析的已知提示，记录角色、阶段、文件和行号；原文仍保留在日志中。
- `summary.json`、`exit-code.txt`：状态和失败原因。成功 0、运行失败 1、参数错误 2、正常取消 130。

### 运行进度与故障判定

运行时 `summary.json` 会原子更新当前阶段、阶段开始时间和耗时；阶段切换及等待
预检、服务就绪、请求 / 评测期间每 15 秒输出进度，无需等整个测试结束才知道运行到了哪里。
最终保留每个阶段的 `stage_durations`。精度未达阈值也保留实际分数和样本数。
`test_results` 分别记录所选测试项的状态，便于区分 GSM8K 失败和性能测试通过。

结果分别记录 `inference_status`、`shutdown_status` 和 `cleanup_status`：
即使评测成功、服务主进程退出码为 0，退出时出现 ERROR、异常或强制 SIGKILL，整体仍判失败；
本次进程全部退出时，清理状态可以通过；设备释放由 `npu run` 或调用方负责。
Ctrl-C 后如退出或清理失败，返回 1。
日志检查按 ERROR 级别、异常结尾及强制结束提示匹配；普通 WARNING 不判失败。
profiler 的已知完整提示 “daemon process 中不能解析，请使用 offline parsing” 单独记录，
不触发服务故障判定；仍须完成 profiler 的离线解析。其他 profiler 错误、NPU / 算子错误和退出异常均判失败，
同一行混入其他错误时也不会被当成提示。退出、进程清理和报告写入失败都有明确的 `failed_stage`。

预检检查端口、网卡和本地模型文件，记录包版本；远程模型加载、设备可用性和算子支持由服务启动检查。
端口检查允许上次连接留下的 TIME_WAIT，但仍拒绝已有监听服务。
服务就绪通过 `/v1/models` 判断；smoke 检查并发请求和有效响应，GSM8K 检查精度与样本数，
性能测试检查完整流式响应和 usage。日志错误、异常退出和清理失败继续判失败。
`/metrics` 在负载前后采集，不在测量请求过程中轮询；404、超时等记录为 unavailable，不影响测试结果。
工具不解析版本相关的 Prometheus 指标，也不推断内部 Graph、DBO 或 Layered 执行覆盖。
profiler 可通过目标版本的 serve 参数启用，trace 收集和离线解析由外部工具完成。
每次输出目录必须是新目录，避免误读旧精度结果。

### 批量查看状态

无需 NPU 环境，也无需设备配置：

```bash
python run.py status runs
python run.py status runs/<运行编号>
python run.py status runs/async-off runs/async-dbo-on --json
```

可传一个运行目录、多个运行目录，或包含多次运行的父目录；父目录只扫描直接子目录，
不遍历模型、profiler 或转储文件。表格显示运行目录、Case 配置、整体结果、阶段、推理 / 退出 / 清理状态、
GSM8K 分数与条数、性能状态及耗时。运行中的测试显示当前阶段；退出失败显示 shutdown。
历史报告缺少 `failed_stage` 时按已有阶段状态显示失败位置，不改动历史文件。
`--json` 返回完整汇总及实际读取的 `result_dir`，用于已有批量脚本。
`status` 读取成功返回 0，目录无结果或报告损坏返回 2；测试是否通过仍查看汇总内的状态。

### 打包结果

测试完成后可在 Pod 内或下载后打包结果，无需 NPU 环境或设备配置：

```bash
python run.py pack runs/<运行编号>
# 或指定新的压缩包路径
python run.py pack runs/<运行编号> --output v4-async-results.tar.gz
# 大日志先留在 Pod：只带报告、逐条结果和错误上下文
python run.py pack runs/<运行编号> --compact
```

默认生成 `results.tar.gz` 和对应 `.sha256` 校验文件，不覆盖已有文件。
压缩包保留根目录的 JSON / 日志 / 文本报告、GSM8K 结果、逐条样本和本地任务配置，
以及性能测试的逐请求数据和外部 metrics 快照；
`extra-info` 等目录的大型 NPU 转储不加入压缩包，原文件保留。
包内 `archive-inventory.json` 记录收录 / 排除文件的路径和大小，终端输出排除总大小与 SHA256。
文件及目录符号链接不打包。iTask 的 Pod 创建、文件传输继续由外部操作完成。

`--compact` 默认生成 `reports.tar.gz`，省去完整的 Attention / FFN / baseline 实例日志，保留其他报告、
评测日志和逐请求数据。包内 `log-evidence.json` 保存原日志的字节数、行数、SHA256、
每条采样错误 / 提示前 8 行后 40 行的上下文，以及末尾 40 行；方便直接定位算子错误和 traceback。
它在打包时生成，原日志和历史汇总不修改；已下载的 `log-evidence.json` 原样收录。
完整日志仍可用默认 pack 打包；profiler trace 和大型转储单独导出。

## 开发验证

```bash
python -m pip install -e '.[test]'
python -m pytest -q
```

测试涵盖参数组合、设备选择与环境继承、精确样本数、严格精度门禁、外部观测、取消及进程清理、
退出报错但进程返回 0、报告写入失败、双模式 V4 答案校验及转储排除打包；
性能测试覆盖长度 / 并发 / RPS 矩阵、混合和突发到达、分块解析、指标计算、并发限制、
完整请求超时、预热失败、取消时的部分结果，以及 V4 同步 / 异步生命周期；
还覆盖 async+DBO 中 profiler 提示不打断请求、真正错误仍失败、后续根因采样、小体积打包和批量状态读取；
还覆盖 serve 参数替换/移除、内部功能支持判定委托、无推理包导入的预检、可选 metrics 和原生多实例日志打包；
生命周期测试使用本地模拟 HTTP 服务和真实子进程，需允许 localhost socket。
这些 CPU 测试不能代替真实 NPU / 模型评测。

`process_utils.py` 从 `afd-plugin/tests/e2e/process_utils.py` 复制，保留 Apache-2.0 许可。
其余运行逻辑独立，不导入仓库的 `tests` 包或历史 async 脚本。
入口为 `plan` / `run` / `status` / `pack`，可通过已有脚本批量调用。
