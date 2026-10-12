# AFD E2E

独立运行的 NPU 测试工具。复制整个 `afd-e2e` 目录到其他平台，将 `afd-plugin`
放在同级目录即可。工具在现有主机或 Pod 内执行。

```text
work/
├── afd-plugin/
└── afd-e2e/
    ├── run.py
    └── platforms/manager.example.yaml
```

## 环境与安装

使用已经安装 CANN、torch-npu、vLLM、vLLM-Ascend 和同级 `afd-plugin` 的推理环境；
自定义 CAM / Layered 算子须在该环境编译安装。工具不改动插件源码。
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
安装在平台已验证的推理环境中，确认依赖解析不会替换该平台固定的 torch / vLLM。
`python run.py`、`python -m afd_e2e` 和安装后的 `afd-e2e` 是同一个入口。
从任意工作目录调用源目录中的 `run.py`，默认仍寻找与工具同级的插件。

## 直接指定设备

不需要平台配置。设备列表的前几张用于 Attention，随后用于 FFN。
V2 的 2A1F 需要 3 张，2A2F 需要 4 张；V4 需要 16 张。

```bash
# V2 Lite，同步，2A1F，MRV1，graph，原生 DBO，GSM8K 严格测 30 条
python run.py run --model v2-lite --model-path /weights/DeepSeek-V2-Lite \
  --topology 2a1f --device 0,1,2 --comm sync --mrv mrv1 \
  --execution graph --dbo on --limit 30

# V2 Lite，异步，2A2F，MRV2，eager
python run.py run --model-path /weights/DeepSeek-V2-Lite \
  --topology 2a2f --device 0,1,2,3 --comm async --mrv mrv2 --limit 30

# V4 Flash，Attention DP2/TP4 + FFN DP8/TP1/EP8，双侧 MRV2，Layered GMM
python run.py run --model v4-flash --model-path /weights/DeepSeek-V4-Flash \
  --device 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 \
  --comm async --mrv mrv2 --layered-gmm on --limit all
```

将 `run` 改成 `plan` 可以先查看最终参数、设备切分、命令及环境变量；
`plan` 不申请设备、不启动服务。异步模式中 `async` 指 AFD 的 CAM 通信方式。

| 参数 | 选择 / 默认 |
| --- | --- |
| `--comm` | `sync` / `async`，默认 sync |
| `--model` | `v2-lite` / `v4-flash`，默认 V2；V4 必须指定权重路径 |
| `--topology` | V2：`2a1f` / `2a2f`（默认）；V4：自动 `dp2tp4-ep8` |
| `--mrv` | `mrv1` / `mrv2`，默认 mrv1，应用于双侧 |
| `--ffn-mrv` | 可单独覆盖 FFN 的 MRV 选择 |
| `--dbo` | 原生 vLLM DBO：`on` / `off`，默认 off |
| `--layered-gmm` | `on` / `off`，默认 off |
| `--moe-ubatching` | Async CAM 两段 token 拆分：`on` / `off`，异步默认 on |
| `--execution` | `eager` / `graph`，默认 eager |
| `--limit` | GSM8K 条数 `1..1319`，默认 30；`all` 为 1319 |
| `--test` | `smoke` / `gsm8k` / `all` / `perf`，默认 all；all 仍是 smoke + GSM8K |
| `--device` / `--devices` | 逗号分隔的设备编号；也可继承 `ASCEND_RT_VISIBLE_DEVICES` |
| `--platform` | 可选，仅配置设备管理命令 |
| `--npu-count` | 管理工具申请数量，默认按拓扑推导 |

当前分支的组合限制在申请设备前校验：Async CAM 仅支持 eager；同步和异步均可开启原生 DBO。
DBO 要求 FFN MRV1，Attention 可选 MRV1 / MRV2，MRV2 DBO 要求 eager。
MRV2 原生双微批需满足当前插件对 Attention DP 和拓扑的限制。
Layered GMM 要求 V4、async、eager、dynamicQuant=1 和 A3/910C
及 W4A8 权重，FFN MRV1 / MRV2 均可开启。FFN 的实际类名为
`AFDNPUFFNModelRunner`，报告同时记录 MRV 开关及实际执行的 Layered 算子。

DBO 开启时会额外发出并发覆盖请求，不增加 GSM8K 的 `--limit`。
MRV2 DBO 的覆盖检查还要求每个 Attention rank 记录真实 `NPUUBatchRunner`
执行、两段容量与各段实际 token 数；配置开关本身不能充当执行证据。
`--moe-ubatching` 对应异步 CAM 自身的拆分机制，不能用原生 DBO 开关替代。
比较 MRV2 的 DBO on / off 时，两组均显式设置 `--moe-ubatching off`，只改变 `--dbo`；
否则异步默认开启的拆分仍会触发双微批。Async DBO 只在 Attention 开启原生 DBO，FFN 消费 CAM 工作项。
`--baseline` 保留仓库中原生 V2 Lite 的 4 卡基线测试。

## 性能测试

使用 `--test perf`，复用原有模型、通信、MRV、DBO、Graph、设备和清理配置。
默认依次测试各输入长度，再测试短 / 长请求混合；每种负载扫描并发上限和 RPS。
`plan` 输出完整的 `performance_points`，可以先核对测试点数量。

```bash
# 固定长度矩阵：每个输入长度 × 输出长度 × 并发上限 × RPS 分别测试
python run.py run --device 0,1,2,3 --model-path /weights/DeepSeek-V2-Lite \
  --test perf --perf-mode matrix --perf-input-lens 128,512,2048 \
  --perf-output-lens 32,128 --perf-concurrencies 1,8,32 --rps 2,8,inf \
  --perf-requests 100 --perf-warmup 3

# V4 异步 CAM：短 / 长输入按 8:1:1 混合，输出长度随机混合，分批突发
python run.py run --model v4-flash --model-path /weights/DeepSeek-V4-Flash \
  --device 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 \
  --comm async --mrv mrv2 --execution eager --test perf \
  --perf-mode mixed --perf-input-lens 128,512,4096 --perf-weights 8,1,1 \
  --perf-output-lens 32,256 --perf-concurrencies 8,32 --rps 4,16 \
  --perf-arrival burst --perf-requests 200
```

第二条命令改成 `--comm sync` 即可测试同步 CAM；设备管理方式也可以换成
`--platform platforms/my-platform.yaml`，省去 `--device`。

| 参数 | 选择 / 默认 |
| --- | --- |
| `--perf-input-lens` | 输入 token 数，逗号分隔；默认 `128,512,2048` |
| `--perf-output-lens` | 输出 token 数，逗号分隔；默认 `128` |
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

逐请求原始数据在 `performance/requests.jsonl`，包含预热 / 正式标记、长度、随机种子、
到达和发送时间、延迟、usage、分块数及失败原因。取消或超时也保留已产生的数据，
并沿用原有进程清理、设备释放流程。`report.txt` 提供简短指标，`pack` 会收录这些结果。
这是合成 token 负载的客户端性能测量，包含 HTTP 和流式缓冲的影响；精度仍用 GSM8K 测试。
逐请求数据和精确分位数在内存中保存，超大请求数应分批运行。

## 使用平台设备管理工具

复制示例文件，填入该平台实际的申请 / 释放命令：

```yaml
allocator:
  allocate: [python3, /opt/platform/allocate.py, --count, "{count}"]
  release: [python3, /opt/platform/release.py, --lease-id, "{lease_id}"]
  timeout: 120
```

```bash
python run.py run --platform platforms/my-platform.yaml --npu-count 4 \
  --model-path /weights/DeepSeek-V2-Lite --topology 2a2f --limit 30
```

配置只涉及当前 Pod 内的设备申请和释放。平台文件支持 YAML / JSON；
YAML 需要 PyYAML，JSON 无额外依赖。也可直接传入
`--allocate-command '...' --release-command '...'`，两条命令必须同时提供。
命令支持 `{count}`、`{run_id}`、`{lease_id}` 占位符；按参数列表执行，不经过 shell。
如原有管理工具输出格式不同，平台自行提供两个短适配脚本。

申请命令 stdin 收到：

```json
{"count":4,"run_id":"20261011T000000Z-example"}
```

申请命令 stdout 必须只返回一个 JSON 对象，设备数量与申请数量一致；日志写 stderr：

```json
{"lease_id":"lease-123","devices":[0,1,2,3]}
```

释放命令 stdin 收到整个申请响应，外加 `count`、`run_id`。两条命令均以退出码 0
表示成功；同时设置环境变量 `AFD_E2E_RUN_ID`、`AFD_E2E_REQUESTED_DEVICES`。
工具检查重复编号、数量及继承的可见设备范围。列表超出拓扑所需部分不使用并记录在报告中。
不能同时指定管理工具和 `--device`。

正常完成、精度失败和 Ctrl-C 都先清理本次子进程，再释放设备。
仍有本次进程存活时保留租约，并将测试标记为失败，避免占用中的设备被重新分配。
申请命令超时或返回无法解析的 JSON 时，工具无法获得租约编号；平台适配脚本须保证
申请过程可取消或自身回滚，不留下未知租约。

## 校准、离线与结果

其他平台通常只需指定设备和本地权重。网卡与通信用 IP 默认沿用环境或从默认路由检测；
有多网卡时使用 `--nic eth0 --afd-host 10.0.0.1`。API / AFD 端口可分别通过
`--api-port`、`--afd-port` 修改。已有 CANN 的 `PYTHONPATH` 和自定义算子库路径保留。

内存与批量配置可用 `--max-model-len`、`--max-batched-tokens`、`--max-seqs`、
`--memory-utilization` 校准；graph 的 `--capture-sizes` 默认 `1,2,4,8,16`。
其他 vLLM 参数按单个 token 传入，例如 `--vllm-arg=--block-size --vllm-arg=128`，
也可只给 Attention / FFN 传 `--attention-arg` / `--ffn-arg`。
核心拓扑和功能参数由工具统一管理，不能通过附加参数覆盖。

离线评测使用：

```bash
python run.py run --device 0,1,2,3 --model-path /weights/DeepSeek-V2-Lite \
  --offline --dataset-path /datasets/gsm8k --limit 30
```

本地数据支持根目录或 `main/` 中的 `train-00000-of-00001.parquet`（7473 行）和
`test-00000-of-00001.parquet`（1319 行）。仅替换官方任务的数据加载路径，保留评分规则。
默认 8-shot、512 生成 token、12 并发、strict-match 阈值 0.27；可分别调整
`--fewshot`、`--max-gen-tokens`、`--concurrency`、`--threshold`。
V4 的同步 / 异步 smoke 都检查算术答案与正常生成结束，不能仅凭 HTTP 200 通过。

结果默认写入 `runs/<唯一运行编号>/`，或指定新的 `--output` 目录：

- `resolved-config.json`、`commands.json`：配置、启动命令与环境。
- `manifest.json`：工具 / 插件源文件哈希、插件 Git 状态、分配设备、模型元数据和依赖版本。
- `preflight.log`、`attention.log`、`ffn.log`、评测日志：原始输出。
- `smoke.json`、`dbo-workload.json`：请求、响应及错误。
- `gsm8k/`：lm-eval 原始结果和逐条样本；精确校验数量和 strict-match。
- `performance.json`、`performance/requests.jsonl`：性能汇总及逐请求原始测量。
- `coverage.json`：所有预期 rank 的实际 MRV、live graph / 双段拆分 / Layered 执行证据。
- `report.txt`：简短结果，包含精度、条数、阈值、耗时及失败原因，结束时也输出到终端。
- `diagnostics.json`：Attention / FFN 按推理和退出阶段记录错误，附原始日志文件名与行号；
  每个角色、阶段最多 20 条，超过时保留最早 10 条和最新 10 条，避免重复消息挤掉后面的根因。
- `advisories.json`：profiler 要求离线解析的已知提示，记录角色、阶段、文件和行号；原文仍保留在日志中。
- `summary.json`、`exit-code.txt`：状态和失败原因。成功 0、运行失败 1、参数错误 2、正常取消 130。

运行时 `summary.json` 会原子更新当前阶段、阶段开始时间和耗时；阶段切换及等待
预检、服务就绪、请求 / 评测期间每 15 秒输出进度，无需等整个测试结束才知道运行到了哪里。
最终保留每个阶段的 `stage_durations`。精度未达阈值也保留实际分数和样本数。

结果分别记录 `inference_status`、`shutdown_status` 和 `cleanup_status`：
即使评测成功、服务主进程退出码为 0，退出时出现 ERROR、异常或强制 SIGKILL，整体仍判失败；
进程全部退出且设备释放成功时，清理状态可以通过。Ctrl-C 后如退出或清理失败，返回 1。
日志检查按 ERROR 级别、异常结尾及强制结束提示匹配；普通 WARNING 不判失败。
profiler 的已知完整提示 “daemon process 中不能解析，请使用 offline parsing” 单独记录，
不触发服务故障判定；仍须完成平台的离线解析。其他 profiler 错误、NPU / 算子错误和退出异常均判失败，
同一行混入其他错误时也不会被当成提示。退出、设备释放和报告写入失败现在都有明确的 `failed_stage`。

工具先检查端口和网卡，再检查来源、可见 NPU、矩阵运算、模型文件及同步 / 异步算子，最后启动服务。
端口检查允许上次连接留下的 TIME_WAIT，但仍拒绝已有监听服务。
不能用启动成功替代功能执行证据；只在实际请求完成后统计覆盖。
每次输出目录必须是新目录，避免误读旧精度结果。

## 查看多个测试的状态

无需 NPU 环境，也无需设备或平台配置：

```bash
python run.py status runs
python run.py status runs/<运行编号>
python run.py status runs/async-off runs/async-dbo-on --json
```

可传一个运行目录、多个运行目录，或包含多次运行的父目录；父目录只扫描直接子目录，
不遍历模型、profiler 或转储文件。表格显示运行目录、Case 配置、整体结果、阶段、推理 / 退出 / 清理状态、
覆盖、GSM8K 分数与条数、性能状态及耗时。运行中的测试显示当前阶段；退出失败显示 shutdown。
历史报告缺少 `failed_stage` 时按已有阶段状态显示失败位置，不改动历史文件。
`--json` 返回完整汇总及实际读取的 `result_dir`，用于已有批量脚本。
`status` 读取成功返回 0，目录无结果或报告损坏返回 2；测试是否通过仍查看汇总内的状态。

测试完成后可在 Pod 内或下载后打包结果，无需 NPU 环境或设备配置：

```bash
python run.py pack runs/<运行编号>
# 或指定新的压缩包路径
python run.py pack runs/<运行编号> --output /tmp/v4-async-results.tar.gz
# 大日志先留在 Pod：只带报告、逐条结果和错误上下文
python run.py pack runs/<运行编号> --compact
```

默认生成 `results.tar.gz` 和对应 `.sha256` 校验文件，不覆盖已有文件。
压缩包保留根目录的 JSON / 日志 / 文本报告、GSM8K 结果、逐条样本和本地任务配置，
以及性能测试的逐请求数据；
`extra-info` 等目录的大型 NPU 转储不加入压缩包，原文件保留。
包内 `archive-inventory.json` 记录收录 / 排除文件的路径和大小，终端输出排除总大小与 SHA256。
文件及目录符号链接不打包。iTask 的 Pod 创建、文件传输继续由外部操作完成。

`--compact` 默认生成 `reports.tar.gz`，省去完整的 Attention / FFN 日志，保留其他报告、
评测日志和逐请求数据。包内 `log-evidence.json` 保存原日志的字节数、行数、SHA256、
每条采样错误 / 提示前 8 行后 40 行的上下文，以及末尾 40 行；方便直接定位算子错误和 traceback。
它在打包时生成，原日志和历史汇总不修改；已下载的 `log-evidence.json` 原样收录。
完整日志仍可用默认 pack 打包；profiler trace 和大型转储仍由平台单独导出。

## 开发验证

```bash
python -m pip install -e '.[test]'
python -m pytest -q
```

测试涵盖参数组合、设备管理协议、精确样本数、严格精度门禁、运行证据、取消及进程清理、
退出报错但进程返回 0、报告写入失败、双模式 V4 答案校验及转储排除打包；
性能测试覆盖长度 / 并发 / RPS 矩阵、混合和突发到达、分块解析、指标计算、并发限制、
完整请求超时、预热失败、取消时的部分结果，以及 V4 同步 / 异步生命周期；
还覆盖 async+DBO 中 profiler 提示不打断请求、真正错误仍失败、后续根因采样、小体积打包和批量状态读取；
生命周期测试使用本地模拟 HTTP 服务和真实子进程，需允许 localhost socket。
这些 CPU 测试不能代替目标平台上的真实 NPU / 模型评测。

`process_utils.py` 从 `afd-plugin/tests/e2e/process_utils.py` 复制，保留 Apache-2.0 许可。
其余运行逻辑独立，不导入仓库的 `tests` 包或历史 async 脚本。
入口为 `plan` / `run` / `status` / `pack`，使用平台自身方式批量调用即可。
