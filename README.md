# Compass

Compass 是一个仅在本机运行的 A 股量化研究工具，覆盖标的池、日线行情、账户持仓、今日信号、策略管理、组合回测和参数调优。

应用默认只监听 `127.0.0.1`，运行数据与 Git 仓库分离。代码可以由多人协作，行情、账户、信号和实验数据则保存在每位使用者自己的电脑上。

> 当前项目采用“从源码运行”的方式，还没有独立安装包或系统服务。

## 主要功能

- 维护唯一的关注标的池，当前支持上证和深证市场；
- 从腾讯证券、东方财富和 BaoStock 增量同步日线行情；
- 检查行情区间、连续性、实际来源和任务历史；
- 维护多个账户方案和共享持仓；
- 根据最新完整交易日生成今日信号与调仓建议；
- 创建和版本化策略模板；
- 组合多个策略和多个 ETF 进行回测，并比较基准；
- 使用训练、验证和冻结测试区间进行策略参数调优；
- 可选接入 Kronos K 线基础模型，将价格预测转换为可回测目标仓位；
- 对齐 Kronos 预测与未来实际/可交易收益，展示方向命中率、收益相关性和入选标的表现；
- 查看本地脱敏日志，排查代理、网络和行情源问题。

## 系统要求

- Windows 10/11、macOS 或主流 Linux 发行版；
- Python `3.12`；
- [uv](https://docs.astral.sh/uv/)；
- Git（协作或从远程仓库安装时需要）。

检查环境（PowerShell、Terminal 或其他 shell 均可）：

```powershell
python --version
uv --version
git --version
```

## 安装

### 普通使用

Windows PowerShell：

```powershell
git clone https://github.com/sikasjc/compass.git compass
Set-Location compass
uv sync
```

macOS / Linux：

```bash
git clone https://github.com/sikasjc/compass.git compass
cd compass
uv sync
```

`uv sync` 会根据 `uv.lock` 创建项目独立的 `.venv`，不会把依赖安装到系统 Python 中。

### 参与开发

开发环境还需要测试、格式和类型检查工具：

```bash
git clone https://github.com/sikasjc/compass.git compass
cd compass
uv sync --extra dev
```

### 可选：安装 Kronos 模型策略

Kronos 不是 Compass 基础安装的必需依赖。CPU 或 macOS 环境运行：

```bash
uv sync --extra kronos
```

Windows/Linux NVIDIA GPU 环境运行：

```bash
uv sync --extra kronos-cuda
```

Windows 推荐使用自检脚本；代理只作用于本次安装，不修改系统设置：

```powershell
.\scripts\install-kronos.ps1 -Mode CUDA -Proxy http://127.0.0.1:7897 -IncludeDev
```

脚本会依次检查 `uv`、NVIDIA 驱动，安装依赖，然后输出 PyTorch 版本、CUDA
运行时、GPU 可用状态和设备名称。下载中断后可直接重跑同一命令复用缓存。

两个选项互斥，切换时直接执行对应命令即可。`kronos-cuda` 从 PyTorch 官方 CUDA 13.2
索引安装 GPU 版本；NVIDIA 驱动必须兼容该运行时，不需要单独安装完整 CUDA Toolkit。
可用下面的命令检查结果：

```bash
uv run python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

最后一项为 `True` 即表示 Compass 的“自动”推理设备会选择 NVIDIA GPU；否则页面会明确显示
“当前仅 CPU”。

首次运行该策略会从 Hugging Face 下载模型权重。默认使用
`NeoQuasar/Kronos-mini` 和 `NeoQuasar/Kronos-Tokenizer-2k`；模型缓存由 Hugging Face
管理，不写入 Git 仓库。需要更高容量时可在策略参数中选择 `small` 或 `base`。Kronos
代码及模型采用 MIT 许可，但模型输出仅作为实验信号，不能视为收益保证。

如果 `huggingface.co` 在当前网络不稳定，可在个人 `.env` 中设置：

```text
HF_ENDPOINT=https://hf-mirror.com
```

模型默认缓存在 `~/.cache/huggingface`；可用 `HF_HOME` 改到空间更充足的磁盘。
Windows 没有开启开发者模式时缓存仍可用，只是无法用符号链接去重。

首次提交代码前建议运行：

```bash
uv run ruff check src tests
uv run mypy src/compass
uv run pytest -q
```

## 启动与停止

默认使用端口 `8080`：

```bash
uv run python -m compass.ui.app
```

也可以使用启动脚本；脚本会以非清理模式同步基础依赖、保留已经安装的 Kronos
CPU/CUDA 可选环境，并读取仓库根目录下未跟踪的 `.env`：

```powershell
# Windows PowerShell
.\scripts\start.ps1
```

```bash
# macOS / Linux
sh scripts/start.sh
```

打开：

```text
http://127.0.0.1:8080/
```

需要使用其他端口时：

```bash
uv run python -m compass.ui.app --port 8081
```

macOS / Linux 启动脚本可通过环境变量设置端口：

```bash
COMPASS_PORT=8081 sh scripts/start.sh
```

然后打开 `http://127.0.0.1:8081/`。根路径是“开始”页面，不需要记忆各功能路径。

停止应用：回到启动应用的终端，按 `Ctrl+C`。应用不是系统服务，关闭终端或结束对应 Python 进程后不会继续在后台运行。

如果提示端口已占用，可以换一个端口，或者先检查占用进程：

```powershell
# Windows
Get-NetTCPConnection -LocalPort 8080 -State Listen
```

```bash
# macOS / Linux
lsof -iTCP:8080 -sTCP:LISTEN
```

## 更新

更新代码不会覆盖个人运行数据：

```bash
cd <Compass 仓库目录>
git pull
uv sync --extra dev --inexact
```

只作为普通用户运行时，可以把最后一条改为 `uv sync --inexact`。`--inexact` 会保留
已经单独安装的 Kronos CPU/CUDA 环境；如果希望严格重建环境，则明确运行
`uv sync --extra kronos` 或 `uv sync --extra kronos-cuda`。

更新后重新启动应用。数据库结构由应用启动时自动检查和创建；重要更新前仍建议备份运行数据目录。

## 卸载

Compass 当前不写入 Windows 注册表，也不安装 Windows、launchd 或 systemd 服务。卸载分为两部分。

### 只删除程序，保留个人数据

1. 使用 `Ctrl+C` 停止应用；
2. 删除克隆下来的代码仓库目录；macOS / Linux 可以使用文件管理器，或在确认绝对路径后使用系统删除命令。

个人行情、账户和实验数据仍保留在本地运行数据目录，之后重新克隆代码即可继续使用。

### 同时删除全部个人数据

先确认应用已经停止，并备份需要保留的内容，再删除：

删除的数据目录因系统而异，见下一节的“数据保存在哪里”。

这是不可恢复操作，会删除数据库、行情、账户、信号、回测报告、策略实验和日志。不要通过仓库清理命令代替这一步，也不要在应用运行时删除数据目录。

## 配置

### 命令行参数

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `--port` | `8080` | 本地 Web 页面端口，范围为 `1～65535` |

服务地址固定为 `127.0.0.1`，不会绑定局域网或公网地址。

### 环境变量

| 环境变量 | 必填 | 说明 |
|---|---|---|
| `COMPASS_DATA_DIR` | 否 | 覆盖运行数据根目录，必须在启动前设置 |
| `COMPASS_GIT_COMMIT` | 否 | 构建或发布时记录版本来源，普通用户无需设置 |

启动脚本还支持：Windows `start.ps1` 的 `-Port`、`-EnvironmentFile` 参数，以及
macOS/Linux `start.sh` 的 `COMPASS_PORT`、`COMPASS_ENV_FILE` 环境变量。

Windows PowerShell（仅对当前会话生效）：

```powershell
$env:COMPASS_DATA_DIR = "D:\CompassData"
uv run python -m compass.ui.app --port 8081
```

macOS / Linux shell（仅对当前会话生效）：

```bash
export COMPASS_DATA_DIR="$HOME/CompassData"
uv run python -m compass.ui.app --port 8081
```

也可以复制 `.env.example` 为未跟踪的 `.env` 作为个人配置参考，但应用当前以进程环境变量和页面设置为准，不会主动把密钥写入仓库。

### 页面设置

“设置”页面可以配置：

- 应用启动、定时或收盘后自动增量同步行情；
- 行情请求超时时间；
- 不使用代理、系统代理或自定义代理 IP/端口；
- 数据源与代理连接测试；
- 本地日志级别。

设置页面顶部会显示应用当前实际使用的运行数据目录。

## 数据保存在哪里

默认运行数据目录：

| 系统 | 默认路径 |
|---|---|
| Windows | `%LOCALAPPDATA%\Compass` |
| macOS | `~/Library/Application Support/Compass` |
| Linux | 设置了 `XDG_DATA_HOME` 时为 `$XDG_DATA_HOME/compass`，否则为 `~/.local/share/compass` |

Windows 示例：

```text
%LOCALAPPDATA%\Compass
```

通常展开为：

```text
C:\Users\<用户名>\AppData\Local\Compass
├─ data
│  ├─ compass.db             # SQLite：标的、任务、策略、回测等结构化数据
│  ├─ market                       # 行情对象、清单和交易日历
│  ├─ signal_accounts.json         # 信号账户方案
│  ├─ signal_executions.json       # 建议采用与执行记录
│  ├─ strategy-drafts.json         # 未发布的自定义规则草稿
│  └─ strategy_optimizations.json  # 策略调优实验
├─ reports                         # 导出或生成的报告
└─ logs
   └─ compass.log            # 本地脱敏诊断日志
```

运行数据目录不属于 Git 仓库。仓库中的 `/data`、`/reports`、`/logs` 也被 `.gitignore` 排除，但新版本默认不会再向仓库写入这些目录。

### 备份与迁移

1. 停止应用；
2. 完整复制当前系统的数据根目录；
3. 在目标电脑恢复到相同位置，或用 `COMPASS_DATA_DIR` 指向恢复目录；
4. 启动应用并检查“设置”页面显示的数据路径。

不要只复制 SQLite 文件：行情对象和部分账户、信号、实验记录保存在同一根目录的其他文件中。

## 基本使用流程

面向日常操作的完整页面说明见 [`docs/user-guide.md`](docs/user-guide.md)。

1. 打开根地址 `/`，从“开始”页面进入功能；
2. 在“标的池”添加关注的指数或 ETF；
3. 在“行情数据”选择数据源和区间，执行增量同步并确认覆盖情况；
4. 在“策略实验室”创建策略模板，必要时运行参数调优；
5. 在“策略回测”组合策略与标的，检查收益、回撤、成交和比较基准；
6. 在“账户”创建账户方案并维护真实持仓；
7. 在“今日信号”生成调仓建议，选择采用或不采用并记录实际执行；
8. 遇到网络、任务或数据异常时，在“设置”测试连接并到“日志”查看详情。

行情采用增量同步：已有数据默认可信，只请求目标区间中的缺失部分。单个标的失败不会回滚其他已成功标的，失败标的继续保留原有可信数据。

## 基本架构

```text
浏览器 / NiceGUI 页面
        │
        ▼
ui/                 页面、表单、图表和页面模型
        │
        ▼
services/           用例编排、任务、自动同步、信号中心、策略实验和调优
   │             │
   ▼             ▼
data/          strategies/ + backtest/ + portfolio/ + risk/
行情提供方       策略决策、撮合、组合分配、费用和风险规则
   │             │
   └──────┬──────┘
          ▼
storage/            SQLite、规范 JSON、Parquet 行情对象和可复现快照
          │
          ▼
Windows: %LOCALAPPDATA%\Compass
macOS: ~/Library/Application Support/Compass
Linux: $XDG_DATA_HOME/compass 或 ~/.local/share/compass
```

主要目录职责：

| 目录 | 职责 |
|---|---|
| `src/compass/domain` | 标的、交易意图、运行和质量等核心领域模型 |
| `src/compass/data` | 行情请求、标准化、质量检查、交易日历和数据源适配器 |
| `src/compass/strategies` | 内置策略、指标、DSL 与策略注册表 |
| `src/compass/backtest` | 回测引擎、订单、撮合、市场规则和运行快照 |
| `src/compass/portfolio` | 多策略目标合并、仓位分配和归因轨迹 |
| `src/compass/risk` | 风险规则与风险引擎 |
| `src/compass/services` | 应用用例、后台任务、本地网关、自动同步、信号与调优 |
| `src/compass/storage` | SQLite 仓储、Parquet 行情对象、规范 JSON 与完整性校验 |
| `src/compass/ui` | NiceGUI 应用、导航、页面模型和可视化组件 |
| `tests/unit` | 领域、服务和 UI 模型的快速单元测试 |
| `tests/integration` | 数据存储、行情同步和跨模块集成测试 |

设计原则：

- 本地优先：不依赖远程业务服务器；
- 数据可追溯：行情清单、内容哈希、回测快照和策略版本可复现；
- 无未来函数：收盘信号最早在下一交易日执行；
- 部分成功可保留：批量任务不会因一个标的失败浪费其他成功结果；
- 数据与代码分离：个人数据不进入 Git，协作者使用各自的运行目录；
- 边界校验：外部行情、持久化文件和页面输入进入核心逻辑前均需验证。

## 协作开发

推荐流程：

```bash
git switch -c codex/<功能名称>
uv sync --extra dev
uv run ruff check src tests
uv run mypy src/compass
uv run pytest -q
```

协作约定：

- 不提交 `.env`、令牌、数据库、行情、报告或日志；
- 测试显式使用临时目录，不读取 Windows、macOS 或 Linux 的个人运行数据目录；
- 不依赖某位协作者已有的本地账户、行情或策略实验；
- 修改数据库、规范 JSON 或回测结果结构时同步增加兼容性和完整性测试；
- 提交前检查 `git status`，确认没有本地数据或密钥进入变更列表。

## 常用检查命令

```bash
# 代码规范
uv run ruff check src tests

# 严格类型检查
uv run mypy src/compass

# 全量测试
uv run pytest -q
```

应用问题优先查看页面“日志”，开发问题再查看终端输出和测试结果。
