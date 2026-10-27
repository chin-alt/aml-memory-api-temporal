# 五套文本基准评测

本仓库的 `benchmarks/` 提供五个可直接运行的 PowerShell 入口。脚本固定使用本仓库对应的记忆链路，不接受运行时切换链路，便于在三个仓库之间做同题对比。

## 首次准备

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\benchmarks\setup.ps1
Copy-Item .\adapter\config\longmemeval.local.env.example .\adapter\config\longmemeval.local.env -ErrorAction SilentlyContinue
```

编辑 `adapter/config/longmemeval.local.env`，填写 LLM 与 embedding 地址及密钥。若本机尚无 Redis、Elasticsearch 和 Milvus，启动评测专用基础设施；端口与示例环境文件一致：

```powershell
docker compose -f .\benchmarks\docker-compose.yml up -d
```

不加 `-Run` 只生成计划；加 `-Run` 才会写记忆、检索和调用 Answer/Judge 模型。若模型端点是明文 HTTP，显式增加 `-AllowHttpModels`。

## 五个入口

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\benchmarks\run-longmemeval.ps1 -Concurrency 3 -Run -AllowHttpModels
powershell -NoProfile -ExecutionPolicy Bypass -File .\benchmarks\run-personamem-v2.ps1 -PerUser 5 -Run -AllowHttpModels
powershell -NoProfile -ExecutionPolicy Bypass -File .\benchmarks\run-clbench.ps1 -PerCategory 1 -Sampling smoke -Run -AllowHttpModels
powershell -NoProfile -ExecutionPolicy Bypass -File .\benchmarks\run-beam.ps1 -Conversations 1 -PerType 1 -Run -AllowHttpModels
powershell -NoProfile -ExecutionPolicy Bypass -File .\benchmarks\run-locomo-refined.ps1 -Experiment smoke16 -WriteConcurrency 3 -QaConcurrency 3 -Run -AllowHttpModels
```

- LongMemEval：首次运行下载并校验公开 Oracle 数据，固定种子抽取六类各 5 题，共 30 题；沿用仓库原始 LongMemEval Runner、Answer/Judge 与 Top-50/10/20 判分。
- PersonaMem-v2：默认 5 个用户、每人 5 题，同时统计 Top-10/20/50/100。
- CL-Bench：冒烟默认四类各 1 题；正式小样可用 `-PerCategory 10 -Sampling stratified`。
- BEAM：默认一个 100K 对话、十类各 1 题，并自动启动隔离的 Milvus（端口 21530/9191）。
- LoCoMo-Refined：仅评测文本题；`smoke16`、`pilot120`、`full` 分别为 16、120、861 题。

每次运行的选择清单、逐题产物、失败原因和 `summary.md` 都写入 `adapter/outputs/`。断点续跑使用原命令并追加 `-Resume "<批次目录>"`；不希望重跑已失败题时再加 `-SkipFailed`。

数据集和官方评测代码会在首次运行时从上游下载并锁定 revision/checksum；`adapter/data/`、`adapter/vendor/`、`adapter/outputs/` 和本地密钥文件不会提交到 Git。
