# 普通主链路 BM25 + 抽取保真联合实验

日期：2026-09-23。

`memory-bm25-fidelity.yml` 叠加在 `memory.yml` 上，由 `MemoryProfile=bm25-fidelity` 加载。
使用普通 `agent-memory` 工作副本和原来的 llm/orchestrating 抽取构建链路。
Schema、entity、graph、rerank、rewrite 均保持基线设置（关闭）。

- fuser 从 weighted_rrf 换为仓库注册的 BM25_scored_fuser。
- BM25 使用 k1=1.2、b=0.75、lexical_weight=1，tokenizer 与查询器共享 default。
- 保留 vector=2、keyword=1 通道权重；融合算法为通道归一化与池内 BM25 的 CombMAX。
- globals.extraction_fidelity_mode=true 由 runner 映射为 HTTP 子进程的
  AML_EXTRACTION_FIDELITY=true，再由后端逐次写入 system_metadata.extraction_fidelity_mode。
- 原仓库 run.py 的报告字段按 HTTP 配置记录实际 fidelity 值，QA 提示词和判分不变。

新建实验，重新写入记忆，禁止复用普通基线或 Schema 批次的 Resume：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\run-longmemeval-batch.ps1 -Experiment oracle5 -MemoryProfile bm25-fidelity -Concurrency 6 -Run -AllowHttpModels
```

新目录名为 oracle-30-bm25-fidelity-<时间>-<ID>。仍使用每类 5 题、共 30 题、seed=42 的
Oracle 清单，通过比赛 HTTP Add/Search，Top-200 召回，原仓库 QA @50/10/20。
每题使用新随机隔离范围，基线记忆和成绩不复用。该联合实验的收益不能单独归因于 BM25 或保真。

单题失败继续执行其他题，最终列出失败题。本批次中断后用相同 MemoryProfile 加
`-Resume '<新批次目录>'` 续跑；若不想重跑已失败题，再加 `-SkipFailed`。

离线验证覆盖真实 Runner→HTTP→内存内核、BM25 实例装配、保真写入元数据及抽取提示词、
原 QA 提示词和判分参数一致性；真实公司模型实验由用户手动启动。
