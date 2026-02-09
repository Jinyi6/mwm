# Code & Doc Review (AgentMemoryBench)

## 代码实现概览

- 入口在 `src/runner/main.py`：读取实验配置、进行训练模式约束校验、探测/加载 locomo 任务、构建调度序列、初始化记忆机制与执行引擎，并驱动训练/测试循环执行样本。该流程包含对 replay/transfer/offline/online 训练模式的分支处理，以及 locomo 的特殊 session 注入逻辑。【F:src/runner/main.py†L1-L214】
- 调度构建由 `src/runner/builders.py` 统一负责：先获取各任务索引，再基于训练模式决定调度方式（包括 transfer/replay/locomo session 混合或默认调度），offline 模式再进行 train/test 划分。【F:src/runner/builders.py†L52-L167】
- 调度细节在 `src/runner/schedule_utils.py`：实现 locomo session 顺序调度、transfer 调度、replay 调度及 offline locomo 一次性注入 schedule 等逻辑，并提供 locomo 判断与实例加载辅助方法。【F:src/runner/schedule_utils.py†L1-L679】
- 默认调度器在 `src/client/scheduler.py`：支持单任务顺序、单任务 shuffle 以及跨任务全局 shuffle，并在 `cross_task=True` 且 `shuffle=False` 时显式报错。【F:src/client/scheduler.py†L1-L56】

## 代码存在的主要问题 / 风险点

1. **记忆机制配置路径的绝对路径兼容性不足**
   - `build_memory_from_config` 无论传入的 `config_path` 是否为绝对路径，都直接拼接到 `ROOT_DIR`，导致绝对路径会被错误地再拼一层项目根目录，从而无法加载配置文件。建议在拼接前检测 `Path(config_path).is_absolute()` 并有条件使用原始路径。【F:src/runner/builders.py†L29-L44】

2. **Locomo 增强消息与历史记录同步存在缺口**
   - `LocomoSessionWrapper.sync_action` 中在 `use_memory` 后仅对比 `messages` 与 `enhanced_messages` 的内容差异，并通过 `zip(messages, enhanced_messages)` 逐条更新 `history`。若记忆机制插入/删除消息（而非仅修改内容），`zip` 会丢弃新增消息或忽略删减，导致 `history` 与实际 prompt 不一致，进而影响保存结果或后续评估。建议显式处理长度变化或直接用 `enhanced_messages` 重建对应的 history 片段。【F:src/runner/main.py†L381-L458】

3. **Offline + Locomo 的 train/test 切分可能切到 session 注入段**
   - Offline locomo 调度将所有 session 注入标记放在序列开头，随后追加全部 QA。【F:src/runner/schedule_utils.py†L528-L588】
   - 但 `_split_train_test` 仅按位置切分 `schedule`，如果 `train_size` 太小，可能把部分 session 注入留在 test 子集，导致 test 中出现 “仅注入” 或 memory 不完整的情况。建议将 session 注入统一放入 train 段或在切分时跳过注入标记。【F:src/runner/builders.py†L189-L215】

## 文档合理性与准确性检查

1. **README 中记忆机制配置段落重复**
   - README 在同一处连续重复了两次 `memory_mechanism` 的配置片段，容易让读者误以为需要重复配置。建议保留一处并说明 `config_path` 可选即可。【F:README.md†L314-L333】

2. **README “View Results” 对 locomo 分析脚本的指向不准确**
   - README 写明 locomo 任务使用 `analyze_results_for_system_memory.py`，但仓库中明确提供了 `analyze_results_for_personal_memory.py` 用于 locomo，并在脚本头部说明了该用途。建议将 README 的 locomo 指令改为 `python -m src.utils.analyze_results_for_personal_memory ...`。【F:README.md†L364-L372】【F:src/utils/analyze_results_for_personal_memory.py†L1-L6】
