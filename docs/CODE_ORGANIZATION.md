# 代码整理记录（2026-10-19）

> 历史快照：本文记录 2026-10-19 的目录和验证状态。2026-10-29 又移除了已完成的一次性 `tools/` 脚本；当前入口以仓库根目录 `README.md` 和各工具索引为准。

## 当前入口

正式清洗从 `src/clean_tools/run_618_pipeline.ps1` 开始；基础转换器为
`tools/conversion/00_convert_618_to_lerobot.py`，公共能力放在 `packages/`。
通用与特定批次转换工具的边界见 `tools/conversion/README.md`。

## 本次整理

- 根目录的训练、启动和 `scripts_stage/` 部署文件归入 `experiments/h20/`。
- 根目录四个评估辅助脚本归入 `tools/diagnostics/h20/`，五个下载脚本归入 `tools/transfer/h20/`。
- 旧 `package_for_transfer.py` 归入 `tools/legacy/deployment/`；其依赖的旧目录已迁出。
- 转换器和成品合并器中重复的腰部顺序数值检测归入 `packages/body_order.py`，保持已有返回语义。
- 清理成品合并器被下一行立即覆盖的重复赋值，以及未使用的 action 列名常量。
- 修正 adapter 中过时的身体顺序注释。显式顺序参数仍是 `TODO.md` 中的待办。
- 将 `artifacts/`、嵌套旧工作目录和测试缓存排除出 Git，保留其本地文件。

没有直接删除相似的批次脚本：训练入口的 1008 与 1015 版本、不同批次合并和
action 重推导工具均存在语义差异。历史快照明确标记用途，避免混入正式主流程。
原始数据、转换结果、模型、远端作业和本机 Python 依赖未改动。

## 验证

- 整理前：现有 18 项回归测试全部通过。
- 整理后：25 项回归测试全部通过（新增顺序提示与歧义检查）。
- `tools/testing/test_process_routing.py`：Windows 进程池路由及结果顺序通过。
- 转换、成品合并和完整轨迹划分三个 CLI 的 `--help` 加载通过。
- 扫描时 113 个 Python/JSON 文件语法解析通过。
- `git diff --check` 通过。

测试临时目录使用仓库内 `output/code_cleanup_tmp/`，避免当前沙箱系统 TEMP 的权限问题。
上述是源码整理验证，不代表重新执行了真实数据批量转换、GPU 训练或远端历史脚本。
