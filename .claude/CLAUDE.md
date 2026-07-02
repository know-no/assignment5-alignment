<!-- CODEGRAPH_START -->
## CodeGraph

In repositories indexed by CodeGraph (a `.codegraph/` directory exists at the repo root), reach for it BEFORE grep/find or reading files when you need to understand or locate code:

- **MCP tools** (when available): `codegraph_explore` answers most code questions in one call — the relevant symbols' verbatim source plus the call paths between them. `codegraph_node` returns one symbol's source + callers, or reads a whole file with line numbers. If the tools are listed but deferred, load them by name via tool search.
- **Shell** (always works): `codegraph explore "<symbol names or question>"` and `codegraph node <symbol-or-file>` print the same output.

If there is no `.codegraph/` directory, skip CodeGraph entirely — indexing is the user's decision.
<!-- CODEGRAPH_END -->

## 知识蒸馏到 notes/

用户希望对话里浮现的非琐碎技术知识点被沉淀下来,方便日后按图索骥、按外延继续学习。所有蒸馏文档放在仓库根的 `notes/` 目录。

**何时写**:在对话中讲清楚了一个值得回顾的概念后,**主动**询问或直接落盘 —— 不要等下一次用户提起。判断"值得回顾"的标准:

- 涉及论文 / 算法 / 设计权衡(不是单纯的 API 用法)
- 容易记错或反直觉(像 `dim=None` 那种)
- 用户在追问"为什么",而不是"怎么做"

**何时不写**:一次性的调试问题、纯仓库结构问题、可以 grep 出来的事实。

**写入约定**:

- 每个文件聚焦一个或一组紧密相关的知识点;文件名 kebab-case,主题清晰
- 用 Markdown,优先用**对比表格**和**形状/伪代码**而不是大段散文 —— 蒸馏的目的是回顾速度
- 用 `[文件名:行号](相对路径#L行号)` 链接锚定到仓库里的真实代码
- 文末必须有 **"外延拓展方向"** 一节,列 5–10 条可勾选的延伸学习线索(论文、对比概念、相关代码、易混淆点),让用户可以按图索骥
- 同步更新 `notes/README.md` 的索引表(主题 + 关联代码锚点)

**风格**:用户母语是中文,蒸馏文档默认中文。代码、术语、专有名词保留英文原文。
