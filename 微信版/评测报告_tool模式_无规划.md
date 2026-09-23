# Agent 评测报告

- 评测时间：2026-09-23 09:21:57
- RAG 模式：**工具检索 tool（Agentic RAG）**
- 用例总数：23

## 汇总指标

| 指标 | 数值 |
| --- | --- |
| 任务达成率（用户侧结果） | 95.7% (22/23) |
| 工具调用完全正确率（行为） | 82.6% (19/23) |
| 需要工具的召回率 | 84.2% (16/19) |
| 无需工具的不误调率 | 100.0% (4/4) |
| 多步任务达成率 | 100.0% (3/3) |
| 模型自主判断免工具并答对 | 1 条 |
| 规划触发 / 反思修订 | 0 次 / 0 次 |
| 平均延迟 / P50 / P95 | 1.70s / 1.47s / 4.19s |
| 平均每轮工具调用 | 0.96 次 |
| 平均每轮 token | 2110（输入 46727 / 输出 1805） |

## 分类表现

| 类别 | 任务达成率 | 工具正确率 | 用例数 |
| --- | --- | --- | --- |
| 工具-时间 | 100.0% | 100.0% | 2 |
| 工具-计算 | 100.0% | 100.0% | 4 |
| 工具-天气 | 50.0% | 50.0% | 2 |
| 工具-知识库 | 100.0% | 80.0% | 5 |
| 无需工具 | 100.0% | 100.0% | 4 |
| 多工具组合 | 100.0% | 66.7% | 3 |
| 多步任务 | 100.0% | 66.7% | 3 |

## 用例明细

| # | 类别 | 输入 | 期望工具 | 实际工具 | 工具正确 | 任务达成 | 延迟 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 工具-时间 | 现在几点了？ | get_current_time | get_current_time | Y | Y | 1.36s |
| 2 | 工具-时间 | 今天星期几？ | get_current_time | get_current_time | Y | Y | 1.50s |
| 3 | 工具-计算 | 帮我算一下 128 乘以 37 再加 456 | calculate | calculate | Y | Y | 1.47s |
| 4 | 工具-计算 | (85+15)*3 等于多少 | calculate | calculate | Y | Y | 1.54s |
| 5 | 工具-计算 | 100 除以 8 是多少 | calculate | calculate | Y | Y | 1.44s |
| 6 | 工具-计算 | 2 的 10 次方等于几 | calculate | calculate | Y | Y | 1.66s |
| 7 | 工具-天气 | 北京今天天气怎么样 | get_weather | get_current_time,get_weather | N | N | 4.19s |
| 8 | 工具-天气 | 上海现在冷不冷 | get_weather | get_weather | Y | Y | 3.52s |
| 9 | 工具-知识库 | 专业版多少钱？ | search_knowledge | search_knowledge | Y | Y | 1.33s |
| 10 | 工具-知识库 | 团队版的价格是多少 | search_knowledge | search_knowledge | Y | Y | 1.71s |
| 11 | 工具-知识库 | 我想退款，怎么操作？ | search_knowledge | search_knowledge | Y | Y | 2.12s |
| 12 | 工具-知识库 | 客服邮箱是什么 | search_knowledge | - | N | Y | 0.74s |
| 13 | 工具-知识库 | 免费版有什么限制 | search_knowledge | search_knowledge | Y | Y | 1.10s |
| 14 | 无需工具 | 你好呀 | - | - | Y | Y | 0.59s |
| 15 | 无需工具 | 今天心情还不错 | - | - | Y | Y | 0.65s |
| 16 | 无需工具 | 陪我说说话吧 | - | - | Y | Y | 0.74s |
| 17 | 无需工具 | 你喜欢喝茶吗 | - | - | Y | Y | 1.12s |
| 18 | 多工具组合 | 现在几点？顺便帮我算一下 25 乘以 4 | calculate,get_current_time | calculate,get_current_time | Y | Y | 1.11s |
| 19 | 多工具组合 | 北京今天天气怎么样，适合出门吗 | get_weather | get_weather | Y | Y | 4.60s |
| 20 | 多工具组合 | 专业版和团队版差多少钱？ | search_knowledge | calculate | N | Y | 1.68s |
| 21 | 多步任务 | 先查一下专业版多少钱，再帮我算买三年一共多少 | calculate,search_knowledge | calculate,search_knowledge | Y | Y | 1.72s |
| 22 | 多步任务 | 帮我查一下团队版的价格，然后算一下比专业版贵多少 | calculate,search_knowledge | calculate | N | Y | 1.87s |
| 23 | 多步任务 | 现在几点？顺便算一下 15 的平方是多少 | calculate,get_current_time | calculate,get_current_time | Y | Y | 1.25s |
