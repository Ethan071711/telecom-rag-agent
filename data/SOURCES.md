# 数据来源说明

## 结论

`data/` 下的全部资料都是本项目作者**自行编写**的演示数据，不是从任何运营商网站复制的内容。

- 文档里的运营商“青禾通信”是**虚构**的。套餐名称、价格、时限、App 菜单路径等数字和规则也都是虚构的，只用来演示 RAG 和智能体技术，不代表中国联通、中国移动、中国电信、中国广电或任何真实运营商的资费和业务规则。
- 编写时参考了国内运营商官网常见的栏目组织方式（资费说明、业务办理指南、常见问题），写法上尽量贴近真实客服知识库的风格。
- 唯一的真实信息是 `20_anti_fraud.md` 中的 **96110**（全国反诈预警劝阻专线），属于公开信息。
- 携号转网、实名制等内容只是对常见做法的简化和虚构，不能当作监管规定的原文。真实规则以工业和信息化部及各运营商公布的为准。

## 文件清单

| 文件 | 标题 | 类别 | 来源 |
| --- | --- | --- | --- |
| docs/01_plans_overview.md | 个人及家庭套餐资费总览 | 套餐资费 | 自行编写 |
| docs/02_overage_billing.md | 套外资费与计费规则 | 套餐资费 | 自行编写 |
| docs/03_plan_change.md | 套餐变更规则 | 套餐资费 | 自行编写 |
| docs/04_data_packs.md | 流量包、加油包与流量结转 | 流量 | 自行编写 |
| docs/05_number_portability.md | 携号转网办理指南 | 业务办理 | 自行编写 |
| docs/06_broadband_install.md | 宽带办理、安装与移机 | 宽带 | 自行编写 |
| docs/07_broadband_repair.md | 宽带故障自助排查与报修 | 宽带 | 自行编写 |
| docs/08_intl_roaming.md | 国际及港澳台漫游 | 漫游 | 自行编写 |
| docs/09_suspension.md | 停机保号、欠费停机与复机 | 业务办理 | 自行编写 |
| docs/10_sim_replacement.md | 补卡、换卡与 eSIM 手表 | 业务办理 | 自行编写 |
| docs/11_real_name.md | 实名登记与“一证五号” | 业务办理 | 自行编写 |
| docs/12_invoice_bill.md | 发票开具与账单查询 | 账单发票 | 自行编写 |
| docs/13_recharge.md | 充值缴费与余额 | 账单发票 | 自行编写 |
| docs/14_secondary_cards.md | 副卡与家庭共享 | 套餐资费 | 自行编写 |
| docs/15_contract_phone.md | 合约机与违约金说明 | 业务办理 | 自行编写 |
| docs/16_points_membership.md | 积分与星级会员 | 会员权益 | 自行编写 |
| docs/17_cancel_transfer.md | 销户与过户 | 业务办理 | 自行编写 |
| docs/18_5g_network.md | 5G 网络与 VoNR 高清通话 | 网络 | 自行编写 |
| docs/19_campus_card.md | 青春校园卡办理说明 | 套餐资费 | 自行编写 |
| docs/20_anti_fraud.md | 防范电信网络诈骗与骚扰拦截 | 安全 | 自行编写（96110 为公开信息） |
| docs/21_faq.md | 常见问题 FAQ | 常见问题 | 自行编写 |
| plans.json | 结构化套餐数据（供工具调用） | 套餐资费 | 自行编写，与 01/02/14 号文档一致 |
| userdict.txt | jieba 分词自定义词典 | — | 自行编写 |

## 如何替换成自己的资料

1. 把 Markdown 文件放进 `data/docs/`，开头写上 `title`、`category`、`source` 三项元信息（格式参考现有文件）。
2. 重新运行 `python run.py index --rebuild`，或者直接运行 `python run.py`（程序发现文档有变化会自动重建索引）。
3. 如果要接入真实资费，记得同步更新 `plans.json`，并重写 `eval/eval_set.jsonl` 里的标准答案。
