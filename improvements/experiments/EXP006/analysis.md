# EXP006：实验协议公平性核查（Protocol A vs B）

## 核查结论

EXP003/EXP004/EXP005 实际使用的是 **Protocol A（Frozen-model Transfer）**，
不是标准 SMORE 的 **Protocol B（Subject-specific Self-supervised Adaptation）**。

- **EXP003**：复用 `day4_l1`（在 g_gaussian 上训练）的 checkpoint，在 g_gaussian /
  g_box / g_fermi 上推理。→ 冻结模型跨退化迁移。
- **EXP004**：`IXI002_multideg` 在 `_sinc_r2.nii.gz`（即 g_gaussian）上训练，在三种
  实际退化上推理。→ 训练数据源始终是 g_gaussian。
- **EXP005**：同 EXP004，4 名受试者。→ 同上。

## 影响范围

"固定 Gaussian 模型在 Box/Fermi 上差于 Sinc"这一结论，严格来说是**冻结模型的跨
退化迁移能力有限**，不能直接等同于"标准单受试者自监督 SMORE 在失配条件下表现差"。

原因：标准 SMORE 会**针对当前待重建的 LR（g_box / g_fermi）自身重新做自监督适配**，
而 Protocol A 没有做这件事。两类协议回答的问题不同：

- Protocol A：一个已训练模型能否跨退化复用？（跨退化迁移）
- Protocol B：针对每种实际 LR 做个体自监督适配后，方法是否稳健？（标准 SMORE）

## 修正

- 不删除 EXP003–005，保留为 Protocol A 的探索性证据。
- 上一轮"多退化修复失配"的结论**必须在 Protocol B 下重新验证**才能成立。
- 本轮以 Protocol B 为主协议，Protocol A 为补充。

## 代码修正

- `scripts/train_smore.py` 的 `--g-path` 已支持传入任意 g（含 g_box/g_fermi），
  因此 Protocol B 无需改核心代码，只需为每种实际退化生成对应的 g 并各自训练。
