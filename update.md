下面是结合代码和 `summary.md` 实验结果后的修改意见。

## 最重要的发现：condition 训练与生成不一致

这是当前最优先修复的问题，很可能比“HSIC 权重不够”更能解释反事实风格失败。

XMIDI adapter 无论训练 genre 还是 factorial，都保存了 `style`、`genre`、`emotion` 三个标签：[xmidi.py](src/cfmusic/data/adapters/xmidi.py)。

在非 factorial 的 genre 训练中，[conditions_from_batch](src/cfmusic/training/transport_trainer.py) 实际构造的是：

$$
c_{\text{train}}
=
e_{\text{style}}(\text{genre})
+
e_{\text{genre}}(\text{genre})
+
e_{\text{emotion}}(\text{emotion}),
$$

因为它虽然把 `style_id` 设成 genre，却仍然把 batch 里的 `genre_id`、`emotion_id` 传给 condition embedding。

但 [generate_counterfactuals.py](src/cfmusic/commands/generate_counterfactuals.py) 构造 source/target condition 时只提供 `style_id`：

$$
c_{\text{generation}}
=
e_{\text{style}}(\text{genre}).
$$

也就是说，当前 E20/E22 生成是在模型没有训练过的 condition 组合上运行。

更严重的是，非 factorial 的 `contrasting_conditions()` 只修改 `style_id`，却保留原来的 `genre_id` 和 `emotion_id`。wrong-condition 事实上变成：

$$
e_{\text{style}}(\text{wrong genre})
+
e_{\text{genre}}(\text{correct genre})
+
e_{\text{emotion}}(\text{factual emotion}),
$$

这是自相矛盾的 condition。因此 summary 中 100% condition accuracy 不能证明模型真正遵循 genre condition。

### 必须修改

建立唯一的 task-aware condition builder，训练、验证、生成和评价全部调用它：

```text
genre task:
    dataset + task + style=genre
    genre=None
    emotion=None

emotion task:
    dataset + task + style=emotion
    genre=None
    emotion=None

factorial task:
    dataset + task + style=constant/sentinel
    genre=genre_id
    emotion=emotion_id
```

同时：

* non-factorial 的 inactive condition 必须严格为 `None`；
* factorial 不再重复加入 `style=genre`；
* wrong condition 只修改当前激活的轴；
* checkpoint 保存 `condition_schema_version`；
* generation 检查 checkpoint schema，拒绝静默加载旧语义 checkpoint；
* 修复后 E20/E22 需要重新训练，旧 checkpoint 不能作为干净的 genre-only 主结果。

## 在重训前，先对现有 checkpoint 做一次定位实验

现有模型仍可用于判断 condition mismatch 是否为主要原因。建议做一个 \(2\times2\) 检查：

| 条件                      | 权重        |
| ----------------------- | --------- |
| 目前 style-only condition | raw model |
| 目前 style-only condition | EMA       |
| 训练时完整 condition         | raw model |
| 训练时完整 condition         | EMA       |

“完整 condition”具体是：

* abduction：`style=source genre, genre=source genre, emotion=source emotion`；
* prediction：`style=target genre, genre=target genre, emotion=source emotion`。

分别用 4/8/16/32-step Heun，比较：

* latent 与 MIDI 层面的风格 NShift；
* target KAD；
* content ExcessDrift；
* leakage；
* round-trip。

如果完整 condition 明显改善风格迁移，基本即可确认训练/生成 mismatch 是主因。注意：这只能作为诊断，不能把旧模型重新解释成规范的 factorial 模型。

## Style 没有从 noise 中剥离的代码原因

### 1. 32768 维 noise 只约束固定的 128 维投影

[abduction_trainer.py](src/cfmusic/training/abduction_trainer.py) 对 \(64\times512=32768\) 维 noise 使用一个固定 `32768→128` projector，然后仅在这 128 维上计算 HSIC 和 prior matching。

这留下了 32640 维的巨大无约束子空间。模型完全可以把 style 信息迁移到 projector 的零空间，而不明显增加训练 loss。

建议改为多视图、每步重采样：

* 2–4 个随机 Rademacher/orthogonal projections，每个 128–256 维；
* token-wise mean/std 表征；
* 随机 token×channel block；
* 每步或每若干步重新采样投影；
* validation 使用不同 seed、不同 projector。

训练目标可改成：

$$
\mathcal L_{\text{exo}}
=
\lambda_h \operatorname{HSIC}(u,s)
+
\lambda_p \operatorname{SWD}(u,\mathcal N)
+
\lambda_c
\sum_{a<b}\operatorname{MMD}(u_a,u_b).
$$

不要直接把当前固定投影上的 HSIC 权重调大；模型仍可能绕过 projector，或者以牺牲 style control 的方式降低 HSIC。

### 2. prior loss 只匹配均值和逐维方差

当前 `classwise_prior_matching` 没有约束：

* covariance；
* higher-order structure；
* multimodality；
* 不同 style noise 分布之间的整体差异。

建议用随机投影 SWD/MMD 补足，而均值/方差 loss 只作为稳定器。仓库已有 adversary/GRL 模块，但 Stage‑2 trainer 并没有真正连接 `adversarial_weight`；`configs/independence/adversarial.yaml` 目前属于无效配置。可以把 adversary 接入，但最终 leakage probe 必须重新训练、与训练 adversary 独立。

### 3. DDP 下 independence loss 只看单卡 64 个样本

每个 rank 只看到 4 个 style、每类 16 个样本，无法直接估计全局 6 类分布。应当：

* differentiable all-gather projected noise 和 labels；
* 在全局 batch 上计算 HSIC/SWD/MMD；
* 或保证每个 rank 都覆盖所有 style。

否则多卡训练降低的是局部相关性，不一定降低全局 style leakage。

### 4. sampler 在一个 shard 内抽取大量相关 segment

Stage‑2 把 `dataset.shard_ids` 作为 group。每个 style 在一次 batch 中先选一个 shard，再从该 shard 抽多个 segment。连续窗口可能来自同一首作品，使 independence regularizer 把作品身份和风格混在一起。

改成分层采样：

```text
style → unique song/sample_id → one segment
```

每个 batch 中同一首歌至多一个窗口。Stage‑1 也建议使用 genre→song→segment 的层级采样，避免长曲因 segment 多而获得过高权重。

## CFM 本身怎样训练得更好

### 1. 先移除目前的 wrong-condition margin 主损失

当前 condition contrast 对同一条 factual flow path 输入错误 style，并要求其 velocity error 更高。错误 style 下这条 path 并不是一个有效 conditional FM training pair，因此模型可能学到“识别矛盾 condition 的水印”，而不是目标分布的传输机制。

主实验建议先设：

```yaml
condition_objective.weight: 0
```

改为用真正的 endpoint 指标验证条件有效性：

$$
F_s(\epsilon)\sim p(z\mid s).
$$

如果要增强条件控制，可在短程 differentiable generation 后，对每个 style 的生成 endpoint 与真实 style latent 计算 class-conditional MMD/SWD。它仍不需要 paired target。

### 2. 改进 latent normalization

当前统计把 `[B,64,512]` reshape 成 `[-1,512]`，只学习一组共享的 512 维 mean/std。但 VAE 的 64 个 latent token 来自不同 learned queries，位置分布可能显著不同。

优先改为：

$$
\mu,\sigma\in\mathbb R^{64\times512},
$$

即每个 latent position 独立统计 mean/std，并更新 latent-normalization schema/hash。之后再消融：

* per-token normalization；
* per-token normalization + channel whitening；
* 当前共享 normalization。

summary 中只有约 46.9% flattened latent dimensions 活跃，也说明目标分布可能是较薄的低维流形。可以进一步比较：

* deterministic posterior mean；
* posterior sample；
* posterior mean + 小幅 Gaussian jitter；
* flow path 的 `sigma_min=0.01/0.03`。

### 3. 修正 Stage‑2 EMA

Stage‑2 trainer 内部直接使用默认 `EMA decay=0.9999`，没有暴露配置。12k step 后，EMA 对初始 Stage‑1 权重仍约保留：

$$
0.9999^{12000}\approx 30.1\%.
$$

而完整正则到 step 2500 才进入稳定阶段。因此最终生成使用的 EMA 很可能没有充分反映 Stage‑2 改动。

建议：

* 把 Stage‑2 EMA decay 写入配置；
* 比较 `0.999`、`0.9995`；
* validation 同时评价 raw/EMA；
* 按 style–content–leakage Pareto 选 checkpoint；
* 不再无条件默认使用 EMA；
* 记录每项 loss 对模型参数的 gradient norm 和 gradient cosine，避免 RT loss 压过 exogeneity loss。

### 4. 公平训练 OT-CFM

当前 `ot_cfm.yaml` 与基础 CFM 的 hidden size、层数、batch size、训练长度不同，因此 E20 vs E23 同时改变了模型容量、优化预算和 coupling，不能作为论文中的干净 ablation。

OT-CFM 对比必须保持：

```text
同一网络
同一 batch
同一 optimizer
同一训练步数
同一 sampler
同一 Stage-2
只改变 Gaussian–factual latent coupling
```

OT-CFM 的确可能产生更直的路径、降低低 NFE 误差，但它应排在 condition 修复和 normalization 之后。[Conditional Flow Matching / OT-CFM](https://arxiv.org/abs/2302.00482v2)。

另外，Sinkhorn 分支用 `argmax` 选配对可能重复选择同一个 noise，不一定保持 Gaussian 端的排列性质；正式使用前应改成 permutation-preserving rounding。当前 Hungarian 路径不受此问题影响。

## 评价代码

Evaluation方面不再训练Transformer分类器，这一分类器本身训练精度不足，且评估时对于新样本可能OOD，不能作为评估指标。主要判断反事实是否成功的指标改为利用CLaMP 2，将MIDI和Style Text映射到同一Embedding Space，比较相似度，确定音乐的风格。Style Text可以将原来的Style Label编入一个固定Template来操作，例如“This is a piece of rock music"。具体Template怎么写你可以参考其他类似工作的做法。

第二，应当有pitch-class histogram cosine,melody contour或者其他类似指标来评价模型对风格以外的因素如旋律的保留程度。

第三，通过在noise上做probe判断风格和外生噪声的解耦程度。

第四，评估最后生成的MIDI质量的指标。

## 必须补的单元测试

* train/generation 对同一 metadata 构造完全相同的 factual condition；
* non-factorial 的 genre/emotion condition 必须为 `None`；
* factorial intervention 每次只改变一个轴；
* wrong condition 不产生 style/genre 相互矛盾；
* 旧 `condition_schema_version` checkpoint 被拒绝；
* balanced batch 中每类样本来自不同 `sample_id`；
* 多卡 exogeneity loss 与单卡全 batch 结果一致；
* dynamic projector 不同 step 的方向不同；
* iid Gaussian leakage probe 接近 chance；
* raw/EMA 都进入 validation；
* Sinkhorn coupling 保持一对一 assignment。

整体结论是：当前最该做的不是继续加大 HSIC，而是先修复 condition 语义。现有结果可以概括为“VAE 很强、CFM 拟合了条件分布，但反事实接口与训练条件不一致；Stage‑2 的固定低秩约束也允许 style 从未约束维度泄漏”。修复 condition 后再评价 E20，才能判断剩余问题究竟来自 CFM transport、latent geometry，还是外生性正则。
