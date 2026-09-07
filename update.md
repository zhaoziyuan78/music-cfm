## 2. 最优先修正：不要在 source abduction 中使用 CFG

当前 [`ConditionalFlow`](src/cfmusic/transport/conditional_flow.py) 的一个全局 `guidance_scale` 同时作用于：

1. source latent 的反演；
2. source reconstruction；
3. target prediction。

也就是现在实际上做：

$$
u=\widetilde F^{-1}_{s,w=1.5}(z),\qquad
z^{cf}=\widetilde F_{t,w=1.5}(u).
$$

但 CFG 的 guided field 对应的是一个被 sharpen 的分布，而不再是原始 \(p(z\mid s)\)。[Guided Flows](https://arxiv.org/abs/2311.13443) 给出的形式是：

$$
\tilde v_s=(1-w)v_\emptyset+w v_s,
$$

其终点分布近似满足：

$$
\tilde p_s\propto p^{1-w}p_s^w.
$$

因此，用 \(w>1\) 反演来自普通 \(p(z\mid s)\) 的 factual sample，会使得到的 \(u\) 不再可靠地服从共享标准高斯，并可能保留 source style。这与当前 pop/country source stickiness 和剩余 leakage 是一致的。

应改成：

$$
u=F^{-1}_{s,w_{\rm abd}=1}(z),
\qquad
z^{cf}=\widetilde F_{t,w_{\rm pred}}(u),
\quad w_{\rm pred}>1.
$$

具体修改 API：

```yaml
classifier_free_guidance: true
condition_dropout: 0.1

abduction_guidance_scale: 1.0
reconstruction_guidance_scale: 1.0
prediction_guidance_scale: 2.0
```

在验证集扫描：

```text
prediction scale: 1.0, 1.5, 2.0, 3.0, 4.0
abduction scale: 固定 1.0
```

这是最适合你当前取舍的改动：**提高 target guidance 只改变 \(u\rightarrow z^{cf}\)，不会改变已经得到的 abducted noise，因此能牺牲内容换风格，而不直接增加 noise leakage。**

## 3. 现有重标注仍然有一层 mismatch

[`relabel_xmidi_clamp2.py`](src/cfmusic/commands/relabel_xmidi_clamp2.py) 的流程是：

1. 对 107,975 首完整 MIDI 计算 CLaMP2 embedding；
2. 用单个文本模板的最近 prompt 硬分配标签；
3. 把整首曲子的标签复制给它的所有短 segment。

但 CFM 实际训练和生成的是短 segment，当前生成样本平均只有约 30 beats。完整乐曲的 CLaMP2 标签不保证对每个局部 segment 仍成立。因此数据和 evaluator 仍未真正对齐。

而且代码虽然计算了 `nearest_prompt_margin`，却：

* 没有过滤低 margin 歌曲；
* 没有把置信度传入 latent index；
* 没有 prompt ensemble；
* 没有保证六个 pseudo-class 平衡。

建议重新构造训练集：

* 对实际 latent segment 做 CLaMP2 标注；
* 或至少只保留“segment 与整曲标签一致”的窗口；
* 每类分别保留 top 50%–70% margin 的高置信样本；
* 按 unique song 平衡采样，而不是按 segment 平衡；
* 使用 8–16 个 prompt template，平均文本 embedding；
* 对每个 prompt 的 logit bias 做真实验证集校准；
* traditional 若仍无法形成紧致簇，应考虑从六类主实验中删除或重新定义。

尤其需要先补三个 ceiling：

1. 原始训练 segment 对其 pseudo-label 的 CLaMP2 top-1；
2. VAE reconstruction 的 pseudo-label retention；
3. 真实 target segment 的 per-class CLaMP2 top-1。

如果 traditional 的真实 segment ceiling 本身只有 20%，生成模型的 6% 就不能全部归因于 CFM。

## 4. 不重训就可以做的第二个增强：source-repulsive guidance

逐样本结果显示失败样本主要表现为 target similarity 不够，同时 source similarity 仍偏高。可以在 target forward 中使用三分支 guidance：

$$
v_{\rm edit}
=
v_\emptyset
+\omega(v_t-v_\emptyset)
-\rho(v_s-v_\emptyset).
$$

其中：

* \(\omega\) 拉向 target；
* \(\rho\) 排斥 source；
* abduction 仍固定使用 \(w=1\)。

推荐扫描：

```text
ω: 1.5, 2.0, 3.0
ρ: 0.0, 0.25, 0.5, 1.0
```

这比重复执行反事实更合适。仓库里的 [`diagnose_repeated_intervention.py`](scripts/diagnose_repeated_intervention.py) 已经明确记录：第二次干预可以强化 target signal，但新 abducted noise 更容易被 source label 分类，恰好违背你限制 leakage 的目标。

## 5. 下一版 CFM 的训练方案

当前 [`cfm.yaml`](configs/transport/cfm.yaml) 使用 independent coupling、uniform time，而 `condition_objective` 完全关闭。高维 independent CFM 中，模型容易依赖 \(z_t\) 本身预测 factual endpoint，而忽略 style condition，特别是在接近数据端的 \(t\) 上。

建议下一版采用：

### 数据与 batch

```yaml
sampling:
  balance_by_style: true
  unique_song_per_batch: true
  confidence_weighted: true
```

不要只使用当前 `class_balance_exponent=0.5` 的 loss weighting；需要让每个 batch 真正包含平衡的 style 和不同歌曲。

### Conditional OT-CFM

仓库已经实现了按 style 分组的 Hungarian OT coupling：

[`ot_coupling.py`](src/cfmusic/transport/ot_coupling.py)

它保持高斯样本的一对一排列，可直接组合 CFG。Minibatch OT coupling 能降低训练方差并产生更直的 flow trajectory，[Multisample Flow Matching](https://arxiv.org/abs/2304.14772) 和 [OT-CFM](https://arxiv.org/abs/2302.00482) 都给出了相应依据。

保持网络、batch 和训练预算不变，只改：

```yaml
flow:
  path: ot
  ot:
    solver: hungarian
    cost_projection_dim: 128
```

### 增强靠近 noise 端的条件学习

实现混合 time sampling：

$$
t\sim0.5\,U(0,1)+0.5\,\mathrm{Beta}(0.5,1).
$$

靠近 \(t=0\) 时，state 几乎没有 factual-style 信息，模型必须依靠 condition 区分类别。这比直接提高现有 wrong-condition margin 更干净。

### 加入真正的 endpoint matching

不要直接用 CLaMP2 反向训练，否则会造成 evaluator hacking。利用仓库已有的 MMD/SWD，在少量可微生成 endpoint 上训练：

$$
\mathcal L_{\rm endpoint}
=
\sum_s
\operatorname{SWD}\bigl(F_s(u),Z_s^{real}\bigr),
\qquad u\sim\mathcal N(0,I).
$$

建议：

```yaml
endpoint_matching:
  enabled: true
  interval: 8
  solver_steps: 4
  samples_per_style: 4
  weight: [0.01, 0.05, 0.1]
```

这里的 \(u\) 是独立采样的，所以只能通过 condition 产生 style，不会鼓励把 style 塞进 noise。

### 保留一阶段，但恢复轻量 exogeneity regularization

不必恢复独立 Stage 2。把它变成统一训练中的稀疏正则：

$$
L=L_{\rm CFM}
+\lambda_eL_{\rm endpoint}
+\lambda_xL_{\rm exo}
+\lambda_rL_{\rm RT}.
$$

其中 \(L_{\rm exo}\) 每 8 step 执行一次、使用无 CFG 的 4-step abduction，并采用多个动态随机投影上的：

* HSIC；
* cross-class SWD/MMD；
* prior matching。

当前 round-trip 权重 0.25 偏大，而且训练时只有 2-step guided inversion。建议消融：

```text
roundtrip weight: 0, 0.05, 0.25
roundtrip steps: 4
roundtrip guidance: 1.0
```

逐样本结果中 round-trip MSE 的中位数为 0.026，但 90% 分位达到 0.761、最大 3.665；它有明显长尾，却与 style success 没有强关系，所以不是当前提升迁移率的主杠杆。

