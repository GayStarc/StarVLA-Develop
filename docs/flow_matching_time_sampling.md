# Flow Matching 中时间 t 的采样与裁剪说明

## 为什么采样出的 t 需要裁剪 / 会出现“异常值”？

### 1. 采样方式

- **starVLA**：`t = (noise_s - sample) / noise_s`，其中 `sample ~ Beta(noise_beta_alpha=1.5, noise_beta_beta=1.0)`，`noise_s=0.999`。
- **vla-scratch**：先用 `Beta(time_dist_beta=1.5, time_dist_alpha=1.0)` 采样得到 `sample`，再 `t = sample * 0.999 + 0.001`（见 `vla_scratch/policies/utils/diffusion.py`）。

Beta 分布定义在 **(0, 1)** 上，采样值可以**任意接近 0 或 1**（只是概率小）。因此：

- 在 starVLA 中，当 `sample` 接近 0 时，`t` 接近 **1**；当 `sample` 接近 0.999 时，`t` 接近 **0**。
- 也就是说，**未做任何限制时，t 可以无限接近 0 或 1**，这就是“异常值”的来源——并非公式错误，而是分布本身的边界行为。

### 2. 数值问题出在哪里？

- **t 非常接近 0**：  
  `noisy_trajectory = (1-t)*noise + t*actions` 几乎全是 `noise`，`velocity = actions - noise` 的尺度可能很大，若再参与平方损失或反向传播，容易梯度爆炸或出现 NaN。

- **t 非常接近 1**：  
  轨迹几乎全是 `actions`，另一端（noise）的贡献极小，对模型和优化来说也是极端情况；同时 `t_discretized = t * num_timestep_buckets` 可能顶到 1000，若没有 clamp，离散 timestep 可能越界或加重嵌入的不稳定。

- **t 严格等于 0 或 1**：  
  某些实现里可能涉及除以 t、除以 (1-t)、或 log(t) 等，会变成除零或 log(0)，直接导致 NaN/Inf。

因此，**从数值稳定性出发，需要避免 t 太靠近 0 或 1**，做法就是对 t 做范围限制（等价于对采样结果做裁剪或线性映射）。

### 3. vla-scratch 的实现（参考）

路径：`/mnt/cpfs/guchenyang/Code/vla-scratch/vla_scratch/policies/utils/diffusion.py`

```python
def sample_clamped_time(
    time_dist: torch.distributions.Distribution,
    shape: torch.Size,
) -> torch.Tensor:
    """Sample diffusion timesteps with a small clamp to avoid numerical issues."""
    return time_dist.sample(shape) * 0.999 + 0.001
```

- 直接从 Beta 采样得到 `sample in (0, 1)`，再线性映射到 **[0.001, 1.0]**。
- 注释明确写明：*"Sample diffusion timesteps with a small clamp to avoid numerical issues."*
- 这样 t 永远不会等于 0，且下界固定为 0.001，避免上述数值问题。

Flow 公式（同文件所在 policy 中）：

- `u_t = noise - actions`，`noisy_actions = actions + timestep * u_t`，即 `x_t = (1-t)*actions + t*noise`（t 从 0 到 1 表示从 data 到 noise）。
- 与 starVLA 的 `(1-t)*noise + t*actions` 只是 t 的语义方向相反，数学上等价。

### 4. starVLA 的对应修改

在 `LayerwiseFM_ActionHeader.py` 中：

1. **`sample_time`**  
   - 保留 `t_raw = (noise_s - sample) / noise_s` 的语义（与原有 flow 方向一致）。  
   - 将 `t_raw` 线性映射到 **[t_min, t_max]**，例如 `t_min=1e-5`、`t_max=1-1e-5`，即  
     `t = t_min + (t_max - t_min) * t_raw`。  
   - 这样 t 不会触及 0 或 1，与 vla-scratch 的“小 clamp 避免数值问题”一致。

2. **`t_discretized`**  
   - 在 `t_discretized = (t * num_timestep_buckets).long()` 之后，对 `t_discretized` 做  
     `clamp(0, num_timestep_buckets - 1)`，  
   - 避免离散 timestep 越界或对 timestep embedding 产生异常索引。

## 小结

| 项目         | 说明 |
|--------------|------|
| t 为何会异常 | Beta 采样可任意接近 0/1，导致 t 接近 0 或 1，带来大 velocity、边界 timestep、或除零/log(0)。 |
| vla-scratch  | 用线性映射把 Beta 采样压到 [0.001, 1.0]，注释写明为避免数值问题。 |
| starVLA      | 对 raw t 做线性映射到 [t_min, t_max]，并对 t_discretized 做 clamp，与 vla-scratch 思路一致。 |

这样既保证了和 vla-scratch 相同的“避免极端 t”的设计，又保持了 starVLA 原有的 flow 方向和配置（如 `noise_s`）语义。
