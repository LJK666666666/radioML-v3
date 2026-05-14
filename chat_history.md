# Chat History

## 2026-04-09 会话 1

**用户提示词**:添加 SNR 出现偏差下的鲁棒性实验(模拟 SNR 随机误差下的预测准确率);将 SNR 预测神经网络的损失函数改成距离感知损失(参考 guide/update1.md)。

**工作内容**:
- `src/snr_predict.py`:
  - 新增 `make_gaussian_soft_label_matrix(C, sigma)` 构建 (C,C) 高斯软标签矩阵。
  - 新增 `GaussianSoftLabelLoss`:log_softmax + 软标签交叉熵(update1.md 方法1)。相邻类别误判轻罚,远距误判重罚。
  - `train_snr_predictor` 增加 `soft_label_sigma=1.0` 参数,默认启用软标签 CE;传 0 回退到硬 CE。
- `src/main.py`:
  - 新增 `--mode snr_robust`,新增 CLI `--snr_noise_stds`(默认 `0 1 2 4 6 10` dB)。
  - 流程:raw split → 对 `snr_test` 加高斯噪声并吸附到最近合法 SNR 档位 → 用扰动后 SNR 对 `X_test` 重新去噪 → 调用 `evaluate_selected_models` 评估(per-SNR 分组仍用真实 SNR)。
  - 每个噪声 std 的结果保存至 `results/snr_robust{suffix}/noisestd_{σ}/`,并存 `perturbed_snr.npz`。
  - `--models` 必填校验扩展到 `snr_robust`。
  - `denoise_split` 从 `denoise` 模块导入。

## 2026-04-09 会话 2

**用户提示词**:运行 `--mode evaluate --use_predicted_snr` 时,模型加载路径错误。训练时始终用真实 SNR,不应存在 `iqformer_model_..._predsnr_snr_cnn_stratified.pt` 这样的权重名。

**工作内容**:解耦模型权重后缀与结果目录后缀。
- `evaluate_model_variants` / `evaluate_selected_models` 新增 `results_suffix` 参数(默认等于 `suffix`)。评估结果子目录改用 `results_suffix`,模型权重仍用 `suffix`。
- train 模式:`suffix` 不再拼接 `_predsnr_{snr_model}`(训练一律真实 SNR)。
- evaluate 模式:`suffix` 用于定位权重(无 predsnr);`results_suffix` 在 `--use_predicted_snr` 时附加 `_predsnr_{snr_model}`,避免覆盖真实 SNR 评估结果。
- `snr_robust` 模式已使用无 predsnr 的 `train_suffix`,无需改动。

## 2026-04-10 会话 3

**用户提示词**:按照 CLAUDE.md 当前任务 7 项要求修改论文 `paper/CL/double3/double_done_3.tex`（移除旋转增强、移除复数残差网络改用IQFormer消融、更新模型列表、添加Related Work、添加SNR鲁棒性章节、合并2016a/2016b删除2018/2022、重新benchmark推理时间）。将PETCGDNN指标调整为原数据63.28%。修改benchmark代码支持PyTorch模型并重新测试。

**工作内容**:
- `paper/CL/double3/double_done_3.tex` 大幅重写:
  - 移除旋转数据增强（小节、星座图、公式、表格Aug列、贡献条目、结论）
  - 移除复数残差神经网络（小节、架构图、表格、收敛曲线分析）
  - 更新模型列表:移除MCNET/CGDNN/ComplexCNN/ResNet/Complex-ResNet-mini/Complex-ResNet,保留ULCNN/MCLDNN/PETCGDNN,添加AMC-Net/FEA-T/IQFormer
  - 新增 Related Work 章节（DL for AMC + Signal Preprocessing）
  - 新增 SNR Robustness 章节（σ_err=0~10dB, IQFormer+FEA-T结果表）
  - 合并 RML2016.10a/10b 结果为单表,删除 RML2018.10a/RML2022.01a 图文
  - 消融实验改用 IQFormer（Low/Medium/High SNR 准确率对比）
  - 引言更新认知无线电 SNR 估计主题,贡献列表更新
  - 效率表填入 benchmark 实测数据,移除 FLOPs 列
  - PETCGDNN +GPR 精度改为 63.28%（原论文值）
- `paper/CL/double3/references.bib`: 添加 FEA-T (chen2024feat) 和 IQFormer (shao2025iqformer) 引用
- `script/snr_accuracy/plot_iqformer_snr_accuracy.py`: 新建脚本生成 IQFormer SNR 准确率对比图
- `paper/CL/double3/figure/snr_accuracy/iqformer_snr_accuracy.png`: 生成图片
- `src/model_benchmark.py`: 重写支持 PyTorch 模型,添加 `--model_name`/`--batch_all` 模式
  - Keras改用`@tf.function`编译前向、PyTorch用`torch.no_grad()`，两边均为各自最优推理路径
  - 最终结果:AMC-Net 1.350ms, MCLDNN 1.573ms, IQFormer 0.032ms, FEA-T 0.026ms, PETCGDNN 0.414ms, ULCNN 0.109ms
  - 发现 Keras vs PyTorch 框架差异导致不公平比较
  - 利用 IQFormer 仓库已有的 PyTorch 版 MCLDNN/PETCGDNN/AMC-Net，新写 PyTorch 版 ULCNN (`src/model/ulcnn_torch_model.py`)
  - 新建 `src/benchmark_pytorch_all.py`：统一 PyTorch 框架 + IQFormer 原论文计时方式（完整 DataLoader 评估流程）
  - 最终结果(bs=400, 3 runs avg): AMC-Net 0.028ms, MCLDNN 0.028ms, IQFormer 0.039ms, FEA-T 0.020ms, PETCGDNN 0.010ms, ULCNN 0.016ms

## 2026-04-10 会话 3 (续前次上下文压缩)

**用户提示词**: 论文修改三项任务

**工作内容**:

### 任务1: 论文18dB分类器设计思路说明
- `paper/CL/double3/double_done_3.tex`: 在超参数优化段落中强调设计动机——高噪信号去噪的目标是接近低噪信号的特征域，因此用18dB（最高SNR）数据训练的分类器作为"干净信号特征空间"的参考模型来确定最优核函数超参数

### 任务3: 基线去噪方法公平对比（微调后再测试）
- 新建 `script/finetune_experiments.py`: 实验脚本，支持任务2(SNR误差微调)和任务3(基线去噪对比)
- 从无去噪预训练权重出发，在移动平均/小波去噪训练数据上微调IQFormer后再测试
- 结果:
  - None (baseline): 63.08%
  - Moving Average (w=5, finetuned): 63.51% (+0.43%)
  - Wavelet (db4, L=2, finetuned): 63.20% (+0.12%)
  - GPR (proposed, full training): 67.48% (+4.40%)
- 更新论文中去噪方法对比段落（取消注释并更新数据）

### CRB分析: SNR估计误差的理论下界
- 引用Pauluzzi 2000论文的CRB公式，在论文鲁棒性分析部分添加说明
- 以-10dB为例: N_sym=1024时误差约±0.6dB，N_sym=64时误差约±3dB
- 将CRB分析结果与鲁棒性实验结论关联起来

### 任务2: SNR误差微调实验 (已完成)
- 运行 `script/finetune_experiments.py --task 2 --sigma_err 3.0 --finetune_epochs 50`
- 从精确SNR的GPR去噪权重出发，在精确+误差(σ=3dB)合并训练集上微调，误差去噪验证集做调度和早停
- 微调15个epoch后早停，best val_acc=0.6525
- 结果：微调模型在所有误差水平下均优于原始模型
  - σ_err=2dB: +0.52%, σ_err=4dB: +1.58%, σ_err=10dB: +2.56%
  - 交叉点从~4dB延伸到~6dB（微调模型在σ_err≤6dB时仍高于无去噪基线63.08%）
- 更新 `script/plot_snr_robustness.py`: 添加微调曲线和改进区域填充
- 重新生成 `paper/CL/double3/figure/snr_robustness.png`
- 更新论文 `paper/CL/double3/double_done_3.tex`:
  - 鲁棒性分析段落添加error-aware fine-tuning方法描述和结果
  - 贡献列表更新：添加误差感知微调策略
  - 结论更新：4dB→6dB容忍范围

## 2026-04-11 会话 4

**用户提示词**: 运行任务2（SNR误差微调实验）并将结果添加到论文中。

**工作内容**:
- 运行 `script/finetune_experiments.py --task 2 --sigma_err 3.0`，微调IQFormer模型
- 更新 `script/plot_snr_robustness.py` 添加原始vs微调双曲线
- 重新生成 `paper/CL/double3/figure/snr_robustness.png` 含双曲线和改进区域
- 更新 `paper/CL/double3/double_done_3.tex` 鲁棒性分析、贡献列表、结论
- 补充GPR在GPU上速度(0.004ms)与CPU(0.005ms)接近的说明
- 修正谱分解加速比说明：254×为渐近复杂度比（省略常数因子），838×实测加速包含算法复杂度降低+向量化批处理实现优势两部分
  - benchmark验证：GPU批量标准求解(无谱分解)与谱分解CPU速度几乎相同(0.98s vs 0.98s)，说明加速主要来自向量化消除22万次Python循环
  - 重写论文中加速比段落，明确区分渐近复杂度比和实测加速比

## 2026-04-11 会话 5

**用户提示词**: 将 letter1.tex 拆分为匿名化的 Main Manuscript 和 Title Page 两份文件。

**工作内容**:
- 创建 `letter1_manuscript.tex`：匿名化处理（作者→Anonymous，移除简介/照片，GPU型号泛化，代码链接改为"upon acceptance"）
- 创建 `letter1_titlepage.tex`：包含标题、作者姓名与单位、通讯地址邮箱、致谢

## 2026-04-11 会话 6

**用户提示词**: 以资深审稿人视角，结合模拟审稿意见(opinion1.md)，对letter1论文进行学术写作深度重构。

**工作内容**:
- 全面重写 `letter1_manuscript.tex` 和 `letter1.tex`，核心改进：
  - **叙事主线重构**：提出"task-entanglement"概念，将GPR去噪定位为"decoupled denoising-then-classification"范式
  - **Abstract**：从问题本质出发（端到端网络被迫同时去噪+判别），补充10b最佳结果(68.89%)
  - **Introduction**：分析AMC领域去噪预处理空白的两个原因（非自适应方法不适配38dB动态范围 + GPR计算量禁止性），弱化"inevitably learn noise"为更严谨的optimization trajectory论述
  - **Methodology**：将超参数优化重新包装为"domain-alignment"准则，论证其优于MSE等重建指标；明确线性长度尺度是数据驱动发现而非先验假设；添加谱分解适用条件说明（固定信号长度，AMC自然满足）
  - **Results**：重命名为"Architecture-Agnostic Effectiveness"，强调12/12 model-dataset组合全部提升；添加实验公平性说明（统一2×128 I/Q输入，统一训练条件）；解释高SNR不过度平滑的原因；量化GPR开销为最快分类器的13%
  - **Robustness**：添加GPR对SNR扰动天然鲁棒的机理解释（谱加权的平滑性）；解释error-aware fine-tuning暴露分类器于SNR mismatch分布
  - **Conclusion**：升华为"task decoupling"视角，定位为"architecture-agnostic performance amplifier"
- 在 `references.bib` 中补充 `dobre2007survey` 引用

## 2026-04-12 会话 7

**用户提示词**: 根据opinion2.md第二轮模拟审稿意见修改论文或反驳。

**工作内容**:
- 论文修改（同步letter1_manuscript.tex和letter1.tex）：
  - Datasets段落：补充RML2016通过GNU Radio生成，包含multipath fading/SRO/CFO等复合信道损伤
  - 长度尺度段落：补充同参数直接迁移至RML2016.10b的跨数据集验证
  - 谱分解段落：补充变长信号按长度分组的退化方案
  - 基线对比段落：补充DnCNN为2D图像去噪不适用于1D I/Q、DAE需额外训练且效果有限的说明
  - 微调段落：补充σ_err=3dB选择的CRB理论依据
  - Conclusion：添加AWGN假设的局限性说明
- references.bib补充zhang2017dncnn引用
- 对各条意见的合理性判断和反驳要点（见下方）

## 2026-04-12 会话 8

**用户提示词**: 将迭代审稿-修改流程整理为可复用的skill。

**工作内容**:
- 创建 `~/.claude/skills/paper-review/SKILL.md`
- 5阶段工作流：Preparation → Review(subagent) → Triage(editor) → Revision → Convergence Check
- 最多3轮迭代，防止无限循环
- 使用方式：`/paper-review path/to/paper.tex [journal-name]`
- 后续将叙事升华提示词整合为 Phase 0.5 (Narrative Elevation)，通过 `--elevate` 标志触发
  - 与后续的缺陷修补(defect patching)本质不同：升华是抬ceiling，修补是堵漏洞
  - 设计为先输出rewrite plan → 主agent过滤over-claim → 用户确认 → 执行

## 2026-04-12 会话 9

**用户提示词**: 论文中不应点名DnCNN说不适用，应强调AMC领域去噪研究空白。

**工作内容**:
- 删除两个tex文件中点名DnCNN/DAE的防御性表述
- 替换为：AMC领域去噪预处理研究空白 → 非自适应基线失败解释了空白成因 → SNR自适应去噪的必要性

## 2026-04-12 会话 10

**用户提示词**: 分析论文可压缩之处和突兀表述。

**工作内容**:
- 压缩贡献列表：3条bullet从冗长句改为精炼句（去除Abstract中已有的重复数字）
- 删除Results段末与Introduction重复的"AMC领域去噪空白"论述
- Datasets "Importantly,..." → 去掉旗帜词"Importantly"，改为自然陈述
- 长度尺度段末 "demonstrating cross-dataset transferability of the learned parameterization" → 删除过度包装的尾巴
- Conclusion "We acknowledge that..." → "The current formulation targets..."（自信而非自谦）

## 2026-04-14 会话 11

**用户提示词**: 将5页论文压缩到4页（含参考文献），总篇幅4页免页面费。Letter1作为备份不修改，复制到Letter2进行压缩。非匿名版只记录不同信息，匿名版定稿后复制共同内容。references.bib不精简（未引用条目不会出现在PDF中）。

**工作内容**:
- 复制 Letter1 → Letter2 作为工作副本
- 合并匿名/非匿名版：letter1_manuscript.tex 为完整匿名版（主工作文件），letter1.tex 仅记录差异（作者、biography）待定稿后填充
- 正文压缩（5页→4页）：
  - 删除 Figure 3（时域图），保留星座图（功能重叠）
  - 删除 Table 2（SNR分层表），关键数字并入正文
  - 压缩 Spectral Decomposition 实现细节（254×推导+Python overhead解释→一行结论）
  - 压缩非自适应基线对比段（15行→4行）
  - Introduction 模型列举缩减（删DenseNet引用，合并引用）
  - Conclusion future work 三项→一句话
  - 压缩 Robustness 段落（CRB展开→一行、鲁棒性解释+fine-tuning细节精简）
  - 压缩 Computational Overhead 段落（删GPU测量细节）
  - 压缩 domain-alignment 和 adaptive L 段落冗余解释
- 投递前修正：Experimental Setup 加一句解释 GPR 在非纯AWGN数据集上仍有效的原因
- 编译确认4页，无LaTeX警告

## 2026-04-14 会话 12

**用户提示词**: 编辑退回修改：(1) Figure 1文字太小模糊 (2) 作者名应为"Author 1"而非"Anonymous"

**工作内容**:
- 作者名修改：`Anonymous` → `Author 1`（`\author` 和 `\markboth` 两处）
- Figure 1 重新生成（`script/snr_accuracy/plot_iqformer_snr_accuracy_letter2.py`）：
  - 移除内嵌小图（文字最小的部分，7pt刻度标签）
  - 增大所有字体：标签18pt、刻度14pt、图例15pt
  - DPI从300提升到600
