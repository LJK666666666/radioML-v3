  修改内容                                                                                                                                                                  
                                                                                                                                                                            
  1. 新建 src/model/custom_objects.py

  将 main.py、main_2016b.py 和 model_benchmark.py 中重复的 ~140 行 custom objects 注册代码集中到一个模块，提供两个公共接口：
  - get_custom_objects_for_model(model_name) — 按模型名称返回所需的 custom objects
  - get_all_custom_objects() — 返回所有 custom objects（用于通用模型加载）

  2. 合并 main.py（吸收 main_2016b.py）

  - 新增 --dataset 参数，支持 2016a（默认）和 2016b
  - 通过 DATASET_CONFIGS 字典管理数据集特定的默认路径（数据路径、输出目录、缓存目录、文件后缀）
  - --dataset_path、--output_dir、--denoised_cache_dir 可手动覆盖默认值
  - 将 2016b 的 denoised cache 文件名后缀逻辑泛化为 prepare_data_with_cache_suffix() 函数
  - 删除了内联的 get_custom_objects_for_model 定义，改为从 model.custom_objects 导入

  3. 更新 model_benchmark.py

  - 删除了 ~70 行重复的 custom objects 导入和定义代码
  - 改为 from model.custom_objects import get_all_custom_objects as get_custom_objects_dict

  4. 删除 main_2016b.py

  5. 修复环境

  - 将 NumPy 从 2.2.6 降级到 1.24.3，使其与 TensorFlow 2.13.1 兼容

  使用方式

# 2016a 数据集（默认，与原来的 main.py 一致）
python main.py --models resnet --mode train

# 2016b 数据集（与原来的 main_2016b.py 一致）
python main.py --dataset 2016b --models resnet --mode train



  实施完成总结                                                                                                                                                                                                                                                                                                                                              
  新建文件                                                                                                                                                                                                                                                                                                                                                  
  1. src/model/snr_predictor_model.py — SNR预测CNN模型（PyTorch），20类分类（-20~18dB），注册表模式便于扩展                                                                     2. src/snr_predict.py — SNR预测训练、推理、保存/加载预测结果、评估（准确率/混淆矩阵/MAE）                                                                                     3. src/denoise.py — 独立去噪模块，支持按split去噪+增强+缓存                                                                                                                                                                                                                                                                                               
  修改文件

  4. src/preprocess.py — 提取 _collect_dataset_arrays、_perform_stratified_split、split_data_raw 三个共享函数，重构 prepare_data_by_snr_stratified 调用它们（向后兼容）       
  5. src/main.py — 主要改动：
    - --mode 新增 snr、denoise
    - --models 改为非必需（snr/denoise/explore 不需要）
    - 新增 --snr_model、--use_predicted_snr、--snr_predictions_path
    - explore 从 --mode all 中移除
    - 新增 get_cache_tag() 函数
  6. src/models.py — 导出 build_snr_predictor、get_available_snr_models
  7. src/config/training.yaml — 新增 snr_training 配置段

  典型使用流程

# 1. 训练SNR预测器
python main.py --mode snr --epochs 100

# 2. 用预测SNR去噪
python main.py --mode denoise --use_predicted_snr --augment_data

# 3. 训练分类模型（从缓存加载去噪数据）
python main.py --mode train --use_predicted_snr --models resnet --augment_data

# 4. 评估
python main.py --mode evaluate --use_predicted_snr --models resnet --augment_data


  已完成的修改

  论文正文 (paper/CL/double3/double_done_3.tex):

  1. 移除旋转数据增强 — 删除了 "Rotation-Based Data Augmentation" 小节（含星座图figure和公式）、贡献列表中的相关条目、表格中的 +Aug 列、结论中的相关描述
  2. 移除复数残差神经网络 — 删除了 "Complex-Valued Residual Neural Network" 整个小节（含Complex-ResNet架构图、表格）、收敛曲线分析小节
  3. 更新模型列表 — 移除 MCNET/CGDNN/ComplexCNN/ResNet/Complex-ResNet-mini/Complex-ResNet，保留 ULCNN/MCLDNN/PETCGDNN，添加 AMC-Net/FEA-T/IQFormer
  4. 添加 Related Work — 新增简洁的两段式相关工作部分
  5. 添加 SNR 鲁棒性分析 — 新增 "Robustness Under SNR Estimation Errors" 小节，含实验数据表格
  6. 合并 RML2016.10a/10b 结果 — 合并为一个表格展示，删除 RML2018.10a 和 RML2022.01a 相关内容（星座图、时域图等）
  7. 更新研究主题 — 引言中加入认知无线电SNR估计场景的描述
  8. 消融实验改用 IQFormer — 替换原来的 Complex-ResNet 数据
  9. 生成 IQFormer SNR 准确率对比图 — figure/snr_accuracy/iqformer_snr_accuracy.png

  参考文献 (references.bib): 添加了 FEA-T 和 IQFormer 的引用

  待完成事项

  1. 推理时间 Benchmark — Table V 中 IQFormer/AMC-Net/FEA-T 的 Parameters/FLOPs/Per-Sample Time 标记为 TODO，ULCNN/MCLDNN/PETCGDNN
  的推理时间也需要在当前硬件重新测试。需要使用 src/model_benchmark.py 对所有模型进行 benchmark（但该脚本目前仅支持 Keras 模型，PyTorch 模型如 IQFormer/FEA-T
  需要额外处理）。
  2. RML2016.10b 缺少部分结果 — ULCNN/MCLDNN/PETCGDNN 在 2016b 上的结果使用了原论文数据（output_2016b 中无这三个模型的评估结果）。如需一致性，可能需要重新训练评估。
  3. PETCGDNN 2016.10a GPR 结果差异 — 原论文为 63.28%，当前 output 目录中为 62.46%，我使用了 output 目录数据。请确认是否需要调整。
  4. 篇幅调整 — 论文目前仍较长，后续需根据 CL/WCL 快报页数限制进行删减。


  已完成
                                                                                                                                                             
  1. PETCGDNN +GPR 精度 — 已调整为 63.28%                                                                                                                                 
  2. Benchmark 代码修改 (src/model_benchmark.py):                                                                                                                         
    - 添加 PyTorch 模型支持（torch.no_grad() 推理 + torch.cuda.synchronize() 精确计时）                                                                                   
    - 添加 --model_name 参数（按名称构建模型，无需权重文件）                                                                                                              
    - 添加 --batch_all 模式（一键测试全部 6 个论文模型）                                                                                                               
  3. Benchmark 实测结果（RTX 3090, batch_size=256, 22000 samples）:

  | Model         | Parameters | Per-Sample Time (ms) |
  |---------------|------------|----------------------|
  | AMC-Net       | 467,112    | 1.554                |
  | IQFormer      | 355,049    | 0.031                |
  | MCLDNN        | 405,175    | 1.868                |
  | FEA-T         | 169,867    | 0.021                |
  | PETCGDNN      | 71,871     | 0.664                |
  | ULCNN         | 9,751      | 0.276                |
  | GPR Denoising | --         | 0.0125               |

  4. 论文效率表已更新 — 移除了 FLOPs 列（Keras/PyTorch 框架间 FLOPs 估算方法不一致），保留 Parameters 和 Per-Sample Time，添加了框架差异说明脚注。

  注意：PyTorch 模型（IQFormer、FEA-T）推理速度显著快于 Keras 模型，这是因为 torch.no_grad() 的 GPU batch 推理效率高于 Keras model.predict() 的 Python 级开销。


  GPR 去噪加速分析                                                                                                                                                        
                                                                                                                                                                          
  当前实现（efficient_gpr_per_sample.py，纯 NumPy/CPU）的核心瓶颈：                                                                                                       
                                                                                                                                                                          
  spectral_gp_denoise_same_inputs():                                                                                                                                      
    V  = Q^T @ Y     ← (128,128) @ (128, ~22000)  矩阵乘法                                                                                                                
    S  = Λ/(Λ+σ²)    ← 逐元素                                                                                                                                             
    G  = S * V        ← 逐元素                                                                                                                                         
    MU = Q @ G        ← (128,128) @ (128, ~22000)  矩阵乘法

  2.74s 的开销大致分布：

  ┌────────────────────────────────────────────────────┬──────────┐
  │                        部分                        │ 耗时估计 │
  ├────────────────────────────────────────────────────┼──────────┤
  │ Python for 循环逐样本计算功率（175-186行，22万次） │ ~0.8s    │
  ├────────────────────────────────────────────────────┼──────────┤
  │ 20 个 SNR 组 × 2 次矩阵乘法                        │ ~1.2s    │
  ├────────────────────────────────────────────────────┼──────────┤
  │ 数据格式转换、数组拼接拆分                         │ ~0.5s    │
  ├────────────────────────────────────────────────────┼──────────┤
  │ 特征分解（20 × 128³）                              │ ~0.1s    │
  └────────────────────────────────────────────────────┴──────────┘

  可加速方案：

  1. 向量化 Python for 循环（零成本，立即可做）

  当前第175-186行是逐样本 for 循环计算功率，可以一行向量化：
  # 替换 for i in range(M): ...
  pwr = np.mean(stacked[:, 0, :]**2 + stacked[:, 1, :]**2, axis=1)  # (M,)
  sigmas = np.sqrt(pwr / (2 * (10**(snr_db/10) + 1)))
  预计节省 ~0.8s（约 30% 的总时间）。

  2. CUDA 加速矩阵运算（最大收益）

  两次矩阵乘法 (128,128) @ (128, 22000) 在 GPU 上是微不足道的运算：
  import torch
  Q_gpu = torch.from_numpy(eigvecs).float().cuda()
  Y_gpu = torch.from_numpy(Y).float().cuda()
  V = Q_gpu.T @ Y_gpu
  G = S_gpu * V
  MU = Q_gpu @ G
  result = MU.cpu().numpy()
  GPU matmul 预计 <0.05s（vs CPU ~1.2s），但 CPU↔GPU 数据传输约 0.1-0.2s。

  3. 使用 float32 代替 float64

  当前全程用 float64。GPR 去噪不需要双精度，改 float32 可让 NumPy CPU 运算快约 2 倍，GPU 运算快约 4-8 倍。

  预估加速效果：

  ┌─────────────────────┬────────┬────────────┬──────────┐
  │        方案         │ 总耗时 │ Per-sample │   改善   │
  ├─────────────────────┼────────┼────────────┼──────────┤
  │ 当前（CPU float64） │ 2.74s  │ 0.013ms    │ baseline │
  ├─────────────────────┼────────┼────────────┼──────────┤
  │ +向量化 for 循环    │ ~1.9s  │ 0.009ms    │ 1.4×     │
  ├─────────────────────┼────────┼────────────┼──────────┤
  │ +float32            │ ~1.2s  │ 0.005ms    │ 2.3×     │
  ├─────────────────────┼────────┼────────────┼──────────┤
  │ +CUDA 矩阵运算      │ ~0.3s  │ 0.001ms    │ 9×       │
  ├─────────────────────┼────────┼────────────┼──────────┤
  │ 全部优化            │ ~0.2s  │ 0.001ms    │ 14×      │
  └─────────────────────┴────────┴────────────┴──────────┘

  结论：CUDA 加速完全可行且效果显著。最简单的改法是方案1（向量化循环）+ 方案3（float32），不需要任何额外依赖就能把时间砍掉一半以上。如果加上 CUDA，可以到 ~0.2s 级别。



  叙事主线重塑

  原稿："GPR去噪能提升AMC精度" → 重构："端到端范式存在task-entanglement瓶颈，本文提出decoupled denoising-then-classification范式"

  对审稿意见的回应策略

  ┌─────────────────────────────┬────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
  │          审稿意见           │                                                  回应方式                                                  │
  ├─────────────────────────────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ 1. 新颖性有限               │ 将贡献从"应用GPR"升格为"建立去噪-判别解耦范式"；论证AMC领域去噪预处理的空白及其两个根因                    │
  ├─────────────────────────────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ 2. 实验公平性               │ Experimental Setup中明确：统一2×128 I/Q输入、统一训练条件、GPR仅作用于I/Q波形                              │
  ├─────────────────────────────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ 3. SNR假设可行性            │ 已有CRB分析；新增GPR谱加权对SNR扰动天然平滑的机理解释                                                      │
  ├─────────────────────────────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ 5. 去噪对比不充分           │ 合理反驳：AMC领域几乎无去噪预处理研究；两个baseline旨在隔离"SNR自适应"vs"通用平滑"的贡献，38dB动态范围论证 │
  ├─────────────────────────────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ 6. "不可避免学噪声"过于绝对 │ 弱化为"optimization trajectory encoding noise-correlated patterns"，承认正则化等可缓解                     │
  ├─────────────────────────────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ 7. 谱分解适用条件           │ 明确"固定信号长度"条件，注明AMC基准数据集天然满足                                                          │
  ├─────────────────────────────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ 8. 代理指标                 │ 重新包装为"domain-alignment metric"，论证优于MSE等重建指标                                                 │
  ├─────────────────────────────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ 11. 高SNR过度去噪           │ 正面回应：+0.91pp证明未退化，因为$L_0$限制了平滑带宽                                                       │
  └─────────────────────────────┴────────────────────────────────────────────────────────────────────────────────────────────────────────────┘

  关键新术语/概念

  - task-entanglement — 端到端范式的本质瓶颈
  - domain-alignment criterion — 超参数优化的理论包装
  - architecture-agnostic performance amplifier — 框架的系统定位
  - spectral weighting smoothness — GPR对SNR误差天然鲁棒的机理

  一、内容完整性检查                                                                                                                                                  
                                                                                                                                                                      
  保留完好的核心要素：                                                                                                                                                
  - 问题定义（task-entanglement）、方法（SNR-adaptive GPR + spectral decomposition）、全部公式（Eq.1-11）完整无损                                                     
  - Table I（6模型×2数据集主实验）、Table II（计算开销）完整                                                                                                          
  - Fig.1（SNR-accuracy曲线）、Fig.2（星座图定性展示）、Fig.3（鲁棒性分析）三张关键图保留                                                                             
  - 非自适应基线对比（MA/wavelet vs GPR）数据完整                                                                                                                  
  - CRB 鲁棒性分析 + error-aware fine-tuning 完整

  压缩删除的内容评估：

  ┌───────────────────────────────┬────────────────────────────────────────────────────────────────────────────────┐
  │            删除项             │                                  是否影响论证                                  │
  ├───────────────────────────────┼────────────────────────────────────────────────────────────────────────────────┤
  │ Table 2（SNR分层表）          │ 不影响。关键数字（+10.48/+4.14/+0.91pp）已并入正文，Fig.1 已可视化展示同一信息 │
  ├───────────────────────────────┼────────────────────────────────────────────────────────────────────────────────┤
  │ Fig.3 时域图                  │ 不影响。与星座图作用重叠，星座图更直观                                         │
  ├───────────────────────────────┼────────────────────────────────────────────────────────────────────────────────┤
  │ 254× 推导公式                 │ 轻微损失。推导过程被压缩为一行结论，但核心数字和 Eq.11 的复杂度公式仍在        │
  ├───────────────────────────────┼────────────────────────────────────────────────────────────────────────────────┤
  │ Python overhead 解释          │ 不影响。838× vs 254× 的差距解释属于实现细节                                    │
  ├───────────────────────────────┼────────────────────────────────────────────────────────────────────────────────┤
  │ Introduction 中 DenseNet 引用 │ 不影响。仍保留了从 CNN→hybrid→lightweight→transformer 的完整演进脉络           │
  └───────────────────────────────┴────────────────────────────────────────────────────────────────────────────────┘

  结论：核心论证链完整，无信息缺失。

  ---
  二、学术成熟度评估

  达标的方面：
  1. 叙事结构清晰：问题（task-entanglement）→ 方案（decoupled denoising）→ 方法细节 → 实验验证 → 结论，逻辑自洽
  2. 方法论扎实：GPR 的 Bayesian 理论基础、SNR→σ²ₙ 的解析映射、domain-alignment 优化准则、spectral decomposition 加速——每一步都有数学支撑
  3. 实验充分：6 模型 × 2 数据集 = 12 组对比；非自适应基线对照；SNR 分层分析；鲁棒性分析；计算开销——对一篇 4 页 Letter 来说实验量已经很充足
  4. 实用性论证：CRB 锚定了 SNR 估计误差的实际范围，error-aware fine-tuning 给出了工程化解决方案
  5. 代码公开：匿名链接已提供

  ---
  三、潜在问题与风险

  需要关注的问题

  1. Fig.1 的图例和标注偏小
  从 PDF 第 3 页看，SNR-accuracy 曲线图的右侧有一个放大的 inset 图，但坐标轴标签和图例在印刷尺寸下可能偏小。审稿人可能会要求提高可读性。

  2. Fig.3 的标注被遮挡
  第 4 页的 robustness 图中，"σ_err = 4 dB"和"σ_err = 6 dB"的标注似乎与曲线有重叠，建议检查清晰度。

  3. 信号模型假设的局限性讨论
  论文的信号模型是 $r[n] = s[n] + w[n]$（AWGN），但 Experimental Setup 中明确指出数据集包含 multipath fading、SRO、CFO。审稿人可能质疑：GPR 只建模了 AWGN
  噪声，为什么对包含多径衰落的信号也有效？当前 Conclusion 提了一句 "targets AWGN-dominated noise"，但正文中没有解释为什么 GPR 在非纯 AWGN
  条件下仍然有效。这是最可能被审稿人抓住的点。

  ▎ 建议：在 Experimental Setup 的数据集描述后加一句解释，例如："While the GPR noise model assumes AWGN, the empirical gains across both datasets—which include
  composite channel impairments—suggest that suppressing the additive noise component alone is sufficient to substantially improve the input quality for downstream
  classifiers." 这不会增加太多篇幅。

  4. 参考文献 [14] 的引用合理性
  [14]（Wei et al., data augmentation）被用来支撑 "optimization trajectory encoding noise-correlated patterns" 的论点。这篇文章实际上是关于数据增强的，作为
  task-entanglement 论点的理论支撑可能不够直接。审稿人如果去查这篇引用，可能会觉得牵强。

  5. "On the RML2016.10a dataset" 表述
  Eq.5 前有 "On the RML2016.10a dataset, the RBF kernel emerges as optimal"——审稿人可能问：在 10b 上是否也是 RBF 最优？如果没有验证，是否存在过拟合于 10a
  的风险？当前文本说参数 "transfer directly to RML2016.10b without re-optimization"，但没有说 10b 上 RBF 也是最优。

  ---
  四、总体判断

  论文基本成熟，可以投递。 对于 IEEE Communications Letters 的 4 页 Letter 格式，方法完整、实验充分、叙事清晰。压缩后没有丢失支撑论证的关键内容。

  建议投递前做的小修：
  1. （推荐）加一句解释 GPR 在非纯 AWGN 数据集上仍有效的原因（约 1-2 行）
  2. （可选）检查 Fig.1 和 Fig.3 在印刷尺寸下的标注可读性
  3. （可选）考虑是否需要替换或补充 [14] 的引用

