 Implementation Plan: SNR Prediction + Standalone Denoising                                                                                                                                                                                                                                                                                                  Context                                                                                                                                                                      

 当前系统的去噪流程是：加载全部数据 → 使用真实SNR去噪全部数据 → 分割train/val/test → 增强 → 训练/评估。
 用户需要：
 1. 训练一个神经网络预测离散SNR值（-20到18dB，步长2dB，共20类）
 2. 支持用预测的SNR对val/test集进行GPR去噪（train集始终用真实SNR）
 3. 去噪可独立调用、去噪结果可缓存
 4. explore从--mode all中移除，保留为独立模式

 CLI 接口变更

 新增 --mode 选项

 --mode {explore, snr, denoise, train, evaluate, all}
 - snr: 训练SNR预测器，保存val/test预测结果
 - denoise: 独立去噪步骤，可使用预测SNR

 新增参数

 --snr_model {snr_cnn}       # SNR预测器架构，默认snr_cnn，可扩展
 --use_predicted_snr          # 使用预测SNR去噪val/test
 --snr_predictions_path PATH  # 手动指定SNR预测文件路径（否则自动检测）

 --models 改为非必需

 - train/evaluate/all 模式必须提供
 - snr/denoise/explore 模式不需要

 各模式的典型用法

 # 步骤1：训练SNR预测器
 python main.py --mode snr --snr_model snr_cnn --epochs 100

 # 步骤2：用预测SNR去噪（train用真实SNR，val/test用预测SNR）
 python main.py --mode denoise --use_predicted_snr --augment_data

 # 步骤3：训练分类模型（加载步骤2缓存的去噪数据）
 python main.py --mode train --use_predicted_snr --models resnet --augment_data

 # 步骤4：评估
 python main.py --mode evaluate --use_predicted_snr --models resnet --augment_data

 # 传统流程（不变）
 python main.py --mode all --models resnet

 新建文件

 1. src/model/snr_predictor_model.py — SNR预测模型

 - SNR预测视为20类分类任务（离散dB值 → 类别索引）
 - 使用PyTorch（与现有iqformer/fea_t一致）
 - 注册表模式便于后续扩展

 # 核心接口
 class SNRPredictorCNN(nn.Module):
     def __init__(self, input_channels=2, seq_len=128, num_snr_classes=20): ...
     def forward(self, x): ...  # x: (B, 2, 128) → logits: (B, num_snr_classes)

 SNR_MODEL_BUILDERS = {'snr_cnn': SNRPredictorCNN}

 def build_snr_predictor(model_name, input_channels, seq_len, num_snr_classes): ...
 def get_available_snr_models() -> list: ...

 架构：Conv1d(2→64, k=7) → Conv1d(64→128, k=5) → Conv1d(128→256, k=3) → AdaptiveAvgPool → FC(256→128) → FC(128→20)

 2. src/snr_predict.py — SNR预测训练/推理/保存

 def train_snr_predictor(model, X_train, snr_labels_train, X_val, snr_labels_val,
                         model_path, batch_size, epochs, lr, patience_lr, patience_es, factor):
     """训练SNR分类器。每epoch保存best+last模型。"""
     # 模式类似 train_torch_model，但标签是SNR类别索引（不是one-hot）
     # CrossEntropyLoss, AdamW, ReduceLROnPlateau(monitor val_accuracy)
     ...

 def predict_snr(model, X_data, snr_classes, batch_size=256, device=None):
     """推理，返回预测的dB值数组 shape=(N,)"""
     ...

 def save_snr_predictions(pred_val, pred_test, true_val, true_test, save_path):
     """保存为 .npz 文件"""
     ...

 def load_snr_predictions(load_path) -> dict:
     """加载 .npz，返回 {snr_pred_val, snr_pred_test, snr_true_val, snr_true_test}"""
     ...

 def evaluate_snr_predictor(snr_true, snr_pred, snr_classes, output_dir):
     """生成：总体准确率、per-SNR准确率、混淆矩阵、MAE(dB)"""
     ...

 3. src/denoise.py — 独立去噪模块

 def denoise_split(X, y_int, snr_values, mods, denoising_method, split_name=""):
     """去噪单个数据分片，复用现有 efficient_gpr/gpr/gpr_fft 实现"""
     # X: (N,2,128), y_int: (N,), snr_values: (N,) — 可以是真实或预测的SNR
     ...

 def denoise_and_cache_splits(X_train, X_val, X_test, y_train_int, y_val_int, y_test_int,
                               snr_train, snr_val, snr_test, mods,
                               denoising_method, cache_dir, cache_tag,
                               augment_data, snr_val_for_denoise, snr_test_for_denoise):
     """去噪3个分片 → 增强train → one-hot编码y → 保存到缓存文件"""
     # train始终使用snr_train（真实SNR）
     # val使用snr_val_for_denoise（可能是预测值或真实值）
     # test使用snr_test_for_denoise（同上）
     # 缓存文件路径: {cache_dir}/{cache_tag}_denoised_splits.pkl
     # 缓存内容: X_train/val/test, y_train/val/test(one-hot), snr_train/val/test(真实值), mods
     # 注意：snr_val/snr_test 存储的是**真实值**（用于evaluate_by_snr评估分组）
     ...

 def load_cached_splits(cache_dir, cache_tag) -> dict or None:
     """加载缓存的去噪分片数据"""
     ...

 修改现有文件

 4. src/preprocess.py — 提取原始分割函数

 从 prepare_data_by_snr_stratified 中提取分割逻辑为共享函数：

 def _collect_dataset_arrays(dataset, specific_snrs=None):
     """收集数据集 → X_all, y_all(int), snr_all, composite_labels, mods"""
     # 当前代码行 202-239

 def _perform_stratified_split(X_all, y_all, snr_all, mods, test_size=0.2, validation_split=0.1):
     """确定性分层分割（random_state=42）。y 保持为 int，不做 one-hot。"""
     # 当前代码行 316-371

 def split_data_raw(dataset, test_size=0.2, validation_split=0.1, specific_snrs=None):
     """[NEW] 加载 + 分割，不去噪、不增强、不one-hot。供 snr/denoise 模式使用。"""
     X_all, y_all, snr_all, _, mods = _collect_dataset_arrays(dataset, specific_snrs)
     return _perform_stratified_split(X_all, y_all, snr_all, mods, test_size, validation_split) + (mods,)
     # 返回: X_train, X_val, X_test, y_train(int), y_val(int), y_test(int),
     #       snr_train, snr_val, snr_test, mods

 重构 prepare_data_by_snr_stratified 内部调用 _collect_dataset_arrays 和 _perform_stratified_split，保持接口不变（向后兼容）。

 关键保证：split_data_raw 和 prepare_data_by_snr_stratified 使用同一份分割逻辑、同一个 random_state=42，对相同原始数据产生完全一致的分割。

 5. src/main.py — 添加新模式和参数

 主要变更：
 1. --mode 增加 snr, denoise 选项
 2. --models 改为非必需（snr/denoise/explore 模式不需要）
 3. 新增 --snr_model, --use_predicted_snr, --snr_predictions_path 参数
 4. --mode all 不再包含 explore
 5. 新增 get_cache_tag() 函数用于去噪缓存命名
 6. 新增 --mode snr 处理块
 7. 新增 --mode denoise 处理块
 8. --mode train/evaluate 支持 --use_predicted_snr 分支（从缓存加载去噪数据）

 6. src/models.py — 导出SNR模型

 from model.snr_predictor_model import build_snr_predictor, get_available_snr_models

 7. src/config/training.yaml — 添加SNR训练配置

 snr_training:
   epochs: 100
   batch_size: 256
   learning_rate: 0.001
   patience_lr: 5
   patience_es: 20
   factor: 0.5

 数据流图

 mode=snr

 load_dataset → split_data_raw (无去噪)
     → 构建SNR类映射 (snr→class_idx)
     → build_snr_predictor → train_snr_predictor
     → predict_snr(val) → predict_snr(test)
     → save_snr_predictions(.npz)
     → evaluate_snr_predictor (准确率/混淆矩阵/MAE)

 mode=denoise --use_predicted_snr

 load_dataset → split_data_raw (无去噪)
     → load_snr_predictions(.npz)
     → denoise_split(train, real_snr)
     → denoise_split(val, predicted_snr)
     → denoise_split(test, predicted_snr)
     → augment_train (if --augment_data)
     → one_hot_encode(y)
     → save cache: {cache_dir}/{cache_tag}_denoised_splits.pkl

 mode=train --use_predicted_snr

 load_cached_splits(cache_dir, cache_tag)  [若未找到则报错提示先运行denoise]
     → train_selected_models (与现有流程相同)

 mode=train (无--use_predicted_snr，向后兼容)

 prepare_data_with_cache_suffix (现有流程，去噪全部→分割→增强)
     → train_selected_models

 文件输出位置

 ┌──────────────┬────────────────────────────────────────────────────────────┐
 │     产物     │                            路径                            │
 ├──────────────┼────────────────────────────────────────────────────────────┤
 │ SNR模型权重  │ {output_dir}/models/snr_{snr_model}{suffix}.pt             │
 ├──────────────┼────────────────────────────────────────────────────────────┤
 │ SNR预测结果  │ {output_dir}/snr_predictions_{snr_model}{suffix}.npz       │
 ├──────────────┼────────────────────────────────────────────────────────────┤
 │ SNR评估结果  │ {output_dir}/results/snr_{snr_model}_eval_{split}{suffix}/ │
 ├──────────────┼────────────────────────────────────────────────────────────┤
 │ 去噪分片缓存 │ {denoised_cache_dir}/{cache_tag}_denoised_splits.pkl       │
 ├──────────────┼────────────────────────────────────────────────────────────┤
 │ SNR训练曲线  │ {output_dir}/training_plots/snr_{snr_model}{suffix}.png    │
 └──────────────┴────────────────────────────────────────────────────────────┘

 其中 suffix 对于SNR模式 = {dataset_suffix_prefix}_stratified（不含去噪方法，因为SNR预测用原始数据）。

 cache_tag 示例：efficient_gpr_per_sample_predsnr_snr_cnn_augment 或 efficient_gpr_per_sample_realsnr_augment

 实施顺序

 1. preprocess.py: 提取 _collect_dataset_arrays, _perform_stratified_split, split_data_raw；重构 prepare_data_by_snr_stratified
 2. model/snr_predictor_model.py: 创建SNR预测模型
 3. snr_predict.py: 创建训练/推理/保存/评估函数
 4. denoise.py: 创建独立去噪模块
 5. main.py: 添加新模式、新参数、处理逻辑
 6. models.py: 导出SNR模型
 7. config/training.yaml: 添加SNR训练配置

 验证方法

# 1. 向后兼容测试：确保现有流程不受影响
python main.py --mode train --models cnn1d --epochs 1 --batch_size 64

# 2. SNR预测测试
python main.py --mode snr --batch_size 256

# 3. 预测SNR去噪测试
python main.py --mode denoise --use_predicted_snr --augment_data
python main.py --mode denoise --use_predicted_snr

# 4. 使用预测SNR缓存训练
python main.py --mode train --use_predicted_snr --models cnn1d --epochs 1 --augment_data

# 5. 独立去噪测试（不使用预测SNR）
python main.py --mode denoise
