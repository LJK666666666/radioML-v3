# 完成RML2016.10a数据集的分类预测任务过程

搜集了5篇论文，作为baseline。

为了和多数论文保持一致，更好的进行对比，将测试集比例划分为0.2。
经过实验对比，将训练集、验证集比例设置为0.72和0.08，更好地训练神经网络。

尝试了fcnn、cnn1d、cnn2d、resnet、transformer、complexcnn架构，实验表明resnet的效果最好，于是在resnet的基础上进行改进。

由于噪声主要是加性高斯白噪声，所以可以认为每点处信号都服从高斯分布且相互独立，那么信号就可以视作高斯过程，因此决定采用高斯过程回归进行去噪。根据信号I/Q通道数据和信噪比计算噪声标准差，设置自适应的length_scale。实验表明分类准确率提升很大。

通过观察星座图，可以可以看出信号的旋转属性，借鉴论文《Ultra Lite Convolutional Neural Network for FastAutomatic Modulation Classification inLow-Resource Scenarios》中提出的旋转数据增强，在训练集上分别对信号旋转90°、180°、270°，进行数据增强。

发现complexcnn(复数卷积神经网络)不仅收敛速度快，而且准确率仅次于resnet，比cnn1d、cnn2d准确率都要高，于是将complexcnn融合到resnet中，即将resnet的输入层改为复数层，直接对复数进行处理，得到了resnet和complexcnn的混合神经网络。

经过以上3次改进后分类准确率达到65.38%，超越了previous SOTA的分类准确率指标。
