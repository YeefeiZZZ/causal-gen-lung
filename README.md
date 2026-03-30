Causal and Active Learning-Based Counterfactual Chest X-ray Generation for Supporting Clinical Decision-Making in Lung Disease

本仓库包含了基于 因果生成模型 (Causal Generative Modelling) 和 结构变分分析 (SVP) 的研究实现，专注于肺部影像数据集（如 MIMIC-CXR），旨在通过高质量的反事实影像生成支持临床决策。

该项目扩展了模块化的深度结构因果模型 (DSCM) 框架，将其应用于医疗影像领域，实现了高保真度的反事实生成以及基于主动学习的数据效率分析。

📦 项目结构

.
┣ 📂causal                      # 因果生成模型核心代码
┃ ┣ 📂pgm                       # 概率图模型与 SCM 机制
┃ ┃ ┣ 📜dscm.py                 # 深度结构因果模型模块
┃ ┃ ┣ 📜flow_pgm.py             # 基于 Pyro 的 Flow 机制实现
┃ ┃ ┣ 📜pgm_train.sh            # PGM 模型训练脚本
┃ ┃ ┣ 📜aux_train.sh            # AUX 辅助模型训练脚本
┃ ┃ ┣ 📜cf_train_test.sh        # 反事实 (CF) 模型综合训练与测试脚本
┃ ┃ ┗ 📜utils_pgm.py            # 图模型工具类
┃ ┣ 📜vae.py                    # 分层 VAE (HVAE) 定义
┃ ┣ 📜trainer.py                # 影像因果机制训练逻辑
┃ ┣ 📜datasets.py               # 数据集定义 (MIMIC 等)
┃ ┣ 📜main.py                   # 训练主入口
┃ ┣ 📜run_mimic.sh              # MIMIC-CXR HVAE 实验启动脚本
┃ ┗ 📜run.sh                    # 通用启动脚本
┃
┣ 📂svp                         # 结构变分分析与主动学习
┃ ┣ 📂svp/mimic                 # MIMIC-CXR 特定分析包
┃ ┃ ┣ 📜active.py               # 主动学习/筛选逻辑
┃ ┃ ┣ 📜coreset.py              # 核心集 (Coreset) 选择方法
┃ ┃ ┣ 📜models.py               # SVP 相关模型架构
┃ ┃ ┗ 📜train.py                # SVP 训练脚本
┃ ┣ 📜run_svp.sh                # SVP 实验运行脚本（包含筛选逻辑）
┃ ┗ 📜setup.py                  # 环境安装脚本
┃
┗ 📜README.md                   # 项目文档


🚀 核心概述

本框架采用模块化设计，将结构化变量（影像 $\mathbf{x}$）的因果机制与因果图中的元数据/标签机制（$\mathbf{y}$）解耦：

因果模块 (/causal):

使用 Pyro (概率编程语言) 进行 SCM 机制的建模。

采用 HVAE (Hierarchical VAE) 作为影像生成的核心机制，确保高保真度。

支持复杂的反事实推理与约束性反事实训练。

SVP 模块 (/svp):

结合主动学习 (Active Learning) 策略，用于分析数据效率。

通过核心集选择 (Coreset Selection) 筛选关键影像样本。

🛠️ 环境要求

建议在虚拟环境中安装依赖：

# 安装核心依赖
pip install -r requirements.txt

# 以可编辑模式安装 SVP 包
cd svp
pip install -e .


🏃 运行指南

项目的训练遵循严格的阶段性流程：

1. 因果模型训练阶段

训练分为三个连续步骤：

步骤 A: 训练 HVAE 基础模型 首先训练影像生成的因果机制（基础 Baseline）：

cd causal
bash run_mimic.sh your_experiment_name


步骤 B: 训练 PGM 与 AUX 模型 在 HVAE 训练完成后，分别运行脚本训练概率图机制与辅助模型：

cd causal/pgm
bash pgm_train.sh   # 训练 PGM
bash aux_train.sh   # 训练 AUX


步骤 C: 综合训练 CF 反事实模型 最后，将上述模型综合，进行反事实细化训练：

cd causal/pgm
bash cf_train_test.sh


2. SVP 筛选与主动学习

在首个 Baseline 模型（HVAE）训练完成后，需要执行以下流程：

利用训练好的模型生成候选影像。

使用 SVP 模块对生成的影像进行筛选：

cd svp
bash run_svp.sh


📝 引用信息

如果您在研究中使用了本代码，请考虑引用原始论文：

@InProceedings{pmlr-v202-de-sousa-ribeiro23a,
  title={High Fidelity Image Counterfactuals with Probabilistic Causal Models},
  author={De Sousa Ribeiro, Fabio and Xia, Tian and Monteiro, Miguel and Pawlowski, Nick and Glocker, Ben},
  booktitle={Proceedings of the 40th International Conference on Machine Learning},
  pages={7390--7425},
  year={2023},
  volume={202},
  series={Proceedings of Machine Learning Research},
  url={[https://proceedings.mlr.press/v202/de-sousa-ribeiro23a.html](https://proceedings.mlr.press/v202/de-sousa-ribeiro23a.html)}
}
