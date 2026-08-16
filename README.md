# KuaiRankLab: Ranking on KuaiRand

KuaiRankLab 是一个基于 **KuaiRand** 数据集的推荐排序学习项目，目标是从数据处理和 LR 基线开始，逐步实现并比较 DeepFM、DCN、DIN、MMoE 等常见推荐模型。

## 开发设备与运行环境

项目通过 GitHub 在两台设备之间同步和维护。两台设备承担不同职责：

两台设备均包含一个conda创建的环境kuairanklab，包含一些必要的模块。

| 设备 | 环境 | 主要职责 | 依赖文件 |
|---|---|---|---|
| MacBook Air | macOS | 修改代码、Notebook EDA、小样本冒烟测试、检查数据管道 | `requirements-mac.txt` |
| Windows 11 主机 | WSL，NVIDIA RTX 5070 | 完整预处理、全量训练、超参数实验和正式评估 | `requirements.txt` |

`requirements-mac.txt` 和 `requirements.txt` 应分别维护，尤其不要直接复用两端的 PyTorch 安装方式：Mac 主要使用 CPU/MPS，WSL 端需要与 NVIDIA 驱动和 CUDA 兼容的 PyTorch 构建。

`requirements.txt` 记录wsl中环境的依赖。深度模型使用的 CUDA PyTorch 需要在 WSL 中根据实际驱动兼容情况单独安装并锁定，不能直接复用 Mac 的 PyTorch 构建。

### MacBook Air

Mac 主要承担开发和冒烟测试，尽量不以全量训练速度作为目标。

### Windows 11 + WSL + RTX 5070

WSL 是正式训练环境。运行全量任务前至少检查：

```bash
nvidia-smi
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CUDA unavailable')"
```

如果正式训练时 `torch.cuda.is_available()` 为 `False`，应直接停止并修复 CUDA/PyTorch 环境，避免全量任务意外退回 CPU。

## 数据集

本项目使用的 KuaiRand 数据集来自 CIKM 2022：

```bibtex
@inproceedings{gao2022kuairand,
  title = {KuaiRand: An Unbiased Sequential Recommendation Dataset with Randomly Exposed Videos},
  author = {Gao, Chongming and Li, Shijun and Zhang, Yuan and Chen, Jiawei and Li, Biao and Lei, Wenqiang and Jiang, Peng and He, Xiangnan},
  url = {https://doi.org/10.1145/3511808.3557624},
  doi = {10.1145/3511808.3557624},
  booktitle = {Proceedings of the 31st ACM International Conference on Information and Knowledge Management},
  series = {CIKM '22},
  location = {Atlanta, GA, USA},
  numpages = {5},
  year = {2022},
  pages = {3953–3957}
}
```

当前主数据集为 **KuaiRand-1K**。这里的 “1K” 指 1,000 个用户，不代表只有 1,000 条交互或 1,000 个视频。

数据集相关链接：

知乎介绍：https://zhuanlan.zhihu.com/p/566393720

官方网站：https://kuairand.com/

github：https://github.com/chongminggao/KuaiRand

数据包含：

- 用户属性和脱敏类别特征；
- 视频、作者、音乐、标签等基础特征；
- 用户曝光和行为时间序列；
- Click、Like、Long View、Follow、Comment 等反馈；
- 随机干预曝光日志；
- 视频全局统计特征。

数据目录已被 `.gitignore` 忽略，不上传 GitHub。



## 计划的数据集划分

两份标准日志先合并，再按时间划分。不要随机拆分行，否则 DIN 历史序列和离线评估会泄漏未来信息。

| 集合 | 日期 | 来源 | 全场景样本 | `tab=1` 样本 | 用途 |
|---|---|---|---:|---:|---|
| Train | 04-08～04-27 | Standard | 7,235,595 | 4,730,487 | 模型训练 |
| Validation | 04-28～05-02 | Standard | 2,045,412 | 1,360,773 | 调参、早停 |
| Standard Test | 05-03～05-08 | Standard | 2,432,038 | 1,626,341 | 标准曝光分布评估 |
| Random Test | 05-03～05-08 | Random | 23,752 | 23,649 | 随机曝光额外评估 |

第一版使用 `tab=1` 作为主实验口径。所有类别词表、归一化参数和特征编码只能在 Train 上拟合，验证或测试期新出现的类别映射为 OOV。

## 模型数据使用方案

### LR / DeepFM / DCN

- 第一版以 `is_click` 为标签；
- 每条曝光本身就是正负样本，不额外构造随机负样本；
- 使用用户特征、视频基础特征、场景和时间特征；
- 暂不使用整月视频统计特征。

### DIN

- 按 `user_id, time_ms` 排序；
- 当前样本只能使用当前时间之前的行为；
- 第一版用历史 `is_click=1` 或 `long_view=1` 的视频构造兴趣序列；
- 从最大序列长度 50 开始，再比较 20、50、100。

### MMoE

第一版建议使用以下任务：

- `is_click`；
- `long_view`；
- `is_like`；
- `is_profile_enter`。

`is_follow`、`is_comment`、`is_forward` 正例过少，等基础多任务版本跑通后再配合加权 BCE、Focal Loss 或重采样加入。

## 评估指标

主要离线指标：

- AUC；
- LogLoss；
- GAUC；
- MMoE 各任务独立 AUC / LogLoss。

Standard Test 和 Random Test 的曝光分布、候选视频集合不同，应分别报告和解释，不直接比较两者绝对指标的高低。

## 当前计划

- [x] 全量的EDA
- [ ] 数据处理
- [ ] 数据集的划分
- [ ] LR
- [ ] DeepFM
- [ ] DCN
- [ ] DIN
- [ ] MMoE

## Git 与数据管理

- 代码、配置、Notebook 和小型实验结果通过 GitHub 同步；
- `data/`、模型 checkpoint、日志和大型中间文件不提交 Git；
- 两台设备使用相同的时间划分、随机种子和实验配置；
- 正式实验记录 Python/PyTorch 版本、Git commit、设备名称、参数和指标，保证结果可复现。

## 状态

🚧 Work in progress.
