# KuaiRankLab: Ranking on KuaiRand

KuaiRankLab 是一个基于 **KuaiRand** 数据集的推荐排序学习项目，目标是从数据处理和 LR 基线开始，逐步实现并比较 DeepFM、DCN、DIN、MMoE 等常见推荐模型。

当前主要研究内容（计划）包括：

- CTR / 用户反馈预测；
- 稀疏特征与连续特征处理；
- 特征交叉和高阶交互；
- 用户行为序列建模；
- 多目标、多行为学习；
- 时间切分、曝光偏差和随机曝光评估。

## 开发设备与运行环境

项目通过 GitHub 在两台设备之间同步和维护。两台设备承担不同职责：

两台设备均包含一个conda创建的环境kuairanklab，包含一些必要的模块。

| 设备 | 环境 | 主要职责 | 依赖文件 |
|---|---|---|---|
| MacBook Air | macOS | 修改代码、Notebook EDA、小样本冒烟测试、检查数据管道 | `requirements-mac.txt` |
| Windows 11 主机 | WSL，NVIDIA RTX 5070 | 完整预处理、全量训练、超参数实验和正式评估 | `requirements.txt` |

`requirements-mac.txt` 和 `requirements.txt` 应分别维护，尤其不要直接复用两端的 PyTorch 安装方式：Mac 主要使用 CPU/MPS，WSL 端需要与 NVIDIA 驱动和 CUDA 兼容的 PyTorch 构建。

`requirements.txt` 先记录数据处理和 LR 所需的通用依赖。深度模型使用的 CUDA PyTorch 需要在 WSL 中根据实际驱动兼容情况单独安装并锁定，不能直接复用 Mac 的 PyTorch 构建。

### MacBook Air

Mac 只承担开发和冒烟测试，不以全量训练速度作为目标。

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-mac.txt
```

冒烟测试应使用少量用户，但保留这些用户的完整时间序列。不要直接取 CSV 前 N 行，因为日志大致按用户排列，会严重偏向少数用户。

### Windows 11 + WSL + RTX 5070

WSL 是正式训练环境。运行全量任务前至少检查：

```bash
nvidia-smi
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CUDA unavailable')"
```

如果正式训练时 `torch.cuda.is_available()` 为 `False`，应直接停止并修复 CUDA/PyTorch 环境，避免全量任务意外退回 CPU。

## 数据集

本项目使用的 KuaiRand 数据集来自 CIKM 2022：

## 📚 Citation

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

本地同时保存了 KuaiRand-Pure，但第一阶段模型实验只使用 KuaiRand-1K。数据目录已被 `.gitignore` 忽略，不上传 GitHub。

## 第一轮 EDA 总结

完整分析代码见 [`notebooks/01_data_overview.ipynb`](notebooks/01_data_overview.ipynb)。Notebook 对大 CSV 使用分块读取，已在全部 6 个文件上验证核心统计逻辑。

### 文件规模

| 文件 | 数据行数 | 列数 | 主要内容 |
|---|---:|---:|---|
| `log_standard_4_08_to_4_21_1k.csv` | 5,055,984 | 19 | 2022-04-08 至 2022-04-21 的标准推荐曝光 |
| `log_standard_4_22_to_5_08_1k.csv` | 6,657,061 | 19 | 2022-04-22 至 2022-05-08 的标准推荐曝光 |
| `log_random_4_22_to_5_08_1k.csv` | 43,028 | 19 | 同期随机干预曝光 |
| `user_features_1k.csv` | 1,000 | 31 | 用户 ID 和 30 个用户特征 |
| `video_features_basic_1k.csv` | 4,371,868 | 12 | 视频 ID 和 11 个基础特征 |
| `video_features_statistic_1k.csv` | 4,371,868 | 52 | 视频 ID 和 51 个聚合统计特征 |

两份标准日志合计 **11,713,045** 条曝光，涉及 **4,369,953** 个不同视频。随机日志包含 7,388 个视频，其中 1,915 个没有出现在标准日志中。全部日志涉及的 4,371,868 个视频都可以在两张视频特征表中找到。

### 反馈标签分布

| 日志 | `is_click` | `long_view` | `is_like` | `is_follow` | `is_profile_enter` |
|---|---:|---:|---:|---:|---:|
| 04-08～04-21 标准曝光 | 37.93% | 26.35% | 1.51% | 0.113% | 1.79% |
| 04-22～05-08 标准曝光 | 37.73% | 26.10% | 1.60% | 0.085% | 1.79% |
| 随机曝光 | 17.41% | 8.42% | 0.56% | 0.019% | 0.46% |

随机曝光的点击、长播、点赞和主页进入率都明显低于标准曝光，说明标准推荐日志存在较强的曝光选择偏差。随机日志不适合作为主要训练集，应保留为额外的随机曝光测试集。

`is_click` 的业务含义与 UI 场景相关：在双列页面表示点击，在单列页面更接近 valid play。随机日志中超过 99% 的样本来自 `tab=1`，因此第一版主实验建议限制 `tab=1`，后续再扩展到全场景并将 `tab` 作为特征。

### 用户行为与序列

- 早期标准日志覆盖 983 个用户，后期标准日志覆盖全部 1,000 个用户；
- 早期每用户交互数中位数约 3,421，最大约 49,242；
- 后期每用户交互数中位数约 4,751，最大约 78,315；
- 随机日志每用户交互数中位数只有 22；
- DIN 等序列模型需要截断或采样，第一版可从最大长度 50 开始。

### 数据质量

用户特征：

- `onehot_feat4`、`onehot_feat12`～`onehot_feat17` 各缺失 33 条；
- `is_live_streamer` 有 782 条值为 `-124`，应作为未知类别而不是连续数值；
- 实际 `user_active_degree` 有 7 个类别，多于数据说明中列出的 4 类。

视频基础特征：

- `video_duration` 缺失 573,057 条，约 13.1%；
- `tag` 缺失 137,484 条，约 3.15%；
- `music_type` 缺失 59,952 条，约 1.37%；
- 视频时长中位数约 35.1 秒，95 分位约 267.7 秒，最大值约 4.59 小时，存在明显长尾；
- 视频 ID 中间有少量空洞，不能假设所有 ID 完全连续。

视频统计特征没有缺失值，但它们是整个月、跨日期和场景聚合后的结果，可能包含验证或测试时点之后的信息。严格时间实验第一版不使用 `video_features_statistic_1k.csv`；后续只能将其作为单独标注的潜在泄漏特征消融实验。

## 推荐的数据集划分

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

## 项目结构（初步，逐渐增加）

```text
KuaiRankLab/
│
├── data/
│   ├── raw/
│   │   └── KuaiRand原始文件
│   │
│   └── processed/
│       ├── train.parquet
│       ├── val.parquet
│       └── test.parquet
├── src/
│   ├── data/
│   │   ├── preprocess.py
│   │   ├── split.py
│   │   └── dataset.py
│   │
│   ├── models/
│   │   ├── deepfm.py
│   │   └── din.py
│   │
│   ├── trainer.py
│   └── metrics.py
├── checkpoints/
├── train.py
├── requirements.txt
├── README.md
└── .gitignore
```

## 当前计划

- [x] 数据目录与字段检查
- [x] 第一轮全量 EDA Notebook
- [ ] 时间划分和数据使用方案
- [ ] 数据处理
- [ ] LR
- [ ] DeepFM
- [ ] DCN
- [ ] DIN
- [ ] MMoE
- [ ] 序列长度实验
- [ ] 多行为和时间特征实验
- [ ] 模型比较与消融实验

## Git 与数据管理

- 代码、配置、Notebook 和小型实验结果通过 GitHub 同步；
- `data/`、模型 checkpoint、日志和大型中间文件不提交 Git；
- 两台设备使用相同的时间划分、随机种子和实验配置；
- 正式实验记录 Python/PyTorch 版本、Git commit、设备名称、参数和指标，保证结果可复现。

## 状态

🚧 Work in progress.
