# 贪吃蛇强化学习

这个项目使用深度Q学习 (DQN) 算法来训练AI玩贪吃蛇游戏。AI的目标是最快速度达到最高分数。

## 项目结构

- `snake_env.py`: 贪吃蛇游戏环境
- `dqn_agent.py`: 深度Q学习代理
- `train.py`: 训练和测试脚本
- `models/`: 保存训练好的模型
- `results/`: 保存训练结果和图表

## 环境需求

详见 `requirements.txt` 文件。主要依赖:
- Python 3.6+
- PyTorch
- NumPy
- Matplotlib
- Pygame

## 安装

```bash
# 克隆仓库
git clone <仓库URL>
cd <仓库目录>

# 安装依赖
pip install -r requirements.txt
```

## 使用方法

### 训练模型

```bash
# 默认训练1000轮
python train.py --mode train --episodes 1000

# 训练时显示游戏界面
python train.py --mode train --render

# 从之前的模型继续训练
python train.py --mode train --continue_training

# 从指定模型继续训练
python train.py --mode train --continue_training --model models/custom_model.pth

# 自定义模型保存的前缀名称
python train.py --mode train --model_prefix my_snake
```

### 测试模型

```bash
# 测试最终模型
python train.py --mode test

# 测试指定模型
python train.py --mode test --model models/snake_best_score.pth
```

## 训练过程

训练过程中，AI通过反复尝试游戏来学习最优策略：

1. 首先AI会进行大量探索，随机尝试不同动作
2. 随着训练进行，AI会逐渐减少探索，更多地利用已学到的策略
3. 训练过程中会保存以下模型：
   - 得分最高的模型 (`{前缀}_best_score.pth`)
   - 平均分最高的模型 (`{前缀}_best_avg.pth`)
   - 定期保存的检查点 (`{前缀}_episode_{轮次}.pth`)
   - 最新的检查点模型 (`{前缀}_latest.pth`)
   - 训练结束的最终模型 (`{前缀}_final.pth`)

## 继续训练

本项目支持从之前训练的模型继续训练，有以下几种方式：

1. 使用 `--continue_training` 参数：
   - 如果同时指定了 `--model` 参数，将从该模型继续训练
   - 否则会按以下顺序自动查找模型：
     1. `{前缀}_latest.pth` (最近保存的检查点)
     2. `{前缀}_final.pth` (之前训练结束的模型)

2. 自定义模型名称前缀：
   - 使用 `--model_prefix` 参数可以设置保存的模型名称前缀
   - 这对于训练不同参数或策略的多个模型很有用
   - 所有保存的模型和结果图表都会使用该前缀

## 游戏状态表示

AI通过以下信息来感知游戏状态：
- 危险检测：蛇头周围是否有障碍物
- 蛇的移动方向
- 食物相对于蛇头的位置

## 奖励系统

- 吃到食物: +10分
- 游戏结束: -10分
- 其他情况: 0分

## 自定义参数

可以在代码中调整以下参数来优化训练过程：
- 学习率
- 折扣因子
- 探索率及其衰减
- 网络结构
- 奖励函数 