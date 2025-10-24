# 蒼蠅蜥蜴強化學習模擬 (Fly–Lizard Reinforcement Learning)

> 此程式碼來自於台積電林泰翔主管，感謝他的提供與指導。

這個專案模擬「蒼蠅在蜥蜴威脅下生存」的場景，  
透過強化學習讓蒼蠅學會在有限能量下 **移動、進食、避開蜥蜴**。<br>
<br>模型：['fly_lizard_RL.py'](./fly_lizard_RL.py) <br>
此模型使用兩種強化學習方法：
- **SAC (Soft Actor-Critic)** 負責「移動」策略  
- **DQN (Deep Q-Network)** 負責「進食」策略  
同時加入 **PER (Prioritized Experience Replay)** 讓學習更有效率。  
整個過程以 **Pygame** 呈現動畫，並以 CSV 紀錄訓練結果。

---

## 專案目標
讓智能體（蒼蠅）能在模擬環境中自我學習以下能力：
1. 維持能量並盡量延長生存時間  
2. 有效率地尋找食物並選擇何時進食  
3. 避開移動中的蜥蜴  

最終比較actor和critic的不同超參數（Adam / AdamW、τ 值）的學習表現與穩定性。

---

## 模型架構說明

整體模型包含三個主要部分：

### 1. 移動策略：SAC 模型
使用 **Soft Actor-Critic** 讓蒼蠅決定下一步的移動方向。  
SAC 同時學習兩個網路：
- **Actor**：輸入當前狀態，輸出移動方向（x, y）  
- **Critic**：評估每個動作的價值  

Actor 會輸出連續動作，之後離散化成 8 個方向（上下左右與對角）。  
Critic 採雙網路 (Twin Q-networks)，可避免高估 Q 值。

### 2. 進食策略：DQN 模型
當蒼蠅站在食物上時，進食行為由 **DQN (Deep Q-Network)** 控制。  
DQN 判斷「是否要吃」，輸入是能量與預期收益。
這部分使用 **Double DQN**（主網 + 目標網），並用平滑 L1 loss 更新參數。  
可避免因短期高獎勵而導致的錯誤判斷。

### 3. 優先回放：PER 模組
**PER (Prioritized Experience Replay)** 用來提升訓練效率。  
每次抽樣時，會根據「學習誤差 (TD-error)」決定哪些資料被重複學習。

功能：
- 誤差大的樣本會被優先抽取  
- 動態調整 β 以修正抽樣偏差  
- SAC 與 DQN 各自維護自己的 replay buffer  

---

## 環境設計 (FlySurvivalEnv)

環境模擬一個 20×20 格的世界，包含：
- **食物 (Food)**：蒼蠅吃掉後會隨機生成新食物（每吃 3 個會再生 1–2 個）
- **蜥蜴 (Lizard)**：移動速度是蒼蠅的 1/3，70% 機率朝蒼蠅方向移動
- **蒼蠅 (Fly)**：可移動、進食、躲避蜥蜴，並受到能量限制  

**懲罰與獎勵機制**
| 行為 | 效果 |
|------|------|
| 吃食物 | +10 獎勵，回復能量 |
| 食物全吃完 | 額外 +30 獎勵 |
| 碰到蜥蜴 | 扣 50 能量 |
| 長時間不動或卡在角落 | 懲罰分數 |
| 撞牆或越界 | 懲罰分數 |

## 訓練設定
| 參數 | 值 | 說明 |
|------|------|------|
| Episodes | 400 | 每個版本訓練 400 回合 |
| Optimizer | Adam / AdamW | 比較不同權重衰減效果 |
| τ (tau) | 0.002 / 0.005 | target network 更新速率 |
| Replay buffer | 40000 | 經驗池大小 |
| Batch size | 96 | 每次訓練批次 |
| Sniff λ | 0.25 | 食物氣味導引比例 |

---

## 測試版本比較
| 版本 | Optimizer | τ 值 | 說明 |
|------|------------|------|------|
| v1 | Adam | 0.005 | 基準版本 |
| v2 | Adam | 0.002 | target 更新更慢 |
| v3 | AdamW | 0.005 | 加入權重衰減 |
| v4 | AdamW | 0.002 | 綜合調整版本 |

訓練結果皆輸出為 CSV，欄位如下：
Episode, Steps, Foods, C1, C2, Eat, Actor, Alpha, Reward, EatEps, PRBBeta, Stuck, OnFoodWaitMax

## 分析與結果

完整數據與統計分析可見於 [`Analysis/README.md`](./Analysis/README.md)，  
 
