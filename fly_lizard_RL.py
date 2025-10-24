# fly_main_ontology.py — Round 2 (SAC Move + Simple DQN Eat + PER + CSV + Pygame + SHACL hooks)
# Timestamp: 2025-08-22 17:20 Asia/Taipei
# Version: v2.5.0
#
# 🧾 本版考量與功能
# - MOVE：SAC（雙 Q + Auto-α），連續動作；採 8 向離散取樣，含「直線優先」避免只走斜邊
# - 嗅覺導引：將 actor 連續動作與「食物場梯度」做加權混合（能量越低越依賴嗅覺）；僅影響動作，不直接改 reward
# - EAT：最簡三層 FC（輸入：energy、expected_gain），Double DQN + target net；on-food gating；鄰格貼靠吃
# - 蜥蜴：恢復捕食行為、速度為蒼蠅的 1/3；撞擊扣 50 能量但不直接結束
# - 食物：每吃滿 3 個，隨機補產 1~2 個新食物
# - 能量：最大/初始體力 250，步進扣能量，耗盡才結束
# - 場梯度：state 含 (foodGradX, foodGradY, lizardGradX, lizardGradY) 與 wall_nearness、shacl_warn（供策略感知）
# - PER：優先回放（依 TD-error），本版不含持久化（v2.7.x 才加入 save/load）
# - UI：pygame 視覺化 + trail 漸層；右側資訊欄（以本蒼蠅為準）；空白鍵切換 View/High-Speed；FPS 可調
# - Logging：CSV KPI；checkpoint：最新權重（actor/critics/eat/log_alpha）
# - SHACL：違規僅設 state 旗標，不直接用作 reward（避免干擾）

from __future__ import annotations
import os, sys, csv, random, json, argparse, math
from datetime import datetime
from collections import deque
import numpy as np
import pygame
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# ===== 可選：SHACL/OWL 驗證 =====
try:
    from rdflib import Graph, Namespace, Literal, RDF, XSD
    from pyshacl import validate as shacl_validate
    SHACL_AVAILABLE = True
except Exception:
    SHACL_AVAILABLE = False
    Graph = Namespace = Literal = RDF = XSD = object

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

def to_t(x): return torch.tensor(x, dtype=torch.float32, device=DEVICE)
def now_str(): return datetime.now().strftime("%Y%m%d_%H%M%S")

def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

class CFG:
    # === 在執行不同版本前改這裡 ===
    run_name = "v4_tau002_adamW"
    tau = 0.002
    use_adamw = True   # ← True 時代表使用 AdamW
    actor_lr = 3e-4
    critic_lr = 3e-4
    weight_decay = 1e-4

# ======================
# Networks
# ======================
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim=2, hidden=128):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.mu = nn.Linear(hidden, action_dim)
        self.log_std = nn.Linear(hidden, action_dim)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2)); nn.init.zeros_(m.bias)
    def forward(self, s):
        x = F.relu(self.fc1(s)); x = F.relu(self.fc2(x))
        mu = self.mu(x)
        log_std = torch.clamp(self.log_std(x), -2, 2)
        return mu, torch.exp(log_std)
    def sample(self, s):
        mu, std = self(s)
        dist = torch.distributions.Normal(mu, std)
        z = dist.rsample(); a = torch.tanh(z)
        logp = dist.log_prob(z) - torch.log(1 - a.pow(2) + 1e-7)
        return a, logp.sum(1, keepdim=True)

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim=2, hidden=128):
        super().__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.q = nn.Linear(hidden, 1)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2)); nn.init.zeros_(m.bias)
    def forward(self, s, a):
        x = torch.cat([s, a], dim=1)
        x = F.relu(self.fc1(x)); x = F.relu(self.fc2(x))
        return self.q(x)

class EatMLP(nn.Module):
    """最簡三層 FC：輸入 [energy, expected_gain]，輸出 2 維 Q（不吃/吃）。"""
    def __init__(self, hidden=64):
        super().__init__()
        self.fc1 = nn.Linear(2, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.out = nn.Linear(hidden, 2)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2)); nn.init.zeros_(m.bias)
    def forward(self, x):
        x = F.relu(self.fc1(x)); x = F.relu(self.fc2(x)); return self.out(x)

# ======================
# Prioritized Replay
# ======================
class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6, beta=0.4, beta_increment=1e-4, eps=1e-5):
        self.capacity = capacity; self.alpha = alpha; self.beta = beta; self.beta_inc=beta_increment; self.eps=eps
        self.buffer=[None]*capacity; self.pos=0; self.full=False; self.priorities=np.zeros((capacity,),dtype=np.float32)
    def __len__(self): return self.capacity if self.full else self.pos
    def push(self, transition, td_error=None):
        idx=self.pos; self.buffer[idx]=transition
        max_prio=self.priorities.max() if (self.full or self.pos>0) else 1.0
        base=(abs(float(td_error))+self.eps) if td_error is not None else max_prio
        self.priorities[idx]=base
        self.pos=(self.pos+1)%self.capacity; self.full=self.full or self.pos==0
    def sample(self, batch_size):
        N=self.capacity if self.full else self.pos; assert N>0, "PRB empty"
        prios=self.priorities[:N]; probs=prios**self.alpha; probs/=probs.sum()+1e-12
        idxs=np.random.choice(N,batch_size,p=probs); samples=[self.buffer[i] for i in idxs]
        self.beta=min(1.0,self.beta+self.beta_inc)
        weights=(N*probs[idxs])**(-self.beta); weights/=weights.max()+1e-12
        return idxs, samples, torch.tensor(weights,dtype=torch.float32,device=DEVICE).unsqueeze(1)
    def update_priorities(self, idxs, td_errors):
        arr=td_errors.detach().abs().flatten().tolist() if torch.is_tensor(td_errors) else np.abs(td_errors).flatten().tolist()
        for i,e in zip(idxs,arr): self.priorities[int(i)]=float(e)+self.eps

# ======================
# Environment（含場梯度、蜥蜴 1/3 速、食物補產生）
# ======================
class Food:
    def __init__(self, x, y): self.x, self.y = int(x), int(y); self.eaten=False
    def is_available(self): return not self.eaten

class FlySurvivalEnv:
    def __init__(self, grid_size=20, num_food=None, num_lizards=None,
                 step_cost=1.0, field_reward_scale=0.0,
                 stall_penalty=0.8, wall_intent_penalty=0.6,
                 anti_corner_penalty=0.2, corner_window=12,
                 stagnation_window=18, stagnation_radius=2.0, stagnation_penalty=0.15,
                 trail_len=80, adjacent_eat_snap=True, max_energy=250.0):
        self.grid_size = int(grid_size)
        self.num_food = int(num_food if num_food is not None else max(8, self.grid_size//2))
        self.num_lizards = int(num_lizards if num_lizards is not None else max(4, self.grid_size//3))
        self.step_cost = float(step_cost)
        self.field_reward_scale = float(field_reward_scale)
        self.stall_penalty = float(stall_penalty)
        self.wall_intent_penalty = float(wall_intent_penalty)
        self.anti_corner_penalty = float(anti_corner_penalty)
        self.corner_window = int(corner_window)
        self.stagnation_window = int(stagnation_window)
        self.stagnation_radius = float(stagnation_radius)
        self.stagnation_penalty = float(stagnation_penalty)
        self.adjacent_eat_snap = bool(adjacent_eat_snap)
        self.max_energy = float(max_energy)
        self.shacl_warn = 0.0
        pygame.init(); self.font = pygame.font.Font(None, 26)
        self.trail = deque(maxlen=trail_len)
        self._recent_food_dists = deque(maxlen=self.corner_window)
        self._recent_positions = deque(maxlen=self.stagnation_window)
        self.lizard_step_count = 0  # 控制 1/3 速度
        self.reset()

    # fields
    def _food_field_at(self, pos):
        x, y = pos
        return sum(1/(abs(x-f.x)+abs(y-f.y)+1) for f in self.foods if f.is_available())
    def _lizard_field_at(self, pos):
        x, y = pos
        return sum(1/(abs(x-lx)+abs(y-ly)+1) for (lx, ly) in self.lizards)
    def _field_gradients(self):
        x, y = self.fly
        def clamp(p): return [max(0, min(self.grid_size-1, p[0])), max(0, min(self.grid_size-1, p[1]))]
        left  = self._food_field_at(clamp([x-1,y])); right = self._food_field_at(clamp([x+1,y]))
        up    = self._food_field_at(clamp([x,y-1])); down  = self._food_field_at(clamp([x,y+1]))
        food_gx = (right - left)*0.5; food_gy = (down - up)*0.5
        left  = self._lizard_field_at(clamp([x-1,y])); right = self._lizard_field_at(clamp([x+1,y]))
        up    = self._lizard_field_at(clamp([x,y-1])); down  = self._lizard_field_at(clamp([x,y+1]))
        liz_gx = (right - left)*0.5; liz_gy = (down - up)*0.5
        squash = np.tanh
        return squash(food_gx), squash(food_gy), squash(liz_gx), squash(liz_gy)

    def nearest_food_distance(self):
        return min((abs(self.fly[0]-f.x)+abs(self.fly[1]-f.y) for f in self.foods if f.is_available()), default=999)
    def on_food(self):
        return any(f.is_available() and f.x==self.fly[0] and f.y==self.fly[1] for f in self.foods)
    def available_food_count(self): return sum(1 for f in self.foods if f.is_available())

    def reset(self):
        self.fly = [random.randrange(1, self.grid_size-1), random.randrange(1, self.grid_size-1)]
        self.energy = self.max_energy
        used = {tuple(self.fly)}
        self.foods = []
        while len(self.foods) < self.num_food:
            x = random.randrange(self.grid_size); y = random.randrange(self.grid_size)
            if (x,y) in used: continue
            self.foods.append(Food(x,y)); used.add((x,y))
        self.lizards = []
        while len(self.lizards) < self.num_lizards:
            lx = random.randrange(self.grid_size); ly = random.randrange(self.grid_size)
            if (lx,ly) in used: continue
            self.lizards.append([lx,ly]); used.add((lx,ly))
        self.episode_steps = 0
        self.episode_foods = 0
        self.last_food_dist = self.nearest_food_distance()
        self.last_pos = tuple(self.fly)
        self.on_food_wait = 0
        self.shacl_warn = 0.0
        self.trail.clear(); self.trail.append(tuple(self.fly))
        self._recent_food_dists.clear(); self._recent_food_dists.append(self.last_food_dist)
        self._recent_positions.clear(); self._recent_positions.append(tuple(self.fly))
        self.lizard_step_count = 0
        return self.get_state()

    def _wall_nearness(self):
        fx, fy = self.fly
        d_wall = min(fx, fy, self.grid_size-1-fx, self.grid_size-1-fy) / (self.grid_size-1)
        return 1.0 - d_wall

    def _local_stagnation_penalty(self):
        if len(self._recent_positions) < self.stagnation_window:
            return 0.0
        xs = [p[0] for p in self._recent_positions]; ys = [p[1] for p in self._recent_positions]
        cx, cy = (max(xs)+min(xs))/2.0, (max(ys)+min(ys))/2.0
        rad = max(max(abs(x-cx) for x in xs), max(abs(y-cy) for y in ys))
        if rad <= self.stagnation_radius and self.nearest_food_distance() > 2:
            return -self.stagnation_penalty
        return 0.0

    def get_state(self):
        fx, fy = self.fly
        x_norm = fx/(self.grid_size-1); y_norm = fy/(self.grid_size-1)
        e_norm = self.energy/self.max_energy
        fgx, fgy, lgx, lgy = self._field_gradients()
        wall = self._wall_nearness()
        return np.array([x_norm, y_norm, e_norm, fgx, fgy, lgx, lgy, wall, float(self.shacl_warn)], dtype=np.float32)

    def _step_lizards(self):
        # 1/3 速度：每 3 步移動一次
        self.lizard_step_count = (self.lizard_step_count + 1) % 3
        if self.lizard_step_count != 0:
            return
        new = []
        for lx, ly in self.lizards:
            if random.random() < 0.7:
                dx = np.sign(self.fly[0] - lx); dy = np.sign(self.fly[1] - ly)
            else:
                dx = random.choice([-1,0,1]); dy = random.choice([-1,0,1])
            nx = max(0, min(self.grid_size-1, lx + int(dx)))
            ny = max(0, min(self.grid_size-1, ly + int(dy)))
            new.append([nx, ny])
        self.lizards = new

    def _anti_corner(self):
        wall = self._wall_nearness()
        if wall < 0.8: return 0.0
        if len(self._recent_food_dists) < self.corner_window: return 0.0
        improving = min(self._recent_food_dists) < self._recent_food_dists[0]
        if not improving and self.nearest_food_distance() > 1:
            return -self.anti_corner_penalty
        return 0.0

    def _snap_to_adjacent_food(self):
        fx, fy = self.fly
        for f in self.foods:
            if not f.is_available(): continue
            if abs(fx - f.x) + abs(fy - f.y) == 1:
                if fx < f.x: fx += 1
                elif fx > f.x: fx -= 1
                if fy < f.y: fy += 1
                elif fy > f.y: fy -= 1
                self.fly = [fx, fy]
                return True
        return False

    def _spawn_foods(self, k_min=1, k_max=2):
        k = random.randint(k_min, k_max)
        for _ in range(k):
            tries = 0
            while tries < 20:
                x = random.randrange(self.grid_size); y = random.randrange(self.grid_size)
                if (x,y) != tuple(self.fly) and all(not(f.is_available() and f.x==x and f.y==y) for f in self.foods):
                    self.foods.append(Food(x,y)); break
                tries += 1

    def step(self, move_dir:int, eat_action:int):
        DIRS = {0:(0,-1), 1:(0,1), 2:(-1,0), 3:(1,0), 4:(-1,-1), 5:(1,-1), 6:(-1,1), 7:(1,1)}
        intended = DIRS.get(move_dir, (0,0))
        prev = tuple(self.fly)
        nx = self.fly[0] + intended[0]
        ny = self.fly[1] + intended[1]
        r = -0.01; done = False
        eat_r = 0.0; ate_flag = False

        # 邊界懲罰 + clamp
        if nx < 0 or nx >= self.grid_size or ny < 0 or ny >= self.grid_size:
            r -= self.wall_intent_penalty
        self.fly[0] = max(0, min(self.grid_size-1, nx))
        self.fly[1] = max(0, min(self.grid_size-1, ny))

        if tuple(self.fly) == prev:
            r -= self.stall_penalty

        # 蜥蜴碰撞：扣 50 能量，不結束
        if self.fly in self.lizards:
            self.energy -= 50.0; r -= 10.0

        # 蜥蜴移動（1/3 速）與再次碰撞
        self._step_lizards()
        if self.fly in self.lizards:
            self.energy -= 50.0; r -= 10.0

        # 能量消耗與死亡
        self.energy -= self.step_cost
        if self.energy <= 0:
            r -= 50.0; done = True

        # Anti-corner + 區域停滯
        r += self._anti_corner(); r += self._local_stagnation_penalty()

        # 可選：場位能差（預設 0，不會因附近繞圈加分）
        if self.field_reward_scale != 0.0:
            curr_food_dist = self.nearest_food_distance()
            r += self.field_reward_scale * (self.last_food_dist - curr_food_dist)
            self.last_food_dist = curr_food_dist
        else:
            self.last_food_dist = self.nearest_food_distance()

        # 吃（含鄰格貼靠）
        actually_on_food = self.on_food()
        auto_eat_threshold = 0.50
        if self.adjacent_eat_snap and (eat_action == 1) and (not actually_on_food):
            snapped = self._snap_to_adjacent_food();
            if snapped: actually_on_food = self.on_food()
        if actually_on_food:
            do_eat = (eat_action == 1) or (self.energy / self.max_energy) < auto_eat_threshold
            if do_eat:
                for f in self.foods:
                    if f.is_available() and f.x==self.fly[0] and f.y==self.fly[1]:
                        f.eaten = True; self.episode_foods += 1
                        gain = 10.0 if self.energy < 0.8*self.max_energy else 2.0
                        self.energy = min(self.max_energy, self.energy + gain)
                        r += gain; eat_r += gain; ate_flag = True
                        self.on_food_wait = 0
                        # 每吃滿 3 個，補產 1~2 個
                        if (self.episode_foods % 3) == 0:
                            self._spawn_foods(1,2)
                        break
            else:
                self.on_food_wait += 1
                urgency = 1.0 + max(0.0, (0.8*self.max_energy - self.energy)) / (0.4*self.max_energy)
                skip_pen = 1.0 * urgency * min(self.on_food_wait, 3)
                r -= skip_pen; eat_r -= skip_pen

        # 食物全吃光（理論上補產機制下較少發生）
        if self.available_food_count() == 0 and not done:
            r += 30.0; done = True

        # 記錄/回傳
        self.fly[0] = max(0, min(self.grid_size-1, int(self.fly[0])))
        self.fly[1] = max(0, min(self.grid_size-1, int(self.fly[1])))
        self.trail.append(tuple(self.fly))
        self._recent_food_dists.append(self.nearest_food_distance())
        self._recent_positions.append(tuple(self.fly))
        self.last_pos = tuple(self.fly)
        self.episode_steps += 1
        next_state = self.get_state()
        info = {"eat_r": eat_r, "ate": ate_flag, "on_food": self.on_food(), "foods_left": self.available_food_count()}
        return next_state, r, done, info

# ======================
# Eat features（本版：只 2 維）
# ======================
def build_eat_features(env: FlySurvivalEnv):
    energy = float(env.energy)
    expected_gain = 10.0 if env.energy < 0.8*env.max_energy else 2.0
    return np.array([energy, expected_gain], dtype=np.float32)

# ======================
# SAC action -> discrete dir (8) with straight-line preference + sniff mix（食物嗅覺）
# ======================
def sac_cont_to_dir(a2: torch.Tensor, s_np: np.ndarray, sniff_lambda_base: float) -> int:
    # 嗅覺導引：用食物梯度（state 第 3,4 項）做方向混合
    ax, ay = float(a2[0,0].item()), float(a2[0,1].item())
    fgx, fgy = float(s_np[3]), float(s_np[4])
    norm = math.sqrt(fgx*fgx + fgy*fgy) + 1e-6
    sx, sy = fgx/norm, fgy/norm
    e_norm = float(s_np[2])
    lam = sniff_lambda_base * (1.0 + (1.0 - e_norm))  # 能量越低 → 越靠氣味
    ax, ay = math.tanh((1.0-lam)*ax + lam*sx), math.tanh((1.0-lam)*ay + lam*sy)
    straight_bias = 1.1
    if abs(ax) > straight_bias * abs(ay):
        return 3 if ax > 0 else 2
    if abs(ay) > straight_bias * abs(ax):
        return 1 if ay > 0 else 0
    dirs = [(-1,-1),(1,-1),(-1,1),(1,1)]
    scores = [ax*dx + ay*dy for (dx,dy) in dirs]
    return [4,5,6,7][int(np.argmax(scores))]

# ======================
# SHACL glue（違規只進 state，不當 reward）
# ======================
OWL_PATH = "fly-survival.owl.ttl"
SHAPES_PATH = "fly-survival.shapes.ttl"
FLY = Namespace("http://example.org/fly#") if SHACL_AVAILABLE else None
shapes_graph = None
owl_graph = None

def world_to_rdf_graph(env: FlySurvivalEnv, last_action_class: str):
    if not SHACL_AVAILABLE: return None
    g = Graph(); g.bind("fly", FLY)
    fly = FLY["fly1"]; g.add((fly, RDF.type, FLY.Fly))
    x_norm, y_norm, e_norm, fgx, fgy, lgx, lgy, wall, shw = env.get_state()
    g.add((fly, FLY.xNorm,       Literal(float(x_norm), datatype=XSD.float)))
    g.add((fly, FLY.yNorm,       Literal(float(y_norm), datatype=XSD.float)))
    g.add((fly, FLY.energyNorm,  Literal(float(e_norm), datatype=XSD.float)))
    g.add((fly, FLY.foodGradX,   Literal(float(fgx), datatype=XSD.float)))
    g.add((fly, FLY.foodGradY,   Literal(float(fgy), datatype=XSD.float)))
    g.add((fly, FLY.lizardGradX, Literal(float(lgx), datatype=XSD.float)))
    g.add((fly, FLY.lizardGradY, Literal(float(lgy), datatype=XSD.float)))
    g.add((fly, FLY.onFood,      Literal(bool(env.on_food()), datatype=XSD.boolean)))
    act = FLY["lastAction1"]; g.add((act, RDF.type, FLY[last_action_class])); g.add((fly, FLY.lastAction, act))
    return g

def ensure_shacl_graphs_loaded():
    if not SHACL_AVAILABLE: return
    global shapes_graph, owl_graph
    if shapes_graph is None and os.path.exists(SHAPES_PATH):
        shapes_graph = Graph(); shapes_graph.parse(SHAPES_PATH, format="turtle")
    if owl_graph is None and os.path.exists(OWL_PATH):
        owl_graph = Graph(); owl_graph.parse(OWL_PATH, format="turtle")

def run_shacl_once(env: FlySurvivalEnv, last_action_class: str):
    if not SHACL_AVAILABLE: return ("SKIP", "pySHACL not installed")
    ensure_shacl_graphs_loaded()
    if shapes_graph is None: return ("SKIP", "SHACL shapes not found")
    data_g = world_to_rdf_graph(env, last_action_class)
    conforms, report_graph, report_text = shacl_validate(
        data_graph=data_g,
        shacl_graph=shapes_graph,
        ont_graph=owl_graph,
        inference="rdfs",
        abort_on_first=False,
        allow_infos=True,
        allow_warnings=True,
    )
    return ("OK" if conforms else "WARN/VIOL", str(report_text))

# ======================
# Training main
# ======================
def main():
    parser = argparse.ArgumentParser(description="Fly Survival – v2.5.0")
    parser.add_argument('--episodes', type=int, default=400)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--grid', type=int, default=20)
    parser.add_argument('--foods', type=int, default=0, help='0 表示依 GRID 自動')
    parser.add_argument('--lizards', type=int, default=0, help='0 表示依 GRID 自動')
    parser.add_argument('--field_r_scale', type=float, default=0.0)
    parser.add_argument('--csv_dir', type=str, default='logs')
    parser.add_argument('--save_dir', type=str, default='models')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--high_speed', action='store_true')
    parser.add_argument('--fps', type=int, default=15)
    parser.add_argument('--cell_px', type=int, default=24)
    parser.add_argument('--start_steps', type=int, default=3000)
    parser.add_argument('--cap', type=int, default=40000)
    parser.add_argument('--batch', type=int, default=96)
    parser.add_argument('--sniff_lambda', type=float, default=0.25)
    args = parser.parse_args()
    args.seed = 0

    set_seed(args.seed)
    os.makedirs(args.csv_dir, exist_ok=True); os.makedirs(args.save_dir, exist_ok=True)

    print(f"[Run] {CFG.run_name} | tau={CFG.tau} | opt={'AdamW' if CFG.use_adamw else 'Adam'} "
          f"| actor_lr={CFG.actor_lr} | critic_lr={CFG.critic_lr} | wd={CFG.weight_decay}",
          flush=True)

    env = FlySurvivalEnv(grid_size=args.grid,
                         num_food=(args.foods if args.foods>0 else None),
                         num_lizards=(args.lizards if args.lizards>0 else None),
                         step_cost=1.0, field_reward_scale=args.field_r_scale,
                         stall_penalty=0.8, wall_intent_penalty=0.6,
                         max_energy=250.0)

    state_dim_move = 9; action_dim = 2
    actor = Actor(state_dim_move, action_dim=action_dim).to(DEVICE)
    critic1 = Critic(state_dim_move, action_dim=action_dim).to(DEVICE)
    critic2 = Critic(state_dim_move, action_dim=action_dim).to(DEVICE)
    tcritic1 = Critic(state_dim_move, action_dim=action_dim).to(DEVICE)
    tcritic2 = Critic(state_dim_move, action_dim=action_dim).to(DEVICE)
    tcritic1.load_state_dict(critic1.state_dict()); tcritic2.load_state_dict(critic2.state_dict())
    if CFG.use_adamw:
        opt_actor = torch.optim.AdamW(actor.parameters(), lr=CFG.actor_lr, weight_decay=CFG.weight_decay)
        opt_c1 = torch.optim.AdamW(critic1.parameters(), lr=CFG.critic_lr, weight_decay=CFG.weight_decay)
        opt_c2 = torch.optim.AdamW(critic2.parameters(), lr=CFG.critic_lr, weight_decay=CFG.weight_decay)
    else:
        opt_actor = torch.optim.Adam(actor.parameters(), lr=CFG.actor_lr)
        opt_c1 = torch.optim.Adam(critic1.parameters(), lr=CFG.critic_lr)
        opt_c2 = torch.optim.Adam(critic2.parameters(), lr=CFG.critic_lr)


    log_alpha = torch.tensor(np.log(0.2), dtype=torch.float32, device=DEVICE, requires_grad=True)
    opt_alpha = optim.Adam([log_alpha], lr=3e-4)
    target_entropy = -float(action_dim)

    eat_net = EatMLP(hidden=64).to(DEVICE)
    eat_tgt = EatMLP(hidden=64).to(DEVICE)
    eat_tgt.load_state_dict(eat_net.state_dict())
    opt_eat = optim.Adam(eat_net.parameters(), lr=3e-4)
    eat_eps_start, eat_eps_end, eat_eps_decay = 0.25, 0.02, 1e-3
    eat_eps = eat_eps_start
    eat_update_steps = 0
    eat_target_update_every = 100

    move_buf = PrioritizedReplayBuffer(args.cap, alpha=0.6, beta=0.4, beta_increment=1e-4)
    eat_buf  = PrioritizedReplayBuffer(args.cap,  alpha=0.6, beta=0.4, beta_increment=1e-4)

    csv_path = os.path.join(args.csv_dir, f"{CFG.run_name}_{now_str()}.csv")
    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerow(["Episode","Steps","Foods","C1","C2","Eat","Actor","Alpha","Reward","EatEps","PRBBeta","Stuck","OnFoodWaitMax"])

    # 續訓（僅載入權重；PER 不持久化）
    if args.resume:
        try:
            actor.load_state_dict(torch.load(os.path.join(args.save_dir, "actor_latest.pth"), map_location=DEVICE))
            critic1.load_state_dict(torch.load(os.path.join(args.save_dir, "critic1_latest.pth"), map_location=DEVICE))
            critic2.load_state_dict(torch.load(os.path.join(args.save_dir, "critic2_latest.pth"), map_location=DEVICE))
            tcritic1.load_state_dict(torch.load(os.path.join(args.save_dir, "tcritic1_latest.pth"), map_location=DEVICE))
            tcritic2.load_state_dict(torch.load(os.path.join(args.save_dir, "tcritic2_latest.pth"), map_location=DEVICE))
            eat_net.load_state_dict(torch.load(os.path.join(args.save_dir, "eatnet_latest.pth"), map_location=DEVICE))
            eat_tgt.load_state_dict(eat_net.state_dict())
            if os.path.exists(os.path.join(args.save_dir, "log_alpha_latest.npy")):
                log_alpha.data = torch.tensor(float(np.load(os.path.join(args.save_dir, "log_alpha_latest.npy"))), device=DEVICE)
            print("[Resume] Loaded latest model weights.", file=sys.stderr, flush=True)
        except Exception as e:
            print(f"[Resume] Skip ({e})", file=sys.stderr, flush=True)

    GRID = env.grid_size
    cell_px = args.cell_px; info_w = 260
    screen_w = GRID*cell_px + info_w; screen_h = GRID*cell_px + 40
    screen = pygame.display.set_mode((screen_w, screen_h))
    font = pygame.font.Font(None, 26)
    clock = pygame.time.Clock()
    fps = args.fps; view_mode = not args.high_speed
    shacl_status = "N/A"
    start_steps = args.start_steps

    for ep in range(1, args.episodes+1):
        s = env.reset()
        total_r = 0.0; steps = 0
        c1_losses, c2_losses, eat_losses, actor_losses = [], [], [], []
        done = False
        stuck_steps = 0
        on_food_wait_max = 0
        last_eat_r_disp = 0.0

        if SHACL_AVAILABLE:
            st, rep = run_shacl_once(env, last_action_class="NotEat")
            shacl_status = st
            env.shacl_warn = 0.0 if st in ("OK","SKIP") else 1.0
            if st not in ("OK","SKIP"):
                print(rep, file=sys.stderr, flush=True)

        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit(); sys.exit(0)
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE: view_mode = not view_mode
                    elif event.key in (pygame.K_PLUS, pygame.K_EQUALS): fps = min(120, fps+5)
                    elif event.key == pygame.K_MINUS: fps = max(1, fps-5)

            s_t = to_t(s).unsqueeze(0)
            with torch.no_grad():
                a_cont, _ = actor.sample(s_t)
                move_dir = sac_cont_to_dir(a_cont, s, args.sniff_lambda)
                eat_feat_np = build_eat_features(env)
                eat_feat_t = to_t(eat_feat_np).unsqueeze(0)
                eat_qs = eat_net(eat_feat_t)
                eat_action = int(eat_qs.argmax(dim=1).item())
                # gating：不在食物上 → 不吃
                if not env.on_food():
                    eat_action = 0

            ns, r, done, info = env.step(move_dir, eat_action)
            total_r += r; steps += 1
            last_eat_r_disp = float(info.get("eat_r", 0.0))
            if tuple(env.fly) == env.last_pos: stuck_steps += 1
            on_food_wait_max = max(on_food_wait_max, env.on_food_wait)
            eat_eps = max(eat_eps_end, eat_eps - eat_eps_decay)

            if SHACL_AVAILABLE and (info.get("on_food", False) or steps % 20 == 0):
                st, rep = run_shacl_once(env, last_action_class=("Eat" if info.get("ate", False) else "NotEat"))
                shacl_status = st
                env.shacl_warn = 0.0 if st in ("OK","SKIP") else 1.0
                if st not in ("OK","SKIP"):
                    print(rep, file=sys.stderr, flush=True)

            # Store move sample
            move_buf.push((s, a_cont.squeeze(0).detach().cpu().numpy(), r, ns, float(done)), td_error=1.0)

            # Store eat sample：在食物上時才收樣本
            if info.get("on_food", False):
                next_eat_feat_np = build_eat_features(env)
                r_eat = float(info.get("eat_r", 0.0))
                eat_buf.push((eat_feat_np, eat_action, r_eat, next_eat_feat_np, float(done)), td_error=1.0)

            s = ns

            # ===== train Move (SAC) =====
            if len(move_buf) >= args.batch and len(move_buf) > start_steps:
                idxs, batch, w = move_buf.sample(args.batch)
                b_s, b_a, b_r, b_ns, b_d = zip(*batch)
                b_s  = to_t(np.array(b_s,  dtype=np.float32))
                b_a  = to_t(np.array(b_a,  dtype=np.float32))
                b_r  = to_t(np.array(b_r,  dtype=np.float32)).unsqueeze(1)
                b_ns = to_t(np.array(b_ns, dtype=np.float32))
                b_d  = to_t(np.array(b_d,  dtype=np.float32)).unsqueeze(1)

                alpha = log_alpha.exp().detach()
                with torch.no_grad():
                    na, nlogp = actor.sample(b_ns)
                    tq1 = tcritic1(b_ns, na); tq2 = tcritic2(b_ns, na)
                    tmin = torch.min(tq1, tq2) - alpha * nlogp
                    target_q = b_r + (1.0 - b_d) * 0.99 * tmin

                q1 = critic1(b_s, b_a); q2 = critic2(b_s, b_a)
                c1_loss = (w * F.mse_loss(q1, target_q, reduction="none")).mean()
                c2_loss = (w * F.mse_loss(q2, target_q, reduction="none")).mean()
                opt_c1.zero_grad(); c1_loss.backward(); torch.nn.utils.clip_grad_norm_(critic1.parameters(), 5.0); opt_c1.step()
                opt_c2.zero_grad(); c2_loss.backward(); torch.nn.utils.clip_grad_norm_(critic2.parameters(), 5.0); opt_c2.step()
                td_avg = 0.5*((target_q - q1).abs() + (target_q - q2).abs())
                move_buf.update_priorities(idxs, td_avg)
                c1_losses.append(c1_loss.item()); c2_losses.append(c2_loss.item())

                a_pi, logp_pi = actor.sample(b_s)
                alpha_curr = log_alpha.exp()
                q1_pi = critic1(b_s, a_pi); q2_pi = critic2(b_s, a_pi)
                actor_loss = (alpha_curr * logp_pi - torch.min(q1_pi, q2_pi)).mean()
                opt_actor.zero_grad(); actor_loss.backward(); torch.nn.utils.clip_grad_norm_(actor.parameters(), 5.0); opt_actor.step()
                actor_losses.append(actor_loss.item())

                alpha_loss = -(log_alpha * (logp_pi + target_entropy).detach()).mean()
                opt_alpha.zero_grad(); alpha_loss.backward(); opt_alpha.step()

                with torch.no_grad():
                    tau = CFG.tau
                    for tp, p in zip(tcritic1.parameters(), critic1.parameters()):
                        tp.data.mul_(1.0 - tau).add_(tau * p.data)
                    for tp, p in zip(tcritic2.parameters(), critic2.parameters()):
                        tp.data.mul_(1.0 - tau).add_(tau * p.data)


            # ===== train Eat (DQN) =====
            if len(eat_buf) >= args.batch and len(eat_buf) > start_steps//2:
                idxs_e, batch_e, w_e = eat_buf.sample(args.batch)
                b_s, b_a, b_r, b_ns, b_d = zip(*batch_e)
                b_s  = to_t(np.array(b_s,  dtype=np.float32))
                b_a  = torch.tensor(b_a, dtype=torch.long, device=DEVICE)
                b_r  = to_t(np.array(b_r,  dtype=np.float32))
                b_ns = to_t(np.array(b_ns, dtype=np.float32))
                b_d  = to_t(np.array(b_d,  dtype=np.float32))
                q = eat_net(b_s).gather(1, b_a.unsqueeze(1)).squeeze(1)
                with torch.no_grad():
                    a_star = eat_net(b_ns).argmax(1)
                    qn = eat_tgt(b_ns).gather(1, a_star.unsqueeze(1)).squeeze(1)
                    target = b_r + (1.0 - b_d) * 0.99 * qn
                td = q - target
                loss_e = (w_e.squeeze(1) * F.smooth_l1_loss(q, target, reduction='none')).mean()
                opt_eat.zero_grad(); loss_e.backward(); torch.nn.utils.clip_grad_norm_(eat_net.parameters(), 5.0); opt_eat.step()
                eat_buf.update_priorities(idxs_e, td.detach().abs())
                eat_losses.append(loss_e.item())
                eat_update_steps += 1
                if eat_update_steps % eat_target_update_every == 0:
                    eat_tgt.load_state_dict(eat_net.state_dict())

            # Render
            screen.fill((250,250,250))
            if view_mode:
                for y in range(GRID):
                    for x in range(GRID):
                        pygame.draw.rect(screen, (230,230,230), (x*cell_px, y*cell_px, cell_px, cell_px), 1)
                for f in env.foods:
                    if f.is_available():
                        pygame.draw.rect(screen, (0,180,0), (f.x*cell_px+2, f.y*cell_px+2, cell_px-4, cell_px-4))
                for lx,ly in env.lizards:
                    pygame.draw.rect(screen, (200,0,0), (lx*cell_px+3, ly*cell_px+3, cell_px-6, cell_px-6))
                pts = list(env.trail); n = len(pts)
                for i in range(1, n):
                    x1,y1 = pts[i-1]; x2,y2 = pts[i]
                    t = i / max(1,n-1)
                    col = (int(40+80*t), int(110+90*(1-t)), int(255-100*t))
                    pygame.draw.line(screen, col, (x1*cell_px+cell_px//2, y1*cell_px+cell_px//2), (x2*cell_px+cell_px//2, y2*cell_px+cell_px//2), 2)
                pygame.draw.rect(screen, (0,0,200), (env.fly[0]*cell_px+4, env.fly[1]*cell_px+4, cell_px-8, cell_px-8))
            info_x = GRID*cell_px + 12
            alpha_val = float(log_alpha.exp().detach().cpu().item())
            txts = [
                f"Episode: {ep}",
                f"Steps: {steps}",
                f"FoodsEaten: {env.episode_foods}",
                f"FoodLeft: {env.available_food_count()}",
                f"TotalR: {total_r:.1f}",
                f"Energy: {env.energy:.1f}/{env.max_energy:.0f}",
                f"Alpha:  {alpha_val:.3f}",
                f"EatR: {last_eat_r_disp:+.1f}",
                f"EatEps: {eat_eps:.3f}",
                f"Mode: {'View' if view_mode else 'High Speed'}",
                f"FPS: {fps}",
                f"SHACL: {shacl_status if SHACL_AVAILABLE else 'SKIP'}"
            ]
            for i,t in enumerate(txts): screen.blit(font.render(t,True,(0,0,0)), (info_x, 12+i*26))
            legend_y = 12+len(txts)*26+8
            pygame.draw.rect(screen, (0,180,0), (info_x, legend_y, 16, 16)); screen.blit(font.render("Food", True,(0,0,0)), (info_x+24, legend_y))
            pygame.draw.rect(screen, (200,0,0), (info_x, legend_y+22,16,16)); screen.blit(font.render("Lizard",True,(0,0,0)), (info_x+24, legend_y+22))
            pygame.draw.rect(screen, (0,0,200), (info_x, legend_y+44,16,16)); screen.blit(font.render("Fly",  True,(0,0,0)), (info_x+24, legend_y+44))
            bar_y = legend_y+72; pygame.draw.rect(screen, (0,0,0), (info_x, bar_y, 200, 14), 1)
            w = int(200*env.energy/env.max_energy); pygame.draw.rect(screen, (0,180,0), (info_x+1, bar_y+1, max(0,w), 12))
            pygame.display.flip(); clock.tick(fps if view_mode else 60)

        c1m = float(np.mean(c1_losses)) if c1_losses else 0.0
        c2m = float(np.mean(c2_losses)) if c2_losses else 0.0
        em  = float(np.mean(eat_losses)) if eat_losses else 0.0
        am  = float(np.mean(actor_losses)) if actor_losses else 0.0
        with open(csv_path, "a", newline="") as f:
            csv.writer(f).writerow([ep, env.episode_steps, env.episode_foods, c1m, c2m, em, am, float(log_alpha.exp().item()), total_r, eat_eps, move_buf.beta, stuck_steps, on_food_wait_max])
        print(json.dumps({
            "ts": now_str(), "ep": ep, "steps": env.episode_steps, "foods": env.episode_foods,
            "c1": round(c1m,4), "c2": round(c2m,4), "eat": round(em,4), "act": round(am,4),
            "alpha": round(float(log_alpha.exp().item()),4), "R": round(total_r,2),
            "eat_eps": round(eat_eps,3), "per_beta": round(move_buf.beta,3)
        }, ensure_ascii=False), file=sys.stdout, flush=True)

        # save latest weights
        torch.save(actor.state_dict(),   os.path.join(args.save_dir, "actor_latest.pth"))
        torch.save(critic1.state_dict(), os.path.join(args.save_dir, "critic1_latest.pth"))
        torch.save(critic2.state_dict(), os.path.join(args.save_dir, "critic2_latest.pth"))
        torch.save(tcritic1.state_dict(),os.path.join(args.save_dir, "tcritic1_latest.pth"))
        torch.save(tcritic2.state_dict(),os.path.join(args.save_dir, "tcritic2_latest.pth"))
        torch.save(eat_net.state_dict(), os.path.join(args.save_dir, "eatnet_latest.pth"))
        np.save(os.path.join(args.save_dir, "log_alpha_latest.npy"), float(log_alpha.detach().cpu().item()))

if __name__ == "__main__":
    main()
