import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error

# 设置绘图字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ==========================================================
# 1. 基础环境与全局物理/几何参数配置 (推理必备)
# ==========================================================
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"当前使用设备: {device}")

um_to_mm = 1.0 / 1000.0

# 几何尺寸转换 (mm)
W_chip_full, L_chip_full, H_chip = 6000.0 * um_to_mm, 6000.0 * um_to_mm, 250.0 * um_to_mm
W_bump_full, L_bump_full, H_bump = 6000.0 * um_to_mm, 6000.0 * um_to_mm, 80.0 * um_to_mm
W_sub_full, L_sub_full, H_sub = 12000.0 * um_to_mm, 12000.0 * um_to_mm, 600.0 * um_to_mm

# 四分之一模型边界
x_sub_min, x_sub_max = 0.0, W_sub_full / 2.0
y_sub_min, y_sub_max = 0.0, L_sub_full / 2.0
x_bump_min, x_bump_max = 0.0, W_bump_full / 2.0
y_bump_min, y_bump_max = 0.0, L_bump_full / 2.0
x_chip_min, x_chip_max = 0.0, W_chip_full / 2.0
y_chip_min, y_chip_max = 0.0, L_chip_full / 2.0

z0 = 0.0
z1 = H_sub
z2 = z1 + H_bump
z3 = z2 + H_chip

subdomain_x = [(x_sub_min, x_sub_max), (x_bump_min, x_bump_max), (x_chip_min, x_chip_max)]
subdomain_y = [(y_sub_min, y_sub_max), (y_bump_min, y_bump_max), (y_chip_min, y_chip_max)]
subdomain_z = [(z0, z1), (z1, z2), (z2, z3)]

# 归一化缩放参数与时间温度参数
total_time = 7.0
T_init = 22.0
T_max = 250.0
dT = T_max - T_init
SCALE_A = 0.85
BETA_HC = 3.0

# ==========================================================
# 2. 缩放辅助函数
# ==========================================================
def scale_input(val, val_min, val_max):
    return 2.0 * SCALE_A * (val - val_min) / (val_max - val_min) - SCALE_A

def scale_x(x_phys, sub_idx):
    xmin, xmax = subdomain_x[sub_idx]
    return scale_input(x_phys, xmin, xmax)

def scale_y(y_phys, sub_idx):
    ymin, ymax = subdomain_y[sub_idx]
    return scale_input(y_phys, ymin, ymax)

def scale_z(z_phys, sub_idx):
    zmin, zmax = subdomain_z[sub_idx]
    return scale_input(z_phys, zmin, zmax)

def scale_t(t_phys):
    return scale_input(t_phys, 0.0, total_time)

# ==========================================================
# 3. 网络架构定义 (精简版：仅保留推理功能)
# ==========================================================
class StandardMLP(nn.Module):
    def __init__(self, dim_in=4, dim_out=1, n_layer=8, n_node=96):
        super().__init__()
        self.activation = nn.Tanh()
        layers = [nn.Linear(dim_in, n_node), self.activation]
        for _ in range(n_layer):
            layers += [nn.Linear(n_node, n_node), self.activation]
        layers.append(nn.Linear(n_node, dim_out))
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)

class XPINNInference:
    def __init__(self):
        # 3个子网络分别对应基板、凸块、芯片
        self.nets = [StandardMLP().to(device) for _ in range(3)]

    def _forward_hard(self, net_idx, x_s, y_s, z_s, t_s):
        xyzt = torch.cat([x_s, y_s, z_s, t_s], dim=1)
        n_raw = self.nets[net_idx](xyzt)
        g = 1.0 - torch.exp(-BETA_HC * (t_s + SCALE_A))
        theta = n_raw * g
        return theta

    def predict(self, x_phys, y_phys, z_phys, t_phys):
        out = torch.zeros_like(x_phys)

        # 区域0：Substrate
        mask0 = (
                (z_phys >= z0) & (z_phys <= z1) &
                (x_phys >= x_sub_min) & (x_phys <= x_sub_max) &
                (y_phys >= y_sub_min) & (y_phys <= y_sub_max)
        ).squeeze()
        if mask0.any():
            xp, yp, zp, tp = x_phys[mask0], y_phys[mask0], z_phys[mask0], t_phys[mask0]
            out[mask0] = self._forward_hard(0, scale_x(xp, 0), scale_y(yp, 0), scale_z(zp, 0), scale_t(tp))

        # 区域1：Bump
        mask1 = (
                (z_phys > z1) & (z_phys <= z2) &
                (x_phys >= x_bump_min) & (x_phys <= x_bump_max) &
                (y_phys >= y_bump_min) & (y_phys <= y_bump_max)
        ).squeeze()
        if mask1.any():
            xp, yp, zp, tp = x_phys[mask1], y_phys[mask1], z_phys[mask1], t_phys[mask1]
            out[mask1] = self._forward_hard(1, scale_x(xp, 1), scale_y(yp, 1), scale_z(zp, 1), scale_t(tp))

        # 区域2：Chip
        mask2 = (
                (z_phys > z2) & (z_phys <= z3) &
                (x_phys >= x_chip_min) & (x_phys <= x_chip_max) &
                (y_phys >= y_chip_min) & (y_phys <= y_chip_max)
        ).squeeze()
        if mask2.any():
            xp, yp, zp, tp = x_phys[mask2], y_phys[mask2], z_phys[mask2], t_phys[mask2]
            out[mask2] = self._forward_hard(2, scale_x(xp, 2), scale_y(yp, 2), scale_z(zp, 2), scale_t(tp))

        return out

# ==========================================================
# 4. 验证与可视化函数
# ==========================================================
def evaluate_and_visualize_ansys(model, data_dir):
    print("=" * 80)
    print("开始加载真实数据进行验证，并生成 ANSYS 风格云图...")
    print("=" * 80)

    ansys_cmap = 'jet'
    times = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
    metrics = {"time": [], "mae": [], "max_err": [], "l2_rel": []}

    for t_val in times:
        file_path = os.path.join(data_dir, f"{int(t_val)}.txt")
        if not os.path.exists(file_path):
            print(f"找不到文件: {file_path}，跳过。")
            continue

        try:
            df = pd.read_csv(file_path, sep=r'\s+', skiprows=1, header=None, encoding='gbk', on_bad_lines='skip')
        except Exception as e:
            try:
                df = pd.read_csv(file_path, sep=r'\s+', skiprows=1, header=None, encoding='latin1', on_bad_lines='skip')
            except Exception as e2:
                print(f"读取文件 {file_path} 彻底失败: {e2}")
                continue

        x_real = df.iloc[:, 1].values
        y_real = df.iloc[:, 2].values
        z_real = df.iloc[:, 3].values
        T_real = df.iloc[:, 4].values

        x_t = torch.tensor(x_real, dtype=torch.float32).view(-1, 1).to(device)
        y_t = torch.tensor(y_real, dtype=torch.float32).view(-1, 1).to(device)
        z_t = torch.tensor(z_real, dtype=torch.float32).view(-1, 1).to(device)
        time_t = (torch.ones_like(x_t) * t_val).to(device)

        with torch.no_grad():
            theta_pred = model.predict(x_t, y_t, z_t, time_t).cpu().numpy().flatten()

        T_pred = T_init + theta_pred * dT

        error_abs = np.abs(T_pred - T_real)
        mae = mean_absolute_error(T_real, T_pred)
        max_err = np.max(error_abs)
        l2_rel = np.linalg.norm(T_pred - T_real) / (np.linalg.norm(T_real) + 1e-8)

        metrics["time"].append(t_val)
        metrics["mae"].append(mae)
        metrics["max_err"].append(max_err)
        metrics["l2_rel"].append(l2_rel)

        print(f"[Time {t_val}s] 节点数: {len(T_real)} | MAE: {mae:.4f} °C | Max Error: {max_err:.4f} °C | Rel L2: {l2_rel:.4%}")

        # --- 绘图 (为避免挂机循环时内存泄漏，可视情况注释掉绘图部分) ---
        fig = plt.figure(figsize=(24, 7))
        fig.suptitle(f"Thermal Field at t={int(t_val)}s  (MAE: {mae:.4f} °C, Max Err: {max_err:.4f} °C)", fontsize=16)

        box_aspect = (np.ptp(x_real), np.ptp(y_real), np.ptp(z_real) * 2)

        ax1 = fig.add_subplot(1, 3, 1, projection='3d')
        p1 = ax1.scatter(x_real, y_real, z_real, c=T_real, cmap=ansys_cmap, s=8, alpha=0.9)
        ax1.set_title("Ground Truth (ANSYS)", fontsize=14)
        ax1.set_box_aspect(box_aspect)
        fig.colorbar(p1, ax=ax1, fraction=0.03, pad=0.1, label='Temperature (°C)')

        ax2 = fig.add_subplot(1, 3, 2, projection='3d')
        vmin_T, vmax_T = np.min(T_real), np.max(T_real)
        p2 = ax2.scatter(x_real, y_real, z_real, c=T_pred, cmap=ansys_cmap, s=8, alpha=0.9, vmin=vmin_T, vmax=vmax_T)
        ax2.set_title("PINN Prediction", fontsize=14)
        ax2.set_box_aspect(box_aspect)
        fig.colorbar(p2, ax=ax2, fraction=0.03, pad=0.1, label='Temperature (°C)')

        ax3 = fig.add_subplot(1, 3, 3, projection='3d')
        p3 = ax3.scatter(x_real, y_real, z_real, c=error_abs, cmap=ansys_cmap, s=8, alpha=0.9)
        ax3.set_title(f"Absolute Error (Max: {max_err:.2f}°C)", fontsize=14)
        ax3.set_box_aspect(box_aspect)
        fig.colorbar(p3, ax=ax3, fraction=0.03, pad=0.1, label='Error (°C)')

        for ax in [ax1, ax2, ax3]:
            ax.set_xlabel('X (mm)')
            ax.set_ylabel('Y (mm)')
            ax.set_zlabel('Z (mm)')
            ax.view_init(elev=25, azim=-45)

        plt.tight_layout()
        save_path = f"ANSYS_Comparison_t{int(t_val)}s.png"
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close()

    # --- 综合评分逻辑 ---
    if len(metrics["time"]) > 0:
        mean_mae = float(np.mean(metrics["mae"]))
        mean_max_err = float(np.mean(metrics["max_err"]))
        absolute_max_err = float(np.max(metrics["max_err"]))

        # 新增混合评分：50% 全局精度 + 30% 平均局部精度 + 20% 极限最差情况
        score = (mean_mae * 0.5) + (mean_max_err * 0.3) + (absolute_max_err * 0.2)
        
        print(f"\n{'='*60}")
        print(f"  AUTORESEARCH SCORE REPORT")
        print(f"  [50%] Mean MAE                 : {mean_mae:.4f} °C")
        print(f"  [30%] Mean of Max Errors       : {mean_max_err:.4f} °C")
        print(f"  [20%] Absolute Max Error       : {absolute_max_err:.4f} °C")
        print(f"{'-'*60}")
        
        # 写入 JSON
        import json
        result = {
            "score": round(score, 4),
            "mean_mae": round(mean_mae, 4),
            "mean_max_err": round(mean_max_err, 4),
            "absolute_max_err": round(absolute_max_err, 4)
        }
        with open("autoresearch_score.json", "w") as f:
            json.dump(result, f, indent=2)

        # 专门给 Claude 留出的唯一锚点（放最后一行，不可更改格式）
        print(f"FINAL_AUTORESEARCH_SCORE: {score:.4f}")
        print(f"{'='*60}\n")

# ==========================================================
# 5. 主程序入口
# ==========================================================
if __name__ == "__main__":
    model = XPINNInference()
    model_path = r"D:\cy\芯片基板pinn\三维\三维1.0.1\best_model_xpinn_3d_quarter.pt"

    if os.path.exists(model_path):
        state_dicts = torch.load(model_path, map_location=device)
        for i, net in enumerate(model.nets):
            net.load_state_dict(state_dicts[i])
            net.eval() 
        print(f"[OK] 成功加载模型权重: {model_path}")
    else:
        print(f"[ERROR] 错误：未找到模型文件 {model_path}")
        exit()

    data_directory = r"D:\cy\芯片基板pinn\三维\data"
    evaluate_and_visualize_ansys(model, data_directory)