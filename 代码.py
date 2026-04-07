import os
import copy
import random
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import grad
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pyDOE import lhs
import matplotlib.pyplot as plt
from tqdm import tqdm

plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f"Random seed fixed: {seed}")


# ==========================================================
# 1. 设备与物理参数 — 三维四分之一模型，单位统一为 mm
# ==========================================================
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"Device: {device}")

# --------------------------
# 几何尺寸（原始给定为 um，这里转成 mm）
# --------------------------
um_to_mm = 1.0 / 1000.0

# 全模型尺寸
W_chip_full = 6000.0 * um_to_mm   # 6.0 mm
L_chip_full = 6000.0 * um_to_mm   # 6.0 mm
H_chip = 250.0 * um_to_mm         # 0.25 mm

W_bump_full = 6000.0 * um_to_mm   # 6.0 mm
L_bump_full = 6000.0 * um_to_mm   # 6.0 mm
H_bump = 80.0 * um_to_mm          # 0.08 mm

W_sub_full = 12000.0 * um_to_mm   # 12.0 mm
L_sub_full = 12000.0 * um_to_mm   # 12.0 mm
H_sub = 600.0 * um_to_mm          # 0.6 mm

# 四分之一模型尺寸：仅保留 x>=0, y>=0
x_sub_min = 0.0
x_sub_max = W_sub_full / 2.0      # 6.0
y_sub_min = 0.0
y_sub_max = L_sub_full / 2.0      # 6.0

x_bump_min = 0.0
x_bump_max = W_bump_full / 2.0    # 3.0
y_bump_min = 0.0
y_bump_max = L_bump_full / 2.0    # 3.0

x_chip_min = 0.0
x_chip_max = W_chip_full / 2.0    # 3.0
y_chip_min = 0.0
y_chip_max = L_chip_full / 2.0    # 3.0

z0 = 0.0
z1 = H_sub
z2 = z1 + H_bump
z3 = z2 + H_chip

# --------------------------
# 热导率 (W / (mm·K))
# --------------------------
k_chip = 150.0 / 1000.0
k_bump = 15.0 / 1000.0
k_sub = 0.5 / 1000.0
k_values = [k_sub, k_bump, k_chip]

# --------------------------
# 热扩散系数 (mm²/s)
# --------------------------
alpha_chip = 150.0 / (2330.0 * 700.0) * 1e6
alpha_bump = 15.0 / (2450.0 * 260.0) * 1e6
alpha_sub = 0.5 / (1800.0 * 1000.0) * 1e6
alpha_values = [alpha_sub, alpha_bump, alpha_chip]

# --------------------------
# 时间与温度参数
# --------------------------
total_time = 7.0
T_init = 22.0
T_max = 250.0
dT = T_max - T_init

# --------------------------
# 输入缩放参数
# --------------------------
SCALE_A = 0.85
BETA_HC = 3.0

# --------------------------
# 子域边界
# --------------------------
subdomain_x = [
    (x_sub_min, x_sub_max),
    (x_bump_min, x_bump_max),
    (x_chip_min, x_chip_max),
]
subdomain_y = [
    (y_sub_min, y_sub_max),
    (y_bump_min, y_bump_max),
    (y_chip_min, y_chip_max),
]
subdomain_z = [
    (z0, z1),
    (z1, z2),
    (z2, z3),
]

dx_vals = [x_sub_max - x_sub_min, x_bump_max - x_bump_min, x_chip_max - x_chip_min]
dy_vals = [y_sub_max - y_sub_min, y_bump_max - y_bump_min, y_chip_max - y_chip_min]
dz_vals = [z1 - z0, z2 - z1, z3 - z2]

print("=" * 80)
print("三维四分之一模型几何参数 (mm):")
print(f"Substrate: x in [{x_sub_min}, {x_sub_max}], y in [{y_sub_min}, {y_sub_max}], z in [{z0}, {z1}]")
print(f"Bump:      x in [{x_bump_min}, {x_bump_max}], y in [{y_bump_min}, {y_bump_max}], z in [{z1}, {z2}]")
print(f"Chip:      x in [{x_chip_min}, {x_chip_max}], y in [{y_chip_min}, {y_chip_max}], z in [{z2}, {z3}]")
print(f"Alpha (mm^2/s): Sub={alpha_sub:.6f}, Bump={alpha_bump:.5f}, Chip={alpha_chip:.4f}")
print(f"BETA_HC = {BETA_HC}")
print("=" * 80)


# ==========================================================
# 2. 缩放函数
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
# 3. 温度载荷函数
# ==========================================================
_chip_t = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7], dtype=torch.float32)
_chip_theta = (torch.tensor([22, 250, 250, 160, 160, 100, 100, 22], dtype=torch.float32) - T_init) / dT

_sub_t = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7], dtype=torch.float32)
_sub_theta = (torch.tensor([22, 100, 100, 100, 100, 100, 100, 22], dtype=torch.float32) - T_init) / dT


def _piecewise_linear_interp(t_query, t_pts, v_pts):
    t_pts_d = t_pts.to(t_query.device)
    v_pts_d = v_pts.to(t_query.device)
    t_q = t_query.squeeze()
    result = torch.zeros_like(t_q)

    for i in range(len(t_pts_d) - 1):
        t_lo, t_hi = t_pts_d[i], t_pts_d[i + 1]
        v_lo, v_hi = v_pts_d[i], v_pts_d[i + 1]

        if i < len(t_pts_d) - 2:
            mask = (t_q >= t_lo) & (t_q < t_hi)
        else:
            mask = (t_q >= t_lo) & (t_q <= t_hi)

        frac = (t_q - t_lo) / (t_hi - t_lo + 1e-12)
        result = torch.where(mask, v_lo + frac * (v_hi - v_lo), result)

    return result.view_as(t_query)


def chip_top_theta(t_phys):
    return _piecewise_linear_interp(t_phys, _chip_t, _chip_theta)


def substrate_bottom_theta(t_phys):
    return _piecewise_linear_interp(t_phys, _sub_t, _sub_theta)


# ==========================================================
# 4. 采样辅助函数
# ==========================================================
def to_tensor(x):
    return torch.tensor(x, dtype=torch.float32, requires_grad=True).reshape(-1, 1).to(device)


def to_tensor_no_grad(x):
    return torch.tensor(x, dtype=torch.float32).reshape(-1, 1).to(device)


def sample_internal_scaled(num, sub_idx):
    xmin, xmax = subdomain_x[sub_idx]
    ymin, ymax = subdomain_y[sub_idx]
    zmin, zmax = subdomain_z[sub_idx]

    samples = lhs(4, samples=num)
    x_phys = xmin + samples[:, 0] * (xmax - xmin)
    y_phys = ymin + samples[:, 1] * (ymax - ymin)
    z_phys = zmin + samples[:, 2] * (zmax - zmin)
    t_phys = samples[:, 3] * total_time

    x_s = 2.0 * SCALE_A * (x_phys - xmin) / (xmax - xmin) - SCALE_A
    y_s = 2.0 * SCALE_A * (y_phys - ymin) / (ymax - ymin) - SCALE_A
    z_s = 2.0 * SCALE_A * (z_phys - zmin) / (zmax - zmin) - SCALE_A
    t_s = 2.0 * SCALE_A * t_phys / total_time - SCALE_A

    return to_tensor(x_s), to_tensor(y_s), to_tensor(z_s), to_tensor(t_s)


def sample_boundary_z_scaled(num, x_range_phys, y_range_phys, z_val_phys, sub_idx):
    samples = lhs(3, samples=num)
    x_phys = x_range_phys[0] + samples[:, 0] * (x_range_phys[1] - x_range_phys[0])
    y_phys = y_range_phys[0] + samples[:, 1] * (y_range_phys[1] - y_range_phys[0])
    t_phys = samples[:, 2] * total_time

    xmin, xmax = subdomain_x[sub_idx]
    ymin, ymax = subdomain_y[sub_idx]
    zmin, zmax = subdomain_z[sub_idx]

    x_s = 2.0 * SCALE_A * (x_phys - xmin) / (xmax - xmin) - SCALE_A
    y_s = 2.0 * SCALE_A * (y_phys - ymin) / (ymax - ymin) - SCALE_A
    z_s_val = 2.0 * SCALE_A * (z_val_phys - zmin) / (zmax - zmin) - SCALE_A
    z_s = np.ones_like(x_phys) * z_s_val
    t_s = 2.0 * SCALE_A * t_phys / total_time - SCALE_A

    return to_tensor(x_s), to_tensor(y_s), to_tensor(z_s), to_tensor(t_s), to_tensor_no_grad(t_phys)


def sample_boundary_x_scaled(num, x_val_phys, y_range_phys, z_range_phys, sub_idx):
    samples = lhs(3, samples=num)
    y_phys = y_range_phys[0] + samples[:, 0] * (y_range_phys[1] - y_range_phys[0])
    z_phys = z_range_phys[0] + samples[:, 1] * (z_range_phys[1] - z_range_phys[0])
    t_phys = samples[:, 2] * total_time

    xmin, xmax = subdomain_x[sub_idx]
    ymin, ymax = subdomain_y[sub_idx]
    zmin, zmax = subdomain_z[sub_idx]

    x_s_val = 2.0 * SCALE_A * (x_val_phys - xmin) / (xmax - xmin) - SCALE_A
    x_s = np.ones_like(y_phys) * x_s_val
    y_s = 2.0 * SCALE_A * (y_phys - ymin) / (ymax - ymin) - SCALE_A
    z_s = 2.0 * SCALE_A * (z_phys - zmin) / (zmax - zmin) - SCALE_A
    t_s = 2.0 * SCALE_A * t_phys / total_time - SCALE_A

    return to_tensor(x_s), to_tensor(y_s), to_tensor(z_s), to_tensor(t_s)


def sample_boundary_y_scaled(num, y_val_phys, x_range_phys, z_range_phys, sub_idx):
    samples = lhs(3, samples=num)
    x_phys = x_range_phys[0] + samples[:, 0] * (x_range_phys[1] - x_range_phys[0])
    z_phys = z_range_phys[0] + samples[:, 1] * (z_range_phys[1] - z_range_phys[0])
    t_phys = samples[:, 2] * total_time

    xmin, xmax = subdomain_x[sub_idx]
    ymin, ymax = subdomain_y[sub_idx]
    zmin, zmax = subdomain_z[sub_idx]

    x_s = 2.0 * SCALE_A * (x_phys - xmin) / (xmax - xmin) - SCALE_A
    y_s_val = 2.0 * SCALE_A * (y_val_phys - ymin) / (ymax - ymin) - SCALE_A
    y_s = np.ones_like(x_phys) * y_s_val
    z_s = 2.0 * SCALE_A * (z_phys - zmin) / (zmax - zmin) - SCALE_A
    t_s = 2.0 * SCALE_A * t_phys / total_time - SCALE_A

    return to_tensor(x_s), to_tensor(y_s), to_tensor(z_s), to_tensor(t_s)


def sample_interface_scaled(num, x_range_phys, y_range_phys, z_val_phys, sub_idx_a, sub_idx_b):
    samples = lhs(3, samples=num)
    x_phys = x_range_phys[0] + samples[:, 0] * (x_range_phys[1] - x_range_phys[0])
    y_phys = y_range_phys[0] + samples[:, 1] * (y_range_phys[1] - y_range_phys[0])
    t_phys = samples[:, 2] * total_time

    xa_min, xa_max = subdomain_x[sub_idx_a]
    ya_min, ya_max = subdomain_y[sub_idx_a]
    za_min, za_max = subdomain_z[sub_idx_a]
    x_s_a = 2.0 * SCALE_A * (x_phys - xa_min) / (xa_max - xa_min) - SCALE_A
    y_s_a = 2.0 * SCALE_A * (y_phys - ya_min) / (ya_max - ya_min) - SCALE_A
    z_s_a_val = 2.0 * SCALE_A * (z_val_phys - za_min) / (za_max - za_min) - SCALE_A
    z_s_a = np.ones_like(x_phys) * z_s_a_val
    t_s_a = 2.0 * SCALE_A * t_phys / total_time - SCALE_A

    xb_min, xb_max = subdomain_x[sub_idx_b]
    yb_min, yb_max = subdomain_y[sub_idx_b]
    zb_min, zb_max = subdomain_z[sub_idx_b]
    x_s_b = 2.0 * SCALE_A * (x_phys - xb_min) / (xb_max - xb_min) - SCALE_A
    y_s_b = 2.0 * SCALE_A * (y_phys - yb_min) / (yb_max - yb_min) - SCALE_A
    z_s_b_val = 2.0 * SCALE_A * (z_val_phys - zb_min) / (zb_max - zb_min) - SCALE_A
    z_s_b = np.ones_like(x_phys) * z_s_b_val
    t_s_b = 2.0 * SCALE_A * t_phys / total_time - SCALE_A

    return (
        (to_tensor(x_s_a), to_tensor(y_s_a), to_tensor(z_s_a), to_tensor(t_s_a)),
        (to_tensor(x_s_b), to_tensor(y_s_b), to_tensor(z_s_b), to_tensor(t_s_b)),
    )


# ==========================================================
# 5. 采样生成函数
# ==========================================================
N_INT_SUB = 11000
N_INT_BUMP = 4000
N_INT_CHIP = 7000

N_BC_SUB_BOTTOM = 3500
N_BC_CHIP_TOP = 2800

# substrate 对称面 x=0, y=0
N_SUB_X0 = 1400
N_SUB_Y0 = 1400

# substrate 外侧绝热 x=max, y=max
N_SUB_XMAX = 1400
N_SUB_YMAX = 1400

# substrate 顶面裸露区域 z=z1，分成两个矩形
N_SUB_TOP_EXPOSED_R = 1400   # x∈[3,6], y∈[0,6]
N_SUB_TOP_EXPOSED_U = 1000   # x∈[0,3], y∈[3,6]

# bump 对称面与外侧面
N_BUMP_X0 = 800
N_BUMP_Y0 = 800
N_BUMP_XMAX = 800
N_BUMP_YMAX = 800

# chip 对称面与外侧面
N_CHIP_X0 = 800
N_CHIP_Y0 = 800
N_CHIP_XMAX = 800
N_CHIP_YMAX = 800

# interfaces
N_IF_12 = 2700
N_IF_23 = 2700

# L形拐点边缘加密：靠近x=3和y=3的接口区域
N_IF_12_EDGE_X = 1000   # x∈[2.5, 3.0], y∈[0, 3.0], z=z1
N_IF_12_EDGE_Y = 1000   # x∈[0, 3.0], y∈[2.5, 3.0], z=z1
# L形拐点附近裸露基板顶面Neumann BC加密
N_SUB_TOP_EDGE_X = 700  # x∈[3.0, 3.5], y∈[0, 6.0], z=z1
N_SUB_TOP_EDGE_Y = 700  # x∈[0, 3.0], y∈[3.0, 3.5], z=z1

RESAMPLE_EVERY = 500


def generate_internal_points():
    int_sub = sample_internal_scaled(N_INT_SUB, 0)
    int_bump = sample_internal_scaled(N_INT_BUMP, 1)
    int_chip = sample_internal_scaled(N_INT_CHIP, 2)
    return [int_sub, int_bump, int_chip]


def generate_fixed_bc_if_points():
    # --------------------------
    # Dirichlet
    # --------------------------
    bc_sub_bottom = sample_boundary_z_scaled(
        N_BC_SUB_BOTTOM,
        (x_sub_min, x_sub_max),
        (y_sub_min, y_sub_max),
        z0,
        sub_idx=0
    )

    bc_chip_top = sample_boundary_z_scaled(
        N_BC_CHIP_TOP,
        (x_chip_min, x_chip_max),
        (y_chip_min, y_chip_max),
        z3,
        sub_idx=2
    )

    # --------------------------
    # substrate symmetry planes
    # --------------------------
    bc_sub_x0 = sample_boundary_x_scaled(
        N_SUB_X0, x_sub_min, (y_sub_min, y_sub_max), (z0, z1), sub_idx=0
    )
    bc_sub_y0 = sample_boundary_y_scaled(
        N_SUB_Y0, y_sub_min, (x_sub_min, x_sub_max), (z0, z1), sub_idx=0
    )

    # substrate outer planes
    bc_sub_xmax = sample_boundary_x_scaled(
        N_SUB_XMAX, x_sub_max, (y_sub_min, y_sub_max), (z0, z1), sub_idx=0
    )
    bc_sub_ymax = sample_boundary_y_scaled(
        N_SUB_YMAX, y_sub_max, (x_sub_min, x_sub_max), (z0, z1), sub_idx=0
    )

    # substrate top exposed regions (L-shape split into two rectangles)
    bc_sub_top_exposed_r = sample_boundary_z_scaled(
        N_SUB_TOP_EXPOSED_R,
        (x_bump_max, x_sub_max),   # [3,6]
        (y_sub_min, y_sub_max),    # [0,6]
        z1,
        sub_idx=0
    )
    bc_sub_top_exposed_u = sample_boundary_z_scaled(
        N_SUB_TOP_EXPOSED_U,
        (x_sub_min, x_bump_max),   # [0,3]
        (y_bump_max, y_sub_max),   # [3,6]
        z1,
        sub_idx=0
    )

    # --------------------------
    # bump symmetry + outer
    # --------------------------
    bc_bump_x0 = sample_boundary_x_scaled(
        N_BUMP_X0, x_bump_min, (y_bump_min, y_bump_max), (z1, z2), sub_idx=1
    )
    bc_bump_y0 = sample_boundary_y_scaled(
        N_BUMP_Y0, y_bump_min, (x_bump_min, x_bump_max), (z1, z2), sub_idx=1
    )
    bc_bump_xmax = sample_boundary_x_scaled(
        N_BUMP_XMAX, x_bump_max, (y_bump_min, y_bump_max), (z1, z2), sub_idx=1
    )
    bc_bump_ymax = sample_boundary_y_scaled(
        N_BUMP_YMAX, y_bump_max, (x_bump_min, x_bump_max), (z1, z2), sub_idx=1
    )

    # --------------------------
    # chip symmetry + outer
    # --------------------------
    bc_chip_x0 = sample_boundary_x_scaled(
        N_CHIP_X0, x_chip_min, (y_chip_min, y_chip_max), (z2, z3), sub_idx=2
    )
    bc_chip_y0 = sample_boundary_y_scaled(
        N_CHIP_Y0, y_chip_min, (x_chip_min, x_chip_max), (z2, z3), sub_idx=2
    )
    bc_chip_xmax = sample_boundary_x_scaled(
        N_CHIP_XMAX, x_chip_max, (y_chip_min, y_chip_max), (z2, z3), sub_idx=2
    )
    bc_chip_ymax = sample_boundary_y_scaled(
        N_CHIP_YMAX, y_chip_max, (x_chip_min, x_chip_max), (z2, z3), sub_idx=2
    )

    # --------------------------
    # interfaces
    # --------------------------
    if_12_a, if_12_b = sample_interface_scaled(
        N_IF_12,
        (x_bump_min, x_bump_max),
        (y_bump_min, y_bump_max),
        z1,
        sub_idx_a=0,
        sub_idx_b=1
    )

    if_23_a, if_23_b = sample_interface_scaled(
        N_IF_23,
        (x_chip_min, x_chip_max),
        (y_chip_min, y_chip_max),
        z2,
        sub_idx_a=1,
        sub_idx_b=2
    )

    # L形拐点边缘加密：在靠近x=3(x_bump_max)的接口条带内密集采样
    if_12_edge_x_a, if_12_edge_x_b = sample_interface_scaled(
        N_IF_12_EDGE_X,
        (x_bump_max - 0.5, x_bump_max),   # x∈[2.5, 3.0]
        (y_bump_min, y_bump_max),
        z1,
        sub_idx_a=0,
        sub_idx_b=1
    )
    # 在靠近y=3(y_bump_max)的接口条带内密集采样
    if_12_edge_y_a, if_12_edge_y_b = sample_interface_scaled(
        N_IF_12_EDGE_Y,
        (x_bump_min, x_bump_max),
        (y_bump_max - 0.5, y_bump_max),   # y∈[2.5, 3.0]
        z1,
        sub_idx_a=0,
        sub_idx_b=1
    )

    # 裸露基板顶面Neumann BC：靠近L形边缘加密
    bc_sub_top_edge_x = sample_boundary_z_scaled(
        N_SUB_TOP_EDGE_X,
        (x_bump_max, x_bump_max + 0.5),   # x∈[3.0, 3.5]
        (y_sub_min, y_sub_max),
        z1,
        sub_idx=0
    )
    bc_sub_top_edge_y = sample_boundary_z_scaled(
        N_SUB_TOP_EDGE_Y,
        (x_sub_min, x_bump_max),
        (y_bump_max, y_bump_max + 0.5),   # y∈[3.0, 3.5]
        z1,
        sub_idx=0
    )

    return {
        "bc_sub_bottom": bc_sub_bottom,
        "bc_chip_top": bc_chip_top,

        "bc_sub_x0": bc_sub_x0,
        "bc_sub_y0": bc_sub_y0,
        "bc_sub_xmax": bc_sub_xmax,
        "bc_sub_ymax": bc_sub_ymax,
        "bc_sub_top_exposed_r": bc_sub_top_exposed_r,
        "bc_sub_top_exposed_u": bc_sub_top_exposed_u,

        "bc_bump_x0": bc_bump_x0,
        "bc_bump_y0": bc_bump_y0,
        "bc_bump_xmax": bc_bump_xmax,
        "bc_bump_ymax": bc_bump_ymax,

        "bc_chip_x0": bc_chip_x0,
        "bc_chip_y0": bc_chip_y0,
        "bc_chip_xmax": bc_chip_xmax,
        "bc_chip_ymax": bc_chip_ymax,

        "if_12_a": if_12_a,
        "if_12_b": if_12_b,
        "if_23_a": if_23_a,
        "if_23_b": if_23_b,

        # L形拐点边缘加密接口点
        "if_12_edge_x_a": if_12_edge_x_a,
        "if_12_edge_x_b": if_12_edge_x_b,
        "if_12_edge_y_a": if_12_edge_y_a,
        "if_12_edge_y_b": if_12_edge_y_b,

        # L形拐点附近裸露基板顶面Neumann BC加密点
        "bc_sub_top_edge_x": bc_sub_top_edge_x,
        "bc_sub_top_edge_y": bc_sub_top_edge_y,
    }


print("Generating fixed boundary/interface points ...")
fixed_pts = generate_fixed_bc_if_points()
print("Generating initial internal points ...")
internal_pts = generate_internal_points()
print("Sampling complete.")


# ==========================================================
# 6. 损失权重
# ==========================================================
W_PDE = 6.0
W_DIR = 5.0
W_NEU = 5.0
W_IF_VAL = 15.0
W_IF_FLUX = 15.0


# ==========================================================
# 7. VS-PINN 归一化因子（3D）
# ==========================================================
pde_norm_factors = []
for idx in range(3):
    s_t = 2.0 * SCALE_A / total_time
    s_x2 = (2.0 * SCALE_A / dx_vals[idx]) ** 2
    s_y2 = (2.0 * SCALE_A / dy_vals[idx]) ** 2
    s_z2 = (2.0 * SCALE_A / dz_vals[idx]) ** 2

    coeff_t = s_t
    coeff_x = alpha_values[idx] * s_x2
    coeff_y = alpha_values[idx] * s_y2
    coeff_z = alpha_values[idx] * s_z2
    c_max = max(coeff_t, coeff_x, coeff_y, coeff_z)

    pde_norm_factors.append({
        "coeff_t": coeff_t,
        "coeff_x": coeff_x,
        "coeff_y": coeff_y,
        "coeff_z": coeff_z,
        "C_max": c_max,
        "norm_t": coeff_t / c_max,
        "norm_x": coeff_x / c_max,
        "norm_y": coeff_y / c_max,
        "norm_z": coeff_z / c_max,
    })

C_flux_12 = max(k_values[0] / dz_vals[0], k_values[1] / dz_vals[1])
C_flux_23 = max(k_values[1] / dz_vals[1], k_values[2] / dz_vals[2])


# ==========================================================
# 8. 网络架构
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
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.main(x)


# ==========================================================
# 9. XPINN 三层模型（3D quarter）
# ==========================================================
class XPINN3Layer3DQuarter:
    def __init__(self):
        self.nets = [StandardMLP().to(device) for _ in range(3)]

        self.all_params = []
        for net in self.nets:
            self.all_params += list(net.parameters())

        self.optim_adam = torch.optim.Adam(self.all_params, lr=2e-3)
        self.scheduler = ReduceLROnPlateau(
            self.optim_adam, mode='min', factor=0.7, patience=600, verbose=False
        )

        self.optim_lbfgs = torch.optim.LBFGS(
            self.all_params,
            lr=1.0,
            max_iter=50,
            max_eval=60,
            tolerance_grad=1e-7,
            tolerance_change=1e-9,
            history_size=50,
            line_search_fn="strong_wolfe"
        )

        self.history = {
            "total": [],
            "pde_sub": [], "pde_bump": [], "pde_chip": [],
            "bc_sub": [], "bc_bump": [], "bc_chip": [],
            "if_12": [], "if_23": [],
        }
        self.grad_history = []
        self.best_loss = float("inf")
        self.best_model_state = None

    def _forward_hard(self, net_idx, x_s, y_s, z_s, t_s):
        xyzt = torch.cat([x_s, y_s, z_s, t_s], dim=1)
        n_raw = self.nets[net_idx](xyzt)
        g = 1.0 - torch.exp(-BETA_HC * (t_s + SCALE_A))
        theta = n_raw * g
        return theta

    def _pde_residual(self, x_s, y_s, z_s, t_s, net_idx):
        nf = pde_norm_factors[net_idx]

        theta = self._forward_hard(net_idx, x_s, y_s, z_s, t_s)

        theta_t_s = grad(theta, t_s, torch.ones_like(theta), create_graph=True)[0]
        theta_x_s = grad(theta, x_s, torch.ones_like(theta), create_graph=True)[0]
        theta_y_s = grad(theta, y_s, torch.ones_like(theta), create_graph=True)[0]
        theta_z_s = grad(theta, z_s, torch.ones_like(theta), create_graph=True)[0]

        theta_xx_s = grad(theta_x_s, x_s, torch.ones_like(theta_x_s), create_graph=True)[0]
        theta_yy_s = grad(theta_y_s, y_s, torch.ones_like(theta_y_s), create_graph=True)[0]
        theta_zz_s = grad(theta_z_s, z_s, torch.ones_like(theta_z_s), create_graph=True)[0]

        residual = (
            nf["norm_t"] * theta_t_s
            - nf["norm_x"] * theta_xx_s
            - nf["norm_y"] * theta_yy_s
            - nf["norm_z"] * theta_zz_s
        )
        return torch.mean(residual ** 2)

    def predict(self, x_phys, y_phys, z_phys, t_phys):
        out = torch.zeros_like(x_phys)

        mask0 = (
            (z_phys >= z0) & (z_phys <= z1) &
            (x_phys >= x_sub_min) & (x_phys <= x_sub_max) &
            (y_phys >= y_sub_min) & (y_phys <= y_sub_max)
        ).squeeze()

        if mask0.any():
            xp, yp, zp, tp = x_phys[mask0], y_phys[mask0], z_phys[mask0], t_phys[mask0]
            x_s = scale_x(xp, 0)
            y_s = scale_y(yp, 0)
            z_s = scale_z(zp, 0)
            t_s = scale_t(tp)
            out[mask0] = self._forward_hard(0, x_s, y_s, z_s, t_s)

        mask1 = (
            (z_phys > z1) & (z_phys <= z2) &
            (x_phys >= x_bump_min) & (x_phys <= x_bump_max) &
            (y_phys >= y_bump_min) & (y_phys <= y_bump_max)
        ).squeeze()

        if mask1.any():
            xp, yp, zp, tp = x_phys[mask1], y_phys[mask1], z_phys[mask1], t_phys[mask1]
            x_s = scale_x(xp, 1)
            y_s = scale_y(yp, 1)
            z_s = scale_z(zp, 1)
            t_s = scale_t(tp)
            out[mask1] = self._forward_hard(1, x_s, y_s, z_s, t_s)

        mask2 = (
            (z_phys > z2) & (z_phys <= z3) &
            (x_phys >= x_chip_min) & (x_phys <= x_chip_max) &
            (y_phys >= y_chip_min) & (y_phys <= y_chip_max)
        ).squeeze()

        if mask2.any():
            xp, yp, zp, tp = x_phys[mask2], y_phys[mask2], z_phys[mask2], t_phys[mask2]
            x_s = scale_x(xp, 2)
            y_s = scale_y(yp, 2)
            z_s = scale_z(zp, 2)
            t_s = scale_t(tp)
            out[mask2] = self._forward_hard(2, x_s, y_s, z_s, t_s)

        return out

    def get_total_loss(self):
        global internal_pts, fixed_pts

        comps = {}
        total = torch.tensor(0.0, device=device)

        # --------------------------------------------------
        # PDE
        # --------------------------------------------------
        l_pde_sub = self._pde_residual(*internal_pts[0], 0)
        l_pde_bump = self._pde_residual(*internal_pts[1], 1)
        l_pde_chip = self._pde_residual(*internal_pts[2], 2)

        total = total + W_PDE * (l_pde_sub + l_pde_bump + l_pde_chip)
        comps["pde_sub"] = l_pde_sub.item()
        comps["pde_bump"] = l_pde_bump.item()
        comps["pde_chip"] = l_pde_chip.item()

        # --------------------------------------------------
        # substrate BC
        # --------------------------------------------------
        bc_sub_loss = torch.tensor(0.0, device=device)

        # z=0 Dirichlet
        x_s, y_s, z_s, t_s, t_phys = fixed_pts["bc_sub_bottom"]
        theta_pred = self._forward_hard(0, x_s, y_s, z_s, t_s)
        theta_target = substrate_bottom_theta(t_phys)
        bc_sub_loss = bc_sub_loss + W_DIR * torch.mean((theta_pred - theta_target) ** 2)

        # symmetry x=0
        x_s, y_s, z_s, t_s = fixed_pts["bc_sub_x0"]
        theta_ = self._forward_hard(0, x_s, y_s, z_s, t_s)
        dtheta_dxs = grad(theta_, x_s, torch.ones_like(theta_), create_graph=True)[0]
        bc_sub_loss = bc_sub_loss + W_NEU * torch.mean(dtheta_dxs ** 2)

        # symmetry y=0
        x_s, y_s, z_s, t_s = fixed_pts["bc_sub_y0"]
        theta_ = self._forward_hard(0, x_s, y_s, z_s, t_s)
        dtheta_dys = grad(theta_, y_s, torch.ones_like(theta_), create_graph=True)[0]
        bc_sub_loss = bc_sub_loss + W_NEU * torch.mean(dtheta_dys ** 2)

        # outer x=max
        x_s, y_s, z_s, t_s = fixed_pts["bc_sub_xmax"]
        theta_ = self._forward_hard(0, x_s, y_s, z_s, t_s)
        dtheta_dxs = grad(theta_, x_s, torch.ones_like(theta_), create_graph=True)[0]
        bc_sub_loss = bc_sub_loss + W_NEU * torch.mean(dtheta_dxs ** 2)

        # outer y=max
        x_s, y_s, z_s, t_s = fixed_pts["bc_sub_ymax"]
        theta_ = self._forward_hard(0, x_s, y_s, z_s, t_s)
        dtheta_dys = grad(theta_, y_s, torch.ones_like(theta_), create_graph=True)[0]
        bc_sub_loss = bc_sub_loss + W_NEU * torch.mean(dtheta_dys ** 2)

        # top exposed right strip
        x_s, y_s, z_s, t_s, _ = fixed_pts["bc_sub_top_exposed_r"]
        theta_ = self._forward_hard(0, x_s, y_s, z_s, t_s)
        dtheta_dzs = grad(theta_, z_s, torch.ones_like(theta_), create_graph=True)[0]
        bc_sub_loss = bc_sub_loss + W_NEU * torch.mean(dtheta_dzs ** 2)

        # top exposed upper-left strip
        x_s, y_s, z_s, t_s, _ = fixed_pts["bc_sub_top_exposed_u"]
        theta_ = self._forward_hard(0, x_s, y_s, z_s, t_s)
        dtheta_dzs = grad(theta_, z_s, torch.ones_like(theta_), create_graph=True)[0]
        bc_sub_loss = bc_sub_loss + W_NEU * torch.mean(dtheta_dzs ** 2)

        # L形拐点附近裸露顶面加密Neumann BC（靠近x=3边缘）
        x_s, y_s, z_s, t_s, _ = fixed_pts["bc_sub_top_edge_x"]
        theta_ = self._forward_hard(0, x_s, y_s, z_s, t_s)
        dtheta_dzs = grad(theta_, z_s, torch.ones_like(theta_), create_graph=True)[0]
        bc_sub_loss = bc_sub_loss + W_NEU * torch.mean(dtheta_dzs ** 2)

        # L形拐点附近裸露顶面加密Neumann BC（靠近y=3边缘）
        x_s, y_s, z_s, t_s, _ = fixed_pts["bc_sub_top_edge_y"]
        theta_ = self._forward_hard(0, x_s, y_s, z_s, t_s)
        dtheta_dzs = grad(theta_, z_s, torch.ones_like(theta_), create_graph=True)[0]
        bc_sub_loss = bc_sub_loss + W_NEU * torch.mean(dtheta_dzs ** 2)

        total = total + bc_sub_loss
        comps["bc_sub"] = bc_sub_loss.item()

        # --------------------------------------------------
        # bump BC
        # --------------------------------------------------
        bc_bump_loss = torch.tensor(0.0, device=device)

        # symmetry x=0
        x_s, y_s, z_s, t_s = fixed_pts["bc_bump_x0"]
        theta_ = self._forward_hard(1, x_s, y_s, z_s, t_s)
        dtheta_dxs = grad(theta_, x_s, torch.ones_like(theta_), create_graph=True)[0]
        bc_bump_loss = bc_bump_loss + W_NEU * torch.mean(dtheta_dxs ** 2)

        # symmetry y=0
        x_s, y_s, z_s, t_s = fixed_pts["bc_bump_y0"]
        theta_ = self._forward_hard(1, x_s, y_s, z_s, t_s)
        dtheta_dys = grad(theta_, y_s, torch.ones_like(theta_), create_graph=True)[0]
        bc_bump_loss = bc_bump_loss + W_NEU * torch.mean(dtheta_dys ** 2)

        # outer x=max
        x_s, y_s, z_s, t_s = fixed_pts["bc_bump_xmax"]
        theta_ = self._forward_hard(1, x_s, y_s, z_s, t_s)
        dtheta_dxs = grad(theta_, x_s, torch.ones_like(theta_), create_graph=True)[0]
        bc_bump_loss = bc_bump_loss + W_NEU * torch.mean(dtheta_dxs ** 2)

        # outer y=max
        x_s, y_s, z_s, t_s = fixed_pts["bc_bump_ymax"]
        theta_ = self._forward_hard(1, x_s, y_s, z_s, t_s)
        dtheta_dys = grad(theta_, y_s, torch.ones_like(theta_), create_graph=True)[0]
        bc_bump_loss = bc_bump_loss + W_NEU * torch.mean(dtheta_dys ** 2)

        total = total + bc_bump_loss
        comps["bc_bump"] = bc_bump_loss.item()

        # --------------------------------------------------
        # chip BC
        # --------------------------------------------------
        bc_chip_loss = torch.tensor(0.0, device=device)

        # z=z3 Dirichlet
        x_s, y_s, z_s, t_s, t_phys = fixed_pts["bc_chip_top"]
        theta_pred = self._forward_hard(2, x_s, y_s, z_s, t_s)
        theta_target = chip_top_theta(t_phys)
        bc_chip_loss = bc_chip_loss + W_DIR * torch.mean((theta_pred - theta_target) ** 2)

        # symmetry x=0
        x_s, y_s, z_s, t_s = fixed_pts["bc_chip_x0"]
        theta_ = self._forward_hard(2, x_s, y_s, z_s, t_s)
        dtheta_dxs = grad(theta_, x_s, torch.ones_like(theta_), create_graph=True)[0]
        bc_chip_loss = bc_chip_loss + W_NEU * torch.mean(dtheta_dxs ** 2)

        # symmetry y=0
        x_s, y_s, z_s, t_s = fixed_pts["bc_chip_y0"]
        theta_ = self._forward_hard(2, x_s, y_s, z_s, t_s)
        dtheta_dys = grad(theta_, y_s, torch.ones_like(theta_), create_graph=True)[0]
        bc_chip_loss = bc_chip_loss + W_NEU * torch.mean(dtheta_dys ** 2)

        # outer x=max
        x_s, y_s, z_s, t_s = fixed_pts["bc_chip_xmax"]
        theta_ = self._forward_hard(2, x_s, y_s, z_s, t_s)
        dtheta_dxs = grad(theta_, x_s, torch.ones_like(theta_), create_graph=True)[0]
        bc_chip_loss = bc_chip_loss + W_NEU * torch.mean(dtheta_dxs ** 2)

        # outer y=max
        x_s, y_s, z_s, t_s = fixed_pts["bc_chip_ymax"]
        theta_ = self._forward_hard(2, x_s, y_s, z_s, t_s)
        dtheta_dys = grad(theta_, y_s, torch.ones_like(theta_), create_graph=True)[0]
        bc_chip_loss = bc_chip_loss + W_NEU * torch.mean(dtheta_dys ** 2)

        total = total + bc_chip_loss
        comps["bc_chip"] = bc_chip_loss.item()

        # --------------------------------------------------
        # interface 1: substrate / bump
        # --------------------------------------------------
        x_s_a, y_s_a, z_s_a, t_s_a = fixed_pts["if_12_a"]
        x_s_b, y_s_b, z_s_b, t_s_b = fixed_pts["if_12_b"]

        theta_sub = self._forward_hard(0, x_s_a, y_s_a, z_s_a, t_s_a)
        theta_bump = self._forward_hard(1, x_s_b, y_s_b, z_s_b, t_s_b)

        l_if12_val = W_IF_VAL * torch.mean((theta_sub - theta_bump) ** 2)

        dtheta_sub_dz = grad(theta_sub, z_s_a, torch.ones_like(theta_sub), create_graph=True)[0]
        dtheta_bump_dz = grad(theta_bump, z_s_b, torch.ones_like(theta_bump), create_graph=True)[0]

        flux_sub_norm = (k_values[0] / dz_vals[0] / C_flux_12) * dtheta_sub_dz
        flux_bump_norm = (k_values[1] / dz_vals[1] / C_flux_12) * dtheta_bump_dz
        l_if12_flux = W_IF_FLUX * torch.mean((flux_sub_norm - flux_bump_norm) ** 2)

        l_if12 = l_if12_val + l_if12_flux

        # L形拐点边缘加密接口约束（靠近x=3）
        x_s_a, y_s_a, z_s_a, t_s_a = fixed_pts["if_12_edge_x_a"]
        x_s_b, y_s_b, z_s_b, t_s_b = fixed_pts["if_12_edge_x_b"]
        theta_sub_ex = self._forward_hard(0, x_s_a, y_s_a, z_s_a, t_s_a)
        theta_bump_ex = self._forward_hard(1, x_s_b, y_s_b, z_s_b, t_s_b)
        l_if12_edge_x_val = W_IF_VAL * torch.mean((theta_sub_ex - theta_bump_ex) ** 2)
        dtheta_sub_ex_dz = grad(theta_sub_ex, z_s_a, torch.ones_like(theta_sub_ex), create_graph=True)[0]
        dtheta_bump_ex_dz = grad(theta_bump_ex, z_s_b, torch.ones_like(theta_bump_ex), create_graph=True)[0]
        flux_sub_ex = (k_values[0] / dz_vals[0] / C_flux_12) * dtheta_sub_ex_dz
        flux_bump_ex = (k_values[1] / dz_vals[1] / C_flux_12) * dtheta_bump_ex_dz
        l_if12_edge_x_flux = W_IF_FLUX * torch.mean((flux_sub_ex - flux_bump_ex) ** 2)
        l_if12 = l_if12 + l_if12_edge_x_val + l_if12_edge_x_flux

        # L形拐点边缘加密接口约束（靠近y=3）
        x_s_a, y_s_a, z_s_a, t_s_a = fixed_pts["if_12_edge_y_a"]
        x_s_b, y_s_b, z_s_b, t_s_b = fixed_pts["if_12_edge_y_b"]
        theta_sub_ey = self._forward_hard(0, x_s_a, y_s_a, z_s_a, t_s_a)
        theta_bump_ey = self._forward_hard(1, x_s_b, y_s_b, z_s_b, t_s_b)
        l_if12_edge_y_val = W_IF_VAL * torch.mean((theta_sub_ey - theta_bump_ey) ** 2)
        dtheta_sub_ey_dz = grad(theta_sub_ey, z_s_a, torch.ones_like(theta_sub_ey), create_graph=True)[0]
        dtheta_bump_ey_dz = grad(theta_bump_ey, z_s_b, torch.ones_like(theta_bump_ey), create_graph=True)[0]
        flux_sub_ey = (k_values[0] / dz_vals[0] / C_flux_12) * dtheta_sub_ey_dz
        flux_bump_ey = (k_values[1] / dz_vals[1] / C_flux_12) * dtheta_bump_ey_dz
        l_if12_edge_y_flux = W_IF_FLUX * torch.mean((flux_sub_ey - flux_bump_ey) ** 2)
        l_if12 = l_if12 + l_if12_edge_y_val + l_if12_edge_y_flux

        total = total + l_if12
        comps["if_12"] = l_if12.item()

        # --------------------------------------------------
        # interface 2: bump / chip
        # --------------------------------------------------
        x_s_a, y_s_a, z_s_a, t_s_a = fixed_pts["if_23_a"]
        x_s_b, y_s_b, z_s_b, t_s_b = fixed_pts["if_23_b"]

        theta_bump2 = self._forward_hard(1, x_s_a, y_s_a, z_s_a, t_s_a)
        theta_chip = self._forward_hard(2, x_s_b, y_s_b, z_s_b, t_s_b)

        l_if23_val = W_IF_VAL * torch.mean((theta_bump2 - theta_chip) ** 2)

        dtheta_bump2_dz = grad(theta_bump2, z_s_a, torch.ones_like(theta_bump2), create_graph=True)[0]
        dtheta_chip_dz = grad(theta_chip, z_s_b, torch.ones_like(theta_chip), create_graph=True)[0]

        flux_bump2_norm = (k_values[1] / dz_vals[1] / C_flux_23) * dtheta_bump2_dz
        flux_chip_norm = (k_values[2] / dz_vals[2] / C_flux_23) * dtheta_chip_dz
        l_if23_flux = W_IF_FLUX * torch.mean((flux_bump2_norm - flux_chip_norm) ** 2)

        l_if23 = l_if23_val + l_if23_flux
        total = total + l_if23
        comps["if_23"] = l_if23.item()

        comps["total"] = total.item()
        return total, comps

    def train_step_adam(self):
        self.optim_adam.zero_grad()
        loss, comps = self.get_total_loss()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.all_params, max_norm=10.0)
        self.optim_adam.step()
        self.scheduler.step(loss.item())

        for k, v in comps.items():
            self.history[k].append(v)
        self._record_grads()
        return loss.item(), comps

    def train_step_lbfgs(self):
        loss_val = [None]
        comps_val = [None]

        def closure():
            self.optim_lbfgs.zero_grad()
            loss, comps = self.get_total_loss()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.all_params, max_norm=10.0)
            loss_val[0] = loss.item()
            comps_val[0] = comps
            return loss

        self.optim_lbfgs.step(closure)

        if comps_val[0] is not None:
            for k, v in comps_val[0].items():
                self.history[k].append(v)
        self._record_grads()
        return loss_val[0], comps_val[0]

    def _record_grads(self):
        grads = []
        for net in self.nets:
            for p in net.parameters():
                if p.grad is not None:
                    grads.append(p.grad.view(-1))
        if grads:
            self.grad_history.append(torch.norm(torch.cat(grads)).item())
        else:
            self.grad_history.append(0.0)

    def save_best_model(self):
        self.best_model_state = [copy.deepcopy(n.state_dict()) for n in self.nets]

    def load_best_model(self):
        if self.best_model_state is not None:
            for i, net in enumerate(self.nets):
                net.load_state_dict(self.best_model_state[i])
            print("Best model loaded.")

    def save_to_file(self, filepath):
        torch.save([n.state_dict() for n in self.nets], filepath)
        print(f"Model saved: {filepath}")


# ==========================================================
# 10. 可视化函数
# ==========================================================
def plot_training_process(model, save_path="training_metrics_3d_quarter.png"):
    h = model.history
    epochs = range(len(h["total"]))

    fig = plt.figure(figsize=(20, 12))

    ax1 = fig.add_subplot(2, 3, 1)
    ax1.plot(epochs, h["pde_sub"], label="PDE Substrate", color='blue')
    ax1.plot(epochs, h["pde_bump"], label="PDE Bump", color='green')
    ax1.plot(epochs, h["pde_chip"], label="PDE Chip", color='red')
    ax1.set_yscale('log')
    ax1.set_title("PDE Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2 = fig.add_subplot(2, 3, 2)
    ax2.plot(epochs, h["bc_sub"], label="BC Substrate", color='blue')
    ax2.plot(epochs, h["bc_bump"], label="BC Bump", color='green')
    ax2.plot(epochs, h["bc_chip"], label="BC Chip", color='red')
    ax2.set_yscale('log')
    ax2.set_title("BC Loss by Subnet")
    ax2.set_xlabel("Epoch")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    ax3 = fig.add_subplot(2, 3, 3)
    ax3.plot(epochs, h["total"], label="Total Loss", color='black', lw=1.5)
    ax3.set_yscale('log')
    ax3.set_title("Total Loss")
    ax3.set_xlabel("Epoch")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    ax4 = fig.add_subplot(2, 3, 4)
    ax4.plot(epochs, model.grad_history, label="Global Grad Norm", color='purple', alpha=0.7)
    ax4.set_yscale('log')
    ax4.set_title("Gradient L2 Norm")
    ax4.set_xlabel("Epoch")
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    ax5 = fig.add_subplot(2, 3, 5)
    ax5.plot(epochs, h["if_12"], label="Interface 1", color='orange')
    ax5.plot(epochs, h["if_23"], label="Interface 2", color='magenta')
    ax5.set_yscale('log')
    ax5.set_title("Interface Loss")
    ax5.set_xlabel("Epoch")
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    print(f"Training history saved: {save_path}")
    plt.close()


def plot_temperature_slice_y0(model, t_val, save_path=None):
    """
    画 y=0 对称面上的 x-z 温度切片
    """
    print(f"Generating y=0 slice at t={t_val}s ...")

    nx = 240
    nz_sub, nz_bump, nz_chip = 80, 20, 40

    # substrate
    x_arr = np.linspace(x_sub_min, x_sub_max, nx)
    z_arr = np.linspace(z0, z1, nz_sub)
    Xs, Zs = np.meshgrid(x_arr, z_arr)
    Ys = np.zeros_like(Xs)

    x_phys_t = torch.tensor(Xs.flatten(), dtype=torch.float32).view(-1, 1).to(device)
    y_phys_t = torch.tensor(Ys.flatten(), dtype=torch.float32).view(-1, 1).to(device)
    z_phys_t = torch.tensor(Zs.flatten(), dtype=torch.float32).view(-1, 1).to(device)
    t_phys_t = torch.ones_like(x_phys_t) * t_val

    with torch.no_grad():
        theta_sub = model.predict(x_phys_t, y_phys_t, z_phys_t, t_phys_t).cpu().numpy()
    T_sub = (T_init + theta_sub * dT).reshape(Xs.shape)

    # bump
    x_arr = np.linspace(x_bump_min, x_bump_max, nx // 2)
    z_arr = np.linspace(z1, z2, nz_bump)
    Xb, Zb = np.meshgrid(x_arr, z_arr)
    Yb = np.zeros_like(Xb)

    x_phys_t = torch.tensor(Xb.flatten(), dtype=torch.float32).view(-1, 1).to(device)
    y_phys_t = torch.tensor(Yb.flatten(), dtype=torch.float32).view(-1, 1).to(device)
    z_phys_t = torch.tensor(Zb.flatten(), dtype=torch.float32).view(-1, 1).to(device)
    t_phys_t = torch.ones_like(x_phys_t) * t_val

    with torch.no_grad():
        theta_bump = model.predict(x_phys_t, y_phys_t, z_phys_t, t_phys_t).cpu().numpy()
    T_bump = (T_init + theta_bump * dT).reshape(Xb.shape)

    # chip
    x_arr = np.linspace(x_chip_min, x_chip_max, nx // 2)
    z_arr = np.linspace(z2, z3, nz_chip)
    Xc, Zc = np.meshgrid(x_arr, z_arr)
    Yc = np.zeros_like(Xc)

    x_phys_t = torch.tensor(Xc.flatten(), dtype=torch.float32).view(-1, 1).to(device)
    y_phys_t = torch.tensor(Yc.flatten(), dtype=torch.float32).view(-1, 1).to(device)
    z_phys_t = torch.tensor(Zc.flatten(), dtype=torch.float32).view(-1, 1).to(device)
    t_phys_t = torch.ones_like(x_phys_t) * t_val

    with torch.no_grad():
        theta_chip = model.predict(x_phys_t, y_phys_t, z_phys_t, t_phys_t).cpu().numpy()
    T_chip = (T_init + theta_chip * dT).reshape(Xc.shape)

    vmin = min(T_sub.min(), T_bump.min(), T_chip.min())
    vmax = max(T_sub.max(), T_bump.max(), T_chip.max())
    levels = np.linspace(vmin, vmax, 120)

    fig, ax = plt.subplots(figsize=(12, 5))
    cp = ax.contourf(Xs, Zs, T_sub, levels=levels, cmap='jet', extend='both')
    ax.contourf(Xb, Zb, T_bump, levels=levels, cmap='jet', extend='both')
    ax.contourf(Xc, Zc, T_chip, levels=levels, cmap='jet', extend='both')
    plt.colorbar(cp, ax=ax, label='Temperature (deg C)')

    ax.plot([x_sub_min, x_sub_max, x_sub_max, x_sub_min, x_sub_min],
            [z0, z0, z1, z1, z0], 'k-', lw=1.2)
    ax.plot([x_bump_min, x_bump_max, x_bump_max, x_bump_min, x_bump_min],
            [z1, z1, z2, z2, z1], 'k-', lw=1.2)
    ax.plot([x_chip_min, x_chip_max, x_chip_max, x_chip_min, x_chip_min],
            [z2, z2, z3, z3, z2], 'k-', lw=1.2)

    ax.set_title(f'Temperature Slice at y=0, t={t_val}s')
    ax.set_xlabel('x (mm)')
    ax.set_ylabel('z (mm)')
    ax.set_aspect('equal')

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.close()


def plot_temperature_slice_z(model, t_val, z_val, save_path=None):
    """
    画固定 z 水平面的 x-y 温度分布。
    z_val 所在层必须与对应几何匹配。
    """
    print(f"Generating horizontal slice at z={z_val} mm, t={t_val}s ...")

    if z0 <= z_val <= z1:
        x_min, x_max = x_sub_min, x_sub_max
        y_min, y_max = y_sub_min, y_sub_max
    elif z1 < z_val <= z2:
        x_min, x_max = x_bump_min, x_bump_max
        y_min, y_max = y_bump_min, y_bump_max
    elif z2 < z_val <= z3:
        x_min, x_max = x_chip_min, x_chip_max
        y_min, y_max = y_chip_min, y_chip_max
    else:
        raise ValueError("z_val is outside the geometry range.")

    nx = 180
    ny = 180

    x_arr = np.linspace(x_min, x_max, nx)
    y_arr = np.linspace(y_min, y_max, ny)
    X, Y = np.meshgrid(x_arr, y_arr)
    Z = np.ones_like(X) * z_val

    x_phys_t = torch.tensor(X.flatten(), dtype=torch.float32).view(-1, 1).to(device)
    y_phys_t = torch.tensor(Y.flatten(), dtype=torch.float32).view(-1, 1).to(device)
    z_phys_t = torch.tensor(Z.flatten(), dtype=torch.float32).view(-1, 1).to(device)
    t_phys_t = torch.ones_like(x_phys_t) * t_val

    with torch.no_grad():
        theta = model.predict(x_phys_t, y_phys_t, z_phys_t, t_phys_t).cpu().numpy()
    T = (T_init + theta * dT).reshape(X.shape)

    fig, ax = plt.subplots(figsize=(6, 5))
    cp = ax.contourf(X, Y, T, levels=100, cmap='jet')
    plt.colorbar(cp, ax=ax, label='Temperature (deg C)')
    ax.set_title(f'Temperature Slice at z={z_val:.3f} mm, t={t_val}s')
    ax.set_xlabel('x (mm)')
    ax.set_ylabel('y (mm)')
    ax.set_aspect('equal')

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.close()


# ==========================================================
# 11. 主程序
# ==========================================================
if __name__ == "__main__":
    SEED = 42
    setup_seed(SEED)

    model = XPINN3Layer3DQuarter()

    print("=" * 80)
    print("XPINN 3D Quarter 3-Layer")
    print("Model features:")
    print("  - 3 subnet architecture: substrate / bump / chip")
    print("  - 3D transient heat conduction")
    print("  - quarter model with symmetry planes x=0 and y=0")
    print("  - bottom/top Dirichlet temperature loading")
    print("  - all other faces adiabatic")
    print("  - interface temperature and flux continuity")
    print(f"  - internal collocation resampling every {RESAMPLE_EVERY} Adam epochs")
    print("=" * 80)

    epochs_adam = 16000
    epochs_lbfgs = 1200

    # ---------- Phase 1: Adam ----------
    print(f"\n--- Phase 1: Adam ({epochs_adam} epochs) ---")
    pbar = tqdm(range(epochs_adam))
    for i in pbar:
        if i > 0 and i % RESAMPLE_EVERY == 0:
            internal_pts = generate_internal_points()
            print(f"\n[Resample] Internal points regenerated at Adam epoch {i}")

        total_loss, comps = model.train_step_adam()

        if total_loss < model.best_loss:
            model.best_loss = total_loss
            model.save_best_model()

        if i % 10000 == 0 and i > 0:
            model.save_to_file(f"checkpoint_adam_{i}_3d_quarter.pt")

        if i % 10 == 0:
            pbar.set_description(
                f"[Adam] Tot:{total_loss:.3e} "
                f"PDE:{comps['pde_sub']:.1e}/{comps['pde_bump']:.1e}/{comps['pde_chip']:.1e} "
                f"IF:{comps['if_12']:.1e}/{comps['if_23']:.1e}"
            )

    # ---------- Phase 2: L-BFGS ----------
    print(f"\n--- Phase 2: L-BFGS ({epochs_lbfgs} epochs) ---")
    pbar = tqdm(range(epochs_lbfgs))
    for i in pbar:  # epochs_lbfgs=0 时直接跳过
        total_loss, comps = model.train_step_lbfgs()

        if total_loss is not None and total_loss < model.best_loss:
            model.best_loss = total_loss
            model.save_best_model()

        if i % 1000 == 0 and i > 0:
            model.save_to_file(f"checkpoint_lbfgs_{i}_3d_quarter.pt")

        if i % 5 == 0 and comps is not None:
            pbar.set_description(
                f"[LBFGS] Tot:{total_loss:.3e} "
                f"PDE:{comps['pde_sub']:.1e}/{comps['pde_bump']:.1e}/{comps['pde_chip']:.1e} "
                f"IF:{comps['if_12']:.1e}/{comps['if_23']:.1e}"
            )

    print("=" * 80)
    print("Training complete")
    print("=" * 80)

    model.load_best_model()
    model.save_to_file("best_model_xpinn_3d_quarter.pt")

    plot_training_process(model, "training_history_3d_quarter.png")

    for t_val in [1, 2, 3, 4, 5, 6, 7]:
        plot_temperature_slice_y0(model, t_val, f"temp_slice_y0_t{t_val}s_3d_quarter.png")

    # 额外给几张水平截面图
    plot_temperature_slice_z(model, 2.0, 0.30, "temp_slice_z0p30_t2s.png")   # substrate 内部
    plot_temperature_slice_z(model, 2.0, 0.64, "temp_slice_z0p64_t2s.png")   # bump 内部
    plot_temperature_slice_z(model, 2.0, 0.80, "temp_slice_z0p80_t2s.png")   # chip 内部
    print("All visualizations generated.")
