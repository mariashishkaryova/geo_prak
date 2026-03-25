import os
import glob
import re
import numpy as np
import matplotlib.pyplot as plt

BASE_FOLDER = "geo_prak"
SAVE_PASH = os.path.join(BASE_FOLDER,"lastresult")
THEORY_FOLDER = os.path.join(BASE_FOLDER, "new_theory")
EXPERIMENT_FOLDER = os.path.join(BASE_FOLDER, "experiment")
RESULT_FOLDER = os.path.join(SAVE_PASH, "results_mre")

os.makedirs(RESULT_FOLDER, exist_ok=True)

def prepare_comparison_data_adaptive(t_exp, m_exp, t_th, m_th, threshold=0.1):

    # 1. 寻找峰值索引
    idx_exp_peak = np.argmax(np.abs(m_exp))
    idx_th_peak = np.argmax(np.abs(m_th))

    # 2. 自动确定时间窗口 (以实验数据为准)
    # 向前寻找信号起始点（直到振幅低于阈值或到达开头）
    start_idx = idx_exp_peak
    while start_idx > 0 and np.abs(m_exp[start_idx]) > threshold * np.abs(m_exp[idx_exp_peak]):
        start_idx -= 1
        
    # 向后寻找信号结束点（直到信号衰减到阈值以下）
    end_idx = idx_exp_peak
    while end_idx < len(m_exp) - 1 and np.abs(m_exp[end_idx]) > threshold * np.abs(m_exp[idx_exp_peak]):
        end_idx += 1

    # 适当放宽窗口边缘（比如多取几个点以包含完整形态）
    buffer = 5
    start_idx = max(0, start_idx - buffer)
    end_idx = min(len(m_exp), end_idx + buffer)

    # 3. 提取实验数据的有效窗口
    y_exp_win = m_exp[start_idx:end_idx]
    t_exp_win = t_exp[start_idx:end_idx]
    
    # 4. 对齐时间轴 (将峰值设为 0)
    t_exp_rel = t_exp_win - t_exp[idx_exp_peak]
    t_th_rel = t_th - t_th[idx_th_peak]

    # 5. 【核心改进】插值法 (Interpolation)
    # 将理论模型 m_th 插值到实验数据的时间网格 t_exp_rel 上
    # 这样理论模型就拥有了和实验数据完全一样的离散点坐标
    y_th_interp = np.interp(t_exp_rel, t_th_rel, m_th, left=0, right=0)

    return y_exp_win, y_th_interp, t_exp_rel


def normalize(arr):
    m = np.max(np.abs(arr))
    return arr / m if m != 0 else arr


def align_peak(time, moment):
    idx = np.argmax(moment)
    return time - time[idx], moment



def read_experiment(file):
    data = np.loadtxt(file, skiprows=2)
    return align_peak(data[:,0], normalize(data[:,1]))


def read_model(file):
    with open(file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    data = []

    for line in lines:
        line = line.strip()
        match = re.match(r"([0-9.eE+-]+)\s+([0-9.eE+-]+)", line)

        if match:
            data.append([float(match.group(1)), float(match.group(2))])

    data = np.array(data)

    return align_peak(data[:,1], normalize(data[:,0]))




def read_parameters(file):

    params = {}

    with open(file, "r", encoding="utf-8") as f:

        for line in f:

            if "=" in line:

                parts = line.split("=")

                key = parts[0].strip()

                try:
                    value = float(parts[1].split()[0])
                    params[key] = value
                except:
                    continue

    return params



#MRE
def mre(obs, pred, fraction=0.75, eps=1e-8):

    obs = np.array(obs)
    pred = np.array(pred)

    n = min(len(obs), len(pred))
    cutoff = int(n * fraction)

    obs = obs[:cutoff]
    pred = pred[:cutoff]

    
    obs_safe = np.where(np.abs(obs) < eps, eps, obs)# dont \0

    return np.mean(np.abs((pred - obs) / obs_safe))




def get_all_models(t_exp, m_exp, theory_files):
    all_results = []
    for file in theory_files:
        t_th, m_th = read_model(file)
        
        # 【修改点1】调用新名字，并只接收3个返回值
        # y_exp_win: 裁剪后的实验数据
        # y_th_win: 插值后的理论数据
        # t_rel: 对齐后的时间轴
        y_exp_win, y_th_win, t_rel = prepare_comparison_data_adaptive(t_exp, m_exp, t_th, m_th)
        
        # 【修改点2】SMAPE 计算
        error = mre(y_exp_win, y_th_win)
        
        params = read_parameters(file)
        all_results.append({
            'file': file,
            't_th': t_th,
            'm_th': m_th,
            'mre': error,
            # 这里为了兼容你后面的绘图函数，我们保留原始的实验数据展示
            't_exp_dec': t_exp,  
            'm_exp_dec': m_exp,
            'params': params
        })
    
    all_results.sort(key=lambda x: x['mre'], reverse=False)
    return all_results



#前百分之十的选择，主要是下面这个零点一的参数，修改这个参数就可以改变选择
def select_top_percent(all_models, percent=0.1):

    N = len(all_models)
    top_n = max(1, int(N * percent))

    return all_models[:top_n]



def compute_statistics(best_models):

    all_params = {}

    for model in best_models:

        params = read_parameters(model['file'])

        for k, v in params.items():

            if k not in all_params:
                all_params[k] = []

            all_params[k].append(v)

    stats = {}

    for k, values in all_params.items():

        values = np.array(values)

        stats[k] = (np.mean(values), np.std(values))

    return stats


def plot_all_results(exp_name, all_models, stats):
    import matplotlib.gridspec as gridspec
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    import matplotlib.pyplot as plt
    import numpy as np
    import os

    fig = plt.figure(figsize=(20, 10))
    # 布局：左侧主图，右侧散点图矩阵
    gs_left = fig.add_gridspec(1, 2, width_ratios=[3, 2], left=0.05, right=0.9)
    ax_main = fig.add_subplot(gs_left[0])

    # -------------------- 左侧主图：保留所有模型 --------------------
    best = all_models[0]
    # 遍历所有模型（300条灰线全画），展示整体搜索空间
    for m in all_models:
        ax_main.plot(m['t_th'], m['m_th'], color='gray', alpha=0.1, linewidth=0.4)

    # 绘制最佳模型（红线）
    label_val = f"SMAPE={best['mre']:.2f}" if 'mre' in best else f"mre={best['mre']:.2e}"
    ax_main.plot(best['t_th'], best['m_th'], color='red', linewidth=2.5, label=f"Best ({label_val})")
    
    # 绘制实验数据（黑线）
    t_exp_shifted, m_exp_norm = align_peak(best['t_exp_dec'], best['m_exp_dec'])
    ax_main.plot(t_exp_shifted, m_exp_norm, color='black', linewidth=3, label="Эксперимент")
    active_mask = np.abs(m_exp_norm) > 0.1
    
    if np.any(active_mask):
        # 获取活跃区的时间范围
        t_active = t_exp_shifted[active_mask]
        t_start, t_end = t_active.min(), t_active.max()
        
        # 2. 设置左右边距（Padding），各留 5 秒或 10% 的宽度
        padding = 5 
        ax_main.set_xlim(t_start - padding, t_end + padding)
    else:
        # 如果没抓到（比如全是噪声），给个保底默认值
        ax_main.set_xlim(-15, 35)

    # 设置 Y 轴，稍微留一点顶部空间，不要贴顶
    ax_main.set_ylim(-0.05, 1.1)
    ax_main.set_xlabel("Время относительно пика [с]")
    ax_main.set_ylabel("Нормированная интенсивность")
    ax_main.set_title(f"{exp_name} - mre Сравнение моделей")
    ax_main.grid(True, alpha=0.3)
    ax_main.set_xlim(-15, 35)
    ax_main.set_ylim(0, 1.1)
    ax_main.legend(loc='upper left', frameon=True)

    # -------------------- 左上角直方图 --------------------
    ax_hist = inset_axes(ax_main, width="30%", height="25%", loc='upper right', borderpad=2)
    all_errors = [m['mre'] for m in all_models] if 'mre' in best else [m['mre'] for m in all_models]
    ax_hist.hist(all_errors, bins=20, color='skyblue', edgecolor='white')
    ax_hist.axvline(all_errors[0], color='red', linestyle='--')
    
    # 绿线标注前 10% 阈值
    if 'mre' in best:
        threshold = np.percentile(all_errors, 10)
    else:
        threshold = np.percentile(all_errors, 90)
    ax_hist.axvline(threshold, color='green', linestyle='--')
    ax_hist.set_title("Распределение", fontsize=8)

    # -------------------- 右侧散点图矩阵：仅限前 10% --------------------
    param_keys = list(stats.keys())
    n_params = len(param_keys)
    n_col = min(4, n_params)
    n_row = (n_params + n_col - 1) // n_col
    gs_right = fig.add_gridspec(n_row, n_col, left=0.55, right=0.98, wspace=0.35, hspace=0.5)

    # 定义前 10% 的模型集合
    top_n = max(1, int(len(all_models) * 0.1))
    top_models = all_models[:top_n]

    for i, key in enumerate(param_keys):
        row, col = divmod(i, n_col)
        ax = fig.add_subplot(gs_right[row, col])
        
        # 只提取前 10% 的参数值
        vals = [m['params'][key] for m in top_models if key in m['params']]
        
        if vals:
            mean_v, std_v = np.mean(vals), np.std(vals)
            # 绘制前 10% 的散点
            ax.scatter(np.random.normal(1, 0.04, len(vals)), vals, alpha=0.5, s=20, color='royalblue')
            # 绘制误差棒
            ax.errorbar(1, mean_v, yerr=std_v, fmt='ro', capsize=6, lw=2, markersize=5, color='red')
            
            # 缩放 Y 轴只显示这 10% 的范围，方便观察收敛
            v_min, v_max = min(vals), max(vals)
            if v_min != v_max:
                margin = (v_max - v_min) * 0.2
                ax.set_ylim(v_min - margin, v_max + margin)
            
            ax.set_xticks([])
            ax.text(1, ax.get_ylim()[0], f"μ={mean_v:.2e}\nσ={std_v:.2e}",
                    ha='center', va='top', fontsize=9, color='darkred')
            ax.set_title(f"{key}", fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.2)

    save_path = os.path.join(RESULT_FOLDER, f"{exp_name}.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


    
    
def main():

    theory_files = glob.glob(os.path.join(THEORY_FOLDER, "data_*.txt"))

    print(f"Total Theory models: {len(theory_files)}")

    experiment_dirs = [
        d for d in os.listdir(EXPERIMENT_FOLDER)
        if os.path.isdir(os.path.join(EXPERIMENT_FOLDER, d))
    ]

    for exp in experiment_dirs:

        exp_path = os.path.join(EXPERIMENT_FOLDER, exp)

        txt_files = glob.glob(os.path.join(exp_path, "*.txt"))

        if not txt_files:
            continue

        print(f"\nProcessing: {exp}")

        t_exp, m_exp = read_experiment(txt_files[0])

        all_models = get_all_models(t_exp, m_exp, theory_files)

    # 选前10%
        best_models = select_top_percent(all_models, 0.1)

    # 计算统计
        stats = compute_statistics(best_models)

    # 传入 stats
        plot_all_results(exp, all_models, stats)

        best_models = select_top_percent(all_models, 0.1)

        print(f"Top 10% models: {len(best_models)}")

        stats = compute_statistics(best_models)

        print("Parameter estimation (mean ± std):")

        for k, (mean, std) in stats.items():
            print(f"{k}: {mean:.2f} ± {std:.2f}")


if __name__ == "__main__":
    main()