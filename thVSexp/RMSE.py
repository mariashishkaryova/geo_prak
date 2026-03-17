import os
import glob
import re
import numpy as np
import matplotlib.pyplot as plt


BASE_FOLDER = "geo_prak"
SAVE_PASH = os.path.join(BASE_FOLDER,"thVSexp")
THEORY_FOLDER = os.path.join(BASE_FOLDER, "new_theory")
EXPERIMENT_FOLDER = os.path.join(BASE_FOLDER, "experiment")

RESULT_FOLDER = os.path.join(SAVE_PASH, "results of RMSE")

os.makedirs(RESULT_FOLDER, exist_ok=True)

def prepare_comparison_data(t_exp, m_exp, t_th, m_th, window=20):

    # 算步长进行降采样,例如 3000点/200点 = 15
    step = max(1, len(m_exp) // len(m_th))
    m_exp_dec = m_exp[::step]
    t_exp_dec = t_exp[::step]

    # 找到峰值位置
    idx_exp = np.argmax(m_exp_dec)
    idx_th = np.argmax(m_th)

    # 截取窗口 (idx - 20 : idx + 20)
    # 处理边界防止 [idx-20] 变成负数或超过数组长度
    s1, e1 = max(0, idx_exp - window), min(len(m_exp_dec), idx_exp + window + 1)
    s2, e2 = max(0, idx_th - window), min(len(m_th), idx_th + window + 1)

    y_exp = m_exp_dec[s1:e1]
    y_th = m_th[s2:e2]

    # 强制长度对齐 (计算误差必须要求数组等长)
    min_len = min(len(y_exp), len(y_th))
    

    return y_exp[:min_len], y_th[:min_len], t_exp_dec, m_exp_dec

# in [0,1]
def normalize(arr):

    m = np.max(np.abs(arr))

    if m == 0:
        return arr

    return arr / m

# align peak to t = 0
def align_peak(time, moment):

    idx = np.argmax(moment)
    t_peak = time[idx]

    time_shifted = time - t_peak

    return time_shifted, moment

# read exp
def read_experiment(file):

    data = np.loadtxt(file, skiprows=2)

    time = data[:,0]
    moment = normalize(data[:,1])

    time, moment = align_peak(time, moment)

    return time, moment



# read th
def read_model(file):
    params = {}
    data_list = []
    with open(file, "r", encoding="utf-8") as f:
        lines = f.readlines()
    for line in lines:
        line = line.strip()
        if not line: continue
        # 抓取参数: key = value
        param_match = re.match(r"^([a-zA-Z0-9_]+)\s*=\s*([0-9.eE+-]+)", line)
        if param_match:
            params[param_match.group(1)] = float(param_match.group(2))
            continue
        # 抓取数据
        data_match = re.match(r"^([0-9.eE+-]+)\s+([0-9.eE+-]+)$", line)
        if data_match:
            data_list.append([float(data_match.group(1)), float(data_match.group(2))])
    data = np.array(data_list)
    t_aligned, m_aligned = align_peak(data[:, 1], normalize(data[:, 0]))
    return t_aligned, m_aligned, params



# RMSE
def rmse(a, b, fraction=0.75):

    n = min(len(a), len(b))
    cutoff = int(n * fraction)  # 0 to fraction
    a = a[:cutoff]
    b = b[:cutoff]

    return np.sqrt(np.mean((a - b)**2))


# found best model
def find_best_models(t_exp, m_exp, theory_files):
    results = []
    for file in theory_files:
        t_th, m_th, params = read_model(file)
        y_exp_win, y_th_win, t_dec, m_dec = prepare_comparison_data(t_exp, m_exp, t_th, m_th)
        
        # 计算 rmse
        rmse_val = np.mean(np.abs(y_exp_win - y_th_win))
        
        results.append({
            'file': file, 't_th': t_th, 'm_th': m_th, 'rmse': rmse_val,
            't_dec': t_dec, 'm_dec': m_dec, 'params': params
        })
    results.sort(key=lambda x: x['rmse'])
    
    # 定义阈值: 最小rmse的1.1倍
    min_rmse = results[0]['rmse']
    threshold = min_rmse * 1.1
    return results[:4], min_rmse, threshold

# --- 5. 绘图并自动计算 平均值 +/- 误差 ---
def plot_result(exp_name, top_4, min_rmse, threshold):
    plt.figure(figsize=(12, 7))
    
    # 实验数据
    first = top_4[0]
    t_exp_shifted = first['t_dec'] - first['t_dec'][np.argmax(first['m_dec'])]
    plt.plot(t_exp_shifted, first['m_dec'], lw=3, color='black', label='Experiment', zorder=10)

    # 统计参数
    param_keys = ['a', 'b', 'x0', 'vr', 'A0', 'Xmax', 'sigmaX', 'sigmaY', 'M0', 'Mr_max']
    collected = {k: [] for k in param_keys}
    
    colors = ['red', 'blue', 'green', 'orange']
    for i, res in enumerate(top_4):
        t_th_shifted = res['t_th'] - res['t_th'][np.argmax(res['m_th'])]
        plt.plot(t_th_shifted, res['m_th'], color=colors[i], alpha=0.7, 
                 label=f"Top {i+1} (rmse={res['rmse']:.2e})")
        for k in param_keys:
            if k in res['params']: collected[k].append(res['params'][k])

    # 生成统计文本: 平均值 +/- 绝对误差(标准差)
    stats_text = [f"Criteria: rmse < {threshold:.2e}", "Stats (Mean ± Std):"]
    for k in param_keys:
        if collected[k]:
            m, s = np.mean(collected[k]), np.std(collected[k])
            stats_text.append(f"{k:7s}: {m:.2e} ± {s:.2e}")

    plt.gca().text(1.02, 0.5, "\n".join(stats_text), transform=plt.gca().transAxes,
                   fontsize=9, family='monospace', verticalalignment='center',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.title(f"rmse Best Fit & Parameters: {exp_name}")
    plt.legend(loc='upper right', fontsize='small')
    plt.grid(True, alpha=0.3)
    plt.tight_layout(rect=[0, 0, 0.8, 1]) # 给右侧留空
    
    save_path = os.path.join(RESULT_FOLDER, f"{exp_name}_rmse_Stats.png")
    plt.savefig(save_path, dpi=300)
    plt.close()

    

def main():

    theory_files = glob.glob(
        os.path.join(THEORY_FOLDER,"data_*.txt")
    )

    print("Theory models:",len(theory_files))

    experiment_dirs = os.listdir(EXPERIMENT_FOLDER)

    for exp in experiment_dirs:

        exp_path = os.path.join(EXPERIMENT_FOLDER,exp)

        if not os.path.isdir(exp_path):
            continue

        txt_files = glob.glob(os.path.join(exp_path,"*.txt"))

        if len(txt_files) == 0:
            continue

        exp_file = txt_files[0]

        print("\nProcessing:",exp)

        t_exp,m_exp = read_experiment(exp_file)

        best_models = find_best_models(t_exp,m_exp,theory_files)

        for f,_,_,e,_,_ in best_models:

            print(os.path.basename(f),"rmse =",e)

        plot_result(exp, best_models)


if __name__ == "__main__":

    main()
