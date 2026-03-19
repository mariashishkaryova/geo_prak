import os
import glob
import re
import numpy as np
import matplotlib.pyplot as plt


BASE_FOLDER = "geo_prak"
THEORY_FOLDER = os.path.join(BASE_FOLDER, "new_theory")
EXPERIMENT_FOLDER = os.path.join(BASE_FOLDER, "experiment")
SAVE_BASE = os.path.join(BASE_FOLDER, "thVSexp")


RESULT_MAE = os.path.join(SAVE_BASE, "Results_MAE")
RESULT_RMSE = os.path.join(SAVE_BASE, "Results_RMSE")
os.makedirs(RESULT_MAE, exist_ok=True)
os.makedirs(RESULT_RMSE, exist_ok=True)


def prepare_comparison_data(t_exp, m_exp, t_th, m_th, window=20):
    step = max(1, len(m_exp) // len(m_th))
    #自动计算步长，根据实验数据的长度自动将密度浓缩到对应的区间内，然后去除以理论的长度。
    m_exp_dec = m_exp[::step]
    t_exp_dec = t_exp[::step]
    idx_exp = np.argmax(m_exp_dec)
    idx_th = np.argmax(m_th)
    # 截取窗口 (idx - 20 : idx + 20)
    # 处理边界防止 [idx-20] 变成负数或超过数组长度
    s1, e1 = max(0, idx_exp - window), min(len(m_exp_dec), idx_exp + window + 1)
    s2, e2 = max(0, idx_th - window), min(len(m_th), idx_th + window + 1)
    y_exp, y_th = m_exp_dec[s1:e1], m_th[s2:e2]
    
     # 强制长度对齐 (计算误差必须要求数组等长)
    min_len = min(len(y_exp), len(y_th))
    return y_exp[:min_len], y_th[:min_len], t_exp_dec, m_exp_dec

def normalize(arr):
    m = np.max(np.abs(arr))
    return arr / m if m != 0 else arr

def align_peak(time, moment):
    idx = np.argmax(moment)
    return time - time[idx], moment


def read_model(file):
    params = {}
    data_list = []
    with open(file, "r", encoding="utf-8") as f:
        lines = f.readlines()
    for line in lines:
        line = line.strip()
        if not line: continue
        param_match = re.match(r"^([a-zA-Z0-9_]+)\s*=\s*([0-9.eE+-]+)", line)
        if param_match:
            params[param_match.group(1)] = float(param_match.group(2))
            continue
        data_match = re.match(r"^([0-9.eE+-]+)\s+([0-9.eE+-]+)$", line)
        if data_match:
            data_list.append([float(data_match.group(1)), float(data_match.group(2))])
    data = np.array(data_list)
    t, m = align_peak(data[:, 1], normalize(data[:, 0]))
    return t, m, params

def read_experiment(file):
    data = np.loadtxt(file, skiprows=2)
    return align_peak(data[:, 0], normalize(data[:, 1]))


def process_and_plot(exp_name, theory_data, criterion='mae'):
    

    theory_data.sort(key=lambda x: x[criterion])
    top_4 = theory_data[:4]
    min_val = top_4[0][criterion]
    threshold = min_val * 1.1  # 阈值判定最小值的1.1倍
    
    plt.figure(figsize=(12, 7))

    first = top_4[0]
    t_exp_shifted = first['t_dec'] - first['t_dec'][np.argmax(first['m_dec'])]
    plt.plot(t_exp_shifted, first['m_dec'], lw=3, color='black', label='Experiment', zorder=10)


    param_keys = ['a', 'b', 'x0', 'vr', 'A0', 'Xmax', 'sigmaX', 'sigmaY', 'M0', 'Mr_max']
    collected = {k: [] for k in param_keys}
    colors = ['red', 'blue', 'green', 'orange']
    
    for i, res in enumerate(top_4):
        t_th_shifted = res['t_th'] - res['t_th'][np.argmax(res['m_th'])]
        plt.plot(t_th_shifted, res['m_th'], color=colors[i], alpha=0.7, 
                 label=f"Top {i+1} ({criterion.upper()}={res[criterion]:.2e})")
        for k in param_keys:
            if k in res['params']: collected[k].append(res['params'][k])


    stats_text = [f"Criterion: {criterion.upper()}", f"Threshold: < {threshold:.2e}", "Stats (Mean ± Std):"]
    for k in param_keys:
        if collected[k]:
            m, s = np.mean(collected[k]), np.std(collected[k])
            stats_text.append(f"{k:7s}: {m:.2e} ± {s:.2e}")

    plt.gca().text(1.02, 0.5, "\n".join(stats_text), transform=plt.gca().transAxes,
                   fontsize=9, family='monospace', verticalalignment='center',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.title(f"Best Fit ({criterion.upper()}) & Uncertainty: {exp_name}")
    plt.legend(loc='upper right', fontsize='small')
    plt.grid(True, alpha=0.3)
    plt.tight_layout(rect=[0, 0, 0.8, 1])
    
    folder = RESULT_MAE if criterion == 'mae' else RESULT_RMSE
    plt.savefig(os.path.join(folder, f"{exp_name}_{criterion.upper()}_Report.png"), dpi=300)
    plt.close()


def main():
    theory_files = glob.glob(os.path.join(THEORY_FOLDER, "data_*.txt"))
    print(f"Loading {len(theory_files)} models...")

    for exp in os.listdir(EXPERIMENT_FOLDER):
        exp_path = os.path.join(EXPERIMENT_FOLDER, exp)
        if not os.path.isdir(exp_path): continue
        txt_files = glob.glob(os.path.join(exp_path, "*.txt"))
        if not txt_files: continue

        print(f"Comparing: {exp}")
        t_exp, m_exp = read_experiment(txt_files[0])
        

        all_results = []
        for f in theory_files:
            t_th, m_th, params = read_model(f)
            y_exp_win, y_th_win, t_dec, m_dec = prepare_comparison_data(t_exp, m_exp, t_th, m_th)
            
            mae_val = np.mean(np.abs(y_exp_win - y_th_win))
            rmse_val = np.sqrt(np.mean((y_exp_win - y_th_win)**2))
            
            all_results.append({
                'file': f, 't_th': t_th, 'm_th': m_th, 'params': params,
                'mae': mae_val, 'rmse': rmse_val, 't_dec': t_dec, 'm_dec': m_dec
            })

   
        process_and_plot(exp, all_results.copy(), criterion='mae')
        process_and_plot(exp, all_results.copy(), criterion='rmse')

if __name__ == "__main__":
    main()