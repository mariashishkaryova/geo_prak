import os
import glob
import re
import numpy as np
import matplotlib.pyplot as plt


BASE_FOLDER = "geo_prak"
SAVE_PASH = os.path.join(BASE_FOLDER,"thVSexp")
THEORY_FOLDER = os.path.join(BASE_FOLDER, "theory")
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

    with open(file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    data = []

    for line in lines:

        line = line.strip() # begin from numbers

        match = re.match(r"([0-9.eE+-]+)\s+([0-9.eE+-]+)", line)

        if match:

            moment = float(match.group(1))
            time = float(match.group(2))

            data.append([moment, time])

    data = np.array(data)

    moment = normalize(data[:,0])
    time = data[:,1]

    time, moment = align_peak(time, moment)

    return time, moment



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
        t_th, m_th = read_model(file)

   
        y_exp_win, y_th_win, t_dec, m_dec = prepare_comparison_data(t_exp, m_exp, t_th, m_th)


        error = rmse(y_exp_win, y_th_win) 

  
        results.append((file, t_th, m_th, error, t_dec, m_dec))

    results.sort(key=lambda x: x[3])
    
    return results[:4]



def plot_result(exp_name, best_models): 
    plt.figure(figsize=(8,5))

    _, _, _, _, t_exp_dec, m_exp_dec = best_models[0]
    t_exp_shifted = t_exp_dec - t_exp_dec[np.argmax(m_exp_dec)]

    plt.plot(t_exp_shifted, m_exp_dec, linewidth=3, label="Experiment", color="black")

    colors = ["red","blue","green","orange"]
    for i, (file, t_th, m_th, e, _, _) in enumerate(best_models):
        name = os.path.basename(file)
        t_th_shifted = t_th - t_th[np.argmax(m_th)]
  
        plt.plot(t_th_shifted, m_th, color=colors[i], label=f"{name} corr={e:.2e}")

    plt.xlabel("Time relative to peak [s]")
    plt.ylabel("Normalized Moment Rate")
    plt.title(f"Cross-Correlation Best Matches: {exp_name}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1.1)

    save_path = os.path.join(RESULT_FOLDER, f"{exp_name}_comparison.png")
    plt.savefig(save_path, dpi=300)
    print("Saved:", save_path)
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
