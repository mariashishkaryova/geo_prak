import os
import glob
import re
import numpy as np
import matplotlib.pyplot as plt


BASE_FOLDER = "geo_prak"
SAVE_PASH = os.path.join(BASE_FOLDER,"thVSexp")
THEORY_FOLDER = os.path.join(BASE_FOLDER, "new_theory")
EXPERIMENT_FOLDER = os.path.join(BASE_FOLDER, "experiment")

RESULT_FOLDER = os.path.join(SAVE_PASH, "results of MRE")

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


# found best model
def find_best_models(t_exp, m_exp, theory_files):
    results = []

    for file in theory_files:
        t_th, m_th = read_model(file)


        error = mre(m_exp, m_th, fraction=0.75)

        results.append((file, t_th, m_th, error))


    results.sort(key=lambda x: x[3])


    return results[:4]



def plot_result(exp_name, t_exp, m_exp, best_models):

    plt.figure(figsize=(8,5))

    plt.plot(
        t_exp,
        m_exp,
        linewidth=3,
        label="Experiment",
        color="black"
    )

    colors = ["red","blue","green","orange"]

    for i,(file,t,m,e) in enumerate(best_models):

        name = os.path.basename(file)

        plt.plot(
            t,
            m,
            color=colors[i],
            label=f"{name}  MRE={e:.2e}"
        )

    plt.xlabel("Time [s]")
    plt.ylabel("Normalized Moment Rate")

    plt.title(f"{exp_name} – Experiment vs Theory")

    plt.ylim(0,1)

    plt.legend()
    plt.grid()

    save_path = os.path.join(
        RESULT_FOLDER,
        f"{exp_name}_comparison.png"
    )

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

        for f,_,_,e in best_models:

            print(os.path.basename(f),"MRE =",e)

        plot_result(exp,t_exp,m_exp,best_models)


if __name__ == "__main__":

    main()