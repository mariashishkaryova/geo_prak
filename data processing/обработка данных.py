import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

path_th = r'D:\pythonProject\Geo_prak\theory'
path_exp = r'D:\pythonProject\Geo_prak\experiment'

for i in range(1, 11):
    exp_data = pd.read_csv(f'D:\pythonProject\Geo_prak\experiment\moment_rate_{i}.txt', skiprows=2, sep='\s+', header=None)
    t_exp = exp_data[0].values
    M0_exp = exp_data[1].values

    max_index = np.argmax(M0_exp)
    t_max = t_exp[max_index]
    t_exp_shifted = t_exp - t_max

    t_exp_downsampled = np.round(t_exp_shifted[::10], 1)
    M0_exp_downsampled = M0_exp[::10]
    #print(i, t_exp_downsampled)

    for j in range(0, 400):
        th_data = pd.read_csv(fr'D:\pythonProject\Geo_prak\theory\data_{j}.txt', skiprows=12, sep='\s+', header=None)
        t_th = np.round(th_data[1].values, 1)
        M0_th = th_data[0].values
        m_index = np.argmax(M0_th)
        t_max_th = t_th[m_index]
        t_th_shifted = t_th - t_max_th
        #print(t_th_shifted)

        #print(len(t_th_shifted), ',', len(t_exp_downsampled))

        #нужно считать параметры с первых строк и зафиксировать их
        with open(fr'D:\pythonProject\Geo_prak\theory\data_{j}.txt', 'r', encoding='utf-8') as f:
            lines = f.readlines()[:12]  # читаем только первые 12 строк

        params = {}
        for line in lines:
            # Разделяем строку по знаку "="
            if '=' in line:
                key, value = line.split('=', 1)
                key = key.strip()

                # Извлекаем числовое значение (убираем единицы измерения и пробелы)
                value_str = value.strip()
                # Оставляем только число (до первого пробела)
                num_str = value_str.split()[0]

                # Преобразуем в float
                try:
                    params[key] = float(num_str)
                except ValueError:
                    # Если не получается преобразовать, возможно это научная нотация
                    # Заменяем * на e для корректного преобразования
                    num_str = num_str.replace('*', 'e')
                    params[key] = float(num_str)
        a_th = params.get('a', 0)
        b_th = params.get('b', 0)
        x0_th = params.get('x0', 0)
        vr_th = params.get('vr', 0)
        A0_th = params.get('A0', 0)
        Xmax_th = params.get('Xmax', 0)
        sigmaX_th = params.get('sigmaX', 0)
        sigmaY_th = params.get('sigmaY', 0)
        M0_th = params.get('M0', 0)
        Mr_max_th = params.get('Mr_max', 0)
        t = []
        N = round(0.7 * len(t_th_shifted))
        if t_exp_downsampled[0] > t_th_shifted[0]:
            for s in range(N):
                t.append(t_exp_downsampled[s])
            M0_exp_cut = M0_exp_downsampled[:N]
            # Находим индекс первого совпадающего значения
            k = np.argmin(np.abs(t_th_shifted - t_exp_downsampled[0]))
            M0_th_cut = M0_th[k:N + k]
        else:
            for s in range(N):
                t.append(t_th_shifted[s])
            M0_th_cut = M0_th[:N]
            k = np.argmin(np.abs(t_exp_downsampled - t_th_shifted[0]))
            M0_exp_cut = M0_exp_downsampled[k:N+k]


        #plt.figure()
        #plt.scatter(t, M0_exp_cut, label = 'experiment', s = 20, color = 'red')
        #plt.scatter(t, M0_th_cut, label='theory', s = 20, color = 'black')
        #plt.scatter(t_exp_downsampled, M0_exp_downsampled, label = 'experiment full', s = 4)
        #plt.scatter(t_th_shifted, M0_th, label='theory full', s = 4)
        #plt.title(f'i = {i}, j = {j}')
        #plt.legend()
        #plt.show()

        #for k in range(len(t_th_shifted)):
            #for m in range(len(t_exp_downsampled)):
                #if t_exp_downsampled[m] == t_th_shifted[k]:
                    #print(k, m, t_th_shifted[k])

