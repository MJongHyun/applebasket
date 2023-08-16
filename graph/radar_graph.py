import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc

rc('font', family='AppleGothic')
plt.rcParams['axes.unicode_minus'] = False

def total_graph(x, y, cn, data, column, des, marker = [0,2,4,6,8,10]): # 입력데이터: 가로,세로, 그래프 수, 데이터, 그래프에 나타낼 컬럼,

    labels = column
    markers = marker

    pre = data[data['predict'] == 0]
    graph = pre[column].describe().loc[des]

    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False)
    stats = np.concatenate((graph, [graph[0]]))
    angles = np.concatenate((angles, [angles[0]]))

    fig = plt.figure(figsize=(20, 20))
    ax = fig.add_subplot(x, y, 1, polar=True)
    ax.plot(angles, stats, 'o-', linewidth=2)
    ax.fill(angles, stats, alpha=0.2)
    ax.set_thetagrids(angles * 180 / np.pi, labels)
    plt.yticks(markers)
    ax.grid(True)

    for i in range(1, cn):
        pre = data[data['predict'] == i]
        graph = pre[column].describe().loc[des]

        ax = fig.add_subplot(x, y, i + 1, polar=True)
        ax.plot(angles, np.concatenate((graph, [graph[0]])), 'o-', linewidth=2)
        ax.fill(angles, np.concatenate((graph, [graph[0]])), alpha=0.2)
        ax.set_thetagrids(angles * 180 / np.pi, labels)
        plt.yticks(markers)
        ax.grid(True)


def ind_graph(data, column, entry=False, marker = [0,2,4,6,8,10]):

    if not entry:
        entry = data.predict.unique()
    else:
        entry = entry

    labels = column
    markers = marker

    ax = plt.subplot(1, 1, 1, polar=True)

    for i in entry:

        pre = data[data['predict'] == i]
        graph = pre[column].describe().loc['mean']

        angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False)
        stats = np.concatenate((graph, [graph[0]]))
        angles = np.concatenate((angles, [angles[0]]))
        ax.plot(angles, stats, 'o-', linewidth=2, label=i)
        ax.fill(angles, stats, alpha=0.2)
        ax.set_thetagrids(angles * 180 / np.pi, labels)
        plt.yticks(markers)
        ax.grid(True)

        plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
