from cluster_simulation import Cluster
from mesa import ClusterMESA

class ClusterISO(Cluster):
    def __init__(self, cluster, cluster_data, lamost_data):
        super().__init__(cluster, cluster_data, lamost_data)
        self.iso = 1

    def connect_iso(x, y):
        # 将点按照 x 坐标进行排序
        sorted_indices = sorted(range(len(x)), key=lambda i: x[i])
        sorted_x = [x[i] for i in sorted_indices]
        sorted_y = [y[i] for i in sorted_indices]

        # 绘制散点图
        plt.scatter(sorted_x, sorted_y)

        # 连接点
        for i in range(len(sorted_x) - 1):
            plt.plot([sorted_x[i], sorted_x[i+1]], [sorted_y[i], sorted_y[i+1]], color='blue')

        # 显示连接的图形
        plt.show()

    def extract_iso(filename, omega):
        bp, rp = [], []
        with open(filename, 'r') as file:
            lines = file.readlines()
            for i in range(len(lines) - 1):
                if '{} '.format(omega) in lines[i]:
                    #print(lines[i].strip())
                    bp.append(lines[i+1].strip().split( )[2])
                    rp.append(lines[i+1].strip().split( )[3])
    
        bp, rp = np.array(bp, dtype=float), np.array(rp, dtype=float)
        return bp, bp-rp

