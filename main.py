from SA import SA
from Tabu import Tabu
from GreedyEpsilon import GreedyEpsilon
from MCTS import MCTS
import numpy as np
import matplotlib.pyplot as plt

# ==============================================================================
# --讀取berlin52城市座標檔案-------------------------------------------------------
# ==============================================================================
distance_matrix = []
with open('berlin52.txt', 'r') as file:
    datas = file.readlines()

for data in datas:
    coordinate = data.rstrip().split(' ')
    distance_matrix.append([float(coordinate[1]), float(coordinate[2])])

"""
 distance_matrix = [
    [565.0, 575.0]
    [25.0, 185.0]
    [345.0, 750.0]
    [945.0, 685.0]
    [845.0, 655.0]
    [880.0, 660.0]
    [25.0, 230.0]
    [525.0, 1000.0]
    [580.0, 1175.0]
    [650.0, 1130.0]
    [1605.0, 620.0]
    [1220.0, 580.0]
    [1465.0, 200.0]
    [1530.0, 5.0]
    [845.0, 680.0]
    [725.0, 370.0]
    [145.0, 665.0]
    [415.0, 635.0]
    [510.0, 875.0]
    [560.0, 365.0]
    [300.0, 465.0]
    [520.0, 585.0]
    [480.0, 415.0]
    [835.0, 625.0]
    [975.0, 580.0]
    [1215.0, 245.0]
    [1320.0, 315.0]
    [1250.0, 400.0]
    [660.0, 180.0]
    [410.0, 250.0]
    [420.0, 555.0]
    [575.0, 665.0]
    [1150.0, 1160.0]
    [700.0, 580.0]
    [685.0, 595.0]
    [685.0, 610.0]
    [770.0, 610.0]
    [795.0, 645.0]
    [720.0, 635.0]
    [760.0, 650.0]
    [475.0, 960.0]
    [95.0, 260.0]
    [875.0, 920.0]
    [700.0, 500.0]
    [555.0, 815.0]
    [830.0, 485.0]
    [1170.0, 65.0]
    [830.0, 610.0]
    [605.0, 625.0]
    [595.0, 360.0]
    [1340.0, 725.0]
    [1740.0, 245.0]
 ]
"""


class main:
    def __init__(self):
        # ============SA============
        self.search_algorithm = SA(
            distance_matrix=distance_matrix,
            iteration=100000,  # 迭代次數
            base_temp=100,  # 起始溫度
            t_min=1e-3,  # 最低溫度
            decay=0.999999  # 每次溫度衰減因子
        )

        # ============Tabu============
        # self.search_algorithm = Tabu(
        #     distance_matrix=distance_matrix,
        #     iteration=100000,  # 迭代次數
        #     tabu_list_length=2,  # 禁忌列表長度
        #     neighbors_num=100  # 每次找尋多少個鄰居
        # )

        # ============Greedy_Epsilon============
        # self.search_algorithm = GreedyEpsilon(
        #     distance_matrix=distance_matrix,
        #     iteration=100000,  # 迭代次數
        #     epsilon=0.8,  # 隨機選擇概率
        #     epsilon_min=0.01,  # 最低概率值
        #     decay=0.99999  # 每次概率衰減因子
        # )

        # ============MCTS============
        # self.search_algorithm = MCTS(
        #     distance_matrix=distance_matrix,
        #     iteration=100000,  # 迭代次數
        #     c=100  # 探索權重
        # )

    # 開始搜索
    def Search(self):
        path, best_dis_list = self.search_algorithm.Search()
        self.draw(path, best_dis_list)

    # 繪圖
    def draw(self, path, best_dis_list):
        plt.plot(best_dis_list)
        plt.xlabel("Iteration")
        plt.ylabel("Distance")
        plt.show()
        plt.scatter(x=np.array(distance_matrix)[:, 0], y=np.array(distance_matrix)[:, 1])
        plt.plot(np.array(distance_matrix)[path, 0], np.array(distance_matrix)[path, 1])
        plt.show()

if __name__ == "__main__":
    main().Search()