import numpy as np
import random
import copy
import matplotlib.pyplot as plt

class SA:
    def __init__(self, distance_matrix, iteration, base_temp, t_min, decay):
        self.distance_matrix = distance_matrix  # 各城市座標
        self.iteration = iteration  # 迭代次數
        self.base_temp = base_temp  # 起始溫度
        self.t_min = t_min  # 最低溫度
        self.decay = decay  # 溫度衰減係數
        self.iteration_count = 0

    def _initial_solution(self):
        # Greedy algorithm
        city_number = len(self.distance_matrix)  # 所有城市數量
        start = np.random.randint(city_number)  # 起始點
        path = [start]  # 路徑
        total_dis = 0  # 總距離
        now = start  # 目前位置

        unvisited_list = list(self.distance_matrix)  # 未拜訪節點
        unvisited_list.remove(self.distance_matrix[now])  # 移除起點

        for i in range(city_number - 1):
            min_dis = np.inf
            next_idx = -1

            for unvisited in unvisited_list:
                x1, y1 = self.distance_matrix[now]
                x2, y2 = unvisited
                dis = self.euler_distance(x1, y1, x2, y2)

                if dis < min_dis:
                    next_idx = list(self.distance_matrix).index(unvisited)
                    min_dis = dis

            now = next_idx
            total_dis += min_dis
            unvisited_list.remove(self.distance_matrix[now])
            path.append(now)

        return path

    def Search(self):
        path = [i for i in range(len(self.distance_matrix))]  # 初始解

        start_point = path[0]  # 起點
        end_point = path[len(path) - 1]  # 終點
        ori_dis = self.calculate_distance(path)  # 對初始解計算總距離
        best_dis = np.inf
        best_dis_list = []
        best_path = []

        # 開始迭代
        while self.base_temp >= self.t_min and self.iteration_count < self.iteration:
            # 從目前路徑隨機選取兩個點出來，且不為起點和終點
            p_1, p_2 = random.sample(path, 2)
            if p_1 == start_point or p_1 == end_point or p_2 == start_point or p_2 == end_point:
                continue

            # 調換兩點
            copy_path = copy.deepcopy(path)
            copy_path[p_1], copy_path[p_2] = copy_path[p_2], copy_path[p_1]

            # 計算調換後的總距離
            new_dis = self.calculate_distance(copy_path)
            print('Iteration :', self.iteration_count, 'distance:', new_dis, 'best_dis:', best_dis)

            # 比較是否有比原本的好，有的話直接替代，沒有的話依照概率判斷
            diff = new_dis - ori_dis
            if diff < 0:
                path = copy.deepcopy(copy_path)
                ori_dis = new_dis
                if new_dis < best_dis:
                    best_dis = new_dis
                    best_path = copy.deepcopy(copy_path)
            else:
                if np.random.random() < np.exp(-diff / self.base_temp):
                    path = copy.deepcopy(copy_path)
                    ori_dis = new_dis

            # 降低溫度
            self.base_temp *= self.decay

            # 增加迭代次數
            self.iteration_count += 1

            best_dis_list.append(best_dis)

        # 將起始點再加入路徑裡
        best_path.append(start_point)
        return best_path, best_dis_list

    # 計算路徑所經過的距離
    def calculate_distance(self, path):
        total_dis = 0

        for i in range(len(path) - 1):
            x1, y1 = self.distance_matrix[path[i]]
            x2, y2 = self.distance_matrix[path[i + 1]]
            total_dis += self.euler_distance(x1, y1, x2, y2)

        return total_dis

    # 歐式距離
    def euler_distance(self, x1, y1, x2, y2):
        return round(((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5)
