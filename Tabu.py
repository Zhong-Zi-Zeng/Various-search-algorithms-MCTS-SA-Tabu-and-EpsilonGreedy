import numpy as np
import random
import copy
from collections import deque

class Tabu:
    def __init__(self, distance_matrix, iteration, tabu_list_length, neighbors_num):
        self.distance_matrix = distance_matrix  # 各城市座標
        self.iteration = iteration  # 迭代次數
        self.neighbors_num = neighbors_num # 鄰近解數量
        self.tabu_list = deque(maxlen=tabu_list_length)  # 禁忌列表

    def Search(self):
        best_path = [i for i in range(len(self.distance_matrix))]  # 最優解路徑
        best_dis = self.calculate_distance(best_path)  # 最優解距離
        self.tabu_list.append(best_path) # 將目前最優解加入到禁忌列表中
        best_dis_list = []

        # 開使迭代
        for i in range(self.iteration):
            # 利用目前最優解取得鄰近解
            neighbors_path = self.get_neighbors(best_path)

            # 挑第一個鄰近解當作本次候選最優解
            best_neighbor = neighbors_path[0]

            # 找出所有不在禁忌列表中的最優解
            for neighbor in neighbors_path:
                neighbor_total_distance = self.calculate_distance(neighbor)
                if neighbor not in self.tabu_list and neighbor_total_distance < best_dis:
                    best_neighbor = neighbor

            # 如果best_neighbor 比歷史最優解還要好則更新
            best_neighbor_total_distance = self.calculate_distance(best_neighbor)
            if best_neighbor_total_distance < best_dis:
                best_dis = best_neighbor_total_distance
                best_path = best_neighbor

            print('Iteration :',i, 'distance:', best_neighbor_total_distance, 'best_dis:', best_dis)


            # 將此解加入到禁忌列表中
            self.tabu_list.append(best_neighbor)

            best_dis_list.append(best_dis)
        best_path.append(best_path[0])

        return best_path, best_dis_list

    # 取得鄰近解
    def get_neighbors(self, best_path):
        neighbors_path = []

        for i in range(self.neighbors_num):
            # 從目前路徑隨機交換兩個點
            p_1, p_2 = random.sample(best_path, 2)

            # 調換兩點
            copy_path = copy.deepcopy(best_path)
            copy_path[p_1], copy_path[p_2] = copy_path[p_2], copy_path[p_1]

            neighbors_path.append(copy_path)

        return neighbors_path

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