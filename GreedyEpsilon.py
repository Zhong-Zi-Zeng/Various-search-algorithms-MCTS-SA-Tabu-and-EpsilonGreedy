import numpy as np
import random
import copy


class GreedyEpsilon:
    def __init__(self, iteration, distance_matrix, epsilon, epsilon_min, decay):
        self.iteration = iteration # 迭代次數
        self.epsilon = epsilon  # epsilon
        self.distance_matrix = distance_matrix  # 各城市座標
        self.epsilon_min = epsilon_min  # 最低epsilon
        self.decay = decay  # 衰減係數

    def Search(self):
        total_dis_list = []
        best_dis = np.inf

        for i in range(self.iteration):
            city_number = len(self.distance_matrix)  # 所有城市數量
            start = np.random.randint(city_number)  # 起始點
            path = [start]  # 路徑
            now = start  # 目前位置

            unvisited_list = list(self.distance_matrix)  # 未拜訪節點
            unvisited_list.remove(self.distance_matrix[now])  # 移除起點

            for _ in range(city_number - 1):
                min_dis = np.inf
                next_idx = -1

                if np.random.random() < self.epsilon:
                    rand_city = random.choice(unvisited_list)
                    next_idx = list(self.distance_matrix).index(rand_city)

                else:
                    # 找出離目前點最短路徑
                    for unvisited in unvisited_list:
                        x1, y1 = self.distance_matrix[now]
                        x2, y2 = unvisited
                        dis = self.euler_distance(x1, y1, x2, y2)

                        if dis < min_dis:
                            next_idx = list(self.distance_matrix).index(unvisited)
                            min_dis = dis

                now = next_idx
                unvisited_list.remove(self.distance_matrix[now])
                path.append(now)

                if self.epsilon > self.epsilon_min:
                    self.epsilon *= self.decay

            path.append(start)
            total_dis = self.calculate_distance(path)

            if total_dis < best_dis:
                best_dis = total_dis
                best_path = copy.deepcopy(path)

            total_dis_list.append(best_dis)
            print('Iteration :', i, 'distance:', total_dis, 'best_dis:', best_dis)

        return best_path, total_dis_list

    # 歐式距離
    def euler_distance(self, x1, y1, x2, y2):
        return round(((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5)

    # 計算路徑所經過的距離
    def calculate_distance(self, path):
        total_dis = 0

        for i in range(len(path) - 1):
            x1, y1 = self.distance_matrix[path[i]]
            x2, y2 = self.distance_matrix[path[i + 1]]
            total_dis += self.euler_distance(x1, y1, x2, y2)

        return total_dis
