import numpy as np
import random
import copy
import matplotlib.pyplot as plt


class Node:
    def __init__(self):
        self.parent = None  # 父節點
        self.visit_times = 0  # 此節點的拜訪次數
        self.q_value = 0  # 此節點的Quality值
        self.child_list = []  # 用來存放子節點用
        self.city_index = None  # 對應到哪個城市

class MCTS:
    def __init__(self, distance_matrix, iteration, c):
        self.distance_matrix = distance_matrix  # 各城市座標
        self.iteration = iteration  # 迭代次數
        self.c = c  # 用來平衡探索與利用的常數

    def Search(self):
        best_dis = np.inf
        root = Node()  # 設定根節點
        best_path = []  # 紀錄最優路徑
        best_dis_list = []  # 紀錄最優距離

        for stage in range(len(self.distance_matrix)):
            # 向下搜尋
            for i in range(self.iteration):
                next_node = self.get_next_node(root.child_list, root)  # 找尋目前需要向下搜索的節點
                path, total_dis, last_node = self.play(next_node)  # 開始進行探索
                self.backpropagation(last_node, -total_dis)  # 反向更新剛剛經過節點的q值。這裡用負號是代表總距離越長越不好

                if total_dis < best_dis:
                    best_dis = total_dis

                best_dis_list.append(best_dis)
                print("Stage:", stage, "iter:", i, "best_dis:", best_dis)

            # 從目前根節點選取一個最優的子節點作為新的根節點後繼續向下搜索
            best_child_q_value = -np.inf
            best_child_node = None

            for child in root.child_list:
                if child.q_value > best_child_q_value:
                    best_child_q_value = child.q_value
                    best_child_node = child

            root = best_child_node  # 更新根節點
            best_path.append(root.city_index)  # 將此節點加入到最優路徑中

        best_path.append(best_path[0])
        print('best:', self.calculate_distance(best_path))

        return best_path, best_dis_list

    def get_unvisited_list(self, node):
        # 所有城市索引值
        all_city_index = [i for i in range(len(self.distance_matrix))]

        # 找出此節點拜訪過的所有節點
        visited_city_idx_list = self.get_visited_list(node)

        # 未拜訪過的節點列表
        unvisited_city_idx_list = [city for city in all_city_index if city not in visited_city_idx_list]

        return unvisited_city_idx_list

    # 取得已拜訪過的節點
    def get_visited_list(self, node):
        # 將此節點所有拜訪過的節點紀錄下來
        visited_city_idx_list = []
        temp_parent = copy.copy(node)

        # 往上搜尋
        while temp_parent.parent is not None:
            visited_city_idx_list.append(temp_parent.city_index)
            temp_parent = temp_parent.parent

        # 往下搜尋
        child_city_list = [child.city_index for child in node.child_list]

        # 將所有拜訪過的節點相加
        visited_city_idx_list += child_city_list

        return visited_city_idx_list

    # 找出此節點之前經過的路徑
    def find_footprint(self, start_node):
        path = []
        temp_node = copy.copy(start_node)
        while temp_node.parent is not None:
            path.append(temp_node.city_index)
            temp_node = temp_node.parent

        return path

    # 針對給定的起點進行探索
    def play(self, start_node):
        # 找出此節點還有哪些節點沒探索過
        unvisited_city_idx_list = self.get_unvisited_list(start_node)

        # 找出之前走過的路徑
        path = self.find_footprint(start_node)

        # 目前節點
        now_node = start_node

        # 若已遍歷所有節點則停止
        while len(path) != len(self.distance_matrix):
            # 如果還有未拜訪過的節點，則從目前節點隨意選則下一個節點
            if len(unvisited_city_idx_list) != 0:
                rand_city_idx = random.choice(unvisited_city_idx_list)

                # 創建隨機節點並加入到目前節點的子節點中
                rand_node = Node()
                rand_node.parent = now_node
                rand_node.city_index = rand_city_idx

                # 將選出的節點加入路徑中
                path.append(rand_city_idx)

                # 將此節點加入到子節點列表中
                now_node.child_list.append(rand_node)

                # 將目前點改為選出的節點
                now_node = rand_node

                # 更新未拜訪節點
                unvisited_city_idx_list = self.get_unvisited_list(now_node)
            else:
                # 利用ucb取出最優的節點
                best_node = self.ucb(now_node.child_list, now_node, self.c)
                path.append(best_node.city_index)
                now_node = best_node

        last_node = now_node  # 取出最後一個節點
        path.append(start_node.city_index)  # 將起點加入到路徑的最後面
        total_dis = self.calculate_distance(path)  # 計算此路徑總距離

        return path, total_dis, last_node

    # 計算路徑所經過的距離
    def calculate_distance(self, path):
        total_dis = 0

        for i in range(len(path) - 1):
            x1, y1 = self.distance_matrix[path[i]]
            x2, y2 = self.distance_matrix[path[i + 1]]
            total_dis += round(((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5)

        return total_dis

    # 搜索策略
    def get_next_node(self, child_list, node):
        # 找出此節點還有哪些節點沒探索過
        unvisited_city_idx_list = self.get_unvisited_list(node)

        # 判斷還有哪個節點沒拜訪過，若有則新增一子節點並將其回傳
        if len(unvisited_city_idx_list) != 0:
            next_node = Node()
            next_node.parent = node
            next_node.city_index = unvisited_city_idx_list.pop(0)  # 取出第一個未拜訪過的節點
            node.child_list.append(next_node)
            return next_node

        # 若全部都被拜訪過則依照UCB進行選取
        else:
            return self.ucb(child_list, node, self.c)

    # Upper Confidence bounds
    def ucb(self, child_list, parent, C):
        ucb_list = []  # 儲存每個子節點的UCB值
        total_visit_times = parent.visit_times  # 總拜訪次數由父節點提供

        for child_node in child_list:
            #  UCB = quality / times + C * sqrt(2 * ln(total_times) / times)
            left = child_node.q_value / total_visit_times
            right = np.sqrt(C * (np.log(total_visit_times) / child_node.visit_times))
            ucb_value = float(left + right)
            ucb_list.append(ucb_value)

        max_idx = np.argmax(np.array(ucb_list))
        best_child = child_list[max_idx]

        return best_child

    # 反向更新q值
    def backpropagation(self, last_node, total_dis):
        total_dis /= 1000.  # 以防數值太大
        # 更新各節點的q值
        for i in range(len(self.distance_matrix) + 1):
            last_node.visit_times += 1  # 增加探索次數
            last_node.q_value += total_dis  # 增加q值
            last_node = last_node.parent  # 往上尋找節點
