# tracker.py
import math
import numpy as np

class EuclideanDistTracker:
    def __init__(self, dist_threshold=100, max_disappeared=5):
        self.center_points = {} # Lưu {id: center}
        self.disappeared = {}   # Lưu {id: số frame đã mất dấu}
        self.id_count = 0
        self.dist_threshold = dist_threshold
        self.max_disappeared = max_disappeared

    def update(self, objects_rect):
        objects_bbs_ids = []

        # Lấy tâm các box mới phát hiện
        input_centroids = []
        rects_dict = {} # Map index centroid -> rect
        for i, rect in enumerate(objects_rect):
            x, y, w, h = rect
            cx = (x + x + w) // 2
            cy = (y + y + h) // 2
            input_centroids.append((cx, cy))
            rects_dict[i] = rect

        # Nếu không có object nào đang track
        if len(self.center_points) == 0:
            for i in range(len(input_centroids)):
                self.register(input_centroids[i], rects_dict[i], objects_bbs_ids)
        
        # Nếu có object cần matching
        else:
            objectIDs = list(self.center_points.keys())
            objectCentroids = list(self.center_points.values())

            # Tính ma trận khoảng cách giữa TẤT CẢ object cũ và mới
            # (Logic này nhanh hơn loop lồng nhau)
            D = []
            for i in range(len(objectCentroids)):
                row = []
                for j in range(len(input_centroids)):
                    dist = math.hypot(objectCentroids[i][0] - input_centroids[j][0],
                                      objectCentroids[i][1] - input_centroids[j][1])
                    row.append(dist)
                D.append(row)
            D = np.array(D)

            # Tìm cặp ghép đôi tốt nhất (Matching)
            if D.size > 0:
                rows = D.min(axis=1).argsort()
                cols = D.argmin(axis=1)[rows]

                usedRows = set()
                usedCols = set()

                for (row, col) in zip(rows, cols):
                    if row in usedRows or col in usedCols:
                        continue
                    
                    if D[row][col] > self.dist_threshold:
                        continue

                    # Cập nhật ID cũ với vị trí mới
                    objectID = objectIDs[row]
                    self.center_points[objectID] = input_centroids[col]
                    self.disappeared[objectID] = 0 # Reset đếm mất dấu
                    
                    # Thêm vào output
                    x, y, w, h = rects_dict[col]
                    objects_bbs_ids.append([x, y, w, h, objectID, input_centroids[col][0], input_centroids[col][1]])

                    usedRows.add(row)
                    usedCols.add(col)

                # Xử lý các object cũ KHÔNG tìm thấy cặp (Mất dấu)
                for row in range(D.shape[0]):
                    if row not in usedRows:
                        objectID = objectIDs[row]
                        self.disappeared[objectID] += 1
                        if self.disappeared[objectID] > self.max_disappeared:
                            self.deregister(objectID)

                # Xử lý các object mới chưa có ID
                for col in range(D.shape[1]):
                    if col not in usedCols:
                        self.register(input_centroids[col], rects_dict[col], objects_bbs_ids)
            else:
                # Trường hợp đặc biệt khi D rỗng
                for col in range(len(input_centroids)):
                    self.register(input_centroids[col], rects_dict[col], objects_bbs_ids)

        return objects_bbs_ids

    def register(self, centroid, rect, output_list):
        self.center_points[self.id_count] = centroid
        self.disappeared[self.id_count] = 0
        x, y, w, h = rect
        output_list.append([x, y, w, h, self.id_count, centroid[0], centroid[1]])
        self.id_count += 1

    def deregister(self, objectID):
        del self.center_points[objectID]
        del self.disappeared[objectID]