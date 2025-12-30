# tracker.py
import numpy as np

class EuclideanDistTracker:
    def __init__(self):
        self.center_points = {}
        self.id_count = 0

    def update(self, objects_rect):
        objects_bbs_ids = []
        for rect in objects_rect:
            x, y, w, h = rect
            cx = (x + x + w) // 2
            cy = (y + y + h) // 2

            same_object_detected = False
            for id, pt in self.center_points.items():
                dist = np.linalg.norm(np.array((cx, cy)) - np.array(pt))
                if dist < 50: # Ngưỡng khoảng cách (pixel)
                    self.center_points[id] = (cx, cy)
                    objects_bbs_ids.append([x, y, w, h, id, cx, cy])
                    same_object_detected = True
                    break

            if not same_object_detected:
                self.center_points[self.id_count] = (cx, cy)
                objects_bbs_ids.append([x, y, w, h, self.id_count, cx, cy])
                self.id_count += 1

        new_center_points = {}
        for obj in objects_bbs_ids:
            _, _, _, _, object_id, cx, cy = obj
            new_center_points[object_id] = (cx, cy)

        self.center_points = new_center_points.copy()
        return objects_bbs_ids