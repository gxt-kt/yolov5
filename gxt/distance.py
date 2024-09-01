import numpy as np

from export import main


class DistanceEstimator:
    def __init__(self, camera_params, img_width, img_height):
        self.img_width = img_width
        self.img_height = img_height
        self.camera_matrix = np.array(camera_params)

        # 不同类别的平均身高 (单位: 米)
        self.HEIGHT_LOOKUP = {"Ren": 1.75, "Dianti": 2.2}

    def estimate_distance(self, bbox, class_name):
        """
        根据给定的归一化检测框和目标类别估算目标距离。
        @params:
        bbox (tuple): 检测框的归一化坐标 (x, y, w, h)
        class_name (str): 目标类别 ("person", "vehicle", 等)
        @ret:
        距离 (float): 目标距离相机的估算距离 (单位: 米)
        """
        x, y, w, h = bbox

        # 计算检测框的中心点坐标 (像素)
        center_x = x * self.img_width
        center_y = y * self.img_height

        # 计算图像平面上的检测框高度(像素)
        bbox_height_px = h * self.img_height

        # 根据目标类别查找平均身高
        if class_name not in self.HEIGHT_LOOKUP:
            print("error {} not find", class_name)
        target_height = self.HEIGHT_LOOKUP[class_name]

        # 根据相似三角形原理计算距离
        distance = (target_height * self.camera_matrix[0, 0]) / bbox_height_px

        return distance

    def get_3d_position(self, bbox, class_name):
        """
        根据给定的归一化检测框和目标类别,计算目标在相机坐标系下的3D位置。

        参数:
        bbox (tuple): 检测框的归一化坐标 (x, y, w, h)
        class_name (str): 目标类别 ("Ren", "Dianti", 等)

        返回:
        3D位置 (numpy.ndarray): 目标在相机坐标系下的3D位置 (x, y, z)
        """
        x, y, w, h = bbox

        # 计算检测框的中心点坐标 (像素)
        center_x = x * self.img_width
        center_y = y * self.img_height

        # 计算距离
        distance = self.estimate_distance(bbox, class_name)
        if distance is None or distance < 0:
            return None

        # 根据相机内参和距离计算3D位置
        Z = distance
        X = (center_x - self.camera_matrix[0, 2]) * Z / self.camera_matrix[0, 0]
        Y = (center_y - self.camera_matrix[1, 2]) * Z / self.camera_matrix[1, 1]
        return np.array([X, Y, Z])

def calculate_distance(point1, point2):
    """
    计算两个 3D 点之间的欧式距离
    参数:
    point1 (np.ndarray): 第一个点的 3D 坐标, 形状为 (3,)
    point2 (np.ndarray): 第二个点的 3D 坐标, 形状为 (3,)
    返回:
    float: 两个点之间的距离
    """
    distance = np.sqrt(np.sum((point1 - point2)**2))
    return distance



# 示例使用
img_width = 1280  # 图像宽度 (像素)
img_height = 640  # 图像高度 (像素)
# 相机内参
camera_params = [
    [787.22316869, 0.0, 628.91534144],
    [0.0, 793.45182, 313.46301416],
    [0.0, 0.0, 1.0],
]
# 畸变参数
# distortion_params = [[0.05828191, -0.11750722, -0.014249, 0.00087086, -0.01850911]]

estimator = DistanceEstimator(camera_params, img_width, img_height)

if __name__=="__main__" :
    # 估算人的距离
    bbox1 = (0.4, 0.3, 0.2, 0.5)  # 检测框的归一化坐标 (x, y, w, h)
    distance_person = estimator.estimate_distance(bbox1, "Ren")
    print(f"估算的人距离: {distance_person:.2f} 米")
    position_person = estimator.get_3d_position(bbox1, "Ren")
    print(f"估算的人位置: {position_person} 米")

    # 估算车的距离
    bbox2 = (0.8, 0.3, 0.2, 0.1)  # 检测框的归一化坐标 (x, y, w, h)
    distance_vehicle = estimator.estimate_distance(bbox2, "Dianti")
    print(f"估算的车距离: {distance_vehicle:.2f} 米")
    positon_vehicle = estimator.get_3d_position(bbox2, "Dianti")
    print(f"估算的车位置: {positon_vehicle} 米")

    dd=calculate_distance(position_person,positon_vehicle)
    print(dd)
