import numpy as np
import cv2

def analyze_polygon(polygon):
    polygon = np.array(polygon, dtype=np.float32)

    # 检查是否闭合
    if not np.array_equal(polygon[0], polygon[-1]):
        polygon = np.vstack([polygon, polygon[0]])  # 闭合

    # 计算面积（带符号）用于判断方向
    def signed_area(poly):
        x = poly[:, 0]
        y = poly[:, 1]
        return 0.5 * np.sum(x[:-1] * y[1:] - x[1:] * y[:-1])

    area = signed_area(polygon)
    orientation = "Clockwise" if area < 0 else "Counter-Clockwise"

    # 找第一个点的位置信息（最左上）
    xs = polygon[:-1, 0]
    ys = polygon[:-1, 1]
    idx_min_xy = np.argmin(xs + ys)
    idx_min_y = np.lexsort((xs, ys))[0]

    def point_position(index):
        x, y = polygon[index]
        h, w = np.max(polygon[:, 1]), np.max(polygon[:, 0])
        if x < w / 2 and y < h / 2:
            return "Top-Left"
        elif x >= w / 2 and y < h / 2:
            return "Top-Right"
        elif x < w / 2 and y >= h / 2:
            return "Bottom-Left"
        else:
            return "Bottom-Right"

    # 输出信息
    first_point = polygon[0]
    first_position = point_position(0)

    print("First point:", first_point)
    print("Heuristic location:", first_position)
    print("Index of min(x + y):", idx_min_xy, "->", polygon[idx_min_xy])
    print("Index of min(y, then x):", idx_min_y, "->", polygon[idx_min_y])
    print("Point order:", orientation)

    return {
        "first_point": first_point,
        "first_position": first_position,
        "orientation": orientation,
        "min_xy_point": polygon[idx_min_xy],
        "min_yx_point": polygon[idx_min_y]
    }

# 示例 polygon（逆时针矩形）
example_polygon = [
    [100, 100],
    [200, 100],
    [200, 200],
    [100, 200]
]

analyze_polygon(example_polygon)