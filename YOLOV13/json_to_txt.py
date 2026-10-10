import json
import os
import cv2
import numpy as np
from pathlib import Path


class LabelMeToYOLOSegmentation:
    def __init__(self, class_names):
        """
        初始化转换器

        参数:
            class_names: 类别名称列表，顺序对应YOLO的类别ID
        """
        self.class_names = class_names

    def convert_single_file(self, json_path, output_dir=None, img_dir=None):
        """
        转换单个LabelMe JSON文件

        参数:
            json_path: JSON文件路径
            output_dir: 输出目录（可选，默认为JSON文件所在目录）
            img_dir: 图像文件目录（可选，用于获取图像尺寸）

        返回:
            bool: 转换是否成功
        """
        try:
            # 读取JSON文件
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # 获取图像尺寸
            img_width, img_height = self._get_image_size(data, json_path, img_dir)

            # 确定输出目录
            if output_dir is None:
                output_dir = os.path.dirname(json_path)
            else:
                os.makedirs(output_dir, exist_ok=True)

            # 创建输出文件名
            base_name = Path(json_path).stem
            output_path = os.path.join(output_dir, f"{base_name}.txt")

            # 处理所有标注形状
            yolo_lines = []
            for shape in data['shapes']:
                label = shape['label']
                shape_type = shape.get('shape_type', 'polygon')
                points = shape['points']

                # 获取类别ID
                if label not in self.class_names:
                    print(f"警告: 类别 '{label}' 不在类别列表中，跳过")
                    continue

                class_id = self.class_names.index(label)

                # 转换坐标到YOLO格式
                normalized_points = self._convert_points_to_yolo(
                    points, shape_type, img_width, img_height
                )

                if normalized_points:
                    line = f"{class_id} " + " ".join([f"{p:.6f}" for p in normalized_points])
                    yolo_lines.append(line)

            # 写入输出文件
            with open(output_path, 'w', encoding='utf-8') as f:
                for line in yolo_lines:
                    f.write(line + '\n')

            print(f"转换成功: {output_path}")
            return True

        except Exception as e:
            print(f"转换失败 {json_path}: {str(e)}")
            return False

    def convert_directory(self, json_dir, output_dir=None, img_dir=None):
        """
        批量转换目录中的所有JSON文件

        参数:
            json_dir: JSON文件目录
            output_dir: 输出目录（可选）
            img_dir: 图像文件目录（可选）

        返回:
            int: 成功转换的文件数量
        """
        json_dir = Path(json_dir)
        json_files = list(json_dir.glob("*.json"))

        if not json_files:
            print(f"在目录 {json_dir} 中未找到JSON文件")
            return 0

        print(f"找到 {len(json_files)} 个JSON文件")
        print(f"类别列表: {self.class_names}")

        success_count = 0
        for json_file in json_files:
            if self.convert_single_file(str(json_file), output_dir, img_dir):
                success_count += 1

        print(f"批量转换完成! 成功处理 {success_count}/{len(json_files)} 个文件")
        return success_count

    def _get_image_size(self, data, json_path, img_dir=None):
        """
        获取图像尺寸

        返回:
            tuple: (width, height)
        """
        # 方法1: 直接从JSON中获取
        if 'imageWidth' in data and 'imageHeight' in data:
            return data['imageWidth'], data['imageHeight']

        # 方法2: 从图像文件获取
        try:
            image_path = data.get('imagePath', '')
            if not image_path:
                raise ValueError("JSON中没有imagePath信息")

            # 处理相对路径
            if not os.path.isabs(image_path):
                # 首先尝试在JSON文件所在目录查找
                json_dir = os.path.dirname(json_path)
                possible_path = os.path.join(json_dir, image_path)

                if os.path.exists(possible_path):
                    image_path = possible_path
                elif img_dir and os.path.exists(os.path.join(img_dir, image_path)):
                    image_path = os.path.join(img_dir, image_path)
                else:
                    raise FileNotFoundError(f"无法找到图像文件: {image_path}")

            # 使用OpenCV读取图像尺寸
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"无法读取图像文件: {image_path}")

            height, width = img.shape[:2]
            return width, height

        except Exception as e:
            print(f"警告: 无法获取图像尺寸 ({str(e)}), 使用默认尺寸640x480")
            return 640, 480

    def _convert_points_to_yolo(self, points, shape_type, img_width, img_height):
        """
        将点坐标转换为YOLO格式的归一化坐标

        返回:
            list: 归一化坐标列表 [x1, y1, x2, y2, ...]
        """
        normalized_points = []

        if shape_type in ['polygon', 'linestrip']:
            # 多边形或折线 - 直接转换所有点
            for x, y in points:
                x_norm = max(0, min(1, x / img_width))
                y_norm = max(0, min(1, y / img_height))
                normalized_points.extend([x_norm, y_norm])

        elif shape_type == 'rectangle':
            # 矩形转换为多边形（四个点）
            if len(points) >= 2:
                x1, y1 = points[0]
                x2, y2 = points[1]
                rect_points = [
                    [x1, y1], [x2, y1], [x2, y2], [x1, y2]
                ]

                for x, y in rect_points:
                    x_norm = max(0, min(1, x / img_width))
                    y_norm = max(0, min(1, y / img_height))
                    normalized_points.extend([x_norm, y_norm])

        elif shape_type == 'circle':
            # 圆近似为多边形（24个点）
            if len(points) >= 2:
                center_x, center_y = points[0]
                edge_x, edge_y = points[1]
                radius = ((edge_x - center_x) ** 2 + (edge_y - center_y) ** 2) ** 0.5

                for i in range(24):
                    angle = 2 * np.pi * i / 24
                    x = center_x + radius * np.cos(angle)
                    y = center_y + radius * np.sin(angle)
                    x_norm = max(0, min(1, x / img_width))
                    y_norm = max(0, min(1, y / img_height))
                    normalized_points.extend([x_norm, y_norm])

        else:
            print(f"警告: 不支持的形状类型: {shape_type}")
            return None

        return normalized_points

    def save_class_names(self, output_dir):
        """
        保存类别名称到文件

        参数:
            output_dir: 输出目录
        """
        class_file = os.path.join(output_dir, 'classes.txt')
        with open(class_file, 'w', encoding='utf-8') as f:
            for class_name in self.class_names:
                f.write(class_name + '\n')
        print(f"类别文件已保存: {class_file}")


# 使用示例
if __name__ == "__main__":
    # 1. 定义类别名称（顺序很重要！）
    CLASS_NAMES = ["cat", "dog"]

    # 2. 创建转换器实例
    converter = LabelMeToYOLOSegmentation(CLASS_NAMES)

    # 3. 设置路径
    JSON_DIR = r"D:\硕士准备\YOLOV13\Image_segmentation\Image_segmentation\json"  # 替换为你的JSON文件目录
    OUTPUT_DIR = r"D:\硕士准备\YOLOV13\Image_segmentation\Image_segmentation\txt1"  # 替换为输出目录


    # 4. 批量转换（方法1）
    print("=== 批量转换 ===")
    converter.convert_directory(JSON_DIR, OUTPUT_DIR)



    # 6. 保存类别文件
    print("\n=== 保存类别文件 ===")
    converter.save_class_names(OUTPUT_DIR)






    print("\n所有转换操作完成！")