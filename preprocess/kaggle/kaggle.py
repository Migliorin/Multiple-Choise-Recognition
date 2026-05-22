import cv2
import numpy as np


class Kaggle():
    def __init__(
        self,
        min_radius=12,
        max_radius=14,
        hough_param2=12,
        min_fill_ratio=0.5,
        kernel_blur=21,
        hough_min_dist=22
    ):

        self.min_radius = min_radius
        self.max_radius = max_radius

        self.hough_param2 = hough_param2
        self.hough_min_dist = hough_min_dist

        self.min_fill_ratio = min_fill_ratio

        self.kernel_blur = kernel_blur

    def __open_image(self, image_path: str):
        img = cv2.imread(image_path)

        if img is None:
            raise FileNotFoundError(f"Imagem não encontrada: {image_path}")

        return img

    def __convert_image(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        return cv2.medianBlur(gray, self.kernel_blur)
    
    def __detect_circles(self, blur):
        circles = cv2.HoughCircles(
            blur,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=self.hough_min_dist,
            param1=80,
            param2=self.hough_param2,
            minRadius=self.min_radius,
            maxRadius=self.max_radius,
        )

        circle_list = []

        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")

            for x, y, r in circles:
                circle_list.append((x, y, r))

        return circle_list
    
    def __binary_mask(self, blur):
        return cv2.adaptiveThreshold(
            blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 10
        )
    
    def __extract_coords(self, circle_list, binary):
        marked = []

        for x, y, r in circle_list:
            mask = np.zeros(binary.shape, dtype=np.uint8)

            inner_radius = int(r * 0.75)

            cv2.circle(mask, (x, y), inner_radius, 255, -1)

            filled_pixels = cv2.countNonZero(cv2.bitwise_and(binary, binary, mask=mask))

            total_pixels = cv2.countNonZero(mask)

            if total_pixels == 0:
                continue

            fill_ratio = filled_pixels / total_pixels

            if fill_ratio > self.min_fill_ratio:
                marked.append((x, y, r, fill_ratio))

        marked.sort(key=lambda item: (item[1], -item[0]))

        marked_new = list()

        for idx in range(0, len(marked), 3):
            x, y, r, fill_ratio = marked[idx]
            marked_new.append((x, y, r, fill_ratio))
            x, _, r, fill_ratio = marked[idx + 1]
            marked_new.append((x, y, r, fill_ratio))
            x, _, r, fill_ratio = marked[idx + 2]
            marked_new.append((x, y, r, fill_ratio))

        marked_new.sort(key=lambda item: (item[1], item[0]))

        return marked_new
    
    def __convert_yolo_annotation(self, img_width, img_height, marked, labels):
        yolo_annotations = list()
        convert_label = {"A": 0, "B": 1, "C": 2, "D": 3}

        for idx, mark in enumerate(marked):
            x, y, r, _ = mark
            x_center = x / img_width
            y_center = y / img_height
            width = (2 * r) / img_width
            height = (2 * r) / img_height

            yolo_annotations.append(
                f"{convert_label[labels[idx]]} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
            )

        return yolo_annotations

    def __call__(self,labels:list[str], image_path:str):

        img = self.__open_image(image_path)
        H, W, _ = img.shape

        blur = self.__convert_image(img)

        circle_list = self.__detect_circles(blur)
        binary = self.__binary_mask(blur)

        marked = self.__extract_coords(circle_list, binary)
        
        if len(marked) != 30:
            raise Exception("Anotacoes faltando")

        return self.__convert_yolo_annotation(W, H, marked, labels)