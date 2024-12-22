# import streamlit as st
# from PIL import Image
# import numpy as np
# import cv2
#
# def geometric_operations(image, operation, params):
#     def translate(image, dx, dy):
#         M = np.float32([[1, 0, dx], [0, 1, dy]])
#         return cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
#
#     def rotate(image, angle):
#         center = (image.shape[1] // 2, image.shape[0] // 2)
#         M = cv2.getRotationMatrix2D(center, angle, 1)
#         return cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
#
#     def scale(image, fx, fy):
#         return cv2.resize(image, None, fx=fx, fy=fy)
#
#     def flip(image):
#         return cv2.flip(image, 1)
#
#     operations = {
#         "平移": lambda img, params: translate(img, params['dx'], params['dy']),
#         "旋转": lambda img, params: rotate(img, params['angle']),
#         "缩放": lambda img, params: scale(img, params['fx'], params['fy']),
#         "镜像": lambda img, params: flip(img),
#     }
#
#     return operations[operation](image, params)
#
# def contrast_enhancement(image, operation, params=None):
#     if operation == "灰度变换":
#         return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     elif operation == "直方图均衡化":
#         gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#         return cv2.equalizeHist(gray)
#     elif operation == "CLAHE":
#         gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#         clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
#         return clahe.apply(gray)
#     elif operation == "Gamma 调整":
#         gamma = params.get('gamma', 1.0)
#         lookup_table = np.array([((i / 255.0) ** gamma) * 255 for i in np.arange(256)]).astype("uint8")
#         return cv2.LUT(image, lookup_table)
#     return image
#
# def smoothing_operations(image, operation, kernel_size):
#     if operation == "均值滤波":
#         return cv2.blur(image, (kernel_size, kernel_size))
#     elif operation == "中值滤波":
#         return cv2.medianBlur(image, kernel_size)
#     elif operation == "高斯滤波":
#         return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
#     elif operation == "双边滤波":
#         return cv2.bilateralFilter(image, d=kernel_size, sigmaColor=75, sigmaSpace=75)
#     return image
#
# def image_segmentation(image, method):
#     if method == "边缘法分割":
#         edges = cv2.Canny(image, 100, 200)
#         return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
#     elif method == "阈值法分割":
#         gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#         _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
#         return cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
#     elif method == "区域法分割":
#         gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#         _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
#         contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#         segmented_image = np.zeros_like(image)
#         for contour in contours:
#             color = [np.random.randint(0, 255) for _ in range(3)]
#             cv2.drawContours(segmented_image, [contour], -1, color, -1)
#         return segmented_image
#     elif method == "K-means 聚类":
#         Z = image.reshape((-1, 3))
#         Z = np.float32(Z)
#         _, labels, centers = cv2.kmeans(Z, 3, None, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0), 10, cv2.KMEANS_RANDOM_CENTERS)
#         centers = np.uint8(centers)
#         segmented = centers[labels.flatten()]
#         return segmented.reshape(image.shape)
#
#     return image
#
#
# # 页面设置
# st.set_page_config(page_title="图像处理平台", layout="wide")
#
# # 自定义 CSS 样式
# st.markdown(
#     """
#     <style>
#     .main {
#         padding: 20px;
#     }
#
#     /* 标题样式 */
#     .title {
#         text-align: center;
#         font-size: 2.5em;
#         font-weight: bold;
#         color: #4CAF50;
#         margin-bottom: 20px;
#     }
#
#     /* 操作菜单 */
#     .menu {
#         background-color: #f9f9f9;
#         padding: 20px;
#         border-radius: 10px;
#         box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
#     }
#
#     /* 图像展示 */
#     .image {
#         text-align: center;
#         margin: auto;
#     }
#
#     .image img {
#         max-width: 90%;
#         border-radius: 10px;
#         box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1);
#     }
#
#     /* 底部说明 */
#     .footer {
#         text-align: center;
#         font-size: 0.9em;
#         color: gray;
#         margin-top: 20px;
#     }
#     </style>
#     """,
#     unsafe_allow_html=True,
# )
#
# # 标题
# st.markdown('<div class="title">图像处理平台 - WCH</div>', unsafe_allow_html=True)
#
# # 上传图片
# uploaded_file = st.file_uploader("上传图片", type=["jpg", "png", "bmp"])
#
# if uploaded_file:
#     img = Image.open(uploaded_file)
#     img = np.array(img)
#
#     # 创建三列布局：操作菜单在最左，原始图片在中，处理后图片在最右
#     col1, col2, col3 = st.columns([1, 2, 2])
#
#     with col1:
#         st.markdown("### 操作菜单")
#         operation = st.selectbox("选择操作", ["几何运算", "对比度增强", "平滑处理", "图像分割"])
#
#         params = {}
#         processed_img = img  # 默认使用原始图片
#         if operation == "几何运算":
#             sub_operation = st.radio("几何操作", ["平移", "旋转", "缩放", "镜像"])
#             if sub_operation == "平移":
#                 params["dx"] = st.slider("X 轴平移", -500, 500, 100)
#                 params["dy"] = st.slider("Y 轴平移", -500, 500, 100)
#                 processed_img = geometric_operations(img, "平移", params)
#             elif sub_operation == "旋转":
#                 params["angle"] = st.slider("旋转角度", -180, 180, 45)
#                 processed_img = geometric_operations(img, "旋转", params)
#             elif sub_operation == "缩放":
#                 params["fx"] = st.slider("X 轴缩放", 0.1, 2.0, 1.0)
#                 params["fy"] = st.slider("Y 轴缩放", 0.1, 2.0, 1.0)
#                 processed_img = geometric_operations(img, "缩放", params)
#             elif sub_operation == "镜像":
#                 processed_img = geometric_operations(img, "镜像", {})
#
#         elif operation == "对比度增强":
#             sub_operation = st.radio("增强方法", ["灰度变换", "直方图均衡化", "CLAHE", "Gamma 调整"])
#             if sub_operation == "灰度变换":
#                 processed_img = contrast_enhancement(img, "灰度变换")
#             elif sub_operation == "直方图均衡化":
#                 processed_img = contrast_enhancement(img, "直方图均衡化")
#             elif sub_operation == "CLAHE":
#                 processed_img = contrast_enhancement(img, "CLAHE")
#             elif sub_operation == "Gamma 调整":
#                 gamma = st.slider("Gamma 值", 0.1, 3.0, 1.0, 0.1)
#                 processed_img = contrast_enhancement(img, "Gamma 调整", {"gamma": gamma})
#
#         elif operation == "平滑处理":
#             sub_operation = st.radio("平滑方法", ["均值滤波", "中值滤波", "高斯滤波", "双边滤波"])
#             kernel_size = st.slider("滤波器大小", 3, 15, 5, step=2)
#             processed_img = smoothing_operations(img, sub_operation, kernel_size)
#
#         elif operation == "图像分割":
#             sub_operation = st.radio("分割方法", ["边缘法分割", "阈值法分割", "区域法分割", "K-means 聚类"])
#             processed_img = image_segmentation(img, sub_operation)
#
#     with col2:
#         st.markdown("### 原始图片")
#         st.image(img, caption="原始图片", use_container_width=True)
#
#     with col3:
#         st.markdown("### 处理后图片")
#         st.image(processed_img, caption="处理后图片", use_container_width=True)
#
# # 页面底部
# st.markdown('<div class="footer">© 2024 图像处理平台. All Rights Reserved.</div>', unsafe_allow_html=True)

import io
import cv2
import streamlit as st
from PIL import Image
import numpy as np


def geometric_operations(image, operation, params):
    def translate(image, dx, dy):
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        return cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))

    def rotate(image, angle):
        center = (image.shape[1] // 2, image.shape[0] // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1)
        return cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))

    def scale(image, fx, fy):
        return cv2.resize(image, None, fx=fx, fy=fy)

    def flip(image):
        return cv2.flip(image, 1)

    operations = {
        "平移": lambda img, params: translate(img, params['dx'], params['dy']),
        "旋转": lambda img, params: rotate(img, params['angle']),
        "缩放": lambda img, params: scale(img, params['fx'], params['fy']),
        "镜像": lambda img, params: flip(img),
    }

    return operations[operation](image, params)

def contrast_enhancement(image, operation, params=None):
    def grayscale(image):
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    def histogram_equalization(image):
        gray = grayscale(image)
        return cv2.equalizeHist(gray)

    def clahe(image):
        gray = grayscale(image)
        clahe_obj = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe_obj.apply(gray)

    def gamma_adjustment(image, gamma):
        lookup_table = np.array([((i / 255.0) ** gamma) * 255 for i in np.arange(256)]).astype("uint8")
        return cv2.LUT(image, lookup_table)

    operations = {
        "灰度变换": lambda img, params: grayscale(img),
        "直方图均衡化": lambda img, params: histogram_equalization(img),
        "CLAHE": lambda img, params: clahe(img),
        "Gamma 调整": lambda img, params: gamma_adjustment(img, params.get('gamma', 1.0))
    }

    return operations[operation](image, params)

def smoothing_operations(image, operation, kernel_size):
    def mean_blur(image, kernel_size):
        return cv2.blur(image, (kernel_size, kernel_size))

    def median_blur(image, kernel_size):
        return cv2.medianBlur(image, kernel_size)

    def gaussian_blur(image, kernel_size):
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

    def bilateral_filter(image, kernel_size):
        return cv2.bilateralFilter(image, d=kernel_size, sigmaColor=75, sigmaSpace=75)

    operations = {
        "均值滤波": lambda img, params: mean_blur(img, params),
        "中值滤波": lambda img, params: median_blur(img, params),
        "高斯滤波": lambda img, params: gaussian_blur(img, params),
        "双边滤波": lambda img, params: bilateral_filter(img, params)
    }

    return operations[operation](image, kernel_size)

def image_segmentation(image, method):
    def edge_based(image):
        edges = cv2.Canny(image, 100, 200)
        return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    def threshold_based(image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        return cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)

    def region_based(image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        segmented_image = np.zeros_like(image)
        for contour in contours:
            color = [np.random.randint(0, 255) for _ in range(3)]
            cv2.drawContours(segmented_image, [contour], -1, color, -1)
        return segmented_image

    def kmeans_clustering(image):
        Z = image.reshape((-1, 3))
        Z = np.float32(Z)
        _, labels, centers = cv2.kmeans(Z, 3, None, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0), 10, cv2.KMEANS_RANDOM_CENTERS)
        centers = np.uint8(centers)
        segmented = centers[labels.flatten()]
        return segmented.reshape(image.shape)

    operations = {
        "边缘法分割": lambda img: edge_based(img),
        "阈值法分割": lambda img: threshold_based(img),
        "区域法分割": lambda img: region_based(img),
        "K-means 聚类": lambda img: kmeans_clustering(img)
    }

    return operations.get(method, lambda img: img)(image)


# 页面设置
st.set_page_config(page_title="图像处理平台", layout="wide")

# 自定义 CSS 样式
st.markdown(
    """
    <style>
    body {
        background: linear-gradient(to bottom, #ffffff, #f0f0f0);
        font-family: 'Arial', sans-serif;
    }
    .main {
        padding: 20px;
    }

    /* 标题样式 */
    .title {
        text-align: center;
        font-size: 2.5em;
        font-weight: bold;
        color: #4CAF50;
        margin-bottom: 20px;
        background: -webkit-linear-gradient(#4CAF50, #2e7d32);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    /* 按钮组样式 */
    .button-group {
        display: flex;
        justify-content: space-evenly;
        flex-wrap: wrap;
    }

    .button-group button {
        width: 120px;
        height: 50px;
        margin: 5px;
        padding: 10px 15px;
        font-size: 1em;
        font-weight: bold;
        color: white;
        background-color: #2196F3;
        border: none;
        border-radius: 5px;
        cursor: pointer;
        transition: background-color 0.3s;
        text-align: center;
    }

    .button-group button:hover {
        background-color: #0b79d0;
    }

    .button-group button.selected {
        background-color: #0b79d0;
    }

    /* 图片展示区域 */
    .image img {
        max-width: 90%;
        border-radius: 10px;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1);
    }

    /* 底部说明 */
    .footer {
        text-align: center;
        font-size: 0.9em;
        color: gray;
        margin-top: 20px;
    }    
    </style>
    """,
    unsafe_allow_html=True,
)

# 标题
st.markdown('<div class="title">图像处理平台 - WCH</div>', unsafe_allow_html=True)

# 上传图片
uploaded_file = st.file_uploader("", type=["jpg", "png", "bmp"])
if not uploaded_file:
    st.warning("请上传图片！")
    st.stop()

img = Image.open(uploaded_file)
img = np.array(img)

# 创建布局与功能
col1, col2, col3 = st.columns([1, 2, 2])

with col1:
    st.markdown("### 操作菜单")
    operation = st.selectbox("选择操作类型", ["图像的几何运算", "图像对比度增强", "图像平滑处理", "图像分割"])

    params = {}
    processed_img = None

    def render_buttons(options, session_key):
        current_selection = st.session_state.get(session_key, options[0])
        st.session_state[session_key] = current_selection
        button_layout = st.columns(len(options))

        for idx, option in enumerate(options):
            with button_layout[idx]:
                button_class = "selected" if st.session_state[session_key] == option else ""
                if st.button(option, key=f"{session_key}_{idx}"):
                    st.session_state[session_key] = option

        return st.session_state[session_key]

    if operation == "图像的几何运算":
        action = render_buttons(["平移", "旋转", "缩放", "镜像"], "geometry_ops")

        if action == "平移":
            params["dx"] = st.slider("水平平移", -1000, 1000, 0, step=50)
            params["dy"] = st.slider("垂直平移", -1000, 1000, 0, step=50)
            processed_img = geometric_operations(img, "平移", params)

        elif action == "旋转":
            params["angle"] = st.slider("旋转角度 (°)", -360, 360, 0, step=15)
            processed_img = geometric_operations(img, "旋转", params)

        elif action == "缩放":
            params["fx"] = st.slider("水平缩放系数", 0.1, 5.0, 1.0, step=0.1)
            params["fy"] = st.slider("垂直缩放系数", 0.1, 5.0, 1.0, step=0.1)
            processed_img = geometric_operations(img, "缩放", params)

        elif action == "镜像":
            processed_img = geometric_operations(img, "镜像", {})

    elif operation == "图像对比度增强":
        enhancement = render_buttons(["灰度变换", "直方图均衡化", "CLAHE方法", "Gamma 调整"], "contrast_ops")

        if enhancement == "灰度变换":
            processed_img = contrast_enhancement(img, "灰度变换")

        elif enhancement == "直方图均衡化":
            processed_img = contrast_enhancement(img, "直方图均衡化")

        elif enhancement == "CLAHE方法":
            processed_img = contrast_enhancement(img, "CLAHE")

        elif enhancement == "Gamma 调整":
            gamma = st.slider("Gamma 值", 0.01, 10.0, 1.0, step=0.01)
            processed_img = contrast_enhancement(img, "Gamma 调整", {"gamma": gamma})

    elif operation == "图像平滑处理":
        smoothing = render_buttons(["均值滤波", "中值滤波", "高斯滤波", "双边滤波"], "smoothing_ops")
        kernel = st.slider("滤波器大小", 1, 31, 5, step=2)
        processed_img = smoothing_operations(img, smoothing, kernel)

    elif operation == "图像分割":
        segmentation = render_buttons(["边缘法分割", "阈值法分割", "区域法分割", "K-means"], "segmentation_ops")
        processed_img = image_segmentation(img, segmentation)

with col2:
    st.markdown("### 原始图片")
    st.image(img, caption="原始图片", use_container_width=True)

with col3:
    st.markdown("### 处理后图片")

    if processed_img is not None:
        st.image(processed_img, caption="处理后图片", use_container_width=True)

        # 添加下载选项
        result_image = Image.fromarray(processed_img)
        buffer = io.BytesIO()
        result_image.save(buffer, format="PNG")
        byte_data = buffer.getvalue()

        st.download_button(
            label="下载处理后图片",
            data=byte_data,
            file_name="processed_image.png",
            mime="image/png",
        )

# 页面底部声明
st.markdown('<div class="footer">© 2024 图像处理平台. All Rights Reserved.</div>', unsafe_allow_html=True)




# # 1. 图像几何运算
# def geometric_operations(image, operation, params):
#     if operation == "平移":
#         M = np.float32([[1, 0, params['dx']], [0, 1, params['dy']]])  # 平移矩阵
#         return cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
#     elif operation == "旋转":
#         center = (image.shape[1] // 2, image.shape[0] // 2)
#         M = cv2.getRotationMatrix2D(center, params['angle'], 1)  # 旋转角度
#         return cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
#     elif operation == "缩放":
#         return cv2.resize(image, None, fx=params['fx'], fy=params['fy'])  # 缩放因子
#     elif operation == "镜像":
#         return cv2.flip(image, 1)  # 水平镜像
#     return image
#
#
# # 2. 图像对比度增强
# def contrast_enhancement(image, operation):
#     if operation == "灰度变换":
#         return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     elif operation == "直方图均衡化":
#         gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#         return cv2.equalizeHist(gray)
#     return image
#
#
# # 3. 图像平滑处理
# def smoothing_operations(image, operation, kernel_size):
#     if operation == "均值滤波":
#         return cv2.blur(image, (kernel_size, kernel_size))  # 均值滤波
#     elif operation == "中值滤波":
#         return cv2.medianBlur(image, kernel_size)  # 中值滤波
#     elif operation == "频率低通滤波器":
#         gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # 转为灰度图像
#         dft = cv2.dft(np.float32(gray), flags=cv2.DFT_COMPLEX_OUTPUT)  # 傅里叶变换
#         dft_shift = np.fft.fftshift(dft)
#         rows, cols = gray.shape
#         crow, ccol = rows // 2, cols // 2
#         dft_shift[crow - 30:crow + 30, ccol - 30:ccol + 30] = 0  # 频率域滤波
#         img_back = cv2.idft(np.fft.ifftshift(dft_shift))
#         img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])  # 获取幅度谱
#         img_back = np.uint8(cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX))  # 归一化显示
#         return img_back
#     return image
#
#
# # 4. 图像分割
# def image_segmentation(image, method):
#     if method == "边缘法分割":
#         edges = cv2.Canny(image, 100, 200)
#         edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
#         edges_colored[edges == 255] = [0, 255, 0]
#         return edges_colored
#     elif method == "阈值法分割":
#         gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#         _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
#         return cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
#     elif method == "区域法分割":
#         gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#         _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
#         contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#         segmented_image = np.zeros_like(image)
#         for i, contour in enumerate(contours):
#             color = [np.random.randint(0, 255) for _ in range(3)]
#             cv2.drawContours(segmented_image, [contour], -1, color, -1)
#         return segmented_image
#     return image


# # 页面设置
# st.set_page_config(page_title="图像处理平台", layout="wide")
#
# # 自定义 CSS 样式
# st.markdown(
#     """
#     <style>
#     .main {
#         padding: 20px;
#     }
#
#     /* 标题样式 */
#     .title {
#         text-align: center;
#         font-size: 2.5em;
#         font-weight: bold;
#         color: #4CAF50;
#         margin-bottom: 20px;
#     }
#
#     /* 操作菜单 */
#     .menu {
#         background-color: #f9f9f9;
#         padding: 20px;
#         border-radius: 10px;
#         box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
#     }
#
#     /* 图像展示 */
#     .image {
#         text-align: center;
#         margin: auto;
#     }
#
#     .image img {
#         max-width: 90%;
#         border-radius: 10px;
#         box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1);
#     }
#
#     /* 底部说明 */
#     .footer {
#         text-align: center;
#         font-size: 0.9em;
#         color: gray;
#         margin-top: 20px;
#     }
#     </style>
#     """,
#     unsafe_allow_html=True,
# )
#
# # 标题
# st.markdown('<div class="title">图像处理平台 - WCH</div>', unsafe_allow_html=True)
#
# # 上传图片
# uploaded_file = st.file_uploader("", type=["jpg", "png", "bmp"])
#
# if uploaded_file:
#     img = Image.open(uploaded_file)
#     img = np.array(img)
#
#     # 创建三列布局：操作菜单在最左，原始图片在中，处理后图片在最右
#     col1, col2, col3 = st.columns([1, 2, 2])
#
#     with col1:
#         #st.markdown('<div class="menu">', unsafe_allow_html=True)
#         st.markdown("### 操作菜单")
#         operation = st.selectbox("选择操作", ["几何运算", "对比度增强", "平滑处理", "图像分割"])
#
#         params = {}
#         processed_img = img  # 默认使用原始图片
#         if operation == "几何运算":
#             sub_operation = st.radio("几何操作", ["平移", "旋转", "缩放", "镜像"])
#             if sub_operation == "平移":
#                 params["dx"] = st.slider("X 轴平移", -500, 500, 100)
#                 params["dy"] = st.slider("Y 轴平移", -500, 500, 100)
#                 processed_img = geometric_operations(img, "平移", params)  # 调用函数并更新结果
#             elif sub_operation == "旋转":
#                 params["angle"] = st.slider("旋转角度", -180, 180, 45)
#                 processed_img = geometric_operations(img, "旋转", params)
#             elif sub_operation == "缩放":
#                 params["fx"] = st.slider("X 轴缩放", 0.1, 2.0, 1.0)
#                 params["fy"] = st.slider("Y 轴缩放", 0.1, 2.0, 1.0)
#                 processed_img = geometric_operations(img, "缩放", params)
#             elif sub_operation == "镜像":
#                 processed_img = geometric_operations(img, "镜像", {})
#
#         elif operation == "对比度增强":
#             sub_operation = st.radio("增强方法", ["灰度变换", "直方图均衡化"])
#             if sub_operation == "灰度变换":
#                 processed_img = contrast_enhancement(img, "灰度变换")
#             elif sub_operation == "直方图均衡化":
#                 processed_img = contrast_enhancement(img, "直方图均衡化")
#
#         elif operation == "平滑处理":
#             sub_operation = st.radio("平滑方法", ["均值滤波", "中值滤波", "频率低通滤波器"])
#             if sub_operation in ["均值滤波", "中值滤波"]:
#                 kernel_size = st.slider("滤波器大小", 3, 15, 5, step=2)
#                 processed_img = smoothing_operations(img, sub_operation, kernel_size)
#             elif sub_operation == "频率低通滤波器":
#                 processed_img = smoothing_operations(img, "频率低通滤波器", None)
#
#         elif operation == "图像分割":
#             sub_operation = st.radio("分割方法", ["边缘法分割", "阈值法分割", "区域法分割"])
#             processed_img = image_segmentation(img, sub_operation)
#
#     with col2:
#         st.markdown("### 原始图片")
#         st.image(img, caption="原始图片", use_container_width=True)
#
#     with col3:
#         st.markdown("### 处理后图片")
#         st.image(processed_img, caption="处理后图片", use_container_width=True)
#
# # 页面底部
# st.markdown('<div class="footer">© 2024 图像处理平台. All Rights Reserved.</div>', unsafe_allow_html=True)
