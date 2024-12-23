import io
import cv2
import streamlit as st
from PIL import Image
import numpy as np
import base64


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
    
    /* 滑块的按钮颜色 */
    .stSlider > div[data-baseweb="slider"] > div > div > div {
        background: #007BFF !important; /* 蓝色 */
        border: 2px solid #0056b3 !important; /* 边框深蓝 */
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
st.markdown('<div class="title">数字图像处理算法集成系统</div>', unsafe_allow_html=True)

# 上传图片
uploaded_file = st.file_uploader("选择文件", type=["jpg", "png", "bmp"], label_visibility="collapsed")

if not uploaded_file:
    st.warning("请上传图片！")
    st.stop()

img = Image.open(uploaded_file)
img = np.array(img)

# 定义布局，包括一个分隔列
col1, divider, col2, col3 = st.columns([1, 0.05, 2, 2])

# 在 col1 中放置菜单
with col1:
    st.markdown("### 图像算法菜单")
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
        segmentation = render_buttons(["边缘法分割", "阈值法分割", "区域法分割", "K - means"], "segmentation_ops")
        processed_img = image_segmentation(img, segmentation)

# 在 divider 中放置竖线分割布局
with divider:
    st.markdown(
        """
        <style>
        .vertical-line {
            border-left: 2px solid #cccccc; /* 竖线颜色 */
            height: 500px;
            display: inline-block; /* 保持和内容对齐 */
        }
        </style>
        <div class="vertical-line"></div>
        """,
        unsafe_allow_html=True,
    )
# 在 col2 中放置原始图片
with col2:
    st.image(img, use_container_width=True)
    # 手动添加标题
    st.markdown(
        """
        <div style='text-align: center; font-size: 2em; font-weight: bold; color: #333333; margin-top: 10px;'>
        原始图片
        </div>
        """,
        unsafe_allow_html=True,
    )

# 在 col3 中放置处理后图片
with col3:
    if processed_img is not None:
        st.image(processed_img, use_container_width=True)

        # 动态生成标题
        operation_caption = "处理后图片"
        if operation == "图像的几何运算":
            operation_caption = f"{action} 操作"
        elif operation == "图像对比度增强":
            operation_caption = f"{enhancement} 操作"
        elif operation == "图像平滑处理":
            operation_caption = f"{smoothing} 操作"
        elif operation == "图像分割":
            operation_caption = f"{segmentation} 操作"

        # 手动添加动态标题
        st.markdown(
            f"""
            <div style='text-align: center; font-size: 2em; font-weight: bold; color: #333333; margin-top: 10px;'>
            {operation_caption}
            </div>
            """,
            unsafe_allow_html=True,
        )

        # 添加下载选项，靠右对齐
        result_image = Image.fromarray(processed_img)
        buffer = io.BytesIO()
        result_image.save(buffer, format="PNG")
        byte_data = buffer.getvalue()

        # 自定义按钮样式并靠右对齐
        st.markdown(
            """
            <div style="display: flex; justify-content: flex-end; padding-top: 10px;">
                <a href="data:file/png;base64,{data}" download="processed_image.png" style="
                text-decoration: none;
                padding: 10px 20px;
                color: white;
                background-color: #4CAF50;
                border: none;
                border-radius: 5px;
                text-align: center;
                font-size: 1em;
                font-weight: bold;
                cursor: pointer;">下载处理后图片</a>
            </div>
            """.format(data=base64.b64encode(byte_data).decode()),
            unsafe_allow_html=True,
        )

# 页面底部声明
st.markdown(
    """
    <style>
    .footer-author {
        text-align: center;
        font-size: 1.5em;
        font-weight: bold;
        color: #333333;
        margin-top: 50px;
    }
    </style>
    <div class="footer-author">222241807529 王晨好</div>
    """,
    unsafe_allow_html=True,
)
# 页面底部声明
st.markdown('<div class="footer">© 222241807529 王晨好. All Rights Reserved.</div>', unsafe_allow_html=True)





