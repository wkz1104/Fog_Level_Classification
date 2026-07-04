import os
import torch
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
from models import FogVisibilityModel  

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# 图像预处理（与训练时保持一致）
def get_transform():
    return transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])


# 加载模型
def load_model(model_path, num_classes=3):
    model = FogVisibilityModel(num_classes=num_classes, pretrained=False)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()  # 设置为评估模式
    return model


# 预测单帧（适配图片/视频帧/摄像头帧）
def predict_single_frame(model, frame, class_names=['0_no_fog', '1_light_fog', '2_heavy_fog']):
    """
    通用帧预测函数：支持OpenCV格式的帧（BGR）
    """
    # 转换为PIL图像（RGB）
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).convert('RGB')

    # 预处理
    transform = get_transform()
    input_tensor = transform(image).unsqueeze(0)  # 添加批次维度
    input_tensor = input_tensor.to(device)

    # 预测
    with torch.no_grad():  # 关闭梯度计算
        outputs, _ = model(input_tensor)
        _, preds = torch.max(outputs, 1)
        # 计算置信度
        confidence = torch.softmax(outputs, dim=1)[0][preds.item()].item() * 100

    # 获取预测结果
    predicted_class = class_names[preds.item()]
    return predicted_class, confidence


# 预测单张图像函数（复用predict_single_frame）
def predict_single_image(model, image_path, class_names=['0_no_fog', '1_light_fog', '2_heavy_fog']):
    # 加载图像（OpenCV格式）
    frame = cv2.imread(image_path)
    if frame is None:
        raise ValueError(f"无法读取图像文件：{image_path}")

    # 调用通用帧预测函数
    predicted_class, confidence = predict_single_frame(model, frame, class_names)

    # 转换为RGB用于绘制结果
    original_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return original_image, predicted_class, confidence


# 在图像/帧上绘制预测结果（通用）
def draw_result(image, predicted_class, confidence):
    # 复制图像以避免修改原图
    result_image = image.copy()

    # 设置文字参数（按图像比例动态计算）
    font = cv2.FONT_HERSHEY_SIMPLEX
    # 按图像宽度的0.001比例设置字体大小（可根据需要调整）
    font_scale = image.shape[1] * 0.001
    # 按字体大小比例设置厚度，确保清晰
    thickness = max(2, int(font_scale * 2.5))

    # 根据预测类别设置文字和颜色
    if predicted_class == '0_no_fog':
        text = f"no_fog(>1000m): {confidence:.1f}%"
        color = (0, 255, 0)  # 绿色（RGB）
    elif predicted_class == '1_light_fog':
        text = f"light_fog(200-1000m): {confidence:.1f}%"
        color = (255, 255, 0)  # 黄色（RGB）
    else:  # '2_heavy_fog'
        text = f"heavy_fog(<200m): {confidence:.1f}%"
        color = (255, 0, 0)  # 红色（RGB）

    # 获取文字尺寸，用于背景框绘制
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]

    # 绘制半透明背景框（左上角1%位置）
    x1, y1 = int(image.shape[1] * 0.01), int(image.shape[0] * 0.01)
    x2, y2 = x1 + text_size[0] + 20, y1 + text_size[1] + 20
    cv2.rectangle(result_image, (x1, y1), (x2, y2), (0, 0, 0), -1)
    cv2.addWeighted(result_image, 0.8, result_image, 0.2, 0, result_image)

    # 在图像上添加文字
    cv2.putText(result_image, text, (x1 + 10, y1 + text_size[1] + 10),
                font, font_scale, color, thickness)

    return result_image


# 预测文件夹内所有图像
def predict_images_in_folder(model, folder_path, class_names=['0_no_fog', '1_light_fog', '2_heavy_fog']):
    # 自动创建结果保存目录
    result_dir = r'D:\bishe\fog\asdf'
    os.makedirs(result_dir, exist_ok=True)

    # 获取文件夹内所有图片路径（支持常见图像格式）
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif')
    image_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path)
                   if f.lower().endswith(image_extensions)]

    if not image_paths:
        print(f"警告：文件夹 {folder_path} 中没有找到图像文件")
        return

    # 遍历预测每张图像
    for image_path in image_paths:
        try:
            # 预测单张图像
            original_image, predicted_class, confidence = predict_single_image(model, image_path, class_names)

            # 绘制结果
            result_image = draw_result(original_image, predicted_class, confidence)

            # 保存结果
            image_filename = os.path.basename(image_path)
            filename_without_ext = os.path.splitext(image_filename)[0]
            output_filename = f"{filename_without_ext}_prediction.jpg"
            output_path = os.path.join(result_dir, output_filename)

            # 转换颜色空间并保存
            result_image_bgr = cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(output_path, result_image_bgr)
            print(f"预测完成：{image_path} → 保存至 {output_path}")

        except Exception as e:
            print(f"处理 {image_path} 时出错：{str(e)}")


# 预测视频（逐帧检测）
def predict_video(model, video_path, output_path=None, class_names=['0_no_fog', '1_light_fog', '2_heavy_fog'],
                  show_realtime=True):
    """
    视频逐帧预测
    :param video_path: 视频路径（0表示摄像头）
    :param output_path: 输出视频路径（None则不保存）
    :param show_realtime: 是否实时展示
    """
    # 打开视频/摄像头
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"错误：无法打开视频/摄像头 {video_path}")
        return

    # 获取视频属性（摄像头则用默认值）
    fps = int(cap.get(cv2.CAP_PROP_FPS)) if video_path != 0 else 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) if video_path != 0 else 640
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) if video_path != 0 else 480

    print(f"视频/摄像头信息：FPS={fps}, 分辨率={width}x{height}")

    # 初始化视频写入器（如果需要保存）
    out = None
    if output_path and video_path != 0:  # 摄像头不保存视频
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    try:
        frame_count = 0
        print("按 'q' 键退出检测")
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break  # 视频结束/摄像头断开

            frame_count += 1
            # 逐帧预测
            pred_class, conf = predict_single_frame(model, frame, class_names)

            # 绘制结果（转换为RGB绘制，再转回BGR显示/保存）
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result_frame_rgb = draw_result(frame_rgb, pred_class, conf)
            result_frame_bgr = cv2.cvtColor(result_frame_rgb, cv2.COLOR_RGB2BGR)

            # 保存帧到输出视频
            if out:
                out.write(result_frame_bgr)

            # 实时展示
            if show_realtime:
                cv2.imshow('Fog Detection (Press Q to Exit)', result_frame_bgr)
                # 按q退出
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        print(f"\n处理完成！共检测 {frame_count} 帧")

    finally:
        # 释放资源
        cap.release()
        if out:
            out.release()
        cv2.destroyAllWindows()


# 主函数（支持：无路径→摄像头 | 图片→单图预测 | 文件夹→批量预测 | 视频→逐帧预测）
def main(input_path=None, model_path='best_model2.pth', output_video=None):
    # 加载模型
    print(f"从 {model_path} 加载模型...")
    model = load_model(model_path)

    # 判断输入类型
    if input_path is None:
        # 无输入路径 → 调用摄像头实时检测
        print("未传入路径，启动摄像头实时检测...")
        predict_video(
            model=model,
            video_path=0,  # 0表示默认摄像头
            output_path=None,  # 摄像头不保存视频
            show_realtime=True
        )

    elif input_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
        # 输入是视频 → 逐帧预测视频
        print(f"正在预测视频: {input_path}...")
        # 自动生成输出视频路径（如果未指定）
        if not output_video:
            video_dir = r'D:\bishe\fog\asdf'
            os.makedirs(video_dir, exist_ok=True)
            video_name = os.path.basename(input_path)
            name, ext = os.path.splitext(video_name)
            output_video = os.path.join(video_dir, f"{name}_prediction{ext}")
        predict_video(
            model=model,
            video_path=input_path,
            output_path=output_video,
            show_realtime=True
        )

    elif os.path.isfile(input_path):
        # 输入是图片 → 单图预测
        print(f"正在预测图像: {input_path}...")
        original_image, predicted_class, confidence = predict_single_image(model, input_path)
        result_image = draw_result(original_image, predicted_class, confidence)

        # 保存结果
        result_dir = r'D:\bishe\fog\asdf'
        os.makedirs(result_dir, exist_ok=True)

        image_filename = os.path.basename(input_path)
        filename_without_ext = os.path.splitext(image_filename)[0]
        output_filename = f"{filename_without_ext}_prediction.jpg"
        output_path = os.path.join(result_dir, output_filename)

        result_image_bgr = cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_path, result_image_bgr)
        print(f"预测结果已保存至: {output_path}")

        # 显示结果
        plt.imshow(result_image)
        plt.axis('off')
        plt.show()

    elif os.path.isdir(input_path):
        # 输入是文件夹 → 批量预测图片
        print(f"正在预测文件夹 {input_path} 内所有图像...")
        predict_images_in_folder(model, input_path)

    else:
        print(f"错误：输入路径 {input_path} 不存在或不是文件/文件夹/视频")


if __name__ == '__main__':
    import argparse

    # 解析命令行参数（input_path设为可选）
    parser = argparse.ArgumentParser(description='雾天检测预测脚本（支持摄像头/单图/文件夹/视频）')
    parser.add_argument('input_path', type=str, nargs='?', default=None,
                        help='图像/视频/文件夹路径（不传则调用摄像头）')
    parser.add_argument('--model_path', type=str, default='best_model2.pth',
                        help='训练好的模型权重路径')
    parser.add_argument('--output_video', type=str, default=None,
                        help='输出标注视频的路径（仅视频预测时有效）')

    args = parser.parse_args()

    # 运行预测
    main(args.input_path, args.model_path, args.output_video)