import os
import shutil
from datetime import datetime

def filter_and_copy_images(source_root, dest_dir, target_slices):
    """
    筛选并复制符合条件的图像文件
    
    参数:
        source_root: 源文件夹根目录
        dest_dir: 目标文件夹
        target_slices: 需要筛选的切片值列表
    """
    # 确保目标文件夹存在
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
        print(f"创建目标文件夹: {dest_dir}")
    
    # 遍历源文件夹中的所有日期子文件夹
    for date_folder in os.listdir(source_root):
        date_folder_path = os.path.join(source_root, date_folder)
        
        # 检查是否为文件夹且名称符合日期格式
        if not os.path.isdir(date_folder_path):
            continue
            
        try:
            # 尝试解析日期，验证文件夹名称是否为日期格式
            datetime.strptime(date_folder, "%Y-%m-%d")
        except ValueError:
            print(f"跳过非日期格式文件夹: {date_folder}")
            continue
            
        print(f"处理文件夹: {date_folder}")
        
        # 遍历日期文件夹中的所有文件
        for filename in os.listdir(date_folder_path):
            # 只处理jpg文件
            if not filename.lower().endswith('.jpg'):
                continue
                
            # 确保文件名长度足够进行切片操作
            if len(filename) < 16:  # 假设文件名格式为YYYYMMDDHHMMSS.jpg
                continue
                
            # 获取文件名的[8:12]切片部分（不含扩展名）
            file_base = os.path.splitext(filename)[0]
            slice_part = file_base[8:12]
            
            # 检查是否符合筛选条件
            if slice_part in target_slices:
                source_path = os.path.join(date_folder_path, filename)
                dest_path = os.path.join(dest_dir, filename)
                
                # 处理可能的文件名重复
                counter = 1
                while os.path.exists(dest_path):
                    name, ext = os.path.splitext(filename)
                    dest_path = os.path.join(dest_dir, f"{name}_{counter}{ext}")
                    counter += 1
                
                # 复制文件
                try:
                    shutil.copy2(source_path, dest_path)
                    print(f"复制文件: {filename} -> {dest_dir}")
                except Exception as e:
                    print(f"复制文件失败 {filename}: {str(e)}")

if __name__ == "__main__":
    # 源文件夹根目录
    source_root = r"F:\高速实景监测数据\汕头南澳岛"
    
    # 目标文件夹
    dest_dir = r"C:\Users\nanak\Desktop\汕头南澳岛整点"

    # 需要筛选的切片值
    target_slices = ['0000', '0100', '0200', '0300', '0400', '0500', '0600', '0700', '0800', '0900',
                     '1000', '1100', '1200', '1300', '1400', '1500', '1600', '1700', '1800', '1900',
                     '2000', '2100', '2200', '2300', '2400']
    
    # 执行筛选和复制操作
    filter_and_copy_images(source_root, dest_dir, target_slices)
    print("操作完成！")
    