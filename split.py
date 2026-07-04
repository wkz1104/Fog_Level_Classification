import pandas as pd
import os
import shutil
from datetime import datetime

# 读取 CSV 文件
df = pd.read_csv(r'C:\Users\nanak\Desktop\雾观.csv',encoding='gbk')

# 获取 1 分钟平均能见度小于 500 的时间数据
target_times = df[df['能见度'] < 500]['时间'].tolist()
print(f"符合条件的时间数据数量: {len(target_times)}")

# 源图像文件夹路径
source_folder = r'F:\高速实景监测数据\乳源雾观大桥'
# 目标图像文件夹路径
target_folder = r'C:\Users\nanak\Desktop\many_scene_fog_data'

# 确保目标文件夹存在
if not os.path.exists(target_folder):
    os.makedirs(target_folder)
    print(f"已创建目标文件夹: {target_folder}")

# 获取源文件夹及所有子文件夹中符合条件的图像文件
source_files = []
# 使用os.walk遍历所有子文件夹
for root, dirs, files in os.walk(source_folder):
    for file in files:
        # 筛选.jpg格式且14位数字的文件（14位数字 + .jpg = 18个字符）
        if (file.endswith('.jpg') and 
            len(file) == 18 and 
            file[:-4].isdigit()):  # 文件名前缀是纯数字
            # 存储文件的完整路径
            source_files.append(os.path.join(root, file))

print(f"源文件夹及其子文件夹中符合格式的图像文件数量: {len(source_files)}")

# 定义可能的时间格式列表（根据实际数据扩展）
time_formats = [
    '%Y-%m-%d %H:%M:%S.%f',  # 带毫秒的完整格式
    '%Y-%m-%d %H:%M:%S',     # 带秒的格式
    '%Y-%m-%d %H:%M',        # 不带秒的格式
    '%Y/%m/%d %H:%M:%S.%f',  # 斜杠分隔 + 毫秒
    '%Y/%m/%d %H:%M:%S',     # 斜杠分隔 + 秒
    '%Y/%m/%d %H:%M',        # 斜杠分隔 + 不带秒（你的数据格式）
    '%Y-%m-%d %H:%M:%S.%f',
    '%m/%d/%Y %H:%M',        # 月/日/年格式
]

# 遍历时间数据，查找前12位匹配的图像
for index, time_str in enumerate(target_times):
    print(f"\n正在处理第 {index + 1} 个时间点: {time_str}")
    dt = None
    # 尝试多种时间格式解析
    for fmt in time_formats:
        try:
            dt = datetime.strptime(str(time_str), fmt)
            break  # 解析成功则跳出循环
        except ValueError:
            continue  # 解析失败则尝试下一种格式
    
    if dt is None:
        print(f"❌ 所有时间格式都无法解析: {time_str}，跳过该时间点")
        continue
    
    # 生成12位前缀（年月日时分）
    target_prefix = dt.strftime('%Y%m%d%H%M')
    print(f"需要匹配的前12位前缀: {target_prefix}")

    # 筛选匹配的文件
    matched_files = [f for f in source_files 
                    if os.path.basename(f)[:12] == target_prefix]

    if not matched_files:
        print(f"❌ 未找到前12位为 {target_prefix} 的图像文件")
        continue

    # 复制所有匹配的文件
    for file_path in matched_files:
        file_name = os.path.basename(file_path)
        target_path = os.path.join(target_folder, file_name)
        
        # 处理文件名重复
        counter = 1
        while os.path.exists(target_path):
            name, ext = os.path.splitext(file_name)
            target_path = os.path.join(target_folder, f"{name}_{counter}{ext}")
            counter += 1
            
        try:
            shutil.copyfile(file_path, target_path)
            print(f"✅ 已复制匹配文件: {file_name}（来自: {os.path.dirname(file_path)}）")
        except Exception as e:
            print(f"❌ 复制文件 {file_name} 时出错: {e}")