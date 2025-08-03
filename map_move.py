import os
import shutil

def copy_and_rename_images(source_dir, dest_dir):
    """
    将源文件夹中的所有.png图片复制到目标文件夹，并在文件名前添加"complex_"
    
    参数:
        source_dir: 源文件夹路径
        dest_dir: 目标文件夹路径
    """
    # 检查源文件夹是否存在
    if not os.path.exists(source_dir):
        print(f"错误: 源文件夹 '{source_dir}' 不存在")
        return
    
    # 创建目标文件夹（如果不存在）
    os.makedirs(dest_dir, exist_ok=True)
    print(f"目标文件夹准备就绪: {dest_dir}")
    
    # 计数处理的文件数量
    processed_count = 0
    
    # 遍历源文件夹中的所有文件
    for filename in os.listdir(source_dir):
        # 检查是否是.png文件
        if filename.lower().endswith('.png'):
            # 构建完整的源文件路径
            source_path = os.path.join(source_dir, filename)
            
            # 确保是文件而不是文件夹
            if os.path.isfile(source_path):
                # 构建新的文件名和目标路径
                new_filename = f"complex_{filename}"
                dest_path = os.path.join(dest_dir, new_filename)
                
                try:
                    # 复制文件
                    shutil.copy2(source_path, dest_path)  # 使用copy2保留元数据
                    processed_count += 1
                    print(f"已复制: {filename} -> {new_filename}")
                except Exception as e:
                    print(f"复制 {filename} 时出错: {str(e)}")
    
    print(f"操作完成，共处理 {processed_count} 个文件")

if __name__ == "__main__":
    # 源文件夹路径
    source_directory = r"/data/kjt_data/code/DARE/maps_train"
    
    # 目标文件夹路径，请根据需要修改
    destination_directory = r"/data/kjt_data/code/vitr/maps"  # 替换为你的目标路径
    
    # 执行复制和重命名操作
    copy_and_rename_images(source_directory, destination_directory)
