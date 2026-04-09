import requests
import zipfile
import io
import os
import subprocess
import sys

# 1. 下载 Zip 文件
url = "https://github.com/HouGroup/InterOptimus/archive/refs/heads/master.zip"
print(f"正在下载: {url}...")
r = requests.get(url)

if r.status_code == 200:
    # 2. 解压到当前项目目录
    print("下载成功，正在解压...")
    z = zipfile.ZipFile(io.BytesIO(r.content))
    extract_path = os.getcwd() # 解压到当前工作目录
    z.extractall(extract_path)
    
    # 找到解压后的文件夹名 (通常是 InterOptimus-master)
    pkg_dir = os.path.join(extract_path, "InterOptimus-master")
    
    if os.path.exists(pkg_dir):
        # 3. 使用 pip 安装
        print(f"正在从目录安装: {pkg_dir}")
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg_dir])
        print("安装完成！")
    else:
        print(f"解压后未找到 {pkg_dir} 文件夹，请检查解压内容。")
else:
    print(f"下载失败，状态码: {r.status_code}")