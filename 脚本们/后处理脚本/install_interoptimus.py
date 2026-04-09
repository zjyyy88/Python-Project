import os
import zipfile
import subprocess
import sys
import glob

def install_local_package():
    # 1. 定义路径
    venv_site_packages = r"C:\Users\ZHANGJY02\PycharmProjects\PythonProject\.venv\Lib\site-packages"
    zip_pattern = os.path.join(venv_site_packages, "InterOptimus-master.zip")
    
    # 查找 zip 文件
    zip_files = glob.glob(zip_pattern)
    if not zip_files:
        print(f"❌ 未找到文件: {zip_pattern}")
        # 尝试在当前目录查找
        zip_files = glob.glob("InterOptimus-master.zip")
        if not zip_files:
             print("❌ 在当前目录也没找到 zip 文件。")
             return

    zip_file_path = zip_files[0]
    print(f"✅ 找到安装包: {zip_file_path}")
    
    extract_to = os.path.join(venv_site_packages, "interoptimus_temp_install")
    
    # 2. 解压
    print(f"📦 正在解压到: {extract_to} ...")
    try:
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
    except zipfile.BadZipFile:
        print("❌ 错误: ZIP 文件已损坏。请重新下载。")
        return
    except Exception as e:
        print(f"❌ 解压出错: {e}")
        return

    # 3. 寻找 setup.py
    setup_dir = None
    for root, dirs, files in os.walk(extract_to):
        if "setup.py" in files or "pyproject.toml" in files:
            setup_dir = root
            break
    
    if setup_dir:
        print(f"✅ 找到安装文件在: {setup_dir}")
        print("🚀 开始安装...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "."], cwd=setup_dir)
            print("\n🎉 安装成功！")
            
            # 验证安装
            try:
                import InterOptimus
                print(f"验证导入成功: InterOptimus 版本可能为 {InterOptimus.__version__ if hasattr(InterOptimus, '__version__') else '未知'}")
            except ImportError:
                 # 有时候安装后需要重启解释器才能导入，但这步主要是验证 pip 是否报错
                 print("⚠️ 安装命令完成，但在当前脚本中导入失败，可能需要重启终端生效。")

        except subprocess.CalledProcessError as e:
            print(f"❌ 安装失败: {e}")
    else:
        print("❌ 在解压后的文件夹中未找到 setup.py 或 pyproject.toml。")
        print("📂 目录结构如下:")
        for root, dirs, files in os.walk(extract_to):
            level = root.replace(extract_to, '').count(os.sep)
            indent = ' ' * 4 * (level)
            print(f"{indent}{os.path.basename(root)}/")
            subindent = ' ' * 4 * (level + 1)
            for f in files:
                print(f"{subindent}{f}")

if __name__ == "__main__":
    install_local_package()
