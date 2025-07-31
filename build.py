import os
import shutil
import subprocess
import sys

def get_path(relative_path):
    """获取资源的绝对路径，兼容开发环境和PyInstaller环境"""
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)

def build():
    """使用PyInstaller打包应用"""
    # 清理旧的构建文件
    for folder in ['build', 'dist']:
        if os.path.exists(folder):
            shutil.rmtree(folder)

    # PyInstaller 命令
    python_executable = os.path.join('.venv', 'Scripts', 'python.exe')

    pyinstaller_command = [
        python_executable,
        '-m', 'PyInstaller',
        '--name=ML_Data_Analysis_Platform',
        '--onefile',
        '--add-data', f'{get_path("frontend")}{os.pathsep}frontend',
        '--add-data', f'{get_path("ml_test_data")}{os.pathsep}ml_test_data',
        '--hidden-import=torch',
        '--hidden-import=sklearn.utils._weight_vector',
        '--add-data', f"{os.path.join('.venv', 'Lib', 'site-packages', 'xgboost')}{os.pathsep}xgboost",
        '--add-data', f"{os.path.join('.venv', 'Lib', 'site-packages', 'lightgbm')}{os.pathsep}lightgbm",
        'backend/main.py'
    ]

    print(f"Running command: {' '.join(pyinstaller_command)}")

    try:
        # 在虚拟环境中执行 PyInstaller
        subprocess.run(pyinstaller_command, check=True, shell=False)
        print("\nPyInstaller build completed successfully!")
        print(f"Executable is located in: {os.path.abspath('dist')}")

    except subprocess.CalledProcessError as e:
        print(f"\nPyInstaller build failed with error: {e}")
    except FileNotFoundError:
        print(f"\nError: '{python_executable}' not found.")
        print("Please ensure you have created a virtual environment in the '.venv' directory and installed the requirements.")


if __name__ == '__main__':
    build()
