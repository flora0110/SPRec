# 檔案位置: train/utils.py
import os
import random
import numpy as np
import torch

def set_seed(seed):
    """
    固定所有必要的隨機種子以確保可重現性
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # 如果需要極致的確定性 (會犧牲一點效能)，可以打開下面這兩行
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False

def save_run_script_content(sh_file_path, output_file_path):
    """
    讀取 sh_file_path 的內容，並將其寫入到 output_file_path 所在的目錄中。
    檔名會儲存為 'run_config_backup.sh'
    """
    if not sh_file_path:
        return

    if not os.path.exists(sh_file_path):
        print(f"Warning: Shell script not found at {sh_file_path}")
        return

    try:
        # 取得存放目標檔案的資料夾
        dest_dir = os.path.dirname(output_file_path)
        os.makedirs(dest_dir, exist_ok=True)
        
        dest_path = os.path.join(dest_dir, "run_config_backup.sh")
        
        # 讀取並寫入
        with open(sh_file_path, 'r', encoding='utf-8') as f_src:
            content = f_src.read()
        
        with open(dest_path, 'w', encoding='utf-8') as f_dest:
            f_dest.write(content)
            
        print(f"Shell script has been backed up to: {dest_path}")
        
    except Exception as e:
        print(f"Warning: Failed to backup shell script. Error: {e}")