import os
from tqdm import tqdm

import os
import chardet

def is_text_file(file_path):
    try:
        with open(file_path, 'rb') as f:
            bytes = f.read(512)
            if b'\0' in bytes:
                return False
        return True
    except Exception as e:
        return False

def detect_encoding(file_path):
    try:
        with open(file_path, 'rb') as f:
            raw_data = f.read()
            result = chardet.detect(raw_data)
            return result['encoding']
    except Exception as e:
        return None

def search_keyword_in_files(directory, keyword):
    encodings = ['utf-8', 'latin1', 'iso-8859-1', 'cp1252', 'ascii']
    
    for root, _, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            
            # 检查文件是否为文本文件
            if not is_text_file(file_path):
                print(f"Skipping binary file: {file_path}")
                continue
            
            file_read = False
            for encoding in encodings:
                try:
                    with open(file_path, 'r', encoding=encoding, errors='ignore') as f:
                        for line_num, line in enumerate(f, 1):
                            if keyword in line:
                                print(f"File: {file_path}, Line {line_num}: {line.strip()}")
                    file_read = True
                    break  # 成功读取后跳出编码尝试循环
                except (UnicodeDecodeError, IOError) as e:
                    continue  # 尝试下一个编码
            
            if not file_read:
                detected_encoding = detect_encoding(file_path)
                if detected_encoding:
                    try:
                        with open(file_path, 'r', encoding=detected_encoding, errors='ignore') as f:
                            for line_num, line in enumerate(f, 1):
                                if keyword in line:
                                    print(f"File: {file_path}, Line {line_num}: {line.strip()}")
                        file_read = True
                    except (UnicodeDecodeError, IOError) as e:
                        pass
                
            if not file_read:
                print(f"Could not read file {file_path} with any of the tested encodings or detected encoding")




# 使用示例
search_keyword_in_files('E:/Ivan/Astrophysics/mesa', 'darken')