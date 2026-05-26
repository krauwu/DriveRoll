import os
import ast
from concurrent.futures import ThreadPoolExecutor, as_completed

def find_py_files(root_dir):
    """Traverse directory and find all Python files"""
    py_files = []
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.py'):
                py_files.append(os.path.join(root, file))
    return py_files

def analyze_file(file_path):
    """Analyze a single file, extracting classes, functions, and global variables"""
    with open(file_path, "r", encoding='utf-8') as file:
        file_content = file.read()
    tree = ast.parse(file_content)
    
    classes = [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
    functions = [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]    
    return file_path, classes, functions

def analyze_projects(root_dir):
    """Process all files in parallel"""
    py_files = find_py_files(root_dir)
    results = []

    with ThreadPoolExecutor(max_workers=4) as executor:
        future_to_file = {executor.submit(analyze_file, file_path): file_path for file_path in py_files}
        for future in as_completed(future_to_file):
            file_path = future_to_file[future]
            try:
                results.append(future.result())
            except Exception as exc:
                print(f'{file_path} generated an exception: {exc}')
    return results

# Set project root directory path
root_dir = "./"
results = analyze_projects(root_dir)

result_dict = {}
# Print analysis results
for file_path, classes, functions in results:
    print(f"File: {file_path}")
    print(f"Classes: {classes}")
    print(f"Functions: {functions}")
    result_dict[file_path] = {}
    result_dict[file_path][classes] = 0
    result_dict[file_path][functions] = 0
    result_dict[file_path][functions] = 0
    result_dict[file_path]['count'] = 0

import json
  
out_file = open("myfile.json", "w") 
json.dump(result_dict, out_file, indent = 4) 
out_file.close() 