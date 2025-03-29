import re
import os

pattern = r'''
        ^(?:\.text|CODE)              # 匹配.text或CODE开头
        :[0-9A-F]+                    # 地址部分
        (                             # 机器码捕获组
          (?:\s+[0-9A-F]{2}){1,}      # 至少一个两字节机器码
        )
        \s+                           # 分隔空格
        ([a-z]+)                      # 操作码捕获组
        \b                            # 单词边界确保完整操作码
    '''

def getOpcodeSequence(filename):
    """增强版操作码提取，精确过滤无效数据"""
    opcode_seq = []
    # 优化后的正则表达式（严格匹配两字节机器码格式）

    p = re.compile(pattern, re.VERBOSE | re.IGNORECASE)

    exclude_ops = {"align", "db", "byte", "dd", "dw", "ptr"}

    with open(filename, mode="r", encoding='utf-8', errors='ignore') as f:
        for line in f:
            if line.startswith((".text", "CODE")):
                m = p.search(line)
                if m:
                    # 验证机器码有效性
                    machine_code = m.group(1).strip().split()
                    if all(len(code) == 2 for code in machine_code):
                        opc = m.group(2).lower()
                        if opc not in exclude_ops:
                            opcode_seq.append(opc)
    return opcode_seq


# 测试用例验证
test_cases = [
    ("CODE:00483C4C 33 C8 xor ecx, eax", ["xor"]),  # 正常情况
    ("CODE:00483C4C off 33 C8 invalid_op", []),  # 包含非法机器码
    (".text:00401000 FF mov edi, edi", []),  # 单字节机器码
    ("CODE:00401000 C3 retn", ["retn"]),  # 单机器码
    ("CODE:00401000 8D 45 F0 lea eax, [ebp-10h]", ["lea"]),  # 多机器码
]

for line, expected in test_cases:
    match = re.search(pattern, line, re.VERBOSE | re.IGNORECASE)
    if match:
        machine_code = match.group(1).strip().split()
        valid = all(len(code) == 2 for code in machine_code)
        opc = match.group(2).lower() if valid else None
    else:
        opc = None
    print(f"输入: {line}")
    print(f"提取结果: {opc} | 预期: {expected}")
    print("---")

# 文件处理（保持原有逻辑）
input_dir = r"E:\pycharmcode\get_opcodes"
output_dir = r"E:\pycharmcode\get_opcodes"

os.makedirs(output_dir, exist_ok=True)

for filename in os.listdir(input_dir):
    if not filename.endswith(".asm"):
        continue

    input_path = os.path.join(input_dir, filename)
    opcodes = getOpcodeSequence(input_path)

    if not opcodes:
        continue

    output_filename = os.path.splitext(filename)[0] + ".txt"
    output_path = os.path.join(output_dir, output_filename)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(' '.join(opcodes))