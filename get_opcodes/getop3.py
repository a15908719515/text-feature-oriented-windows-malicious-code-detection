import re
import os


def getOpcodeSequence(filename):
    opcode_seq = []
    # p = re.compile(r'\s([a-fA-F0-9]{2}\s)+\s*([a-z]+)')
    p = re.compile(r'\s+((?:[a-fA-F0-9]{2}\s*)+)\s+([a-z]+)')
    exclude_ops = {"align", "db", "byte","dd","dw","off","offset",}
    with open(filename, mode="r", encoding='utf-8', errors='ignore') as f:
        for line in f:
            if line.startswith((".text","CODE")):
                m = re.findall(p, line)
                if m:
                    opc = m[0][1]
                    if opc not in exclude_ops:
                        opcode_seq.append(opc)
    return opcode_seq


path = r"E:\pycharmcode\get_opcodes"
files = os.listdir(path)

for file in files:
    filename = os.path.join(path, file)
    temp = getOpcodeSequence(filename)
    if not temp:  # 跳过空结果
        continue

    with open(r'E:\pycharmcode\get_opcodes\asm.txt', 'a', encoding='UTF8') as f:
        line =' '.join(temp)  # 文件名+空格+操作码序列
        f.write(line + '\n')