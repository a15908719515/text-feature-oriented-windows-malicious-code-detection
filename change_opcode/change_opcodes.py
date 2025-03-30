import os

# 操作码映射规则
OPCODE_MAPPING = {
    'je': 'je', 'jz': 'je', 'ja': 'je', 'jl': 'je', 'jb': 'je', 'jnz': 'je', 'jle': 'je',
    'ins': 'in', 'in': 'in',
    'outs': 'out', 'out': 'out',
    'fistp': 'fistp', 'fstp': 'fistp',
    'faddp': 'add', 'add': 'add', 'daa': 'add','adc': 'add',
    'sub': 'sub', 'sbb': 'sub', #减法
    'imul': 'imul', 'mul': 'imul',
    'fdivr': 'fdiv', 'fdiv': 'fdiv',
    'shr': 'shr', 'sar': 'shr',
    'sal': 'sal', 'shl': 'sal',
    'mov': 'mov', 'movzx': 'mov','movsd': 'mov',
}
#ja,jl,jb,jnz,jz,jle条件跳转


def process_opcodes(input_file, output_file):
    """处理单个文件的操作码替换和去重"""
    with open(input_file, 'r', encoding='utf-8') as f:
        original = f.read().split()

    # 执行操作码替换
    replaced = [OPCODE_MAPPING.get(op, op) for op in original]

    # 去除相邻重复
    filtered = []
    prev = None
    for op in replaced:
        if op != prev:
            filtered.append(op)
            prev = op

    # 保存处理结果
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(' '.join(filtered))


# 文件路径配置
input_path = "0B2RwKm6dq9fjUWDNIOa.txt"
output_path = "0B2RwKm6dq9fjUWDNIOaw_change.txt"

# 执行处理
process_opcodes(input_path, output_path)
print(f"处理完成，结果已保存至 {output_path}")