import re
import os


def extract_opcodes(input_asm_path):
    opcodes = []
    instr_pattern = re.compile(
        r'^\.text:[0-9A-F]+\s+(?:[0-9A-F]{2}\s+)+([a-zA-Z]+)\b',
        re.IGNORECASE
    )
    data_pattern = re.compile(
        r'^\s*\.text:[0-9A-F]+\s+.*?(?:dd|db|dw|\+)\b',
        re.IGNORECASE
    )

    try:
        with open(input_asm_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith(';'):
                    continue
                if data_pattern.search(line):
                    continue
                match = instr_pattern.match(line)
                if match:
                    opcodes.append(match.group(1).lower())
    except Exception as e:
        print(f"Error: {str(e)}")
        return []

    return opcodes


def save_opcodes_to_txt(opcodes, output_path):
    try:
        with open(output_path, 'w') as f:
            f.write(' '.join(opcodes))
        print(f"Saved to {output_path}")
    except Exception as e:
        print(f"Save error: {str(e)}")


if __name__ == "__main__":
    asm_file = "GEs5ryOXjLYVdAM7qK8z.asm"
    if not os.path.exists(asm_file):
        print(f"File {asm_file} not found!")
        exit()

    opcodes = extract_opcodes(asm_file)
    txt_file = os.path.splitext(asm_file)[0] + "_opcodes.txt"
    save_opcodes_to_txt(opcodes, txt_file)