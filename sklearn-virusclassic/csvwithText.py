import pandas as pd
import os

# 路径配置
csv_path = r"E:\kaggledata\subtrain\subtrainLabels.csv"
txt_dir = r"E:\kaggledata\subtraintxt"
output_path = r"E:\pycharmcode\sklearn-virusclassic\subtrain_with_text.csv"


def add_text_column(row):
    """
    为每行数据添加text列
    返回包含text列的完整行，如果文件不存在返回None
    """
    # 构造txt文件路径
    txt_filename = f"{row['Id']}.txt"
    txt_filepath = os.path.join(txt_dir, txt_filename)

    # 检查文件是否存在
    if not os.path.exists(txt_filepath):
        return None

    # 读取txt文件内容
    try:
        with open(txt_filepath, 'r', encoding='utf-8') as f:
            text_content = f.read().strip()
    except Exception as e:
        print(f"读取文件 {txt_filename} 失败: {str(e)}")
        return None

    # 返回包含text列的完整行
    return pd.Series({
        'Id': row['Id'],
        'Class': row['Class'],
        'text': text_content
    })


# 主处理流程
if __name__ == "__main__":
    # 读取原始CSV
    df = pd.read_csv(csv_path)
    # 添加text列并过滤无效行
    print("开始处理，原始数据量：", len(df))
    processed_df = df.apply(add_text_column, axis=1).dropna()
    print("处理后剩余数据量：", len(processed_df))

    # 保存结果
    processed_df.to_csv(output_path, index=False)
    print(f"处理完成，结果已保存至：{output_path}")