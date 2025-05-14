import pandas as pd

# 讀取 Excel 文件
input_file = '/home/bl515-ml/Documents/shaio_jie/sma/Kpop_challenge_analyze/data/K-POP藝人清單.xlsx'
output_file = '/home/bl515-ml/Documents/shaio_jie/sma/Kpop_challenge_analyze/data/K-POP藝人清單.csv'

# 讀取 Excel 文件
df = pd.read_excel(input_file, sheet_name='idols')

# 將數據保存為 CSV 文件
df.to_csv(output_file, index=False, encoding='utf-8-sig')

print(f'轉換完成！CSV 文件已保存至: {output_file}') 