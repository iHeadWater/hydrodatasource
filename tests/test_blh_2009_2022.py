import pandas as pd
import os

def ExcelFile_to_csv(ExcelFile = 'st_rain_c.xls',csv='st_rain_c.csv',encoding = 'gbk'):
    xls_file = pd.ExcelFile(ExcelFile)
    # 获取所有工作表的名称
    sheet_names = xls_file.sheet_names
    # 初始化一个空的DataFrame来存储所有数据
    all_data = pd.DataFrame()
    # 遍历每个工作表
    for sheet_name in sheet_names:
        # 如果第一个sheet，保存列名
        if sheet_name == sheet_names[0]:
            header = sheet_name
            sheet_data = pd.read_excel(xls_file, sheet_name, header=None)
        else:
            # 对于其他的sheet，从第一个sheet获取列名
            sheet_data = pd.read_excel(xls_file, sheet_name, header=None)

        # 将当前sheet的数据添加到all_data中
        all_data = pd.concat([all_data,sheet_data], ignore_index=True)
    # 上面处理的时候只进行了拼接，原本列名变成了第一行数据，这里需要转换一下
    all_data = all_data.rename(columns=all_data.iloc[0])
    all_data = all_data.iloc[1:].reset_index(drop=True)
    # 去除缺失值和无用的列
    all_data = all_data.loc[all_data['paraid'].notnull()]
    all_data = all_data[['paraid','paravalue','systemtime']]
    # 将结果保存为新的csv文件或者xlsx文件
    all_data.to_csv(csv, index=False,encoding = encoding)

def Excel_to_csv(xls = 'st_stbprp_b.xls',csv = 'st_stbprp_b.csv',encoding = 'gbk'):
    # 处理其他部分数据 读取xls文件
    st_stbprp_b = pd.read_excel(xls)
    st_stpara_r = pd.read_excel('st_stpara_r.xls')
    st_stbprp_b.to_csv(csv, index=False,encoding = encoding )
    st_stpara_r.to_csv('st_stpara_r.csv', index=False,encoding = encoding )

def Distribution_Data(csv_read1 = 'st_rain_c.csv',csv_read2 = 'st_stbprp_b.csv',csv_read3 = 'st_stpara_r.csv',csv_write='Rainfall_Distribution_Data_2009_2022.csv',encoding = 'gbk'):
    # 读取CSV文件
    st_rain_c = pd.read_csv(csv_read1,encoding = encoding)
    st_stbprp_b = pd.read_csv(csv_read2,encoding = encoding)
    st_stpara_r = pd.read_csv(csv_read3,encoding = encoding)
    # 使用stid连接st_stbprp_b和st_stpara_r
    merged1 = pd.merge(st_stbprp_b, st_stpara_r, on='stid')
    # 使用paraid连接merged1和st_rain_c
    merged2 = pd.merge(merged1, st_rain_c, on='paraid')
    # 选择需要的列
    new_table = merged2[['stid','stcd','stname', 'paravalue','systemtime','sttp','lgtd_y', 'lttd_y']]
    # 创建字典修改列名规范
    new_table = new_table.rename(columns={'stid':'STID', 'stcd':'STCD', 'stname':'STNM', 'paravalue':'DRP', 'systemtime':'TM', 'sttp':'STTP', 'lgtd_y':'LGTD', 'lttd_y':'LTTD'})
    # print(new_table)
    # 保存新表为CSV文件
    new_table.to_csv(csv_write, index=False,encoding =encoding)

# 这个函数不用了，没有效果
def STNM_to_STCD(csv_read = 'Rainfall_Distribution_Data_2009_2022.csv',csv_write='new_data.csv',encoding = 'gbk'):
    # 你的字典
    STNM_to_STCD = {
    }

    # 读取CSV文件
    table2 = pd.read_csv(csv_read ,encoding = encoding)

    # 新建一个空的 DataFrame 来保存修改后的数据
    new_table = pd.DataFrame()
    # 遍历每一行，如果 'STNM' 在字典中，则替换 'STCD' 列的值，否则设置为 '0'
    for index, row in table2.iterrows():
        if row['STNM'] in STNM_to_STCD:
            row['STCD'] = STNM_to_STCD[row['STNM']]
        else:
            row['STCD'] = '0'

    new_table = table2  # 将修改后的DataFrame赋值给new_table变量

    # 将修改后的 DataFrame 保存为新的 CSV 文件（请根据实际情况修改文件路径和文件名）
    new_table.to_csv(csv_write, index=False,encoding=encoding)
    #print(new_table)

def SubsetTable(csv = 'Rainfall_Distribution_Data_2009_2022.csv',encoding = 'gbk'):
    # 读取大表
    main_table = pd.read_csv(csv,encoding=encoding)
    # 获取所有唯一的STID值
    stids = main_table['STID'].unique()
    stnms = main_table['STNM'].unique()
    # 遍历每一个STID，创建新的DataFrame，然后保存为csv
    for stnm in stnms:
        # 使用STID过滤出对应的行
        subset_table = main_table[main_table['STNM'] == stnm]
        # 保存为csv，文件名包含STID
        subset_table.to_csv(f'{stnm}_subset.csv', index=False,encoding = encoding)





#ExcelFile_to_csv()
#Excel_to_csv()
#Distribution_Data()
#SubsetTable()
