import pandas as pd
import glob

def test_gen_table_heads():
    file_list0 = glob.glob('/ftproot/iowa_stations/*.csv', recursive=True)
    file_list1 = glob.glob('/ftproot/iowa_stations1/*.csv', recursive=True)
    file_list2 = glob.glob('/ftproot/iowa_stations2/*.csv', recursive=True)
    file_list3 = glob.glob('/ftproot/iowa_stations3/*.csv', recursive=True)
    file_list4 = glob.glob('/ftproot/iowa_stations4/*.csv', recursive=True)
    file_list5 = glob.glob('/ftproot/iowa_stations5/*.csv', recursive=True)
    file_list = file_list0 + file_list1 + file_list2 + file_list3 + file_list4 + file_list5
    head_dict = {}
    for file in file_list:
        file_name = file.split('/')[-1].split('.')[0]
        df = pd.read_csv(file, engine='c')
        head_dict[file_name] = df.columns.tolist()
    with open('heads.txt', 'w') as fp:
        for key in head_dict.keys():
            fp.writelines(key+':'+str(head_dict[key]))

