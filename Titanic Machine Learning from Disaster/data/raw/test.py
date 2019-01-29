
import csv
import os
  
#表示該程式所在路徑
path1=os.path.dirname(os.path.realpath( __file__ ))


# 開啟 CSV 檔案
f =  open("%s/test.csv"%path1,'r')


# 以迴圈輸出每一列
for row in csv.reader(f):
    print(row)

f.close();
"""

import csv
f = open('example.csv', 'r')
for row in csv.reader(f):
    print row
f.close()
"""