import xlrd
import xlwt

# 打开文件 输入表格地址 ‘r’是为了解析带有中文字的文件名称
workbook = xlrd.open_workbook(r'test.xlsx')
# 指定excel表
sheet1_name = workbook.sheet_names()[0]
# 根据sheet索引或者名称获取sheet内容
sheet1 = workbook.sheet_by_name('Sheet1')
# 读取一个表格中的行或列的值， row为行， cols为列
row = sheet1.row_values(3)
cols = sheet1.col_values(2)
# 读取具体行和列的数据
a = sheet1.cell(1, 0).value.encode('utf-8')
b = sheet1.cell_value(1, 0).encode('utf-8')
c = sheet1.row(1)[0].value.encode('utf-8')

print(row[0])
print(cols)
print(a, b, c)

f = xlwt.Workbook()
sheet1 = f.add_sheet('sheet1')
sheet1.write(0, 0, 'I')
sheet1.write(0, 1, 'love')
sheet1.write(0, 2, 'bai')
f.save(r'test1.xls')
