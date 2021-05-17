from collections import OrderedDict
from pyexcel_xls import get_data
from pyexcel_xls import save_data


def read_xls_file():
    xls_data = get_data("test.xlsx")
    print("Get data type: ", type(xls_data))
    for sheet_n in xls_data.keys():
        print(sheet_n, ":", xls_data[sheet_n])
    return xls_data


def save_xls_file():
    data = OrderedDict()
    sheet_1 = []
    row_1_data = [u"ID", u"卷面", u"平时"]
    row_2_data = [4, 5, 6]

    sheet_1.append(row_1_data)
    sheet_1.append(row_2_data)
    data.update({u"这是成绩表": sheet_1})
    save_data("write_test.xls", data)


if __name__ == '__main__':
    save_xls_file()
    read_xls_file()
