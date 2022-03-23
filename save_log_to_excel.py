import xlwt


def set_style(name, height, bold=False):
    style = xlwt.XFStyle()
    font = xlwt.Font()
    font.name = name
    font.bold = bold
    font.color_index = 4
    font.height = height
    style.font = font
    return style


def init_excel():
    workbook = xlwt.Workbook()
    sheet1 = workbook.add_sheet('test', cell_overwrite_ok=True)
    # 通过excel保存训练结果（训练集验证集loss，学习率，训练时间，总训练时间）
    row0 = ["CLS", "Acc", "TP", "TN", "FP", "FN", "NUM_P", "NUM_N", "F1"]
    print('写入test_excel')
    for i in range(0, len(row0)):
        sheet1.write(0, i, row0[i], set_style('Times New Roman', 220, True))
    return workbook, sheet1


def write(workbook, sheet, path):
    f = open(path)
    line = f.readline()
    while line:
        if line[:3] == "Cls":
            line = line.split(",")
            print(line)
            if len(line) > 2:
                idx = line[0].split(":")[-1]
                sheet.write(int(idx) + 1, 0, idx)
                for i in range(1, len(line)):
                    num = line[i].split(":")[-1]
                    sheet.write(int(idx) + 1, i, num)
            else:
                idx = line[0].split(": ")[-1]
                sheet.write(int(idx) + 1, 0, idx)
                F1 = line[1].split(":")[-1]
                sheet.write(int(idx) + 1, 8, F1)
        line = f.readline()
    f.close()


if __name__ == "__main__":
    workbook, sheet = init_excel()
    path = "/home/liu/gyq/workshop/PrewittSingleBranch/log/test_2022_01_19_full.log"
    write(workbook, sheet, path)
    workbook.save(path[:-4]+"_excel.csv")
