import os
import numpy as np


def preprocess(name):
    # f = open("/media/kang/90C42234C4221CCA/Users/yk/Desktop/0b_1.txt", "r")
    f = open("C:/Users/yk/Desktop/data/" + name + ".txt", "r")
    lines = []
    for line in f:
        lines.append(line)
    # print(len(lines))
    f.close()
    # print(lines)

    list = []
    countspace = 0
    countnumber = 0

    for line in lines:
        if line == '\n':
            if countnumber != 0:
                # output countnumber to list value
                list.append(countnumber)
                countnumber = 0
            else:
                countspace += 1
        else:
            if countspace >= 1:
                # output countspace to list index
                for i in range(countspace):
                    list.append(0)
                countspace = 0
            countnumber += 1
    # print(len(list))
    # print(list)
    s = []
    # for i in range(1, len(list)+1):
    #     s.append(sum(list[:i]))

    # print(s)

    f = open("C:/Users/yk/Desktop/data/" + name + "_after.txt", "r")
    # f = open("/media/kang/90C42234C4221CCA/Users/yk/Desktop/0b_1_after.txt", "r")
    lines = []
    for line in f:
        lines.append(line)
    # print(len(lines))
    f.close()

    pos, box = np.zeros((len(lines), 3)), np.zeros((len(lines), 4))
    i = 0
    for line in lines:
        x1 = float(line.split(' ')[0])
        y1 = float(line.split(' ')[1])
        # print(line, line.split(' ')[2])
        x2 = float(line.split(' ')[2])
        y2 = float(line.split(' ')[3])
        if (x1 < 10) | (x2 > 1590) | (y1 < 10) | (y2 > 1190):
            # list 对应位置 - 1
            box[i][0], box[i][1], box[i][2], box[i][3] = -1, -1, -1, -1
            pos[i][0], pos[i][1], pos[i][2] = -1, -1, -1
            i += 1
        else:
            box[i][0], box[i][1], box[i][2], box[i][3] = x1, y1, x2, y2
            pos[i][0], pos[i][1], pos[i][2] = int(int((x2 + x1) * 0.5)), int((y2 + y1) * 0.5), int(np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2) * 0.5)
            i += 1
    # print(pos.shape)
    list_pos = []
    list_box = []
    tmp_pos = []
    tmp_box = []

    for i in range(len(list)):
        if list[i] == 0:
            list_pos.append(None)
            list_box.append(None)
        else:
            for j in range(list[i]):
                # print(list[i])
                tmp_pos.append(pos[sum(list[0:i]) + j])
                tmp_box.append(box[sum(list[0:i]) + j])
            list_pos.append(tmp_pos)
            list_box.append(tmp_box)
            tmp_pos = []
            tmp_box = []
    # print(len(list_box))
    # print(list_box)
    return list_pos, list_box, list


# pos, box, list = preprocess()

