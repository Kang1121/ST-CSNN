import os

# start = {'45a_1': 1, '45a_2': 1862, '45a_3': 3599, '45a_4': 5401, '45b_1': 1, '45b_2': 2126, '45b_3': 4246}
# end = {'45a_1': 1865, '45a_2': 3601, '45a_3': 5403, '45a_4': 7203, '45b_1': 2126, '45b_2': 4246, '45b_3': 6369}
# list = ['45a_1', '45a_2', '45a_3', '45a_4', '45b_1', '45b_2', '45b_3']
# for i in range(len(list)):
#     f = open("/home/kang/numbers/" + list[i] + ".txt", "w")
#     for j in range(start[list[i]], end[list[i]]):
#         f.write('/home/kang/frames/' + list[i].split('_')[0] + '/' + str(j) + '.jpg ')
#     f.close()


f = open("/home/kang/numbers/ppng.txt", "w")
for j in range(1, 44):
    f.write('/home/kang/桌面/ppng' + '/' + str(j) + '.png ')