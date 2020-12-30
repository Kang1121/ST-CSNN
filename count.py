import os


dataPath = 'E:/1VERIWILD/images'
datas = []
agrees = [0]
idx = 1
file = open("C:/Users/yk/Desktop/count.txt")
lines = file.readlines(100)
file.close()
f = open("C:/Users/yk/Desktop/generate.txt", 'a+')
for alphaPath in lines:
    # datas[idx] = []
    #temp = 0
    for samplePath in os.listdir(os.path.join(dataPath, alphaPath.strip('\n'))):
        # filePath = os.path.join(dataPath, alphaPath, samplePath)
        # s = Image.open(filePath).rotate(agree).convert('RGB')
        # s = s.resize((224, 224), Image.ANTIALIAS)
        # datas[idx].append(s)
        f.write(str(alphaPath.strip('\n')) + '/' + str(samplePath.strip('.jpg')) + '\n')
    # datas.append(temp)
    # if temp > 98:
    #     print(idx)
    # idx += 1
f.close()
# for i in range(idx
# datas.sort()
# print(datas)
