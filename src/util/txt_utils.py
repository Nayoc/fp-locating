def writeList(filename, data=[]):
    with open(filename, "w") as f:
        # 遍历列表中的每个元素并写入文件
        for item in data:
            f.write("%s\n" % item)

def writeStr(filename, data:str):
    with open(filename, "w") as f:
        # 遍历列表中的每个元素并写入文件
        f.write(data)
