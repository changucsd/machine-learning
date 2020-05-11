from DecisionTree import DecisionTree

#
# authors:
# Ziqiao Gao 2157371827
# He Chang 5670527576
# Fanlin Qin 5317973858
#

data = []
def readFile():
    readFilePath = 'dt_data.txt'
    readFile = open(readFilePath, 'r')
    lines = readFile.readlines()
    count = 0
    for line in lines:
        if count <= 1:
            count += 1
            continue
        line = line[:-2]
        l = line.split(": ")
        attrs = l[1].split(", ")
        # (Occupied, Price, Music, Location, VIP, Favorite Beer, Enjoy)
        row = {
            "Occupied": attrs[0],
            "Price": attrs[1],
            "Music": attrs[2],
            "Location": attrs[3],
            "VIP": attrs[4],
            "Favorite Beer": attrs[5],
            "Enjoy": attrs[6],
        }
        data.append(row)


if __name__ == '__main__':
    readFile()
    # print(data)
    dt = DecisionTree(data)
    root = dt.build(data,"")
    dt.root = root
    testdata = {
        "Occupied": "Moderate",
            "Price": "Cheap",
            "Music": "Loud",
            "Location": "City-Center",
            "VIP": "No",
            "Favorite Beer": "No"
    }
    print(dt.predict(testdata))
    dt.printTree()

