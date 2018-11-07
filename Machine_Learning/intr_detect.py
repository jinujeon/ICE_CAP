def isIntrusion(colist, x, y):
    #colist = [x1, y1, x2, y2]
    #x2 > x1
    #직선 경계선을 만드는 두 개의 좌표를 각각 0.1 * 화면의 가로 혹은 세로 만큼 빼고 더해서 두 개의 직선을 더 만든다
    #객체의 꼭짓점이 두 직선 사이에 머무르는 시간이 5초 이상이 되면 경고
    warning = False
    newlist  = []
    for i in range(0, len(colist)):
            newlist.append(colist[i] + 0.1)

    for i in range(0, len(colist)):
            newlist.append(colist[i] - 0.1)
    #newlist = [colist[0] + 128, colist[1] + 72, colist[2] + 128, colist[3] + 72,
    #           colist[0] - 128, colist[1] - 72, colist[2] - 128, colist[3] - 72 ]
    #newlist = [0, 1, 2, 3, 4, 5, 6, 7]
    #newlist = [x11, y11, x12, y12, x21, y21, x22, y22]
    gr = (colist[3] - colist[1]) / (colist[2] - colist[0])  # 기울기
    #gr1 = (newlist[3] - newlist[1]) / (newlist[2] - newlist[0]) #기울기
    #gr2 = (newlist[7] - newlist[5]) / (newlist[6] - newlist[4])
    #gradient =  (colist[1][1] - colist[0][1]) / (colist[1][0] - colist[0][0])
    yinter = -1 * gr * colist[0] + colist[1]  # y절편
    #yinter1 = -1 * gr1 * newlist[0] + newlist[1] #y절편
    #yinter2 = -1 * gr2 * newlist[4] + newlist[5]
    yout = gr * x + yinter
    #yout1 = gr1 * x + yinter1
    #yout2 = gr2 * x + yinter2
    #if y < yout1 and y > yout2:
    if y > yout:
            warning = True
    return warning
