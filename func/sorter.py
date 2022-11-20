def InsertionSort(lis):
    for x in range(1,len(lis)):
        y = lis[x]
        k = x-1
        while k >= 0 and ((lis[k][2] < y[2]) or (lis[k][2] == y[2] and lis[k][0] > y[0]) or (lis[k][2] == y[2] and lis[k][0] == y[0] and lis[k][1] > y[1])):
            lis[k+1] = lis[k]
            k -= 1
        lis[k+1] = y
    return lis

#second algorithm
def Merge(lis,a,b,c):
    i1=b-a+1
    i2=c-b
    lis1=[[0,0,0]]
    for i in range(i1-1):
        lis1.append([0,0,0])
    for i in range(i1):
        for j in range(3):
            lis1[i][j]=lis[a+i][j]
    lis2=[[0,0,0]] 
    for i in range(i2-1):
        lis2.append([0,0,0])
    for i in range(i2):
        for j in range(3):
            lis2[i][j]=lis[b+i+1][j]
    i=0
    j=0
    k=a
    while i<i1 and j<i2:
        if ((lis1[i][2]>lis2[j][2]) or (lis1[i][2]==lis2[j][2] and lis1[i][0]<lis2[j][0]) or (lis1[i][2]==lis2[j][2] and lis1[i][0]==lis2[j][0] and lis1[i][2]<lis2[j][2])):           
            lis[k]=lis1[i]
            k+=1
            i+=1
        else:
            lis[k]=lis2[j]
            k+=1
            j+=1
    while i<i1:
        lis[k]=lis1[i]
        i+=1
        k+=1
    while i<i2:
        lis[k]=lis2[j]
        j+=1
        k+=1
    return lis
        
def MergeSort(lis,a,c):
    if a<c:
        b=a+(c-a)//2
        MergeSort(lis,a,b)
        MergeSort(lis,b+1,c)
        return Merge(lis,a,b,c)
    
#third algorithm
def parts(lis,start,end):
    y=start-1
    p0, p1, p2 = lis[end]
    for i in range(start,end):
        if ((lis[i][2]>p2) or (lis[i][2]==p2 and lis[i][0]<p0) or (lis[i][2]==p2 and lis[i][0]==p0 and lis[i][1]<p1)):         
            y+=1
            (lis[y],lis[i])=(lis[i],lis[y])
    (lis[y+1],lis[end])=(lis[end],lis[y+1])
    return y+1

def QuickSort(lis,start,end):
    if start<end:
        p=parts(lis,start,end)
        QuickSort(lis,start,p-1)
        QuickSort(lis,p+1,end)
    return lis