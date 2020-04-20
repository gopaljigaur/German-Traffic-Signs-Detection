x=6 #max neutrons
y=400 #neutrons
n=0
powl=0
while(powl<=y):
    n+=1
    powl=x**n
a = [0]*n
while(y>0):
    a[0]+=1
    for i in range(0,len(a)-1):
        if(a[i]>=6):
            a[i]=0
            a[i+1]+=1
    y=y-1
print(a)
