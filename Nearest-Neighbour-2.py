import numpy as np

labels = [b'NO',b'DH',b'SL']
data = np.loadtxt('column_3c.dat',converters={6: lambda s:labels.index(s)})

x = data[:,0:6]
y = data[:,6]

training_indices = list(range(0,20))+list(range(40,188))+list(range(230,310))
test_indices = list(range(20,40))+list(range(188,230))

trainx = x[training_indices,:]
trainy = y[training_indices]
testx = x[test_indices,:]
testy = y[test_indices]



def NN_L2(trainx,trainy,testx):
    result_y=[]
    for  x in range(len(testx)):
        distances = [np.sum(np.square(testx[x,]-trainx[i,])) for i in range(len(trainx))]
        index = np.argmin(distances)
        result_y.append(trainy[index])
        
    return np.array(result_y)

testy_L2 = NN_L2(trainx, trainy, testx)

assert( type( testy_L2).__name__ == 'ndarray' )
assert( len(testy_L2) == 62 ) 
assert( np.all( testy_L2[50:60] == [ 0.,  0.,  0.,  0.,  2.,  0.,  2.,  0.,  0.,  0.] )  )
assert( np.all( testy_L2[0:10] == [ 0.,  0.,  0.,  1.,  1.,  0.,  1.,  0.,  0.,  1.] ) )

def NN_L1(trainx,trainy,testx):
    result_y = []
    for i in range(len(testx)):
        distance = [np.sum(np.absolute(trainx[x,]-testx[i,])) for x in range(len(trainx))]
        index = np.argmin(distance)
        result_y.append(trainy[index])
    return np.array(result_y)

testy_L1 = NN_L1(trainx, trainy, testx)
testy_L2 = NN_L2(trainx, trainy, testx)
print("l1 l2 differrence in test",sum(testy_L1!=testy_L2))

assert( type( testy_L1).__name__ == 'ndarray' )
assert( len(testy_L1) == 62 ) 
assert( not all(testy_L1 == testy_L2) )
assert( all(testy_L1[50:60]== [ 0.,  2.,  1.,  0.,  2.,  0.,  0.,  0.,  0.,  0.]) )
assert( all( testy_L1[0:10] == [ 0.,  0.,  0.,  0.,  1.,  0.,  1.,  0.,  0.,  1.]) )

def error_rate(testy,testy_fit):
    return float(sum(testy!=testy_fit))/len(testy)

print("Error rate of NN_L1: ",error_rate(testy,testy_L1))
print("Error rate of NN_L2: ",error_rate(testy,testy_L2))

print(testy[10])

def confusion(testy,testy_fit):
    x=[[0,0,0],[0,0,0],[0,0,0]]
    for i in range(len(testy)):
        if(testy[i]!=testy_fit[i]):
            x[int(testy[i])][int(testy_fit[i])]+=1
    return np.array(x)
L1_neo = confusion(testy, testy_L1) 
assert( type(L1_neo).__name__ == 'ndarray' )
assert( L1_neo.shape == (3,3) )
assert( np.all(L1_neo == [[ 0.,  2.,  2.],[ 10.,  0.,  0.],[ 0.,  0.,  0.]]) )
L2_neo = confusion(testy, testy_L2)  
assert( np.all(L2_neo == [[ 0.,  1.,  2.],[ 10.,  0.,  0.],[ 0.,  0.,  0.]]) )

