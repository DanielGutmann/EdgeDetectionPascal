import numpy as np
from random import randint
import cv2
#import matplotlib.pyplot as plt
import time
#from PIL import Image
def pascalTransformMatrix(dimension):
	# zero matrix for dimension
	Matrix = np.zeros((dimension,dimension),np.int64);
	# diagonal alternating ones and first colomn ones
	for i in range(dimension):
		Matrix[i][i] = pow(-1,i);
		Matrix[i][0] = 1;
	# rest calulations from matrix[2][1] => matrix[dimension-1][dimension-1] (lower triangular part!)

	for i in range(2,dimension,1):
		for j in range(1,dimension,1):
			if j < i:
				Matrix[i][j] = (abs(Matrix [i-1][j]) + abs(Matrix[i-1][j-1])) * pow(-1,j)


	return Matrix

def FastDescretePascal(dimension,array):
	Result = [0] * dimension;
	#print('first element in array : ')
	#print(array)
	Result[0] = array[0];
	Buffer = [];
	#print(Buffer)
	for i in range(1,dimension):
		
		#print("Stage number : %d" % i)
		#print("\n")

		Buffer = list(Result)
		#print("Buffer Content:")
		#print(Buffer)
		for j in range (i,dimension,1):
			
			# first stage takes the input array values
			if i==1:
				Result[j] = array[j-1] - array[j];
				#print("Oduzimam %d sa %d i smestam u result[%d] = %d" % (array[j-1],array[j],j,Result[j]))
			# rest of the stages take the buffer values ( partial results for the previous stage )
			else:
				Result[j] =  Buffer[j-1] - Buffer[j];
				#print("else grana :Oduzimam %d sa %d i smestam u result[%d] = %d" % (Buffer[j-1],Buffer[j],j,Result[j]))

	return Result

#splits the list in parts
def split_list(alist, wanted_parts=1):
    length = len(alist)
    return [ alist[i*length // wanted_parts: (i+1)*length // wanted_parts] 
             for i in range(wanted_parts) ]

#checks for alternation of sign (edge pixel) and returns index
def apsoluteMax(dim,array):
    if (array[1] < 0):
        return 1
    if(array[2] < 0):
        return 2
    if(array[3] < 0):
        return 3
    return -1

def printTxt(string):
    f = open('output.txt', 'a')
    f.write(str(string) + "\n")
    f.close()

#basic treshold function
def isoArray(dim,array,treshold):
	edge = True
	midval = 0
	for i in range(0,dim):
		midval = midval + array[i]
	midval = midval/dim

	for i in range(0,dim):
		test = array[i]
		if(test <= midval + treshold and test >= midval - treshold and edge):
			edge = True
		else:
			edge = False
	#print edge
	return edge

def exists(dim,array,element):
	for i in range(dim):
		if(element == array[i]):
			return True
	return False
#threshold pozitivan broj !
def analyser(array,threshold):
	brojac = 0
	m = 0
	n = 1
	test = []
	for i in range(len(array)-1):
		if(i==0):
			if(abs(array[n])>=threshold):
				print("Za prvi")
				print("EdgeValue:")
				print(array[0])
				brojac = brojac +1
		else:
			if(abs((array[m+i])-(array[n+i])) >= threshold):

				print("Edge value:")
				print(array[m+i])
				brojac = brojac +1
	print(test)
	return brojac
def test():
	matrica = pascalTransformMatrix(4)
	print("Start:+++++++++++++++++++++++++++++++")
	for i in range(10):
		array = []
		array.append(np.random.randint(0,255))
		array.append(np.random.randint(0,255))
		array.append(np.random.randint(0,255))
		array.append(np.random.randint(0,255))
		print(array)
		spektar = np.dot(matrica,array)
		print(spektar)
		analyser(spektar,50)
		print("----------------------------------")

def inverseSignal(signalArray):
	length = len(signalArray)
	sup = signalSupremum(signalArray)
	for i in range(0,length):
		signalArray[i] = sup - signalArray[i];

	return signalArray

def signalSupremum(signalArray):
	sup = signalArray[0]
	for i in range(1,len(signalArray)):
		if(signalArray[i] > sup):
			sup = signalArray[i]
	return sup

def test3():
    array = []
    array.append(20);
    array.append(145);
    array.append(155);
    array.append(140);
    print(signalSupremum(array))
    print("==================")   
    print(FastDescretePascal(2,[255,250]))
    print(FastDescretePascal(2,[250,20]))
    print(FastDescretePascal(2,[20,15]))    
    print("==================")
    print("Niz:")
    print(array)
    inversArray = inverseSignal(array)
    print("Inverzan niz:")
    print(inversArray)
    print("Spektar niza:")
    matrixP = pascalTransformMatrix(4)
    #spektar = FastDescretePascal(len(array),array)
    spektar = np.dot(matrixP,array)
    print(spektar)
    print("Spektar inverznog:")
    #invSpektar = FastDescretePascal(len(inversArray),inversArray)
    invSpektar = np.dot(matrixP,inversArray)
    print(invSpektar)    
    #test
    PascalMat = pascalTransformMatrix(4)
    print("test za inverznog:")
    print(np.dot(PascalMat,inversArray))	
def test2():
	matrica = pascalTransformMatrix(4)
	array = []
	array.append(0)
	array.append(0)
	array.append(0)
	array.append(255)
	spektar = np.dot(matrica,array)
	print(apsoluteMax(4,spektar))
	print(array)
	print(spektar)
	print("------------------------")
	array = []
	array.append(0)
	array.append(0)
	array.append(255)
	array.append(0)
	spektar = np.dot(matrica,array)
	print(apsoluteMax(4,spektar))
	print(array)
	print(spektar)
	print("------------------------")
	array = []
	array.append(0)
	array.append(0)
	array.append(255)
	array.append(255)
	spektar = np.dot(matrica,array)
	print(apsoluteMax(4,spektar))
	print(array)
	print(spektar)
	print("------------------------")
	array = []
	array.append(0)
	array.append(255)
	array.append(0)
	array.append(0)
	spektar = np.dot(matrica,array)
	print(apsoluteMax(4,spektar))
	print(array)
	print(spektar)
	print("------------------------")
	array = []
	array.append(0)
	array.append(255)
	array.append(0)
	array.append(255)
	spektar = np.dot(matrica,array)
	print(apsoluteMax(4,spektar))
	print(array)
	print(spektar)
	print("------------------------")
	array = []
	array.append(0)
	array.append(255)
	array.append(255)
	array.append(0)
	spektar = np.dot(matrica,array)
	print(apsoluteMax(4,spektar))
	print(array)
	print(spektar)
	print("------------------------")
	array = []
	array.append(0)
	array.append(255)
	array.append(255)
	array.append(255)
	spektar = np.dot(matrica,array)
	print(apsoluteMax(4,spektar))
	print(array)
	print(spektar)
	print("------------------------")
	array = []
	array.append(255)
	array.append(5)
	array.append(5)
	array.append(5)
	spektar = np.dot(matrica,array)
	print(apsoluteMax(4,spektar))
	print(array)
	print(spektar)
	print("------------------------")
	array = []
	array.append(255)
	array.append(0)
	array.append(0)
	array.append(255)
	spektar = np.dot(matrica,array)
	print(apsoluteMax(4,spektar))
	print(array)
	print(spektar)
	print("------------------------")
	array = []
	array.append(255)
	array.append(0)
	array.append(255)
	array.append(0)
	spektar = np.dot(matrica,array)
	print(apsoluteMax(4,spektar))
	print(array)
	print(spektar)
	print("------------------------")
	array = []
	array.append(255)
	array.append(0)
	array.append(255)
	array.append(255)
	spektar = np.dot(matrica,array)
	print(apsoluteMax(4,spektar))
	print(array)
	print(spektar)
	print("------------------------")
	array = []
	array.append(255)
	array.append(255)
	array.append(0)
	array.append(0)
	spektar = np.dot(matrica,array)
	print(apsoluteMax(4,spektar))
	print(array)
	print(spektar)
	print("------------------------")
	array = []
	array.append(255)
	array.append(255)
	array.append(0)
	array.append(255)
	spektar = np.dot(matrica,array)
	print(apsoluteMax(4,spektar))
	print(array)
	print(spektar)
	print("------------------------")
	array = []
	array.append(255)
	array.append(255)
	array.append(255)
	array.append(0)
	spektar = np.dot(matrica,array)
	print(apsoluteMax(4,spektar))
	print(array)
	print(spektar)
	print("------------------------")
	array = []
	array.append(250)
	array.append(255)
	array.append(230)
	array.append(50)
	spektar = np.dot(matrica,array)
	print(apsoluteMax(4,spektar))
	print(array)
	print(spektar)
	print("------------------------")       	

def signal_processing(signal,threshold,matrix):
	
	if not isoArray(4,signal,threshold):
		spectar = np.dot(matrix,signal)
		indexOfEdgePoint = apsoluteMax(4,spectar)
		return indexOfEdgePoint
	return -1
def main(threshold,image):
	
	#imeSlike = 'photographer.jpg'
	#threshold = 20
	img = cv2.imread(image,0);

	#print(img)

	rows = img.shape[0]
	colomns = img.shape[1]
	dim = rows * colomns

	row = []

	iar = np.asarray(img)

	Pmatrix = pascalTransformMatrix(4)
	
	
	tupleArray = []
	start_time = time.time()
	for i in range(3,rows - 3):
		for j in range(3,colomns - 3):

			tempRowRight = [] 
			tempRowLeft = []

			tempColDown = []
			tempColUp = []

			tempDiagBottomRight = []
			tempDiagBottomLeft = []

			tempDiagTopRight = []
			tempDiagTopLeft = []

			tempRowRight.append(iar[i][j])
			tempRowRight.append(iar[i][j + 1])
			tempRowRight.append(iar[i][j + 2])
			tempRowRight.append(iar[i][j + 3])

			tempColDown.append(iar[i][j])
			tempColDown.append(iar[i + 1][j])
			tempColDown.append(iar[i + 2][j])
			tempColDown.append(iar[i + 3][j])

			tempDiagBottomRight.append(iar[i][j])
			tempDiagBottomRight.append(iar[i + 1][j + 1])
			tempDiagBottomRight.append(iar[i + 2][j + 2])
			tempDiagBottomRight.append(iar[i + 3][j + 3])

			tempDiagBottomLeft.append(iar[i][j])
			tempDiagBottomLeft.append(iar[i + 1][j - 1])
			tempDiagBottomLeft.append(iar[i + 2][j - 2])
			tempDiagBottomLeft.append(iar[i + 3][j - 3])

			tempRowLeft.append(iar[i][j])
			tempRowLeft.append(iar[i][j - 1])
			tempRowLeft.append(iar[i][j - 2])
			tempRowLeft.append(iar[i][j - 3])

			tempDiagTopLeft.append(iar[i][j])
			tempDiagTopLeft.append(iar[i - 1][j - 1])
			tempDiagTopLeft.append(iar[i - 2][j - 2])
			tempDiagTopLeft.append(iar[i - 3][j - 3])

			tempColUp.append(iar[i][j])
			tempColUp.append(iar[i - 1][j])
			tempColUp.append(iar[i - 2][j])
			tempColUp.append(iar[i - 3][j])

			tempDiagTopRight.append(iar[i][j])
			tempDiagTopRight.append(iar[i - 1][j + 1])
			tempDiagTopRight.append(iar[i - 2][j + 2])
			tempDiagTopRight.append(iar[i - 3][j + 3])


			index = signal_processing(tempRowRight,threshold,Pmatrix)
			#print(index)
			if(index != -1):
				tupleVertical = (i,j+index)
				if exists(len(tupleArray),tupleArray,tupleVertical) is False:
					tupleArray.append(tupleVertical)
			#print(index)
			index = signal_processing(tempColDown,threshold,Pmatrix)
			if(index != -1):
				tupleHorizontal = (i+index,j)
				if exists(len(tupleArray),tupleArray,tupleHorizontal) is False:
					tupleArray.append(tupleHorizontal)
			#print(index)
			index = signal_processing(tempDiagBottomRight,threshold,Pmatrix)
			if(index != -1):
				tupleDiagonal = (i+index,j+index)
				if exists(len(tupleArray),tupleArray,tupleDiagonal) is False:
					tupleArray.append(tupleDiagonal)
			index = signal_processing(tempDiagBottomLeft,threshold,Pmatrix)
			if(index != -1):
				tupleDiagonalBL = (i+index,j-index)
				if exists(len(tupleArray),tupleArray,tupleDiagonalBL) is False:
					tupleArray.append(tupleDiagonalBL)
			index = signal_processing(tempRowLeft,threshold,Pmatrix)
			if(index != -1):
				tupleRowLFT = (i,j-index)
				if exists(len(tupleArray),tupleArray,tupleRowLFT) is False:
					tupleArray.append(tupleRowLFT)
			index = signal_processing(tempDiagTopLeft,threshold,Pmatrix)
			if(index != -1):
				tupleDiagTL = (i-index,j-index)
				if exists(len(tupleArray),tupleArray,tupleDiagTL) is False:
					tupleArray.append(tupleDiagTL)
			index = signal_processing(tempColUp,threshold,Pmatrix)
			if(index != -1):
				tupleColUp = (i-index,j)
				if exists(len(tupleArray),tupleArray,tupleColUp) is False:
					tupleArray.append(tupleColUp)
			index = signal_processing(tempDiagTopRight,threshold,Pmatrix)
			if(index != -1):
				tupleDiagTR = (i-index,j+index)
				if exists(len(tupleArray),tupleArray,tupleDiagTR) is False:
					tupleArray.append(tupleDiagTR)


	vremeStr = (time.time() - start_time)
	print("--- %s seconds for Pascal transform ---" % vremeStr)
	print("--------Wait for OpenCV write --------------")
	size = (w,h,channels) = (rows,colomns,1)
	blackAndWhite = np.zeros(size,np.uint8)
	#print(tupleArray)
	for i in range(len(tupleArray)):
		(row,col) = tupleArray[i]
		blackAndWhite[row][col] = 255
	#print(iar)
	cv2.imwrite('Threshold-'+str(threshold)+'-TimeInSec-'+str(vremeStr)+'-'+image, blackAndWhite)
	print("--------OpenCV write done.Image processed : %s --------------" % image)


# za 4 slike u folderu gde je skripta sa zadatim imenom pravi slike sa ivicama za tresholdove od 50 - 5
#for slika in ['bullseye.jpg','lena.jpg','photographer.jpg','wheel.png']:
for slika in ['lena.jpg']:
	for i in range(50,0,-5):
		main(i,slika)
#test4([20,145,155,140,15,30,35,160],8,4)
#test3()