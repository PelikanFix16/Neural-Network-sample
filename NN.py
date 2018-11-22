#!/usr/bin/env python
import numpy as np
import time
class NN():
    def __init__(self,x,y):
        self.x = x
        self.y = y
        self.syn0 = 2*np.random.random((3,4)) - 1
        self.syn1 = 2*np.random.random((4,1)) - 1
    def forward(self):
        self.l0 = self.x
        self.l1 = self.sigmoid(np.dot(self.l0,self.syn0))
        self.l2 = self.sigmoid(np.dot(self.l1,self.syn1))
        self.error()

    def sigmoid(self,x):
        return 1/(1+np.exp(-x))

    def error(self):
        self.errorOut = ((self.y-self.l2)**2)/2
    def dir_error(self,y,out):
        return -(y-out)

    def dir_sigmoid(self,outl2):
        return (outl2)*(1-(outl2))

    def dir_errorl2(self):
        temp = np.empty((len(self.x),4))
        for i in range(len(self.l1)):
            a= (self.dir_sigmoid(self.l2[i])*self.l1[i]*self.dir_error(self.y[i],self.l2[i]))
            temp[i] = a
        self.errorDirL2 = temp

    def minimalize_weightL2(self):
        self.dir_errorl2()
        for i in range(len(self.y)):
            for j in range(len(self.syn1)):
                self.syn1[j] = (self.syn1[j])-((0.5)*(self.errorDirL2[i][j]))
    def dir_errorl1(self):
        temp = np.empty((len(self.x),3,4))
        for i in range(len(self.x)):
            for j in range(len(self.x[i])):
                i1 = self.x[i][j]
                for k in range(len(self.syn1)):
                    error1 = (self.dir_error(self.y[i],self.l2[i])*self.dir_sigmoid(self.l2[i]))*self.syn1[k]
                    error2 = self.dir_sigmoid(self.l1[i][k])
                    error3 = i1#
                    temp[i][j][k] = (error1*error2*error3)
        self.errorDirL1 = temp

    def minimalize_weightL1(self):
        self.dir_errorl1()
        for i in range(len(self.errorDirL1)):
            for j in range(len(self.errorDirL1[i])):
                for k in range(len(self.errorDirL1[i][j])):
                    self.syn0[j][k] = (self.syn0[j][k]-((0.5)*(self.errorDirL1[i][j][k])))








    dir_errorl1
np.random.seed(1)
nn = NN(
    np.array(
        [[1,0,1],
         [1,1,0],
         [0,1,1],
         [0,1,0],
         [0,0,1]

         ]
             ),np.array(
                 [[1],
                  [1],
                  [0],
                  [0],
                  [0]
                  ]
             ))








for i in range(60000):
    nn.forward()
    nn.minimalize_weightL1()
    nn.minimalize_weightL2()
    print "Learn Generation:",i



print "Dane do nauki: ",nn.x," wyniki dla tych danych: ",nn.y


nn.forward()
print "Siec neuronowa po nauce: ",nn.x,"zwaraca wyniki: "
print nn.l2

print "Wrzucam do sieci [1,0,0] oczekiwany wynik to 1"

nn.x = np.array([[1,0,0]])
nn.forward()
print nn.l2



