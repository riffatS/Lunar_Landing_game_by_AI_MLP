
import numpy as np
import pandas as pd
# import neural_networkfinal
from sklearn.preprocessing import MinMaxScaler
class NeuralNetHolder:
  

    def __init__(self):
      self.lmbda = 0.6
      self.eitha=0.9
      self.alpha=0.1
      super().__init__()

    #reading weights
    def readWeights(self):
      weightsInputHidden = np.loadtxt('weights_inputHidden.txt', dtype=float)
      weightsHiddenOutput=np.loadtxt('weights_hiddenToOutput.txt', dtype=float)
      return weightsInputHidden,weightsHiddenOutput

    def predict(self, input_row):
        # WRITE CODE TO PROCESS INPUT ROW AND PREDICT X_Velocity and Y_Velocity
        print("input row from game is",input_row)
        inputValue=[]
        input_rowXSplit=input_row.split(",")
        #splitting the input row
        inputX=input_rowXSplit[0]
        inputX=input_rowXSplit[1]
        #normilizing the input values
        dataRaw= pd.read_csv("ce889_dataCollection.csv",index_col=False,header=None)
        inputValue.append((float(input_rowXSplit[0])-dataRaw[0].min())/(dataRaw[0].max()-dataRaw[0].min()))
        inputValue.append((float(input_rowXSplit[1])-dataRaw[1].min())/(dataRaw[1].max()-dataRaw[1].min()))
        #predicting the output
        outputs=self.predictions(inputValue)
        print("output iss",outputs)

        #Denormilizing output value 
        
        outputY1=(dataRaw[2].min()+((outputs[0]-0)/1-0)*(dataRaw[2].max()-dataRaw[2].min()))
        outputY2=(dataRaw[3].min()+((outputs[1]-0)/1-0)*(dataRaw[3].max()-dataRaw[3].min()))
        
        return outputY1,outputY2




    def predictions(self,inputValue):
        # input to hidden
      wxInputHidden=self.InputIntoWeight(self.weightsInputHidden,self.weightsInputHiddenBias,inputValue)
      activationHiddenLayer=self.sig(wxInputHidden)
        #hidden to output
      wxHiddenOutput=self.InputIntoWeight(self.weightsHiddenOutput,self.weightsHiddenOutputBias,activationHiddenLayer)
      outputPredicted=self.sig(wxHiddenOutput)
      return outputPredicted


      return inputValue
    def InputIntoWeight(self,weights,biasInput,inputValue):
      #assigning weights        
        inputWeightss=[]
        biasInput=[]
        biasInput=inputValue.copy()
        # inserting bias input in input value
        biasInput.insert(0,1)
        sums=0       
        #weights into input  
        for i in range(len(weights[1])):
            for j in range(len(inputValue)):
                sums+=inputValue[j]*weights[j][i]
            
            inputWeightss.append(sums)
            sums=0
        return inputWeightss
    #activation fucntion 
    def sig(self,x):
        sigmoidValues=[]
        for i in range(len(x)):
            sigmoidValues.append(round(1/(1 + np.exp(-(self.lmbda*x[i]))),6))
        self.activation_value=sigmoidValues

        return sigmoidValues
    
    weightsInputHidden=[[  0.10985284 ,  7.7619416,   -2.54147806, -14.1411045,   -5.52683282,
   -5.66009584],
 [  4.61431991 , -0.6162896,   -1.843548,    -0.33102002,  -3.07092385,-4.69358649]]
    weightsHiddenOutput=[[-12.21359861 , -9.1575221 ],
 [  6.6829733 ,   9.7744087 ],
 [ 16.95247933,   1.74109849],
 [ -4.48317781,  20.97070821],
 [  3.83013875, -18.03046986],
 [ -2.26754452, -16.56487089]]
    weightsHiddenOutputBias=[-0.12639373 , 1.24530361]
    weightsInputHiddenBias=[1.1126102 , -0.11582 ,-2.21145677 , 4.659935554, -0.34456775,0.14452115]

