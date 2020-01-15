from tkinter import *
import tkinter as tk
from tkinter import messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split 
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error 
from sklearn import linear_model
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression, BayesianRidge, SGDRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from tabulate import tabulate

df1 = pd.read_csv('WeatherDATA1.csv')    

class main(tk.Tk):
    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)
        container = tk.Frame(self)
        container.pack(side = 'top', fill = 'both', expand = True)
        container.grid_columnconfigure(0, weight = 1)
        container.grid_rowconfigure(0, weight = 1)
        self.title("WEATHER PREDICTION")
        self.geometry("1425x1200")
        self.frames = {}
        for F in (StartPage, plot1, Predict, Rainplot):
            
            frame = F(container, self)
            self.frames[F] = frame
            frame.grid(row = 0, column = 0, sticky = 'nsew')
        
        self.show_frame(StartPage)
        
    def show_frame(self, cont):
        
        frame = self.frames[cont]
        frame.tkraise()

def plotgraph(self,e1,e2,btn2):
    fromdate = e1.get()
    a = fromdate.split('-')
    todate = e2.get()
    b = todate.split('-')
    print(a[0])
    df1 = pd.read_csv('WeatherDATA1.csv')
    col1 = btn2.get()
    # print(df1.loc[(df1['Year'] == int(a[2])) & (df1['Month'] == int(a[1])) & (df1['Date'] == int(a[0]))].index)

    index1 = df1.loc[(df1['Year'] == int(a[2])) & (df1['Month'] == int(a[1])) & (df1['Date'] == int(a[0]))].index[0]
    index2 = df1.loc[(df1['Year'] == int(b[2])) & (df1['Month'] == int(b[1])) & (df1['Date'] == int(b[0]))].index[0]

    print(index1)
    print(index2)
    df1 = df1.iloc[index1:index2 + 1]
    a = list(df1[col1])
    b = list(df1['Date'])
    c = list(df1['Month'])
    d = list(df1['Year'])
    e = []

    df2 = pd.DataFrame(columns=[col1, 'Date'])

    for i in range(len(b)):
        e.append(str(int(b[i])) + '-' + str(int(c[i])) + '-' + str(int(d[i])))
        df2.loc[i] = [a[i]] + [str(e[i])]

    figure1 = plt.Figure(figsize=(7.7, 5), dpi=100)
    ax1 = figure1.add_subplot(111)
    bar1 = FigureCanvasTkAgg(figure1, self)
    bar1.get_tk_widget().place(x=325,y=210)
    df2.plot(kind='line', x = 'Date', y=col1, legend=True, ax=ax1)
    ax1.set_ylabel("Temperature in °C ",fontweight="bold",fontsize = 14)
    ax1.set_title(' Weather Statistics ',fontweight="bold",fontsize = 16)
    ax1.tick_params(axis='x', rotation=30)
    figure1.tight_layout()
    # ax1.set_title('Country Vs. GDP Per Capita')
    
def rainplotgraph(self,e1,e2):
    
    fromdate = e1.get()
    a = fromdate.split('-')
    todate = e2.get()
    b = todate.split('-')
    print(a)
    print(b)
    df1 = pd.read_csv('RainDATA1.csv')
    col1 = 'Rainfall'
    # print(df1.loc[(df1['Year'] == int(a[2])) & (df1['Month'] == int(a[1])) & (df1['Date'] == int(a[0]))].index)

    index1 = df1.loc[(df1['Year'] == int(a[1])) & (df1['Month'] == int(a[0]))].index[0]
    index2 = df1.loc[(df1['Year'] == int(b[1])) & (df1['Month'] == int(b[0]))].index[0]

    #print(index1)
    #print(index2)
    df1 = df1.iloc[index1:index2 + 1]
    a = list(df1[col1])
    #b = list(df1['Date'])
    c = list(df1['Month'])
    d = list(df1['Year'])
    e = []

    df2 = pd.DataFrame(columns=[col1, 'Month'])
    for i in range(len(d)):
        e.append(str(int(c[i])) + '-' + str(int(d[i])))
        df2.loc[i] = [a[i]] + [str(e[i])]

    figure1 = plt.Figure(figsize=(7.7, 5), dpi=100)
    ax1 = figure1.add_subplot(111)
    bar1 = FigureCanvasTkAgg(figure1, self)
    bar1.get_tk_widget().place(x=325,y=210)
    df2.plot(kind='line', x = 'Month', y=col1, legend=True, ax=ax1)
    ax1.set_ylabel("Rainfall in millimetres ",fontweight="bold",fontsize = 14)
    ax1.set_title(' Rainfall Statistics ',fontweight="bold",fontsize = 16)
    ax1.tick_params(axis='x', rotation=30)
    figure1.tight_layout()
    # ax1.set_title('Country Vs. GDP Per Capita')



def predictvalue(self,e3):

    date = e3.get()
    date = date.split('-')
    

    def randomforestinitiate(X,y):
        X = np.array(X)
        y = np.array(y)

        perm = np.random.permutation(X.shape[0])
        X = X[perm]
        y = y[perm]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
        return (X_train, X_test, y_train, y_test)

    #####################################################################################################

    class regression():

        algorithm = ["Model","KNN","Linear","RandomForest","DecisionTree","Bayesian","SVR"]
        mse = ["Mean_Squared_Error",0,0,0,0,0,0]
        predicted_maxtemp = ["Max_Temperature",0,0,0,0,0,0]
        predicted_mintemp = ["Min_Temperature",0,0,0,0,0,0]
        predicted_avgtemp = ["Avg_Temperature",0,0,0,0,0,0] 

        def Bayesian(self,X_train,X_test,y_train,y_test,date,q): #self,Xtrain,Xtest,ytrain,ytest,['dd','mm',yyyy'],indexofalgo]
            model = BayesianRidge(compute_score=True)
            y_train.shape
            X_train.shape

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            df = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': y_pred.flatten()})
            mse = mean_squared_error(y_test, y_pred)
            predictedval = (model.predict([date]))

            self.mse[q] = mse
            self.predicted_avgtemp[q] = predictedval[0]
            self.predicted_mintemp[q] = '-'  # because BAyesian only does one val prediction
            self.predicted_maxtemp[q] = '-'  # because BAyesian only does one val prediction

        def KNeighbors(self,X_train,X_test,y_train,y_test,date,q):
            model = KNeighborsRegressor()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            df = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': y_pred.flatten()})
            mse = mean_squared_error(y_test, y_pred)
            predictedval = model.predict([date])
            
            self.mse[q] = mse
            self.predicted_avgtemp[q] = predictedval[0][0]
            self.predicted_mintemp[q] = predictedval[0][1] 
            self.predicted_maxtemp[q] = predictedval[0][2]

        def Linear(self,X_train,X_test,y_train,y_test,date,q):
            model = LinearRegression()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            df = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': y_pred.flatten()})
            mse = mean_squared_error(y_test, y_pred)
            predictedval = (model.predict([date]))
            
            self.mse[q] = mse
            self.predicted_avgtemp[q] = predictedval[0][0]
            self.predicted_mintemp[q] = predictedval[0][1] 
            self.predicted_maxtemp[q] = predictedval[0][2]
        
        def DecisionTree(self,X_train,X_test,y_train,y_test,date,q):
            model = DecisionTreeRegressor()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            df = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': y_pred.flatten()})
            mse = mean_squared_error(y_test, y_pred)
            predictedval = (model.predict([date]))
            
            self.mse[q] = mse
            self.predicted_avgtemp[q] = predictedval[0][0]
            self.predicted_mintemp[q] = predictedval[0][1] 
            self.predicted_maxtemp[q] = predictedval[0][2]  

        def SVR(self,X_train,X_test,y_train,y_test,date,q):
            model = SVR()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            df = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': y_pred.flatten()})
            mse = mean_squared_error(y_test, y_pred)
            predictedval = (model.predict([date]))
            
            self.mse[q] = mse
            self.predicted_avgtemp[q] = predictedval[0]
            self.predicted_mintemp[q] = '-'  # because SVR only does one val prediction
            self.predicted_maxtemp[q] = '-'  # because SVR only does one val prediction

        def RandomForest(self,X_train,X_test,y_train,y_test,date,q):
            model = RandomForestRegressor()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            df = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': y_pred.flatten()})
            mse = mean_squared_error(y_test, y_pred)
            predictedval = (model.predict([date]))
            
            self.mse[q] = mse
            self.predicted_avgtemp[q] = predictedval[0][0]
            self.predicted_mintemp[q] = predictedval[0][1] 
            self.predicted_maxtemp[q] = predictedval[0][2]  

        def reducefloat(self):
            for q in range(1,7):
                self.mse[q] = round(self.mse[q],2)
                self.predicted_avgtemp[q] = round(self.predicted_avgtemp[q],2)
                if (self.algorithm[q] == "KNN" or self.algorithm[q] == "Linear" or self.algorithm[q] == "RandomForest" or self.algorithm[q] == "DecisionTree"):
                    self.predicted_mintemp[q] = round(self.predicted_mintemp[q],2)
                    self.predicted_maxtemp[q] = round(self.predicted_maxtemp[q],2)
                    
####################################################################################################

    
    class regression2():

        algorithm = ["Model","KNN","Linear","RandomForest","DecisionTree","Bayesian","SVR"]
        mse = ["Mean_Squared_Error",0,0,0,0,0,0]
        predicted_rainfall = ["Avg_Rain(mm)\nfor given month",0,0,0,0,0,0]

        def Bayesian(self,X_train,X_test,y_train,y_test,date,q): #self,Xtrain,Xtest,ytrain,ytest,['dd','mm',yyyy'],indexofalgo]
            model = BayesianRidge(compute_score=True)
            y_train.shape
            X_train.shape

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            df = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': y_pred.flatten()})
            mse = mean_squared_error(y_test, y_pred)
            predictedval = (model.predict([date]))

            self.mse[q] = mse
            self.predicted_rainfall[q] = predictedval[0]

        def KNeighbors(self,X_train,X_test,y_train,y_test,date,q):
            model = KNeighborsRegressor()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            df = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': y_pred.flatten()})
            mse = mean_squared_error(y_test, y_pred)
            predictedval = model.predict([date])
            
            self.mse[q] = mse
            self.predicted_rainfall[q] = predictedval[0]
            

        def Linear(self,X_train,X_test,y_train,y_test,date,q):
            model = LinearRegression()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            df = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': y_pred.flatten()})
            mse = mean_squared_error(y_test, y_pred)
            predictedval = (model.predict([date]))
            
            self.mse[q] = mse
            self.predicted_rainfall[q] = predictedval[0]
            
        
        def DecisionTree(self,X_train,X_test,y_train,y_test,date,q):
            model = DecisionTreeRegressor()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            df = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': y_pred.flatten()})
            mse = mean_squared_error(y_test, y_pred)
            predictedval = (model.predict([date]))
            
            self.mse[q] = mse
            self.predicted_rainfall[q] = predictedval[0]
            

        def SVR(self,X_train,X_test,y_train,y_test,date,q):
            model = SVR()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            df = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': y_pred.flatten()})
            mse = mean_squared_error(y_test, y_pred)
            predictedval = (model.predict([date]))
            
            self.mse[q] = mse
            self.predicted_rainfall[q] = predictedval[0]
            

        def RandomForest(self,X_train,X_test,y_train,y_test,date,q):
            model = RandomForestRegressor()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            df = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': y_pred.flatten()})
            mse = mean_squared_error(y_test, y_pred)
            predictedval = (model.predict([date]))
            
            self.mse[q] = mse
            self.predicted_rainfall[q] = predictedval[0]
            

        def reducefloat(self):
            for q in range(1,7):
                self.mse[q] = round(self.mse[q],2)
                self.predicted_rainfall[q] = round(self.predicted_rainfall[q],2)
                
    ##################################################################################################                
    for i in range(len(date)):
        date[i] = int(date[i])  # string to int
    date2 = date[1:]
    ####################################################################################################

    dataset = pd.read_csv('WeatherDATA1.csv')
    X = dataset.iloc[:, 4:7].values
    y = dataset.iloc[:, 1:4].values
    y_sgd_svr_bayesian = dataset.iloc[:, 1].values

    X_train_r, X_test_r, y_train_r, y_test_r =randomforestinitiate(X,y)  # FOr Random forest

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0) #for decision, linear, knn

    X_train_sgd_svr_bayesian , X_test_sgd_svr_bayesian , y_train_sgd_svr_bayesian , y_test_sgd_svr_bayesian = train_test_split(X, y_sgd_svr_bayesian , test_size=0.2, random_state=0) # for sgd and svr and bayesian

    #########################################################################################################
    ####################################################################################################

    dataset2 = pd.read_csv('RainDATA1.csv')
    X2 = dataset2.iloc[:, 1:].values
    y2 = dataset2.iloc[:, 0].values

    X_train_r2, X_test_r2, y_train_r2, y_test_r2 =randomforestinitiate(X2,y2)  # FOr Random forest

    X_train2, X_test2, y_train2, y_test2 = train_test_split(X2, y2, test_size=0.2, random_state=0) #for decision, linear, knn

    ######################################################################################################

    obj = regression()  # class object declared
    obj2 = regression2()  # class object declared

    #########################################################################################################
    ####################################################################################################
    
    # Ref : algorithm = ["KNeighbors_Regressor","Linear_Regression","RandomForest_Regressor","DecisionTree_Regressor","BayesianRidge_Regressor","SupportVector_Regressor","StochasticGradientDescent_Regressor"] 

    obj.DecisionTree(X_train,X_test,y_train,y_test,date,4)
    obj.KNeighbors(X_train,X_test,y_train,y_test,date,1)
    obj.Linear(X_train,X_test,y_train,y_test,date,2)

    obj.RandomForest(X_train_r,X_test_r,y_train_r,y_test_r,date,3)

    obj.Bayesian(X_train_sgd_svr_bayesian,X_test_sgd_svr_bayesian,y_train_sgd_svr_bayesian,y_test_sgd_svr_bayesian,date,5)
    obj.SVR(X_train_sgd_svr_bayesian,X_test_sgd_svr_bayesian,y_train_sgd_svr_bayesian,y_test_sgd_svr_bayesian,date,6)
    #obj.SGD(X_train_sgd_svr_bayesian,X_test_sgd_svr_bayesian,y_train_sgd_svr_bayesian,y_test_sgd_svr_bayesian,date,7)

    ## Also, values too big, reduce to 2 digit float:
    obj.reducefloat()

    #########################################################################################################
    ## Ref : algorithm = ["KNeighbors_Regressor","Linear_Regression","RandomForest_Regressor","DecisionTree_Regressor","BayesianRidge_Regressor","SupportVector_Regressor","StochasticGradientDescent_Regressor"] 

    obj2.DecisionTree(X_train2,X_test2,y_train2,y_test2,date2,4)
    obj2.KNeighbors(X_train2,X_test2,y_train2,y_test2,date2,1)
    obj2.Linear(X_train2,X_test2,y_train2,y_test2,date2,2)
    obj2.RandomForest(X_train2,X_test2,y_train2,y_test2,date2,3)
    obj2.Bayesian(X_train2,X_test2,y_train2,y_test2,date2,5)
    obj2.SVR(X_train2,X_test2,y_train2,y_test2,date2,6)

    #obj.SGD(X_train_sgd_svr_bayesian,X_test_sgd_svr_bayesian,y_train_sgd_svr_bayesian,y_test_sgd_svr_bayesian,date,7)
    ## Also, values too big, reduce to 2 digit float:
    obj2.reducefloat()

    ####################################################################################################
    ##tabulate works in a row wise fashio. what u have is column wise. so print horizontally

    table = [obj.algorithm, obj.predicted_mintemp, obj.predicted_avgtemp, obj.predicted_maxtemp, obj2.predicted_rainfall, obj.mse]

    #table = [[1,2,3],[4,5,6],[7,8,9]]

    x1 = 150
    y1 = 300
    n=0
    
    label=[]
    for i in range(len(table)):
        for j in range(len(table[i])):
            if i==0:
                label.append(tk.Label(self, text = table[i][j], font = ('arial', 16, 'bold')))
                label[n].place(x=x1, y=y1, anchor="center")
                n = n+1
                x1 = x1 + 170
        
            else:
                label.append(tk.Label(self, text = table[i][j], font = ('arial', 14)))
                label[n].place(x=x1, y=y1, anchor="center")
                n = n+1
                x1 = x1 + 170
        
        y1 = y1+40
        x1 = 150


class StartPage(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        
        label1 = tk.Label(self, text = 'WEATHER ANALYSIS', relief = 'solid', width = 30, font = ('arial', 22, 'bold'))
        label1.place(x=460, y=50)
        
        b1 = tk.Button(self, text='WEATHER STATISTICS PLOT', width = 30, bg = 'brown', fg = 'white', command = lambda: controller.show_frame(plot1))
        b1.place(x=620, y=120)
        
        b2 = tk.Button(self, text='WEATHER PREDICTION', width = 30, bg = 'brown', fg = 'white', command = lambda: controller.show_frame(Predict))
        b2.place(x=620, y=200)
 
        b3 = tk.Button(self, text='RAINFALL STATISTICS PLOT', width = 30, bg = 'brown', fg = 'white', command = lambda: controller.show_frame(Rainplot))
        b3.place(x=620, y=160)

class plot1(tk.Frame):
    
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        e1 = tk.StringVar()
        e2 = tk.StringVar()
        btn2 = tk.StringVar()   
        label1 = tk.Label(self, text = 'WEATHER STATSTICS', relief = 'solid', width = 30, font = ('arial', 22, 'bold'))
        label1.place(x=460, y=50)
        
        t2 = tk.Label(self, text="FROM DATE", fg="black").place(x=312, y=110)
        entry1 = tk.Entry(self, textvar = e1)
        entry1.place(x=412, y=110)
        
        t2 = tk.Label(self, text="TO DATE", fg="black").place(x=912, y=110)
        entry2 = tk.Entry(self, textvar = e2)
        entry2.place(x=1012, y=110)
        
        gumb4 = tk.Radiobutton(self, bg="white", text="Average", value="Average", variable=btn2)
        gumb4.place(x=412, y=150)
        gumb5 = tk.Radiobutton(self, bg="white", text="Minimum", value="Minimum", variable=btn2)
        gumb5.place(x=689, y=150)
        gumb6 = tk.Radiobutton(self, bg="white", text="Maximum", value="Maximum", variable=btn2)
        gumb6.place(x=966, y=150)
                        
        btn = tk.Button(self, text="PLOT", fg="white", bg="black", command=lambda: plotgraph(self,e1,e2,btn2))
        btn.place(x=712, y=180)
        
        btun3 = tk.Button(self, text = 'Back to Home Page', command = lambda: controller.show_frame(StartPage))
        btun3.place(x=1, y=1)
        

class Rainplot(tk.Frame):
    
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        e1 = tk.StringVar()
        e2 = tk.StringVar()
        #btn2 = tk.StringVar()   
        label1 = tk.Label(self, text = 'RAINFALL STATSTICS', relief = 'solid', width = 30, font = ('arial', 22, 'bold'))
        label1.place(x=460, y=50)
        
        t2 = tk.Label(self, text="FROM DATE", fg="black").place(x=312, y=110)
        entry1 = tk.Entry(self, textvar = e1)
        entry1.place(x=412, y=110)
        
        t2 = tk.Label(self, text="TO DATE", fg="black").place(x=912, y=110)
        entry2 = tk.Entry(self, textvar = e2)
        entry2.place(x=1012, y=110)
                        
        btn = tk.Button(self, text="PLOT", fg="white", bg="black", command=lambda: rainplotgraph(self,e1,e2))
        btn.place(x=712, y=150)
        
        btun3 = tk.Button(self, text = 'Back to Home Page', command = lambda: controller.show_frame(StartPage))
        btun3.place(x=1, y=1)



class Predict(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)    
        e3 = tk.StringVar()
        #btn2 = tk.StringVar()
        label3 = tk.Label(self, text = ' WEATHER PREDICTION', relief = 'solid', width = 30, font = ('arial', 22, 'bold'))
        label3.place(x=460, y=50)
    
        t3 = tk.Label(self, text="DATE", fg="black").place(x=625, y=100)
        entry3 = tk.Entry(self, textvar = e3)
        entry3.place(x=725, y=100)    
        
        #gumb4 = tk.Radiobutton(self, bg="white", text="Average", value="Average", variable=btn2)
        #gumb4.place(x=300, y=200)
        #gumb5 = tk.Radiobutton(self, bg="white", text="Minimum", value="Minimum", variable=btn2)
        #gumb5.place(x=577, y=200)
        #gumb6 = tk.Radiobutton(self, bg="white", text="Maximum", value="Maximum", variable=btn2)
        #gumb6.place(x=854, y=200)
        
        btn = tk.Button(self, text="PREDICT", fg="white", bg="black", command = lambda: predictvalue(self,e3))
        btn.place(x=712, y=150)
        
        btun4 = tk.Button(self, text = 'Back to Home Page', command = lambda: controller.show_frame(StartPage))
        btun4.place(x=1, y=1)


if __name__=='__main__':  
    app = main()
    app.mainloop()
    