import tkinter 
import pandas as pd
from tkinter import filedialog
from tkinter import messagebox
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import r2_score,mean_squared_error

class deploy:
    def savef(self):
        self.file = filedialog.askopenfilename(initialdir='/', title="Select File", filetypes=[("csv files", "*.csv")])
        self.entry_file_path.delete(0, tkinter.END) 
        self.entry_file_path.insert(tkinter.END, self.file)

    def x_and_y(self):
        self.file_path = self.entry_file_path.get()
        self.df = pd.read_csv(self.file_path)
        self.columns = self.df.columns.tolist()

    def __init__(self):
        self.window = tkinter.Tk()
        self.window.title("Deployment")
        self.window.geometry("700x400")
        self.window.configure(bg="black")
        self.indep_selected = []
        self.dep_selected = [] 
        self.file = None
        self.window_label = tkinter.Label(self.window, text="Deployment", font=("times new roman", 22, "bold"),
                                          bg="black", fg="white")
        self.window_label.place(x=260, y=50)

        self.label_file_path = tkinter.Label(self.window, text="Select the dataset file",
                                             font=("times new roman", 14, "italic"), bg="black", fg="white")
        self.label_file_path.place(x=230, y=120)
        self.entry_file_path = tkinter.Entry(self.window, width=50)
        self.entry_file_path.place(x=180, y=150)
        self.fileopen = tkinter.Button(self.window, text="Browse", command=self.savef,
                                       font=("times new roman", 12, "bold italic"),
                                       bg="white", fg="black")
        self.fileopen.place(x=520, y=140)
        self.select_dep_and_indep = tkinter.Button(self.window, text="Enter", command=self.next_page,
                                                   font=("times new roman", 12, "bold italic"), bg="black", fg="white")
        self.select_dep_and_indep.place(x=310, y=200)
        self.window.mainloop()

    def next_page(self):
        if not self.file:
            messagebox.showerror("Error", "Please select a file")
            return
    
        self.window2 = tkinter.Toplevel(self.window)
        self.window2.title("Deployment")
        self.window2.geometry("700x400")
        self.window2.configure(bg="black")
        self.window2_label = tkinter.Label(self.window2, text="Select the X and Y",
                                           font=("times new roman", 14, "italic"), bg="black", fg="white")
        self.window2_label.place(x=240, y=30)
    
        self.Indep_var_select()
        self.dep_var_select()
        
    def Indep_var_select(self):
        self.x_and_y()

        label_dep_dropdown = tkinter.Label(self.window2, text="Select the InDependent Variable:", bg="black", fg="white")
        label_dep_dropdown.place(x=50, y=80)
        self.var_Indep = tkinter.StringVar(self.window2)
        self.var_Indep.set(self.columns[0])

        self.listbox1 = tkinter.Listbox(self.window2, selectmode=tkinter.MULTIPLE)
        self.listbox1.place(width=205, height=220, x=40, y=110)

        for col in self.columns:
            self.listbox1.insert(tkinter.END, col)

        scrollbar1 = tkinter.Scrollbar(self.window2)
        scrollbar1.place(x=245, y=110, height=220)
        self.listbox1.config(yscrollcommand=scrollbar1.set)
        scrollbar1.config(command=self.listbox1.yview)

    def dep_var_select(self):
        self.x_and_y()

        label_dep_dropdown2 = tkinter.Label(self.window2, text="Select the Dependent Variables:", bg="black", fg="white")
        label_dep_dropdown2.place(x=420, y=80)
        self.var_dep = tkinter.StringVar(self.window2)
        self.var_dep.set(self.columns[0])

        self.listbox2 = tkinter.Listbox(self.window2, selectmode=tkinter.SINGLE)
        self.listbox2.place(width=205, height=220, x=400, y=110)

        for col in self.columns:
            self.listbox2.insert(tkinter.END, col)

        scrollbar2 = tkinter.Scrollbar(self.window2)
        scrollbar2.place(x=605, y=110, height=220)
        self.listbox2.config(yscrollcommand=scrollbar2.set)
        scrollbar2.config(command=self.listbox2.yview)
        button_choose1 = tkinter.Button(self.window2, text="Choose", fg="white", bg="black", command=self.indep_select,
                                  font=("times new roman", 8, 'bold'))
        button_choose1.place(x=120, y=350, width=60, height=30)
        button_choose2 = tkinter.Button(self.window2, text="Choose",fg="white",bg="black", command=self.dep_select,font=("times new roman", 10, 'bold italic'))
        button_choose2.place(x=500, y=350, width=60, height=30)

        button_submit = tkinter.Button(self.window2, text = "Submit",fg="white",bg="black",command=self.page3,font=("times new roman", 10, 'bold italic'))
        button_submit.place(x=300, y=350, height=40, width=70)
        self.window2.mainloop()

    def indep_select(self):
        self.indep = []
        clicked = self.listbox1.curselection()
        for index in clicked:
            column_name = self.columns[index]
            self.indep.append(column_name)
        if self.indep:
            messagebox.showinfo("Variables Chosen Successfully", f"Selected Independent variable: {', '.join(self.indep)}")
        else:
            messagebox.showinfo("No Selection", "No independent variables selected.")

    def dep_select(self):
        self.dep = []
        clicked = self.listbox2.curselection()
        for index in clicked:
            column_name = self.columns[index]
            self.dep.append(column_name)
        if self.dep:
            messagebox.showinfo("Variable Chosen Successfully", f"Selected dependent variable: {', '.join(self.dep)}")
        else:
            messagebox.showinfo("No Selection", "No dependent variable selected.")

    def page3(self):
        self.window3 = tkinter.Tk()
        self.window3.title("Deployment")
        self.window3.geometry("700x400")
        self.window3.configure(bg="black")
        self.window3_label = tkinter.Label(self.window3, text="Choose the Algorithm",font=("times new roman", 14, "italic"), bg="black", fg="white")
        self.window3_label.place(x=240, y=20)
        self.button_slr=tkinter.Button(self.window3,text="Simple Linear Regression",bg="black",fg="white",command=self.slr,font=("times new roman",12,"bold italic"))
        self.button_slr.place(x=30,y=50)
        self.button_mlr=tkinter.Button(self.window3,text="Multi linear Regression",bg="black",fg="white",command=self.mlr,font=("times new roman",12,"bold italic"))
        self.button_mlr.place(x=30,y=100)
        self.button_poly = tkinter.Button(self.window3,text="Polynominal Regression",bg="black",fg="white",command=self.poly,font=("times new roman",12,"bold italic"))
        self.button_poly.place(x=30,y=150)
        self.button_logi=tkinter.Button(self.window3,text="Logistic Regression",bg="black",fg="white",command=self.logi,font=("times new roman",12,"bold italic"))
        self.button_logi.place(x=30,y=200)
        self.button_svmr = tkinter.Button(self.window3,text="Support Vector Regresssion",bg="black",fg="white",command=self.svmr,font=("times new roman",12,"bold italic"))
        self.button_svmr.place(x=30,y=250)
        self.button_dtr = tkinter.Button(self.window3,text="Decision Tree Regression",bg="black",fg="white",command=self.dtr,font=("times new roman",12,"bold italic"))
        self.button_dtr.place(x=30,y=300)
        self.button_rtr = tkinter.Button(self.window3,text="Random Forest Regression",bg="black",fg="white",command=self.rfr,font=("times new roman",12,"bold italic"))   
        self.button_rtr.place(x=30,y=350)
        self.button_svmc = tkinter.Button(self.window3,text="Support Vector Classification",bg="black",fg="white",command=self.svmc,font=("times new roman",12,"bold italic"))
        self.button_svmc.place(x=450,y=50)  
        self.button_dtc = tkinter.Button(self.window3,text="Decision Tree Classification",bg="black",fg="white",command=self.dtc,font=("times new roman",12,"bold italic"))
        self.button_dtc.place(x=450,y=100) 
        self.button_rfc = tkinter.Button(self.window3,text="Random Forest Classifier",bg="black",fg="white",command=self.rfc,font=("times new roman",12,"bold italic"))
        self.button_rfc.place(x=450,y=150)
        self.button_naive = tkinter.Button(self.window3,text="Naive Bayes",bg="black",fg="white",command=self.naive,font=("times new roman",12,"bold italic"))
        self.button_naive.place(x=450,y=200)
        self.button_KNN = tkinter.Button(self.window3,text="KNN Classification",bg="black",fg="white",command=self.Knn,font=("times new roman",12,"bold italic"))
        self.button_KNN.place(x=450,y=250)
        self.button_kmeans = tkinter.Button(self.window3,text="Kmeans Clustering",bg="black",fg="white",command=self.kmeans,font=("times new roman",12,"bold italic"))
        self.button_kmeans.place(x=450,y=300)
        self.button_hierar = tkinter.Button(self.window3,text="Hierarchical Clustering",bg="black",fg="white",command=self.hierar,font=("times new roman",12,"bold italic"))
        self.button_hierar.place(x=450,y=350)
                                         
    def slr(self):
        self.x_and_y()
    
        indep_indices = [self.columns.index(col) for col in self.indep]
        dep_indices = [self.columns.index(col) for col in self.dep]
    
        self.X = self.df.iloc[:, indep_indices].values
        self.Y = self.df.iloc[:, dep_indices].values
    
        from sklearn.model_selection import train_test_split
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.X, self.Y, test_size=0.30, random_state=0)
        self.predict_slr()

    def predict_slr(self):
        from sklearn.linear_model import LinearRegression
        regressor = LinearRegression()
        regressor.fit(self.x_train, self.y_train)
        y_pred = regressor.predict(self.x_test)
        mse = mean_squared_error(self.y_test,y_pred)
        r2 = r2_score(self.y_test, y_pred)
        self.MSE=mse
        self.R2=r2*100
        self.window4 = tkinter.Tk()
        self.window4.title("Deployment")
        self.window4.geometry("700x500")
        self.window4.configure(bg="black")
        self.window4_label = tkinter.Label(self.window4, text="Predictions",font=("times new roman", 14, "italic"), bg="black", fg="white")
        self.window4_label.place(x=240, y=10)
        self.window4_label4 = tkinter.Label(self.window4,text="Enter the independent Variable:",font=("times new roman",14,"italic"),bg="black",fg="white")
        self.window4_label4.place(x=80,y=50)
        self.window4_entry4=tkinter.Entry(self.window4,width=15,textvariable=self.predict_slr)        
        self.window4_entry4.place(x=340,y=50)

        self.window4_button=tkinter.Button(self.window4,text="Predict",bg="black",fg="white",command=self.predict_slr_F,font=("times new roman",8,"bold italic"))
        self.window4_button.place(x=460,y=45)
        
        self.window4_label1 = tkinter.Label(self.window4, text="Mean Squared Error", font=("times new roman", 14, "italic"), bg="black", fg="white")
        self.window4_label1.place(x=140, y=80)
        self.mse_formatted = "{:.2f}".format(self.MSE)
        self.window4_entry1 = tkinter.Entry(self.window4, width=15)
        self.window4_entry1.insert(0, str(self.mse_formatted))
        self.window4_entry1.place(x=340, y=80)  
        self.window4_label2 = tkinter.Label(self.window4, text="R2 Score", font=("times new roman", 14, "italic"), bg="black", fg="white")
        self.window4_label2.place(x=140, y=110)  
        self.R2_formatted="{:.2f}".format(self.R2)
        self.window4_entry2 = tkinter.Entry(self.window4, width=15)
        self.window4_entry2.insert(0, str(self.R2_formatted))
        self.window4_entry2.place(x=340, y=110)
        self.plot_regression_results(self.x_test,self.y_test,y_pred)
        
    def plot_regression_results(self, x_test, y_test, y_pred):
        plt.figure(dpi=400)
        figure, ax = plt.subplots(figsize=(5, 4))
        ax.scatter(x_test, y_test, color='blue', label='Actual')
        ax.plot(x_test, y_pred, color='red', label='Predicted')
        ax.set_title('Simple Linear Regression')
        ax.set_xlabel('X-axis')
        ax.set_ylabel('Y-axis')
        ax.legend()
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
      
        canvas = FigureCanvasTkAgg(figure, master=self.window4)
        canvas_widget = canvas.get_tk_widget()
        canvas_widget.place(x=120, y=190)
        
    def predict_slr_F(self):
        try:
            x_value = float(self.window4_entry4.get())
        except ValueError:
            messagebox.showerror("Input Error", "Please enter a valid numeric value.")
            return
    
        from sklearn.linear_model import LinearRegression
        regressor = LinearRegression()
        regressor.fit(self.X, self.Y)
        y_pred = regressor.predict([[x_value]])    
        predicted_value = float(y_pred[0])
    
        prediction_result = f"{predicted_value:.2f}"
        self.window4_entry3 = tkinter.Entry(self.window4, width=15)
        self.window4_entry3.insert(0, str(prediction_result))
        self.window4_entry3.place(x=550, y=50)

                
    def mlr(self):
        self.x_and_y()

        indep_indices = [self.columns.index(col) for col in self.indep]
        dep_indices = [self.columns.index(col) for col in self.dep]

        self.X = self.df.iloc[:, indep_indices].values
        self.Y = self.df.iloc[:, dep_indices].values
    
        from sklearn.model_selection import train_test_split
        self.X_train,self.X_test,self.y_train,self.y_test = train_test_split(self.X, self.Y, test_size=0.2, random_state=0)
        self.predict_mlr()
        
    def predict_mlr(self):
        
        from sklearn.linear_model import LinearRegression
        regressor = LinearRegression()
        regressor.fit(self.X_train,self.y_train)  
        y_pred = regressor.predict(self.X_test)              
        acc = regressor.score(self.X_train,self.y_train)
        self.window4 = tkinter.Tk()
        self.window4.title("Deployment")
        self.window4.geometry("700x400")
        self.window4.configure(bg="black")
        self.window4_label = tkinter.Label(self.window4, text="Predictions",font=("times new roman", 14, "italic"), bg="black", fg="white")
        self.window4_label.place(x=240, y=20)
        self.ACC=acc*100
        self.window4_label3 = tkinter.Label(self.window4, text="Accuracy Score", font=("times new roman", 14, "italic"), bg="black", fg="white")
        self.window4_label3.place(x=120, y=170)  
        self.acc_formatted="{:.2f}".format(self.ACC)
        self.window4_entry3 = tkinter.Entry(self.window4, width=10)
        self.window4_entry3.insert(0, str(self.acc_formatted + " %"))
        self.window4_entry3.place(x=300, y=170)
        self.window4_label4 = tkinter.Label(self.window4,text="Enter the independent Variable:",font=("times new roman",14,"italic"),bg="black",fg="white")
        self.window4_label4.place(x=60,y=50)
        self.window4_entry4=tkinter.Entry(self.window4,width=15,textvariable=self.predict_mlr)        
        self.window4_entry4.place(x=340,y=50)
        self.window4_label5 = tkinter.Label(self.window4,text="Enter the independent Variable 2:",font=("times new roman",14,"italic"),bg="black",fg="white")
        self.window4_label5.place(x=60,y=90)
        self.window4_entry5=tkinter.Entry(self.window4,width=15,textvariable=self.predict_mlr)        
        self.window4_entry5.place(x=340,y=90)
        self.window4_label6 = tkinter.Label(self.window4,text="Enter the independent Variable 3:",font=("times new roman",14,"italic"),bg="black",fg="white")
        self.window4_label6.place(x=60,y=130)
        self.window4_entry6=tkinter.Entry(self.window4,width=15,textvariable=self.predict_mlr)        
        self.window4_entry6.place(x=340,y=130)
        self.window4_button=tkinter.Button(self.window4,text="Predict",bg="black",fg="white",command=self.predict_mlr_F,font=("times new roman",8,"bold italic"))
        self.window4_button.place(x=460,y=45)
        
    def predict_mlr_F(self):
        try:
            x_value = float(self.window4_entry4.get())
            y_value = float(self.window4_entry5.get())
            z_value = float(self.window4_entry6.get())
            
        except ValueError:
            messagebox.showerror("Input Error", "Please enter a valid numeric value.")
            return
        
        import seaborn as sns
        from sklearn.linear_model import LinearRegression
        regressor = LinearRegression()
        regressor.fit(self.X, self.Y)
        y_pred = regressor.predict([[x_value,y_value,z_value]])    
     
        predicted_value = float(y_pred[0])
    
        prediction_result = f"{predicted_value:.2f}"
        self.window4_entry3 = tkinter.Entry(self.window4, width=15)
        self.window4_entry3.insert(0, str(prediction_result))
        self.window4_entry3.place(x=550, y=50) 
    
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        fig, ax = plt.subplots(figsize=(4,3))
        sns.heatmap(self.df.corr(), annot=True, cmap='winter', center=0)
        canvas = FigureCanvasTkAgg(fig, master=self.window4)
        canvas_widget = canvas.get_tk_widget()
        canvas_widget.place(x=400, y=170)
    
        
    def poly(self):
        self.x_and_y()
        
        indep_indices = [self.columns.index(col) for col in self.indep]
        dep_indices = [self.columns.index(col) for col in self.dep]
        
        self.X = self.df.iloc[:, indep_indices].values
        self.Y = self.df.iloc[:, dep_indices].values
        
        from sklearn.model_selection import train_test_split
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(self.X, self.Y, test_size=0.3, random_state=0)
        self.predict_poly()
        
    def predict_poly(self):
        
        from sklearn.linear_model import LinearRegression
        lin_regs = LinearRegression()
        lin_regs.fit(self.X, self.Y)
        from sklearn.preprocessing import PolynomialFeatures
        poly_reg = PolynomialFeatures(degree=5) 
        x_poly = poly_reg.fit_transform(self.X)
                
        regressor = LinearRegression()
        regressor.fit(x_poly, self.Y)               
        acc = regressor.score(x_poly, self.Y)
        self.poly_reg = poly_reg
        self.regressor = regressor
        
        self.window4 = tkinter.Tk()
        self.window4.title("Deployment")
        self.window4.geometry("700x400")
        self.window4.configure(bg="black")
        self.window4_label = tkinter.Label(self.window4, text="Predictions", font=("times new roman", 14, "italic"), bg="black", fg="white")
        self.window4_label.place(x=240, y=20)
        self.ACC = acc * 100
        self.window4_label3 = tkinter.Label(self.window4, text="Accuracy Score", font=("times new roman", 14, "italic"), bg="black", fg="white")
        self.window4_label3.place(x=140, y=100)  
        self.acc_formatted = "{:.2f}".format(self.ACC)
        self.window4_entry3 = tkinter.Entry(self.window4, width=10)
        self.window4_entry3.insert(0, str(self.acc_formatted + " %"))
        self.window4_entry3.place(x=310, y=100)
        self.window4_label4 = tkinter.Label(self.window4, text="Enter the independent Variable:", font=("times new roman", 14, "italic"), bg="black", fg="white")
        self.window4_label4.place(x=80, y=50)
        self.window4_entry4 = tkinter.Entry(self.window4, width=15, textvariable=self.predict_poly)        
        self.window4_entry4.place(x=340, y=50)
    
        self.window4_button = tkinter.Button(self.window4, text="Predict", bg="black", fg="white", command=self.predict_poly_F, font=("times new roman", 8, "bold italic"))
        self.window4_button.place(x=460, y=45)
        
    def predict_poly_F(self):
        try:
            x_value = float(self.window4_entry4.get())
        except ValueError:
            messagebox.showerror("Input Error", "Please enter a valid numeric value.")
            return
    
        from sklearn.preprocessing import PolynomialFeatures
        from sklearn.linear_model import LinearRegression
        import numpy as np

        x_poly = self.poly_reg.transform(np.array([[x_value]]))
        y_pred = self.regressor.predict(x_poly)
        predicted_value = float(y_pred[0])
        
        prediction_result = f"{predicted_value:.2f}"
        self.window4_entry3 = tkinter.Entry(self.window4, width=15)
        self.window4_entry3.insert(0, str(prediction_result))
        self.window4_entry3.place(x=550, y=50)

        self.plot_poly_results(self.X_test, self.Y_test, y_pred)
    
    def plot_poly_results(self, X_test, Y_test, y_pred):
        
        x_poly_plot = self.poly_reg.transform(X_test)
        y_poly_plot = self.regressor.predict(x_poly_plot)
        plt.figure(dpi=400)
        figure, ax = plt.subplots(figsize=(4,3))
        ax.scatter(self.X, self.Y, color='blue', label='Actual')
        ax.plot(X_test,y_poly_plot, color='red', label='Polynomial Regression')
        ax.set_title('Polynomial Regression')
        ax.set_xlabel('X-axis')
        ax.set_ylabel('Y-axis')
        ax.legend()
    
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        
        canvas = FigureCanvasTkAgg(figure, master=self.window4)
        canvas_widget = canvas.get_tk_widget()
        canvas_widget.place(x=190, y=150)

    def logi(self):
        
        self.x_and_y()
        indep_indices = [self.columns.index(col) for col in self.indep]
        dep_indices = [self.columns.index(col) for col in self.dep]
    
        self.X = self.df.iloc[:, indep_indices].values
        self.Y = self.df.iloc[:, dep_indices].values
    
        from sklearn.model_selection import train_test_split
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(self.X, self.Y, test_size=0.30, random_state=0)
        self.predict_logi()
        
    def predict_logi(self):

        from sklearn.preprocessing import StandardScaler
        sc=StandardScaler()
        self.x_train=sc.fit_transform(self.X_train)
        self.x_test=sc.transform(self.X_test)

        from sklearn.linear_model import LogisticRegression
        self.classifier=LogisticRegression()
        self.classifier.fit(self.x_train,self.Y_train)

        y_pred=self.classifier.predict(self.x_test)

        from sklearn.metrics import accuracy_score
        acc=accuracy_score(self.Y_test, y_pred)
        
        self.window4 = tkinter.Tk()
        self.window4.title("Deployment")
        self.window4.geometry("700x400")
        self.window4.configure(bg="black")
        self.window4_label = tkinter.Label(self.window4, text="Predictions",font=("times new roman", 14, "italic"), bg="black", fg="white")
        self.window4_label.place(x=240, y=20)
        self.ACC=acc*100
        self.window4_label3 = tkinter.Label(self.window4, text="Accuracy Score", font=("times new roman", 14, "italic"), bg="black", fg="white")
        self.window4_label3.place(x=140, y=120)  
        self.acc_formatted="{:.2f}".format(self.ACC)
        self.window4_entry3 = tkinter.Entry(self.window4, width=10)
        self.window4_entry3.insert(0, str(self.acc_formatted + " %"))
        self.window4_entry3.place(x=310, y=120)
        self.window4_label4 = tkinter.Label(self.window4, text="Enter the independent Variable:", font=("times new roman", 14, "italic"), bg="black", fg="white")
        self.window4_label4.place(x=80, y=50)
        self.window4_label5 = tkinter.Label(self.window4, text="Enter the independent Variable 2 :", font=("times new roman", 14, "italic"), bg="black", fg="white")
        self.window4_label5.place(x=70, y=90)
        self.window4_entry4 = tkinter.Entry(self.window4, width=15, textvariable=self.predict_logi)        
        self.window4_entry4.place(x=340, y=50)
        self.window4_entry5 = tkinter.Entry(self.window4, width=15, textvariable=self.predict_logi)        
        self.window4_entry5.place(x=340, y=90)
        self.window4_button = tkinter.Button(self.window4, text="Predict", bg="black", fg="white", command=self.predict_logi_F, font=("times new roman", 8, "bold italic"))
        self.window4_button.place(x=460, y=75)       
        
    def predict_logi_F(self):
        try:
            x_value = float(self.window4_entry4.get())
            y_value = float(self.window4_entry5.get())
        except ValueError:
            messagebox.showerror("Input Error", "Please enter valid numeric values.")
            return
        feature_vector = np.array([[x_value, y_value]])
    
        y_pred = self.classifier.predict(feature_vector)
    
        # Display the prediction result (0 or 1)
        prediction_result = "1" if y_pred[0] == 1 else "0"
    
        self.window4_entry3 = tkinter.Entry(self.window4, width=15)
        self.window4_entry3.insert(0, prediction_result)
        self.window4_entry3.place(x=550, y=50)
        self.plot_logi_results()
        
    def plot_logi_results(self):
        from matplotlib.colors import ListedColormap
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        x_set, y_set = self.X_train, self.Y_train
    
        x1, x2 = np.meshgrid(
            np.arange(start=x_set[:, 0].min() - 1, stop=x_set[:, 0].max() + 1, step=1),
            np.arange(start=x_set[:, 1].min() - 1, stop=x_set[:, 1].max() + 1, step=1))
        plt.figure(dpi=400)
        fig, ax = plt.subplots(figsize=(4,3))
        ax.contourf(x1, x2, self.classifier.predict(np.array([x1.ravel(), x2.ravel()]).T).reshape(x1.shape),
                    alpha=0.75, cmap=ListedColormap(('purple', 'green')))    
        ax.set_xlim(x1.min(), x1.max())
        ax.set_ylim(x2.min(), x2.max())
    
        for label in np.unique(y_set):
            class_data = x_set[y_set.flatten() == label]
            ax.scatter(class_data[:, 0], class_data[:, 1], label=label,
                       cmap=ListedColormap(('purple', 'green')), edgecolors='k', s=20)
    
        ax.set_title("Logistic Regression")
        ax.legend()
    
        canvas = FigureCanvasTkAgg(fig, master=self.window4)
        canvas_widget = canvas.get_tk_widget()
        canvas_widget.place(x=120, y=150)
        canvas.draw()
        
    def svmr(self):
        self.x_and_y()
    
        indep_indices = [self.columns.index(col) for col in self.indep]
        dep_indices = [self.columns.index(col) for col in self.dep]

        X = self.df.iloc[:, indep_indices].values
        Y = self.df.iloc[:, dep_indices].values
        Y = Y.reshape(-1,1) 
        self.predict_svmr()
    
    def predict_svmr(self):
        from sklearn.preprocessing import StandardScaler
        from sklearn.model_selection import train_test_split
        from sklearn.svm import SVR
        from sklearn.metrics import mean_squared_error, r2_score
    
        X = self.df.drop(columns=self.dep).values
        Y = self.df[self.dep].values
    
        self.sc_x = StandardScaler()
        X_scaled = self.sc_x.fit_transform(X)
    
        X_train, X_test, Y_train, Y_test = train_test_split(X_scaled,Y, test_size=0.2, random_state=0)
    
        self.regressor = SVR(kernel='rbf')
        self.regressor.fit(X_train, Y_train)
            
        Y_pred = self.regressor.predict(X_test)
    
        # Inverse transform to get predictions in the original scale
        Y_pred = self.sc_x.inverse_transform(Y_pred.reshape(-1,1))
          
        mse = mean_squared_error(Y_test,Y_pred)
        r2 = r2_score(Y_test,Y_pred)
    
        self.MSE = mse
        self.R2 = r2 * 100
    
        self.window4 = tkinter.Tk()
        self.window4.title("Deployment")
        self.window4.geometry("700x500")
        self.window4.configure(bg="black")
        self.window4_label = tkinter.Label(self.window4, text="Predictions", font=("times new roman", 14, "italic"), bg="black", fg="white")
        self.window4_label.place(x=240, y=10)
        self.window4_label4 = tkinter.Label(self.window4, text="Enter the independent Variable:", font=("times new roman", 14, "italic"), bg="black", fg="white")
        self.window4_label4.place(x=80, y=50)
        self.window4_entry4 = tkinter.Entry(self.window4, width=15, textvariable=self.predict_svmr)
        self.window4_entry4.place(x=340, y=50)
    
        self.window4_button = tkinter.Button(self.window4, text="Predict", bg="black", fg="white", command=self.predict_svmr_F, font=("times new roman", 8, "bold italic"))
        self.window4_button.place(x=460, y=45)
    
        self.window4_label1 = tkinter.Label(self.window4, text="Mean Squared Error", font=("times new roman", 14, "italic"), bg="black", fg="white")
        self.window4_label1.place(x=140, y=80)
        self.mse_formatted = "{:.2f}".format(self.MSE)
        self.window4_entry1 = tkinter.Entry(self.window4, width=15)
        self.window4_entry1.insert(0, str(self.mse_formatted))
        self.window4_entry1.place(x=340, y=80)
        self.window4_label2 = tkinter.Label(self.window4, text="R2 Score", font=("times new roman", 14, "italic"), bg="black", fg="white")
        self.window4_label2.place(x=140, y=110)
        self.R2_formatted = "{:.2f}".format(self.R2)
        self.window4_entry2 = tkinter.Entry(self.window4, width=15)
        self.window4_entry2.insert(0, str(self.R2_formatted))
        self.window4_entry2.place(x=340, y=110)
    
    def predict_svmr_F(self):
        try:
            x_value = float(self.window4_entry4.get())
        except ValueError:
            messagebox.showerror("Input Error", "Please enter a valid numeric value.")
            return
    
        
        user_input_scaled = self.sc_x.transform([[x_value]])  # Reshape to (1, -1)
        prediction = self.regressor.predict(user_input_scaled)
            
        prediction_F = self.sc_x.inverse_transform(prediction.reshape(-1,1))
    
        prediction_result = f"{float(prediction_F[0][0]):.2f}"
    
        self.window4_entry3 = tkinter.Entry(self.window4, width=15)
        self.window4_entry3.insert(0, str(prediction_result))
        self.window4_entry3.place(x=550, y=50)
        self.plot_svmr_results()
        
    def plot_svmr_results(self):
        import numpy as np
        import matplotlib.pyplot as plt
        
        # Assuming X and Y are your original data
        original_X = self.df[self.indep].values
        original_Y = self.df[self.dep].values
        
        self.min_excepted_range = 18
        self.max_excepted_range = 65
        
        
        x_1 = np.linspace(self.min_excepted_range, self.max_excepted_range, 100).reshape(-1, 1)
                
        y_p = self.regressor.predict(x_1)
        
        # Inverse transform to get predictions in the original scale
        y_original = self.sc_x.inverse_transform(y_p.reshape(-1, 1))
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        plt.figure(dpi=400)
        fig, ax = plt.subplots(figsize=(4,3))
        plt.scatter(original_X, original_Y, color='red', label='Original Data')  
        plt.plot(x_1, y_original, color="blue", label='SVR Predictions')
        ax.set_title("Logistic Regression")
        ax.legend()
      
        canvas = FigureCanvasTkAgg(fig, master=self.window4)
        canvas_widget = canvas.get_tk_widget()
        canvas_widget.place(x=120, y=150)
        canvas.draw()

    def dtr(self):
        self.x_and_y()
    
        indep_indices = [self.columns.index(col) for col in self.indep]
        dep_indices = [self.columns.index(col) for col in self.dep]
    
        X = self.df.iloc[:, indep_indices].values
        Y = self.df.iloc[:, dep_indices].values
    
        from sklearn.model_selection import train_test_split
        self.X_train,self.X_test,self.Y_train,self.Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
        self.predict_dtr()
    
    def predict_dtr(self):
        
        from sklearn.tree import DecisionTreeRegressor
        regressor = DecisionTreeRegressor()
        regressor.fit(self.X_train,self.Y_train)
        acc = regressor.score(self.X_train,self.Y_train)
        y_pred = regressor.predict(self.X_test)
        self.window4 = tkinter.Tk()
        self.window4.title("Deployment")
        self.window4.geometry("700x400")
        self.window4.configure(bg="black")
        self.window4_label = tkinter.Label(self.window4, text="Predictions",font=("times new roman", 14, "italic"), bg="black", fg="white")
        self.window4_label.place(x=240, y=20)
        self.ACC=acc*100
        self.window4_label3 = tkinter.Label(self.window4, text="Score:", font=("times new roman", 14, "italic"), bg="black", fg="white")
        self.window4_label3.place(x=140, y=90)  
        self.acc_formatted="{:.2f}".format(self.ACC)
        self.window4_entry3 = tkinter.Entry(self.window4, width=10)
        self.window4_entry3.insert(0, str(self.acc_formatted + " %"))
        self.window4_entry3.place(x=310, y=90)
        self.window4_label4 = tkinter.Label(self.window4,text="Enter the independent Variable:",font=("times new roman",14,"italic"),bg="black",fg="white")
        self.window4_label4.place(x=80,y=50)
        self.window4_entry4=tkinter.Entry(self.window4,width=15,textvariable=self.predict_dtr)        
        self.window4_entry4.place(x=340,y=50)

        self.window4_button=tkinter.Button(self.window4,text="Predict",bg="black",fg="white",command=self.predict_dtr_F,font=("times new roman",8,"bold italic"))
        self.window4_button.place(x=460,y=45)
        
    def predict_dtr_F(self):
        try:
            x_value = float(self.window4_entry4.get())
        except ValueError:
            messagebox.showerror("Input Error", "Please enter a valid numeric value.")
            return
    
        from sklearn.tree import DecisionTreeRegressor
        regressor = DecisionTreeRegressor()
        regressor.fit(self.X_train, self.Y_train) 
        y_pred = regressor.predict([[x_value]])
        predicted_value = float(y_pred[0])
    
        prediction_result = f"{predicted_value:.2f}"
        self.window4_entry3 = tkinter.Entry(self.window4, width=15)
        self.window4_entry3.insert(0, str(prediction_result))
        self.window4_entry3.place(x=550, y=50) 
        y_pred_test = regressor.predict(self.X_test)
        plt.figure(dpi=400)
        figure, ax = plt.subplots(figsize=(4, 3))
        ax.scatter(self.X_test, self.Y_test, color='blue', label='Actual', alpha=0.6)  # Plot the actual data points
        ax.plot(self.X_test, y_pred_test, color='red', label='Predicted')  # Plot the predicted line
        ax.set_title('Decision tree Regression')
        ax.legend()
       
        # Embed the Matplotlib plot into the Tkinter window
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        canvas = FigureCanvasTkAgg(figure, master=self.window4)
        canvas_widget = canvas.get_tk_widget()
        canvas_widget.place(x=130, y=130)

    def rfr(self):
        self.x_and_y()
        
        indep_indices = [self.columns.index(col) for col in self.indep]
        dep_indices = [self.columns.index(col) for col in self.dep]
        
        X = self.df.iloc[:, indep_indices].values
        Y= self.df.iloc[:, dep_indices].values
        
        from sklearn.model_selection import train_test_split
        self.X_train,self.X_test,self.Y_train,self.Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
        self.predict_rfr()
        
    def predict_rfr(self):
               
        from sklearn.ensemble import RandomForestRegressor
        regressor=RandomForestRegressor(n_estimators=10,random_state=0)
        regressor.fit(self.X_train,self.Y_train)
        y_pred = regressor.predict(self.X_test)
        acc = regressor.score(self.X_train,self.Y_train)
        self.window4 = tkinter.Tk()
        self.window4.title("Deployment")
        self.window4.geometry("700x400")
        self.window4.configure(bg="black")
        self.window4_label = tkinter.Label(self.window4, text="Predictions",font=("times new roman", 14, "italic"), bg="black", fg="white")
        self.window4_label.place(x=240, y=20)
        self.ACC=acc*100
        self.window4_label3 = tkinter.Label(self.window4, text="Score:", font=("times new roman", 14, "italic"), bg="black", fg="white")
        self.window4_label3.place(x=140, y=90)  
        self.acc_formatted="{:.2f}".format(self.ACC)
        self.window4_entry3 = tkinter.Entry(self.window4, width=10)
        self.window4_entry3.insert(0, str(self.acc_formatted + " %"))
        self.window4_entry3.place(x=310, y=90)
        self.window4_label4 = tkinter.Label(self.window4,text="Enter the independent Variable:",font=("times new roman",14,"italic"),bg="black",fg="white")
        self.window4_label4.place(x=80,y=50)
        self.window4_entry4=tkinter.Entry(self.window4,width=15,textvariable=self.predict_rfr)        
        self.window4_entry4.place(x=340,y=50)

        self.window4_button=tkinter.Button(self.window4,text="Predict",bg="black",fg="white",command=self.predict_rfr_F,font=("times new roman",8,"bold italic"))
        self.window4_button.place(x=460,y=45)
    
    def predict_rfr_F(self):
        try:
            x_value = float(self.window4_entry4.get())
        except ValueError:
            messagebox.showerror("Input Error", "Please enter a valid numeric value.")
            return
    
        from sklearn.ensemble import RandomForestRegressor
        regressor = RandomForestRegressor(n_estimators=10, random_state=0)
        regressor.fit(self.X_train, self.Y_train)
        
        
        y_pred_test = regressor.predict(self.X_test)
        plt.figure(dpi=400)
        figure, ax = plt.subplots(figsize=(4, 3))
        ax.scatter(self.X_test, self.Y_test, color='blue', label='Actual', alpha=0.6) 
        ax.plot(self.X_test, y_pred_test, color='red', label='Predicted')  
        ax.set_title('Random Forest Regression')
        ax.legend()
      
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        canvas = FigureCanvasTkAgg(figure, master=self.window4)
        canvas_widget = canvas.get_tk_widget()
        canvas_widget.place(x=130, y=130)
        
        y_pred = regressor.predict([[x_value]])[0] 
    
        prediction_result = f"{y_pred:.2f}"
        self.window4_entry3 = tkinter.Entry(self.window4, width=15)
        self.window4_entry3.insert(0, str(prediction_result))
        self.window4_entry3.place(x=550, y=50)        
        

    def svmc(self):
        
        self.x_and_y()
        indep_indices = [self.columns.index(col) for col in self.indep]
        dep_indices = [self.columns.index(col) for col in self.dep]
        
        X = self.df.iloc[:, indep_indices].values
        Y= self.df.iloc[:, dep_indices].values  
        
        from sklearn.model_selection import train_test_split
        self.x_train,self.x_test,self.y_train,self.y_test=train_test_split(X,Y,test_size=0.20,random_state=0)
        self.predict_svmc()
        
    def predict_svmc(self):
        
        from sklearn.preprocessing import StandardScaler
        sc=StandardScaler()
        x_train=sc.fit_transform(self.x_train)
        x_test=sc.transform(self.x_test)
    
        from sklearn.svm import SVC
        classifier=SVC(kernel="rbf",random_state=0)
        classifier.fit(x_train,self.y_train)        

        y_pred=classifier.predict(x_test)
        
        from sklearn.metrics import accuracy_score
        acc=accuracy_score(self.y_test,y_pred)
        self.window4 = tkinter.Tk()
        self.window4.title("Deployment")
        self.window4.geometry("700x400")
        self.window4.configure(bg="black")
        self.window4_label = tkinter.Label(self.window4, text="Predictions",font=("times new roman", 14, "italic"), bg="black", fg="white")
        self.window4_label.place(x=240, y=20)
        self.ACC=acc*100
        self.window4_label3 = tkinter.Label(self.window4, text="Accuracy Score", font=("times new roman", 14, "italic"), bg="black", fg="white")
        self.window4_label3.place(x=140, y=120)  
        self.acc_formatted="{:.2f}".format(self.ACC)
        self.window4_entry3 = tkinter.Entry(self.window4, width=10)
        self.window4_entry3.insert(0, str(self.acc_formatted + " %"))
        self.window4_entry3.place(x=310, y=120)
        self.window4_label4 = tkinter.Label(self.window4,text="Enter the independent Variable:",font=("times new roman",14,"italic"),bg="black",fg="white")
        self.window4_label4.place(x=80,y=50)
        self.window4_entry4=tkinter.Entry(self.window4,width=15,textvariable=self.predict_svmc)        
        self.window4_entry4.place(x=340,y=50)
        self.window4_label5 = tkinter.Label(self.window4, text="Enter the independent Variable 2 :", font=("times new roman", 14, "italic"), bg="black", fg="white")
        self.window4_label5.place(x=70, y=90)
        self.window4_entry5 = tkinter.Entry(self.window4, width=15, textvariable=self.predict_svmc)        
        self.window4_entry5.place(x=340, y=90)

        self.window4_button=tkinter.Button(self.window4,text="Predict",bg="black",fg="white",command=self.predict_svmc_F,font=("times new roman",8,"bold italic"))
        self.window4_button.place(x=460,y=45)
                
        from matplotlib.colors import ListedColormap
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        plt.figure(dpi=400)
        fig, ax = plt.subplots(figsize=(4, 3))
        x_set, y_set = self.x_train, self.y_train
    
        x1, x2 = np.meshgrid(
        np.arange(start=x_set[:, 0].min() - 1, stop=x_set[:, 0].max() + 1, step=10.0),
        np.arange(start=x_set[:, 1].min() - 1, stop=x_set[:, 1].max() + 1, step=10.0))

    
        ax.contourf(x1, x2, classifier.predict(np.array([x1.ravel(), x2.ravel()]).T).reshape(x1.shape),
                    alpha=0.75, cmap=ListedColormap(('purple', 'green')))
    
        ax.set_xlim(x1.min(), x1.max())
        ax.set_ylim(x2.min(), x2.max())
    
        for label in np.unique(y_set):
            class_data = x_set[y_set.flatten() == label]
            ax.scatter(class_data[:,0], class_data[:,1], label=label,
                       cmap=ListedColormap(('purple','green')), edgecolors='k',s=20)                
        ax.set_title("Support Vector Classification")
        ax.legend()
    
        canvas = FigureCanvasTkAgg(fig, master=self.window4)
        canvas_widget = canvas.get_tk_widget()
        canvas_widget.place(x=130, y=150)
        canvas.draw()
                
    def predict_svmc_F(self):
        try:
            x_value = float(self.window4_entry4.get())
            y_value = float(self.window4_entry5.get())
        except ValueError:
            messagebox.showerror("Input Error", "Please enter valid numeric values.")
            return
    
        feature_vector = np.array([[x_value, y_value]])
        from sklearn.svm import SVC

        self.classifier=SVC(kernel="rbf",random_state=0)
        self.classifier.fit(self.x_train,self.y_train)

        y_pred = self.classifier.predict(feature_vector)
        
        prediction_result = str(int(y_pred[0]))
        
        self.window4_entry3 = tkinter.Entry(self.window4, width=15)
        self.window4_entry3.insert(0, prediction_result)
        self.window4_entry3.place(x=550, y=50)
               
    def dtc(self):
        
        self.x_and_y()
        indep_indices = [self.columns.index(col) for col in self.indep]
        dep_indices = [self.columns.index(col) for col in self.dep]
        
        X = self.df.iloc[:, indep_indices].values
        Y= self.df.iloc[:, dep_indices].values
        
        from sklearn.model_selection import train_test_split
        self.x_train,self.x_test,self.y_train,self.y_test=train_test_split(X,Y,test_size=0.30,random_state=0)
        self.predict_dtc()
        
    def predict_dtc(self):  
        
        from sklearn.preprocessing import StandardScaler
        sc= StandardScaler()
        x_train=sc.fit_transform(self.x_train)
        x_test=sc.transform(self.x_test)
                
        from sklearn.tree import DecisionTreeClassifier
        classifier=DecisionTreeClassifier(criterion="gini",random_state=0)
        classifier.fit(x_train,self.y_train)

        y_pred=classifier.predict(x_test)
        from sklearn.metrics import accuracy_score
        acc=accuracy_score(self.y_test,y_pred)
        self.window4 = tkinter.Tk()
        self.window4.title("Deployment")
        self.window4.geometry("700x500")
        self.window4.configure(bg="black")
        self.window4_label = tkinter.Label(self.window4, text="Predictions",font=("times new roman", 14, "italic"), bg="black", fg="white")
        self.window4_label.place(x=240, y=20)
        self.ACC=acc*100
        self.window4_label3 = tkinter.Label(self.window4, text="Accuracy Score", font=("times new roman", 14, "italic"), bg="black", fg="white")
        self.window4_label3.place(x=140, y=210)  
        self.acc_formatted="{:.2f}".format(self.ACC)
        self.window4_entry3 = tkinter.Entry(self.window4, width=10)
        self.window4_entry3.insert(0, str(self.acc_formatted + " %"))
        self.window4_entry3.place(x=310, y=210)
        self.window4_label4 = tkinter.Label(self.window4,text="Enter the independent Variable:",font=("times new roman",14,"italic"),bg="black",fg="white")
        self.window4_label4.place(x=80,y=50)
        self.window4_entry4=tkinter.Entry(self.window4,width=15,textvariable=self.predict_dtc)        
        self.window4_entry4.place(x=340,y=50)
        self.window4_label5 = tkinter.Label(self.window4, text="Enter the independent Variable 2 :", font=("times new roman", 14, "italic"), bg="black", fg="white")
        self.window4_label5.place(x=70, y=90)
        self.window4_entry5 = tkinter.Entry(self.window4, width=15, textvariable=self.predict_dtc)        
        self.window4_entry5.place(x=340, y=90)
        self.window4_label6 = tkinter.Label(self.window4, text="Enter the independent Variable 3 :", font=("times new roman", 14, "italic"), bg="black", fg="white")
        self.window4_label6.place(x=70, y=130)
        self.window4_entry6 = tkinter.Entry(self.window4, width=15, textvariable=self.predict_dtc)        
        self.window4_entry6.place(x=340, y=130)
        self.window4_label7 = tkinter.Label(self.window4, text="Enter the independent Variable 4 :", font=("times new roman", 14, "italic"), bg="black", fg="white")
        self.window4_label7.place(x=70, y=170)
        self.window4_entry7 = tkinter.Entry(self.window4, width=15, textvariable=self.predict_dtc)        
        self.window4_entry7.place(x=340, y=170)

        self.window4_button=tkinter.Button(self.window4,text="Predict",bg="black",fg="white",command=self.predict_dtc_F,font=("times new roman",8,"bold italic"))
        self.window4_button.place(x=460,y=45)
        
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        from sklearn import tree
        fig = plt.figure(figsize=(4,3))
        tree.plot_tree(classifier, filled=True)
    
        canvas = FigureCanvasTkAgg(fig, master=self.window4)
        canvas_widget = canvas.get_tk_widget()
        canvas_widget.place(x=400,y=230)
                
    def predict_dtc_F(self):
        try:
            x_value = float(self.window4_entry4.get())
            y_value = float(self.window4_entry5.get())
            z_value = float(self.window4_entry6.get())
            t_value = float(self.window4_entry7.get())
        except ValueError:
            messagebox.showerror("Input Error", "Please enter valid numeric values.")
            return
        
        feature_vector = np.array([[x_value, y_value,z_value,t_value]])
        from sklearn.tree import DecisionTreeClassifier
        classifier=DecisionTreeClassifier(criterion="gini",random_state=0)
        classifier.fit(self.x_train,self.y_train)
        
        y_pred = classifier.predict(feature_vector)
        prediction_result = str(str(y_pred[0]))
        
        self.window4_entry3 = tkinter.Entry(self.window4, width=15)
        self.window4_entry3.insert(0, prediction_result)
        self.window4_entry3.place(x=550, y=50)
        
    def rfc(self):
        
        self.x_and_y()
        indep_indices = [self.columns.index(col) for col in self.indep]
        dep_indices = [self.columns.index(col) for col in self.dep]
        
        X = self.df.iloc[:, indep_indices].values
        Y= self.df.iloc[:, dep_indices].values
        
        from sklearn.model_selection import train_test_split
        self.x_train,self.x_test,self.y_train,self.y_test=train_test_split(X,Y,test_size=.3,random_state=0)
        self.predict_rfc()
        
    def predict_rfc(self):
        
        from sklearn.ensemble import RandomForestClassifier
        classifier=RandomForestClassifier(n_estimators=11,criterion='entropy',random_state=0)
        
        classifier.fit(self.x_train,self.y_train)
        
        y_pred=classifier.predict(self.x_test)
        
        from sklearn.metrics import accuracy_score
        acc=accuracy_score(self.y_test, y_pred)
        self.window4 = tkinter.Tk()
        self.window4.title("Deployment")
        self.window4.geometry("700x500")
        self.window4.configure(bg="black")
        self.window4_label = tkinter.Label(self.window4, text="Predictions",font=("times new roman", 14, "italic"), bg="black", fg="white")
        self.window4_label.place(x=240, y=20)
        self.ACC=acc*100
        self.window4_label3 = tkinter.Label(self.window4, text="Accuracy Score", font=("times new roman", 14, "italic"), bg="black", fg="white")
        self.window4_label3.place(x=140, y=210)  
        self.acc_formatted="{:.2f}".format(self.ACC)
        self.window4_entry3 = tkinter.Entry(self.window4, width=10)
        self.window4_entry3.insert(0, str(self.acc_formatted + " %"))
        self.window4_entry3.place(x=310, y=210)
        self.window4_label4 = tkinter.Label(self.window4,text="Enter the independent Variable:",font=("times new roman",14,"italic"),bg="black",fg="white")
        self.window4_label4.place(x=80,y=50)
        self.window4_entry4=tkinter.Entry(self.window4,width=15,textvariable=self.predict_rfc)        
        self.window4_entry4.place(x=340,y=50)
        self.window4_label5 = tkinter.Label(self.window4, text="Enter the independent Variable 2 :", font=("times new roman", 14, "italic"), bg="black", fg="white")
        self.window4_label5.place(x=70, y=90)
        self.window4_entry5 = tkinter.Entry(self.window4, width=15, textvariable=self.predict_rfc)        
        self.window4_entry5.place(x=340, y=90)
        self.window4_label6 = tkinter.Label(self.window4, text="Enter the independent Variable 3 :", font=("times new roman", 14, "italic"), bg="black", fg="white")
        self.window4_label6.place(x=70, y=130)
        self.window4_entry6 = tkinter.Entry(self.window4, width=15, textvariable=self.predict_rfc)        
        self.window4_entry6.place(x=340, y=130)
        self.window4_label7 = tkinter.Label(self.window4, text="Enter the independent Variable 4 :", font=("times new roman", 14, "italic"), bg="black", fg="white")
        self.window4_label7.place(x=70, y=170)
        self.window4_entry7 = tkinter.Entry(self.window4, width=15, textvariable=self.predict_rfc)        
        self.window4_entry7.place(x=340, y=170)

        self.window4_button=tkinter.Button(self.window4,text="Predict",bg="black",fg="white",command=self.predict_rfc_F,font=("times new roman",8,"bold italic"))
        self.window4_button.place(x=460,y=45)

        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        from sklearn import tree
        fig = plt.figure(figsize=(4,3))
        tree.plot_tree(classifier.estimators_[0], feature_names=["Feature1", "Feature2", "Feature3", "Feature4"], class_names=["setosa", "versicolor", "verginica"], filled=True, fontsize=3)
        

        canvas = FigureCanvasTkAgg(fig, master=self.window4)
        canvas_widget = canvas.get_tk_widget()
        canvas_widget.place(x=280, y=240)
        
    def predict_rfc_F(self):
        
        try:
            x_value = float(self.window4_entry4.get())
            y_value = float(self.window4_entry5.get())
            z_value = float(self.window4_entry6.get())
            t_value = float(self.window4_entry7.get())
        except ValueError:
            messagebox.showerror("Input Error", "Please enter valid numeric values.")
            return
        
        feature_vector = np.array([[x_value, y_value,z_value,t_value]])
        from sklearn.ensemble import RandomForestClassifier
        classifier=RandomForestClassifier(n_estimators=11,criterion='entropy',random_state=0)        
        classifier.fit(self.x_train,self.y_train)
        
        y_pred=classifier.predict(feature_vector)

        prediction_result = str(str(y_pred[0]))
        
        self.window4_entry3 = tkinter.Entry(self.window4, width=15)
        self.window4_entry3.insert(0, prediction_result)
        self.window4_entry3.place(x=550, y=50)
        
    def naive(self):
         
         self.x_and_y()
         indep_indices = [self.columns.index(col) for col in self.indep]
         dep_indices = [self.columns.index(col) for col in self.dep]
         
         X = self.df.iloc[:, indep_indices].values
         Y= self.df.iloc[:, dep_indices].values
         
         from sklearn.model_selection import train_test_split
         self.x_train,self.x_test,self.y_train,self.y_test=train_test_split(X,Y,test_size=.30,random_state=0)
         self.naive_predict()
         
    def naive_predict(self):
        
         from sklearn.naive_bayes import GaussianNB #continuous
         classifer=GaussianNB()
         classifer.fit(self.x_train,self.y_train)

         y_pred=classifer.predict(self.x_test)

         from sklearn.metrics import accuracy_score
         acc = accuracy_score(self.y_test,y_pred)
         self.window4 = tkinter.Tk()
         self.window4.title("Deployment")
         self.window4.geometry("700x500")
         self.window4.configure(bg="black")
         self.window4_label = tkinter.Label(self.window4, text="Predictions",font=("times new roman", 14, "italic"), bg="black", fg="white")
         self.window4_label.place(x=240, y=20)
         self.ACC=acc*100
         self.window4_label3 = tkinter.Label(self.window4, text="Accuracy Score", font=("times new roman", 14, "italic"), bg="black", fg="white")
         self.window4_label3.place(x=140, y=210)  
         self.acc_formatted="{:.2f}".format(self.ACC)
         self.window4_entry3 = tkinter.Entry(self.window4, width=10)
         self.window4_entry3.insert(0, str(self.acc_formatted + " %"))
         self.window4_entry3.place(x=310, y=210)
         self.window4_label4 = tkinter.Label(self.window4,text="Enter the independent Variable:",font=("times new roman",14,"italic"),bg="black",fg="white")
         self.window4_label4.place(x=80,y=50)
         self.window4_entry4=tkinter.Entry(self.window4,width=15,textvariable=self.naive_predict)        
         self.window4_entry4.place(x=340,y=50)
         self.window4_label5 = tkinter.Label(self.window4, text="Enter the independent Variable 2 :", font=("times new roman", 14, "italic"), bg="black", fg="white")
         self.window4_label5.place(x=70, y=90)
         self.window4_entry5 = tkinter.Entry(self.window4, width=15, textvariable=self.naive_predict)        
         self.window4_entry5.place(x=340, y=90)
         self.window4_label6 = tkinter.Label(self.window4, text="Enter the independent Variable 3 :", font=("times new roman", 14, "italic"), bg="black", fg="white")
         self.window4_label6.place(x=70, y=130)
         self.window4_entry6 = tkinter.Entry(self.window4, width=15, textvariable=self.naive_predict)        
         self.window4_entry6.place(x=340, y=130)
         self.window4_label7 = tkinter.Label(self.window4, text="Enter the independent Variable 4 :", font=("times new roman", 14, "italic"), bg="black", fg="white")
         self.window4_label7.place(x=70, y=170)
         self.window4_entry7 = tkinter.Entry(self.window4, width=15, textvariable=self.naive_predict)        
         self.window4_entry7.place(x=340, y=170)

         self.window4_button=tkinter.Button(self.window4,text="Predict",bg="black",fg="white",command=self.naive_predict_F,font=("times new roman",8,"bold italic"))
         self.window4_button.place(x=460,y=45)
         
    def naive_predict_F(self):
        try:
            x_value = float(self.window4_entry4.get())
            y_value = float(self.window4_entry5.get())
            z_value = float(self.window4_entry6.get())
            t_value = float(self.window4_entry7.get())
        except ValueError:
            messagebox.showerror("Input Error", "Please enter valid numeric values.")
            return
        feature_vector = np.array([[x_value, y_value,z_value,t_value]])
        from sklearn.naive_bayes import GaussianNB 
        classifer=GaussianNB()
        classifer.fit(self.x_train,self.y_train)

        y_pred=classifer.predict(feature_vector)
        prediction_result = str(str(y_pred[0]))
        
        self.window4_entry3 = tkinter.Entry(self.window4, width=15)
        self.window4_entry3.insert(0, prediction_result)
        self.window4_entry3.place(x=550, y=50)
        
        
    def Knn(self):
        
        self.x_and_y()
        indep_indices = [self.columns.index(col) for col in self.indep]
        dep_indices = [self.columns.index(col) for col in self.dep]
        
        X = self.df.iloc[:, indep_indices].values
        Y= self.df.iloc[:, dep_indices].values
        
        from sklearn.model_selection import train_test_split
        self.x_train,self.x_test,self.y_train,self.y_test=train_test_split(X,Y,test_size=.30,random_state=0)
        self.knn_predict()
        
    def knn_predict(self):
        
        from sklearn.preprocessing import StandardScaler
        sc=StandardScaler()
        x_train=sc.fit_transform(self.x_train)
        x_test=sc.transform(self.x_test)
        
        from sklearn.neighbors import KNeighborsClassifier
        classifier=KNeighborsClassifier(n_neighbors=5,metric="minkowski",p=2)
        classifier.fit(x_train,self.y_train)

        y_pred=classifier.predict(x_test)

        from sklearn.metrics import accuracy_score
        acc=accuracy_score(self.y_test, y_pred)
        self.window4 = tkinter.Tk()
        self.window4.title("Deployment")
        self.window4.geometry("700x500")
        self.window4.configure(bg="black")
        self.window4_label = tkinter.Label(self.window4, text="Predictions",font=("times new roman", 14, "italic"), bg="black", fg="white")
        self.window4_label.place(x=240, y=20)
        self.ACC=acc*100
        self.window4_label3 = tkinter.Label(self.window4, text="Accuracy Score", font=("times new roman", 14, "italic"), bg="black", fg="white")
        self.window4_label3.place(x=140, y=180)  
        self.acc_formatted="{:.2f}".format(self.ACC)
        self.window4_entry3 = tkinter.Entry(self.window4, width=10)
        self.window4_entry3.insert(0, str(self.acc_formatted + " %"))
        self.window4_entry3.place(x=310, y=180)
        self.window4_label4 = tkinter.Label(self.window4,text="Enter the independent Variable:",font=("times new roman",14,"italic"),bg="black",fg="white")
        self.window4_label4.place(x=80,y=50)
        self.window4_entry4=tkinter.Entry(self.window4,width=15,textvariable=self.knn_predict)        
        self.window4_entry4.place(x=340,y=50)
        self.window4_label5 = tkinter.Label(self.window4, text="Enter the independent Variable 2 :", font=("times new roman", 14, "italic"), bg="black", fg="white")
        self.window4_label5.place(x=70, y=90)
        self.window4_entry5 = tkinter.Entry(self.window4, width=15, textvariable=self.knn_predict)        
        self.window4_entry5.place(x=340, y=90)
        self.window4_button=tkinter.Button(self.window4,text="Predict",bg="black",fg="white",command=self.knn_predict_F,font=("times new roman",8,"bold italic"))
        self.window4_button.place(x=460,y=45)
        
        from matplotlib.colors import ListedColormap
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        plt.figure(dpi=400)
        fig, ax = plt.subplots(figsize=(4, 3))
        x_set, y_set = self.x_train, self.y_train
    
        x1, x2 = np.meshgrid(
        np.arange(start=x_set[:, 0].min() - 1, stop=x_set[:, 0].max() + 1, step=10.0),
        np.arange(start=x_set[:, 1].min() - 1, stop=x_set[:, 1].max() + 1, step=10.0))

    
        ax.contourf(x1, x2, classifier.predict(np.array([x1.ravel(), x2.ravel()]).T).reshape(x1.shape),
                    alpha=0.75, cmap=ListedColormap(('purple', 'green')))
    
        ax.set_xlim(x1.min(), x1.max())
        ax.set_ylim(x2.min(), x2.max())
    
        for label in np.unique(y_set):
            class_data = x_set[y_set.flatten() == label]
            ax.scatter(class_data[:,0], class_data[:,1], label=label,
                       cmap=ListedColormap(('purple','green')), edgecolors='k',s=20)                
        ax.set_title("KNN Classification")
        ax.legend()
    
        canvas = FigureCanvasTkAgg(fig, master=self.window4)
        canvas_widget = canvas.get_tk_widget()
        canvas_widget.place(x=130, y=250)
        canvas.draw()
        
    def knn_predict_F(self):
        try:
            x_value = float(self.window4_entry4.get())
            y_value = float(self.window4_entry5.get())

        except ValueError:
            messagebox.showerror("Input Error", "Please enter valid numeric values.")
            return
        feature_vector = np.array([[x_value, y_value]])
        from sklearn.neighbors import KNeighborsClassifier
        classifier=KNeighborsClassifier(n_neighbors=5,metric="minkowski",p=2)
        classifier.fit(self.x_train,self.y_train)

        y_pred=classifier.predict(feature_vector)
        prediction_result = str(str(y_pred[0]))
        
        self.window4_entry3 = tkinter.Entry(self.window4, width=15)
        self.window4_entry3.insert(0, prediction_result)
        self.window4_entry3.place(x=550, y=50)
        
    def kmeans(self):
        
        self.x_and_y()
        indep_indices = [self.columns.index(col) for col in self.indep]
        X = self.df.iloc[:, indep_indices].values
    
        from sklearn.cluster import KMeans
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg    
        wcss = []
    
        for i in range(1, 10):
            kmeans = KMeans(n_clusters=i, init='k-means++', random_state=0)
            kmeans.fit(X)
            wcss.append(kmeans.inertia_)
        plt.figure(dpi=400)
        fig1, ax1 = plt.subplots(figsize=(4, 3))
        plt.plot(range(1, 10), wcss)
        plt.title("The Elbow Method")
        plt.xlabel("number of clusters")
        plt.ylabel("wcss")
    
        self.window4 = tkinter.Tk()
        self.window4.title("Deployment")
        self.window4.geometry("700x500")
        self.window4.configure(bg="black")
        self.window4_label = tkinter.Label(self.window4, text="Plotting", font=("times new roman", 14, "italic"), bg="black", fg="white")
        self.window4_label.place(x=240, y=40)
        self.window4_label4 = tkinter.Label(self.window4, text="Enter the number of clusters:", font=("times new roman", 14, "italic"), bg="black", fg="white")
        self.window4_label4.place(x=80, y=80)
        self.window4_entry4 = tkinter.Entry(self.window4, width=15)
        self.window4_entry4.place(x=340, y=80)
    
        canvas1 = FigureCanvasTkAgg(fig1, master=self.window4)
        canvas_widget1 = canvas1.get_tk_widget()
        canvas_widget1.place(x=50, y=230)
    
        self.window4_button = tkinter.Button(self.window4, text="Plot", bg="black", fg="white", command=self.kmeans_plot_F, font=("times new roman", 8, "bold italic"))
        self.window4_button.place(x=460, y=75)
    
    def kmeans_plot_F(self):
        try:
            num_clusters = int(self.window4_entry4.get())
        except ValueError:
            messagebox.showerror("Input Error", "Please enter a valid numeric value for the number of clusters.")
            return
        if(num_clusters<=5):
            from sklearn.cluster import KMeans
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
            indep_indices = [self.columns.index(col) for col in self.indep]
            X = self.df.iloc[:, indep_indices].values
        
            kmeans = KMeans(n_clusters=num_clusters, random_state=0)
            X_kmeans = X
            y_kmeans = kmeans.fit_predict(X)
        
            fig2, ax2 = plt.subplots(figsize=(4, 3))
            plt.scatter(X_kmeans[y_kmeans == 0, 0], X_kmeans[y_kmeans == 0, 1], c="red", label="cluster1")
            plt.scatter(X_kmeans[y_kmeans == 1, 0], X_kmeans[y_kmeans == 1, 1], c="blue", label="cluster2")
            plt.scatter(X_kmeans[y_kmeans == 2, 0], X_kmeans[y_kmeans == 2, 1], c="orange", label="cluster3")
            plt.scatter(X_kmeans[y_kmeans == 3, 0], X_kmeans[y_kmeans == 3, 1], c="green", label="cluster4")
            plt.scatter(X_kmeans[y_kmeans == 4, 0], X_kmeans[y_kmeans == 4, 1], c="yellow", label="cluster5")        
            
            canvas2 = FigureCanvasTkAgg(fig2, master=self.window4)
            canvas_widget2 = canvas2.get_tk_widget()
            canvas_widget2.place(x=400, y=230)
        else:
            messagebox.showerror("ERROR","please select the num of clusters below or equal to 5")
        
    def hierar(self):
        
        self.x_and_y()
        indep_indices = [self.columns.index(col) for col in self.indep]
        
        X = self.df.iloc[:,indep_indices].values
        
        import scipy.cluster.hierarchy as sch
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        
        plt.figure(dpi=400)
        fig1, ax1 = plt.subplots(figsize=(4, 3))
        dendrogram=sch.dendrogram(sch.linkage(X,method="centroid"))
        plt.title("dendrogram")
        plt.xlabel("customers")
        plt.ylabel("Eclidean distance")
        plt.show()
        
        self.window4 = tkinter.Tk()
        self.window4.title("Deployment")
        self.window4.geometry("700x500")
        self.window4.configure(bg="black")
        self.window4_label = tkinter.Label(self.window4, text="Plotting", font=("times new roman", 14, "italic"), bg="black", fg="white")
        self.window4_label.place(x=240, y=40)
        self.window4_label4 = tkinter.Label(self.window4, text="Enter the number of clusters:", font=("times new roman", 14, "italic"), bg="black", fg="white")
        self.window4_label4.place(x=80, y=80)
        self.window4_entry4 = tkinter.Entry(self.window4, width=15)
        self.window4_entry4.place(x=340, y=80)

        canvas1 = FigureCanvasTkAgg(fig1, master=self.window4)
        canvas_widget1 = canvas1.get_tk_widget()
        canvas_widget1.place(x=50, y=230)
    
        self.window4_button = tkinter.Button(self.window4, text="Plot", bg="black", fg="white", command=self.hierar_plot_F, font=("times new roman", 8, "bold italic"))
        self.window4_button.place(x=460, y=75)

    def hierar_plot_F(self):
        try:
            num_clusters = int(self.window4_entry4.get())
        except ValueError:
            messagebox.showerror("Input Error", "Please enter a valid numeric value for the number of clusters.")
            return
        if(num_clusters<=5):

            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
            indep_indices = [self.columns.index(col) for col in self.indep]
            X = self.df.iloc[:, indep_indices].values
            
            from sklearn.cluster import AgglomerativeClustering
            hc=AgglomerativeClustering(n_clusters=num_clusters,metric="euclidean",linkage="ward")
            #X_kmeans = X
            y_hc=hc.fit_predict(X)
            plt.figure(dpi=400)
            fig2, ax2 = plt.subplots(figsize=(4, 3))
            plt.scatter(X[y_hc==0,0],X[y_hc==0,1],s=50,c='red',label='cluster1')
            plt.scatter(X[y_hc==1,0],X[y_hc==1,1],s=50,c='green',label='cluster2')
            plt.scatter(X[y_hc==2,0],X[y_hc==2,1],s=50,c='blue',label='cluster3')
            plt.scatter(X[y_hc==3,0],X[y_hc==3,1],s=50,c='cyan',label='cluster4')
            plt.scatter(X[y_hc==4,0],X[y_hc==4,1],s=50,c='magenta',label='cluster5')
            canvas2 = FigureCanvasTkAgg(fig2, master=self.window4)
            canvas_widget2 = canvas2.get_tk_widget()
            canvas_widget2.place(x=400, y=230)
        else:
            messagebox.showerror("ERROR","please select the num of clusters below or equal to 5")

deployment = deploy()
