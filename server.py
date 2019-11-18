# -*- coding: utf-8 -*-
"""
Created on Sat Nov 16 23:41:11 2019

@author: nikol
"""

import pickle
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, url_for




app = Flask(__name__)

@app.route('/')
def Index():
   return render_template("index.html")


@app.route('/api',methods=['POST'])
def Api():
    if request.method=='POST':
        dataset = pd.read_csv('heart.csv')
        dataset = pd.get_dummies(dataset, columns = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal'])
        age = float (request.form['age'])
        sex = (request.form['sex'])
        cp = (request.form['cp'])
        chol = float (request.form['chol'])
        fbs = (request.form['fbs'])
        trestbps = float (request.form['trestbps'])
        restecg = (request.form['restecg'])
        thalach =float ( request. form['thalach'])
        exang =  (request.form['exang'])
        oldpeak = float (request.form['oldpeak'])
        slope = (request.form['slope'])
        ca = (request.form['ca'])
        thal = (request.form['thal'])
        
        a = (age-(np.mean(dataset["age"])))
        age = (a/np.std(dataset["age"]))

        tres = (trestbps-(np.mean(dataset["trestbps"])))
        trestbps = (tres/np.std(dataset["trestbps"]))

        c = (chol-(np.mean(dataset["chol"])))
        chol = (c/np.std(dataset["chol"]))

        t = (thalach-(np.mean(dataset["thalach"])))
        thalach = (t/np.std(dataset["thalach"]))

        o = (oldpeak-(np.mean(dataset["oldpeak"])))
        oldpeak = (o/np.std(dataset["oldpeak"]))
       
        
        new_row={"age":age,"sex":sex,"cp":cp,"trestbps":trestbps,
                    "chol":chol,
                    "fbs":fbs,
                    "restecg":restecg,
                    "thalach":thalach,
                    "exang":exang,
                    "oldpeak":oldpeak,
                    "slope":slope,
                    "ca":ca,
                    "thal":thal,
                    "target":1
                    }
        dataset5=pd.read_csv('copia.csv')
        dataset5 = dataset5.append(new_row, ignore_index=True)
        dataset5 = pd.get_dummies(dataset5, columns = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal'])
        dataset5=dataset5.reindex(columns = dataset.columns.values, fill_value=0)
        
        
        
        
     
      
        

        model = pickle.load(open('model.pkl','rb'))
        
        X3 = dataset5.drop(['target'], axis = 1)
        #score =str(model.predict_proba(X3))
     
       
        s = "Llevar una dieta equilibrada y saludable es esencial para prevenir enfermedades cardiovasculares. Elegir los alimentos adecuados ayudará a cuidarte y mantener una buena salud cardiovascular. Aquí podrá encontrar consejos nuetricionales específicos para ayudar en la prevención de problemas del corazón"
        
        #print(X3)
        
        if model.predict(X3)==[1]:
            a= "Enfermedad Cardiaca Detectada con :" +str (model.predict_proba(X3)[:,1]) +"de probabilidad"
            
            b = " Posibles enfermedades: " 
        elif model.predict(X3)==[0]:
             b="No se detectó ninguna enfermedad cardiaca" 

                
    return render_template('diag.html', d = b, c=a)


if __name__=='__main__':
    app.run(port=3000, debug=False)