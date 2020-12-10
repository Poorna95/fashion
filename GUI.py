# -*- coding: utf-8 -*-
"""
Created on Fri Nov  6 23:20:44 2020

@author: Poorna
"""
import tkinter  
from tkinter import*

root = Tk()

#Create a label widget
myLabel1 = tkinter.Label(root, text="New Cloth Type And Color Of the Cloth in Female Fashion Industry! ")
myLabel1.grid(row=0,column=0)
myLabel2 = tkinter.Label(root, text="Type of the cloth: ")
myLabel2.grid(row=3,column=0)
myLabel4 = tkinter.Label(root, text="The type of the cloth is ")
myLabel4.grid(row=4,column=0)
myButton1= tkinter.Button(root, text="Type of the Cloth", padx= 50) 
myButton1.grid(row=3,column=1)
myLabel3 = tkinter.Label(root, text="Color Of the cloth: ")
myLabel3.grid(row=5,column=0)
myLabel5= tkinter.Label(root, text="The color of the cloth is ")
myLabel5.grid(row=6,column=0)
myButton2 =tkinter.Button(root, text="Color of the cloth", padx=50)
myButton2.grid(row=5,column=1)

#Showing it into the screen




root.mainloop()




