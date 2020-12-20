import tkinter as tk
import tkinter.font as tkFont



class App:
    def __init__(self, root):
        #setting title
        root.title("undefined")
        #setting window size
        width=600
        height=500
        screenwidth = root.winfo_screenwidth()
        screenheight = root.winfo_screenheight()
        alignstr = '%dx%d+%d+%d' % (width, height, (screenwidth - width) / 2, (screenheight - height) / 2)
        root.geometry(alignstr)
        root.resizable(width=False, height=False)
        

        GLabel_433=tk.Label(root)
        ft = tkFont.Font(family='Times',size=18)
        GLabel_433["font"] = ft
        GLabel_433["fg"] = "#00ced1"
        GLabel_433["justify"] = "center"
        GLabel_433["text"] = "CURRENT TREND OF CLOTH COLOR AND TYPE"
        GLabel_433["relief"] = "groove"
        GLabel_433.place(x=20,y=40,width=530,height=81)

        GLabel_772=tk.Label(root)
        GLabel_772["bg"] = "#00ced1"
        ft = tkFont.Font(family='Times',size=10)
        GLabel_772["font"] = ft
        GLabel_772["fg"] = "#393d49"
        GLabel_772["justify"] = "center"
        GLabel_772["text"] = "COLOR OF THE CLOTHS"
        GLabel_772.place(x=40,y=170,width=227,height=61)

        GLabel_332=tk.Label(root)
        GLabel_332["bg"] = "#f9fbfb"
        ft = tkFont.Font(family='Times',size=10)
        GLabel_332["font"] = ft
        GLabel_332["fg"] = "#f5fbfb"
        GLabel_332["justify"] = "center"
        GLabel_332["text"] = ""
        GLabel_332.place(x=310,y=170,width=226,height=60)

        GLabel_403=tk.Label(root)
        GLabel_403["bg"] = "#00ced1"
        ft = tkFont.Font(family='Times',size=10)
        GLabel_403["font"] = ft
        GLabel_403["fg"] = "#333333"
        GLabel_403["justify"] = "center"
        GLabel_403["text"] = "TYPE OF THE CLOTHS"
        GLabel_403.place(x=40,y=270,width=226,height=63)

        GLabel_399=tk.Label(root)
        GLabel_399["bg"] = "#f3f9f9"
        ft = tkFont.Font(family='Times',size=10)
        GLabel_399["font"] = ft
        GLabel_399["fg"] = "#333333"
        GLabel_399["justify"] = "center"
        GLabel_399["text"] = ""
        GLabel_399.place(x=310,y=270,width=228,height=60)

        GButton_816=tk.Button(root)
        GButton_816["bg"] = "#00ced1"
        ft = tkFont.Font(family='Times',size=10)
        GButton_816["font"] = ft
        GButton_816["fg"] = "#000000"
        GButton_816["justify"] = "center"
        GButton_816["text"] = "CLOSE"
        GButton_816.place(x=140,y=380,width=99,height=34)
        GButton_816["command"] = self.GButton_816_command

        GButton_52=tk.Button(root)
        GButton_52["bg"] = "#00ced1"
        ft = tkFont.Font(family='Times',size=10)
        GButton_52["font"] = ft
        GButton_52["fg"] = "#000000"
        GButton_52["justify"] = "center"
        GButton_52["text"] = "TRY AGAIN"
        GButton_52.place(x=300,y=380,width=92,height=33)
        GButton_52["command"] = self.GButton_52_command

    def GButton_816_command(self):
        print("command")


    def GButton_52_command(self):
        print("command")

if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
