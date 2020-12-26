import tkinter as tk
import tkinter.font as tkFont

class App:
    def __init__(self, root):
        #setting title
        root.title("undefined")
        #setting window size
        width=685
        height=495
        screenwidth = root.winfo_screenwidth()
        screenheight = root.winfo_screenheight()
        alignstr = '%dx%d+%d+%d' % (width, height, (screenwidth - width) / 2, (screenheight - height) / 2)
        root.geometry(alignstr)
        root.resizable(width=False, height=False)

        GLabel_433=tk.Label(root)
        ft = tkFont.Font(family='Times',size=36)
        GLabel_433["font"] = ft
        GLabel_433["fg"] = "#00ced1"
        GLabel_433["justify"] = "center"
        GLabel_433["text"] = "CURRENT TREND OF CLOTH COLOR AND TYPE"
        GLabel_433["relief"] = "groove"
        GLabel_433.place(x=20,y=40,width=530,height=81)

        GButton_816=tk.Button(root)
        GButton_816["bg"] = "#00ced1"
        ft = tkFont.Font(family='Times',size=10)
        GButton_816["font"] = ft
        GButton_816["fg"] = "#000000"
        GButton_816["justify"] = "center"
        GButton_816["text"] = "CLOSE"
        GButton_816.place(x=230,y=380,width=99,height=34)
        GButton_816["command"] = self.GButton_816_command

        GButton_784=tk.Button(root)
        GButton_784["bg"] = "#00ced1"
        ft = tkFont.Font(family='Times',size=10)
        GButton_784["font"] = ft
        GButton_784["fg"] = "#000000"
        GButton_784["justify"] = "center"
        GButton_784["text"] = "Download Images from Instagram"
        GButton_784.place(x=50,y=190,width=191,height=51)
        GButton_784["command"] = self.GButton_784_command

        GButton_756=tk.Button(root)
        GButton_756["bg"] = "#00ced1"
        ft = tkFont.Font(family='Times',size=10)
        GButton_756["font"] = ft
        GButton_756["fg"] = "#000000"
        GButton_756["justify"] = "center"
        GButton_756["text"] = "Filter the Female Images"
        GButton_756.place(x=370,y=190,width=193,height=51)
        GButton_756["command"] = self.GButton_756_command

        GButton_973=tk.Button(root)
        GButton_973["bg"] = "#00ced1"
        ft = tkFont.Font(family='Times',size=10)
        GButton_973["font"] = ft
        GButton_973["fg"] = "#000000"
        GButton_973["justify"] = "center"
        GButton_973["text"] = "Find the Current Cloth Type"
        GButton_973.place(x=50,y=270,width=192,height=50)
        GButton_973["command"] = self.GButton_973_command

        GButton_649=tk.Button(root)
        GButton_649["bg"] = "#00ced1"
        ft = tkFont.Font(family='Times',size=10)
        GButton_649["font"] = ft
        GButton_649["fg"] = "#000000"
        GButton_649["justify"] = "center"
        GButton_649["text"] = "Find the Current Cloth Color"
        GButton_649.place(x=370,y=270,width=191,height=52)
        GButton_649["command"] = self.GButton_649_command

    def GButton_816_command(self):
        print("command")


    def GButton_784_command(self):
        print("command")


    def GButton_756_command(self):
        print("command")


    def GButton_973_command(self):
        print("command")


    def GButton_649_command(self):
        print("command")

if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
