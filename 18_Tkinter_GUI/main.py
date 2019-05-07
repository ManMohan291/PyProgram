import tkinter as tk  
from tkinter import ttk
from tkinter import messagebox

#################################################################################################
#################################Background Events###############################################
#################################################################################################
def btnOK_click():
    lblLoginId.configure(foreground='red')
    messagebox.showinfo("Error", "Invalid Login \n Server:"+gServerName.get() +"\n Login ID:"+gLoginId.get()+"\n SQL Mode:"+str(gServerType.get())+"\n Remember Me:"+str(gRememberMe.get())+"\n Remember Password:"+str(gRememberPassword.get()) )


    return
def btnCancel_click():
    result=messagebox.askyesno("Quit", "Quit??")
    if(result==True):
        frmMain.quit()

    return

#################################################################################################
#################################Screen Design###################################################
#################################################################################################

#WinForm Initialize
#################################
frmMain=tk.Tk()
frmMain.title("Application Title")
frmMain.geometry("500x500")

#Variables to Input Box
#################################
gLoginId=tk.StringVar()
gServerName=tk.StringVar()
gServerType=tk.IntVar()
gRememberMe=tk.BooleanVar()
gRememberPassword=tk.BooleanVar()


#Frame Initialize
#################################
frameTop=tk.LabelFrame(frmMain,text="Enter Login Details",width=500,height=100)
frameTop.grid(row=0,column=1,sticky=(tk.N, tk.S, tk.W, tk.E))

frameBottom=tk.LabelFrame(frmMain,text="")
frameBottom.grid(row=2,column=1,sticky=(tk.N, tk.S, tk.W, tk.E))


#Label Initialize
#################################
lblLoginId=tk.Label(frameTop,text="Login Id:")
lblLoginId.grid(row=0,column=0,sticky=tk.E)


#TextBox Initialize
#################################
txtLoginId=tk.Entry(frameTop,textvariable=gLoginId)
txtLoginId.grid(row=0,column=1,sticky=tk.W)



#ComboBox Initialize
#################################
lblServer=tk.Label(frameTop,text="Server IP:")
lblServer.grid(row=1,column=0,sticky=tk.E)

cboServer=ttk.Combobox(frameTop,textvariable=gServerName)
cboServer['values']=['192.168.10.133','10.26.78.96']
cboServer.grid(row=1,column=1,sticky=tk.W)


#Option/Radio Buttion
#################################
radMySQL=tk.Radiobutton(frameTop,text="MySQL Server",variable=gServerType,value=11)
radSQLServer=tk.Radiobutton(frameTop,text="MS SQL Server",variable=gServerType,value=22)
radMySQL.grid(row=2,column=1,sticky=tk.W)
radSQLServer.grid(row=3,column=1,sticky=tk.W)

#CheckBox Buttion
#################################
chkRememberMe=tk.Checkbutton(frameTop,text="Remember Me",variable=gRememberMe)
chkRememberMe.select()
chkRememberPassword=tk.Checkbutton(frameTop,text="Remember Password",variable=gRememberPassword,state="disabled")
chkRememberPassword.deselect()
chkRememberMe.grid(row=4,column=0,sticky=tk.E)
chkRememberPassword.grid(row=4,column=1,sticky=tk.W)


#Button Initialize
#################################
btnOK=tk.Button(frameBottom,text="OK",command=btnOK_click)
btnOK.grid(row=0,column=0,sticky=tk.E)
btnCancel=tk.Button(frameBottom,text="Cancel",command=btnCancel_click)
btnCancel.grid(row=0,column=1,sticky=tk.W)


#Show Window
#################################
frmMain.width=300
frmMain.height=300
frmMain.mainloop()


