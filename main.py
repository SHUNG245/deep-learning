# This is a sample Python script.
import pandas as pd
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')
    dic={'a':1,'b':2,'c':3}
    dic1={'c':3.5,'b':5.5,'a':6}
    df=pd.DataFrame.from_dict(dic,orient='index',columns=['r2'])
    df1=pd.DataFrame.from_dict(dic1,orient='index',columns=['r2'])
    df2=pd.concat([df,df1],axis=1)
    print(df)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
