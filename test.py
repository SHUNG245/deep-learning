import torch as t
import pandas as pd
#a = t.arange(0, 12).view(2, 3,2)
a=t.arange(3,5).view(2)
x=a.numpy()
print(len(x))
X=[3,4,5]
print(X[-1])
for e, o in zip([1,2,3],[4,5,6]):
    print(e,o)
ESTIMATES=[1,2,3]
OBSERVATION=[4,5,6]
df_e=pd.DataFrame(ESTIMATES,orient='index',columns=['estimate'])
df_o=pd.DataFrame.from_arrays(OBSERVATION,orient='index',columns=['observation'])
df = pd.concat([df_e, df_o], axis=1)
print(df)
# b = a.view(-1, 2,2)  # 当某一维是-1时，会自动计算它的大小
# print(b)
# print(b.size())
# b=b[:,-1,:]
# a=a[:,-1,:]
# print(b)
# print(a)
