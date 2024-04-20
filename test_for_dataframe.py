import pandas as pd
df1 = pd.DataFrame({'key': ['A', 'B', 'C', 'D'],
                    'value': [1, 2, 3, 4]})

df2 = pd.DataFrame({'key': ['B', 'D', 'D', 'E'],
                    'value': [5, 6, 7, 8]})
merged_df = pd.merge(df1, df2, on='key', how='inner')
print(merged_df)


# 创建两个DataFrame  
df1 = pd.DataFrame({  
    'key': ['A', 'B', 'C', 'D'],  
    'value1': [1, 2, 3, 4]  
})  
  
df2 = pd.DataFrame({  
    'key': ['B', 'C', 'E', 'F'],  
    'value2': [5, 6, 7, 8]  
})  
  
# 使用merge函数进行合并，how='right'  
merged_df = pd.merge(df1, df2, on='key', how='right')  
  
print(merged_df)