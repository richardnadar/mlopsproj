#!/usr/bin/env python
# coding: utf-8

# In[2]:


line_index = 115
lines = None
with open('cnncode.py', 'r') as file_handler:
    lines = file_handler.readlines()

lines.insert(line_index, 'CRP(6)\n')

with open('cnncode.py', 'w') as file_handler:
    file_handler.writelines(lines)


# In[ ]:




