import matplotlib.pyplot as plt
import numpy as np
import json

F1_all = []
F1_ans = []
F1_unans = []
EM_all = []
EM_ans = []
EM_unans = []
for i in range(5):
    with open(f'0{10-(2*i+1)}.json', 'r') as f:
        result = json.load(f)
        F1_all.append(result['overall']['f1'])
        F1_ans.append(result['answerable']['f1'])
        F1_unans.append(result['unanswerable']['f1'])
        EM_all.append(result['overall']['em'])
        EM_ans.append(result['answerable']['em'])
        EM_unans.append(result['unanswerable']['em'])
        
x = [0.1, 0.3, 0.5, 0.7, 0.9]
fig, (ax1, ax2) = plt.subplots(1, 2)

fig.suptitle('performance of different threshold')
ax1.plot(x, F1_all, 'o-', color='blue', label='overall')
ax1.plot(x, F1_ans, 'o-', color='orange', label='answerable')
ax1.plot(x, F1_unans, 'o-', color='green', label='unanswerable')
ax1.set_title('F1')
ax1.set_xlabel('threshold')
ax1.set_xticks(x) 
ax1.set_xticklabels(x)
ax2.plot(x, EM_all, 'o-', color='blue', label='overall')
ax2.plot(x, EM_ans, 'o-', color='orange', label='answerable')
ax2.plot(x, EM_unans, 'o-', color='green', label='unanswerable')
ax2.set_title('EM')
ax2.set_xlabel('threshold')
ax2.set_xticks(x) 
ax2.set_xticklabels(x)
fig.savefig('threshold.png')