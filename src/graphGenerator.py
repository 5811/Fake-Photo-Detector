import re, sys
import matplotlib.pyplot as plt
if len(sys.argv) < 3:
    print(f"usage: {sys.argv[0]} <filename> <plot title>")
    exit()
with open(sys.argv[1], 'r') as f:
    lines = [line for line in f.readlines() if line.startswith('Test set: Accuracy:')]
accuracy = [re.search('([\d\.]*)%', line).group(1) for line in lines]
accuracy = [round(float(a), 5) for a in accuracy]
accuracy.insert(0, 50)
plt.figure(figsize=(50,50))
plt.plot(accuracy)
plt.ylim(bottom=50, top=100)
plt.ylabel('Accuracy (%)', fontsize=20)
plt.xlabel('Number of epochs', fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.title(sys.argv[2], fontsize=25)
plt.show()
