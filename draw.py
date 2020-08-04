import numpy as np
import matplotlib.pyplot as plt
import re

def read_output(filename):
    loss = []
    pattern = "Run: 01, Epoch: (\d+), Loss: (\d?).(\d*),.*"
    with open(filename) as fin:
        for i, line in enumerate(fin):
            if i == 500:
                break

            m = re.match(pattern, line)
            num = float(m.group(2) + '.' + m.group(3))
            loss.append(num)

    return loss

#print(read_output("output_MLP.txt"))
plt.plot(np.arange(1, 501), read_output("output_MLP.txt"), label="MLP")
# plt.plot(np.arange(1, 501), read_output("output_inductive.txt"), label="inductive GCN")
plt.plot(np.arange(1, 501), read_output("output_transductive.txt"), label="transductive GCN")
plt.plot(np.arange(1, 501), read_output("gcn_inductive.txt"), label="new GCN inductive")
plt.legend()
plt.show()

