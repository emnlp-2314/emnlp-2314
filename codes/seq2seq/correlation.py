import json
import matplotlib.pyplot as plt
import numpy as np 
def stat(files):
    for file in files:
        with open(file + '_f1.json') as f:
            f1 = json.load(f)

        with open(file + '_grad_sim.json') as f:
            sim = json.load(f)

        f1_sum = 0
        sim_sum = 0
        for epoch in range(1, 51):
            f1_sum += f1[str(epoch)]
            sim_sum += sum(y for x, y in sim[str(epoch)].items())
        print(max([f1[str(epoch)] for epoch in range(1, 51)]))
        # print(max([sum(y for x, y in sim[str(epoch)].items()) for epoch in range(1, 51)]))
        print(f1_sum / 50, sim_sum / 50)
    # x_data = [f1[str(epoch)] for epoch in range(10, 51)]
    # y_data = [sim[str(epoch)]['02'] for epoch in range(10, 51)]
    # print(np.corrcoef(x_data, y_data))
    # fig = plt.figure()
    # ax = fig.add_subplot(1,1,1)
    # ax.scatter(x_data, y_data)
    # plt.savefig('./' + file + '.jpg')
    # plt.show()

files = ['1260', '1260_1', '1260_2', '1260_3']
stat(files)
print("---------")
files = ['noise_0.15', 'noise_0.15_1', 'noise_0.15_2', 'noise_0.15_3']
stat(files)
print("---------")
files = ['Noise_0.3', 'noise_0.3_1', 'noise_0.3_2', 'noise_0.3_3']
stat(files)
print("---------")
files = ['same', 'same_1', 'same_2']
stat(files)
print("---------")
files = ['noise_0.5_0','noise_0.5_2','noise_0.5_3','noise_0.5_4',]
stat(files)
print("---------")
files = ['NER', 'NER_1', 'NER_2', 'NER_3']
stat(files)
print("---------")
files = ['1260_new', '1260_new_1','1260_new_2','1260_new_3']
stat(files)
files = ['fine_grid_all_0_grad_sim']
stat(files)

