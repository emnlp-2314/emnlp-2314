import json
import numpy as np
def stat(file, parmkey):
    with open(file + '.json') as f:
        examples = json.load(f)
    sim_dic = {}
    for i in range(1, 51):
        for key, dic in examples[str(i)].items():
            if dic != {}:
                if i == 1:
                    sim_dic['shared.weight'] = 0
                    for j in range(0, 12):
                        sim_dic['encoder.block.' + str(j)] = 0
                        sim_dic['decoder.block.' + str(j)] = 0
                        # sim_dic['encoder.block.' + str(j) + 'l'] = []
                        # sim_dic['decoder.block.' + str(j) + 'l'] = []
                sim_dic['shared.weight'] += dic['shared.weight']
                for j in range(0, 12):
                    for key, value in dic.items():
                        if 'encoder.block.' + str(j) in key and parmkey in key:
                            sim_dic['encoder.block.' + str(j)] += value
#                            sim_dic['encoder.block.' + str(j) + 'l'].append(value)
                        if 'decoder.block.' + str(j) in key and parmkey in key:
                            sim_dic['decoder.block.' + str(j)] += value
#                            sim_dic['decoder.block.' + str(j) + 'l'].append(value)
#    for j in range(0, 12):
#        sim_dic['encoder.block.' + str(j) + 'l'] = np.var(sim_dic['encoder.block.' + str(j) + 'l'])
#        sim_dic['decoder.block.' + str(j) + 'l'] = np.var(sim_dic['decoder.block.' + str(j) + 'l'])
    return sim_dic

# stat('fine_grid_min_len_1100_grad_sim', 'SelfAttention')
# stat('fine_grid_original_grad_sim', 'SelfAttention')
# s1 = stat('fine_grid_NER_0_grad_sim', 'DenseReluDense')
# s2 = stat('fine_grid_NER_1_grad_sim', 'DenseReluDense')
# s3 = stat('fine_grid_NER_2_grad_sim', 'DenseReluDense')
# s4 = stat('fine_grid_NER_3_grad_sim', 'DenseReluDense')
s1 = stat('fine_grid_all_0_grad_sim', 'DenseReluDense')
s2 = stat('fine_grid_all_1_grad_sim', 'DenseReluDense')
s3 = stat('fine_grid_all_2_grad_sim', 'DenseReluDense')
s4 = stat('fine_grid_all_3_grad_sim', 'DenseReluDense')
for key, value in s1.items():
    s1[key] += s2[key]
    s1[key] += s3[key]
    s1[key] += s4[key]
    
# stat('fine_grid_min_len_1100_grad_sim', 'layer_norm')
# stat('fine_grid_original_grad_sim', 'layer_norm')

s= sorted(s1.items(),key=lambda x:x[1])
b = [i[0] for i in s]
print(b)