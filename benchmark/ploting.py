import matplotlib.pyplot as plt
import json

path = "./loss"
target_file = path + "/cyclegan_mbs_128_loss_value"
file_format = '.json'
with open(target_file + file_format) as json_file:
    json_data = json.load(json_file)
    
    g_loss = []
    dA_loss = []
    dB_loss = []
    for index in json_data:
        g_loss.append(json_data[index]['g_loss'])
        dA_loss.append(json_data[index]['A_loss'])
        dB_loss.append(json_data[index]['B_loss'])

    plt.subplot(121)
    plt.plot(g_loss)
    plt.legend(['G loss'])
    plt.ylabel('loss-value')
    plt.xlabel('epoch')
    plt.xlim([0, 100])

    plt.subplot(122)
    plt.plot(dA_loss, 'r', dB_loss, 'b')
    plt.legend(['D_A loss', 'D_B loss'])
    plt.ylabel('loss-value')
    plt.xlabel('epoch')
    plt.xlim([0, 100])
    plt.savefig(f'{target_file}.png', dpi=600)