from torchvision.utils import save_image
import torch
import matplotlib.pyplot as plt


class Log:
    def __init__(self):
        self.dic = {
            'G_AA': [],  # Identity loss : L1
            'G_AB': [],  # Discriminator loss : MSE
            'G_ABA': [],  # Double pass loss : L1

            'G_BB': [],
            'G_BA': [],
            'G_BAB': [],

            'D_A_real': [],  # Prediction on real samples
            'D_A_fake': [],  # Prediction on fake samples

            'D_B_real': [],
            'D_B_fake': [],
        }

    def clear(self):
        for key in self.dic:
            self.dic[key] = []

    def update(self, L):
        self.dic['G_AA'].append(L[0].item())
        self.dic['G_AB'].append(L[1].item())
        self.dic['G_ABA'].append(L[2].item())

        self.dic['G_BB'].append(L[3].item())
        self.dic['G_BA'].append(L[4].item())
        self.dic['G_BAB'].append(L[5].item())

        self.dic['D_A_real'].append(L[6].item())
        self.dic['D_B_real'].append(L[7].item())

        self.dic['D_A_fake'].append(L[8].item())
        self.dic['D_B_fake'].append(L[9].item())

    def plot_img(self, c_epoch, data_A, data_B, fake_AB, fake_AA, fake_BA, fake_BB):
        save_image(torch.cat(((data_A*0.5)+0.5, (fake_AB*0.5)+0.5, (fake_AA*0.5)+0.5), dim=3),
                   'img{}_AB.png'.format(c_epoch))
        save_image(torch.cat(((data_B*0.5)+0.5, (fake_BA*0.5)+0.5, (fake_BB*0.5)+0.5), dim=3),
                   'img{}_BA.png'.format(c_epoch))

    def plot_graph(self):
        plt.figure(figsize=(10, 10))
        for key in self.dic:
            plt.plot(self.dic[key], label=key)
        plt.legend()
        plt.savefig('figure.png')
        plt.close()

    def save(self, save_path):
        with open(save_path, 'w') as f:
            f.write(str(self.dic))
