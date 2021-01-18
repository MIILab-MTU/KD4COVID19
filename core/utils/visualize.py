import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import imageio
import numpy as np


class DataVisualizer(object):
    def __init__(self, data_for_per_row, save_path):
        self.data = data_for_per_row
        self.save_path = save_path

    def visualize(self, num_per_row=32):
        f, plots = plt.subplots(len(self.data), num_per_row, sharex='all', sharey='all', figsize=(num_per_row, len(self.data)))

        for row in range(len(self.data)):
            for i in range(self.data[row].shape[2]):
                plots[(i + num_per_row * row) // num_per_row, i % num_per_row].axis('off')
                plots[(i + num_per_row * row) // num_per_row, i % num_per_row].imshow(self.data[row][:, :, i], cmap='gray')

        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0.01, wspace=0.01)
        plt.margins(0, 0)

        plt.savefig(self.save_path)
        plt.close()

    def visualize_np(self, num_per_row=32, patch_size=32, spacing=1):
        height = patch_size * len(self.data) + spacing * (len(self.data) - 1)
        width = patch_size * num_per_row + spacing * (num_per_row - 1)
        img = np.zeros((height, width), dtype=np.uint8)
        for row in range(len(self.data)):
            for col in range(num_per_row):
                patch_data = self.data[row][:, :, col]
                patch_data = np.array((patch_data - np.min(patch_data)) / (np.max(patch_data) - np.min(patch_data)) * 255., dtype=np.uint8)

                img[row*patch_size + row*spacing: (row+1)*patch_size + row*spacing,
                    col*patch_size + col*spacing: (col+1)*patch_size + col*spacing] = patch_data
        imageio.imsave(self.save_path, img)


class DataVisualizerGIF(object):

    def __init__(self, data_for_per_row, save_path, patient_name):
        self.data = data_for_per_row
        self.save_path = save_path
        self.patient_name = patient_name

    def generate_gif(self, file_paths, gif_name, duration=0.2):
        frames = []
        for image_name in file_paths:
            frames.append(imageio.imread(image_name))
        imageio.mimsave(gif_name, frames, 'GIF', duration=duration)

    def visualize(self, num_per_row=32):
        image_file_names = []
        for r in range(num_per_row):
            f, plots = plt.subplots(len(self.data), 1, sharex='all', sharey='all', figsize=(1, len(self.data)))
            for row in range(len(self.data)):
                plots[row].axis('off')
                if len(self.data[row].shape)==3:
                    plots[row].imshow(self.data[row][:, :, r])
                else:
                    plots[row].imshow(self.data[row][:, :, r, :])

            plt.gca().xaxis.set_major_locator(plt.NullLocator())
            plt.gca().yaxis.set_major_locator(plt.NullLocator())
            plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0.01, wspace=0.01)
            plt.margins(0, 0)

            plt.savefig(self.save_path+"/%s_%d.png" % (self.patient_name, r))
            image_file_names.append(self.save_path+"/%s_%d.png" % (self.patient_name, r))
            plt.close()

        self.generate_gif(image_file_names, self.save_path+"/%s_gif.gif" % (self.patient_name))

        return image_file_names
