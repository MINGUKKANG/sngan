import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import time
import os

def plot_images(images, H_n, W_n, save_dir, name):
    '''
    Args
        images: Images that we want to plot, train_xs[0:100]
        H_n: Number of images to stack horizontally, 10
        W_n: Number of images to stack vertically, 10
        save_dir: "./plot/train"
        name: saving name, "ori_images"
    '''
    shape = np.shape(images)

    if len(shape) == 4:
        H = shape[1]
        W = shape[2]
        c = shape[3]
    else:
        H = shape[0]
        W = shape[1]
        c = shape[2]

    x = np.linspace(-2, 2, W_n)
    y = np.linspace(-2, 2, H_n)

    if c == 1:
        canvas = np.empty((H_n * H, W_n * W))
        for i, yi in enumerate(y):
            for j, xi in enumerate(x):
                img = (np.reshape(images[H_n * i + j], [H, W])+1)/2
                canvas[H * i: H * i + H, W * j: W * j + W] = img
        plt.figure(figsize=(8, 8))
        plt.imshow(canvas, cmap="gray")
    else:
        canvas = np.empty((H_n * H, W_n * W, 3))
        for i, yi in enumerate(y):
            for j, xi in enumerate(x):
                img = (np.reshape(images[H_n * i + j], [H, W, c])+1)/2
                canvas[H * i: H * i + H, W * j: W * j + W, :] = img
        plt.figure(figsize=(8, 8))
        plt.imshow(canvas)

    if not tf.gfile.Exists(save_dir):
        tf.gfile.MakeDirs(save_dir)

    name = name + ".png"
    path = os.path.join(save_dir, name)
    if os.path.isfile(path):
        os.remove(path)
    plt.savefig(path)
    print("Plot_images saving location: %s" % (path))
    print("")
    plt.close()

def print_saver(string, dir_txt):
    if not tf.gfile.Exists(dir_txt):
        tf.gfile.MakeDirs(dir_txt)
    
    string = string + "\n"
    dir_txt = dir_txt + "/print_log.txt"    
        
    if os.path.isfile(dir_txt):
        print_writer = open(dir_txt, "a")
    else:
        print_writer = open(dir_txt, "w")
    
    print_writer.write(string)
    print_writer.close()
    print(string)
    
def print_arg(args, dir_txt):
    
    for arg in vars(args):
        log_string = arg
        log_string += "." * (100 - len(arg) - len(str(getattr(args, arg))))
        log_string += str(getattr(args, arg))
        print_saver(log_string, dir_txt)
        
def print_time(start_time, dir_txt):
    
    hour = int((time.time() - start_time) / 3600)
    min = int(((time.time() - start_time) - 3600 * hour) / 60)
    sec = int((time.time() - start_time) - 3600 * hour - 60 * min)
    print_saver("time: %d hour %d min %d sec" % (hour, min, sec), dir_txt)
