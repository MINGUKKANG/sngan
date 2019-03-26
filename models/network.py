from models.GAN_Generator.G_Res_cifar10 import G_Res_cifar10
from models.GAN_Discriminator.D_Res_cifar10 import D_Res_cifar10

def network(network, args):
    if network == 'G_Res_cifar10':
        net = G_Res_cifar10(args, 'G_Res_cifar10')
    elif network == 'D_Res_cifar10':
        net = D_Res_cifar10(args, 'D_Res_cifar10')
    else:
        raise NotImplementedError

    return net
