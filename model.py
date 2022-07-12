import os
from nets import inception_v3, inception_v4, inception_resnet_v2, resnet_v2
from nets.mobilenet import mobilenet_v2
from nets.nasnet import pnasnet, nasnet

id_to_name = {
    '1': 'inception_v3',
    '2': 'adv_inception_v3',
    '3': 'ens3_adv_inception_v3',
    '4': 'ens4_adv_inception_v3',
    '5': 'inception_v4',
    '6': 'inception_resnet_v2',
    '7': 'ens_adv_inception_resnet_v2',
    '8': 'resnet_v2_101',
    '9': 'resnet_v2_152',
    '10': 'mobilenet_v2_1.0',
    '11': 'mobilenet_v2_1.4',
    '12': 'pnasnet-5_mobile',
    '13': 'nasnet-a_mobile',
}

checkpoint_path = './model'
id_to_checkpoint = {
    '1': os.path.join(checkpoint_path, 'inception_v3.ckpt'),
    '2': os.path.join(checkpoint_path, 'adv_inception_v3_rename.ckpt'),
    '3': os.path.join(checkpoint_path, 'ens3_adv_inception_v3_rename.ckpt'),
    '4': os.path.join(checkpoint_path, 'ens4_adv_inception_v3_rename.ckpt'),
    '5': os.path.join(checkpoint_path, 'inception_v4.ckpt'),
    '6': os.path.join(checkpoint_path, 'inception_resnet_v2_2016_08_30.ckpt'),
    '7': os.path.join(checkpoint_path, 'ens_adv_inception_resnet_v2_rename.ckpt'),
    '8': os.path.join(checkpoint_path, 'resnet_v2_101.ckpt'),
    '9': os.path.join(checkpoint_path, 'resnet_v2_152.ckpt'),
    '10': os.path.join(checkpoint_path, 'mobilenet_v2_1.0_224.ckpt'),
    '11': os.path.join(checkpoint_path, 'mobilenet_v2_1.4_224_modify.ckpt'),
    '12': os.path.join(checkpoint_path, 'pnasnet-5_mobile_model_modify.ckpt'),
    '13': os.path.join(checkpoint_path, 'nasnet-a_mobile_model_modify.ckpt'),
}

id_to_scope = {
    '1': 'InceptionV3',
    '2': 'AdvInceptionV3',
    '3': 'Ens3AdvInceptionV3',
    '4': 'Ens4AdvInceptionV3',
    '5': 'InceptionV4',
    '6': 'InceptionResnetV2',
    '7': 'EnsAdvInceptionResnetV2',
    '8': 'resnet_v2_101',
    '9': 'resnet_v2_152',
    '10': 'MobilenetV2',
    '11': 'MobilenetV2_1.4',
    '12': 'pnasnet_mobile',
    '13': 'nasnet_mobile',
}

id_to_saverscope = {
    '1': 'InceptionV3',
    '2': 'AdvInceptionV3',
    '3': 'Ens3AdvInceptionV3',
    '4': 'Ens4AdvInceptionV3',
    '5': 'InceptionV4',
    '6': 'InceptionResnetV2',
    '7': 'EnsAdvInceptionResnetV2',
    '8': 'resnet_v2_101',
    '9': 'resnet_v2_152',
    '10': 'MobilenetV2/',
    '11': 'MobilenetV2_1.4',
    '12': 'pnasnet_mobile',
    '13': 'nasnet_mobile',
}


id_to_arg_scope = {
    '1': inception_v3.inception_v3_arg_scope(),
    '2': inception_v3.inception_v3_arg_scope(),
    '3': inception_v3.inception_v3_arg_scope(),
    '4': inception_v3.inception_v3_arg_scope(),
    '5': inception_v4.inception_v4_arg_scope(),
    '6': inception_resnet_v2.inception_resnet_v2_arg_scope(),
    '7': inception_resnet_v2.inception_resnet_v2_arg_scope(),
    '8': resnet_v2.resnet_arg_scope(),
    '9': resnet_v2.resnet_arg_scope(),
    '10': mobilenet_v2.training_scope(),
    '11': mobilenet_v2.training_scope(),
    '12': pnasnet.pnasnet_mobile_arg_scope(),
    '13': nasnet.nasnet_mobile_arg_scope(),
}

id_to_model = {
    '1': inception_v3.inception_v3,
    '2': inception_v3.inception_v3,
    '3': inception_v3.inception_v3,
    '4': inception_v3.inception_v3,
    '5': inception_v4.inception_v4,
    '6': inception_resnet_v2.inception_resnet_v2,
    '7': inception_resnet_v2.inception_resnet_v2,
    '8': resnet_v2.resnet_v2_101,
    '9': resnet_v2.resnet_v2_152,
    '10': mobilenet_v2.mobilenet,
    '11': mobilenet_v2.mobilenet_v2_140,
    '12': pnasnet.build_pnasnet_mobile,
    '13': nasnet.build_nasnet_mobile,
}