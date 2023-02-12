#!/usr/bin/env python
# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

import sys

sys.path.append('.')
# fastreid 是一个 python 库，用于快速开发和研究基于深度学习的目标检测、关键点检测和指纹识别算法。
# 它提供了一个高度可定制的代码框架，支持快速开发和实验，并具有良好的可扩展性和可读性。
# 它还支持多种深度学习框架，包括 PyTorch，TensorFlow 等。
from fastreid.config import get_cfg
from fastreid.engine import DefaultTrainer, default_argument_parser, default_setup, launch
from fastreid.utils.checkpoint import Checkpointer


def setup(args):
    """
    Create configs and perform basic setups.
    """
    # `get_cfg`是fastreid模块中的内置函数，用于从配置文件中加载配置。
    # 它接受一个参数：要加载的配置文件，并返回一个`FastReIDConfig`对象。
    # `get_cfg`可以用来加载配置文件，以及获取配置参数。
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)

    if args.eval_only:
        # `defrost` 是 fastreid 模块中的内置函数，用于重新加载一个已有的模型，并将其设置为可训练状态，从而可以继续训练模型。
        cfg.defrost()
        cfg.MODEL.BACKBONE.PRETRAIN = False
        model = DefaultTrainer.build_model(cfg)

        # `Checkpointer`是fastreid模块中的内置函数，用于保存和加载模型的状态，以及记录模型的训练进度。
        # 它接受两个参数：要保存的模型和要保存的路径，并返回一个`Checkpointer`对象。
        # `Checkpointer`可以用来保存模型的状态，以及加载模型的状态，以及记录模型的训练进度。
        Checkpointer(model).load(cfg.MODEL.WEIGHTS)  # load trained model

        # `DefaultTrainer`是fastreid模块中的内置类，用于训练模型。
        # 它接受一个参数：要训练的模型配置，并返回一个`DefaultTrainer`实例。
        # `DefaultTrainer`可以用来训练模型，以及设置训练参数和模型参数。
        res = DefaultTrainer.test(cfg, model, 0)
        return res

    trainer = DefaultTrainer(cfg)

    trainer.resume_or_load(resume=args.resume)
    return trainer.train()


if __name__ == "__main__":
    # `default_argument_parser`是 fastreid 模块中的内置函数，用于创建一个默认参数解析器，可以用来解析命令行参数和设置默认参数。
    # 它接受一个参数：要解析的参数，并返回一个参数解析器。
    # `default_argument_parser`可以用来管理和解析命令行参数，以及设置和使用默认参数。
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    # `launch`是fastreid模块中的内置函数，用于启动程序。
    # 它接受一个参数：要启动的参数列表，并返回一个`FastReIDConfig`对象。
    # `launch`可以用来启动程序，以及加载参数文件和配置文件。
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
