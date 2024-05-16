import os
import time

from loguru import logger as logging
import torch
import torch.nn.functional as tnf
from timm import create_model as creat
import deepspeed
import torch.distributed as dist
from mydatasets import getDataLoader

from torch.utils.data import DistributedSampler, DataLoader

# image shape
input_shape = 224
distributed_port = 29500
seed = 1234
lr = 1e-3
time_str = time.strftime('%Y%m%d%H%M%S', time.localtime())  # 全局time

supportDataset = ['miniImageNet100', 'CIFAR10']
# datasets
dataset_name = supportDataset[1]
dataset_path = ''  # 为空则使用对应数据集默认路径

model_name = 'vit_base_patch16_224'
# model_name = 'vgg16'

# pretrain
pretrain = False
pretrain_model_path = '/home/dx/projects/datasets/models/model_vit70.pth'

# output dir
output_dir = f"./model/pth{time_str}"
log_dir = f"./timeLog/tim{time_str}.log"

# whether save model
isSave = False
# save model gap
save_interval = 10

# step gap show config
step_interval = 10
# epoch size
epoch = 200

test_interval = 1


def initlizeEnv():
    # create dir to save model
    if isSave:
        os.makedirs(output_dir, exist_ok=True)
    os.makedirs('timeLog', exist_ok=True)
    torch.manual_seed(seed)


def getModel(channelCount):
    if pretrain:
        pretrained_cfg = creat(model_name, pretrained=True, num_classes=channelCount).default_cfg
        pretrained_cfg['file'] = pretrain_model_path
        model = creat(model_name, pretrained=True, num_classes=channelCount, pretrained_cfg=pretrained_cfg)
    else:
        model = creat(model_name, pretrained=True, num_classes=channelCount)
    return model


def parse_arguments():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_rank',
                        type=int,
                        default=-1,
                        help='local rank passed from distributed launcher')
    parser = deepspeed.add_config_arguments(parser)
    return args


def train():
    initlizeEnv()

    deepspeed.init_distributed(distributed_port=distributed_port)

    trainSet, testSet, channelCount = getDataLoader(dataset_name, dataset_path, input_shape)

    sampler = DistributedSampler(testSet)
    test_dataloader = DataLoader(testSet, sampler=sampler, batch_size=32)

    model = getModel(channelCount)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    args = parse_arguments()

    engine, optimizer, training_dataloader, lr_scheduler = deepspeed.initialize(
        args=args,
        model=model,
        optimizer=optimizer,
        model_parameters=model.parameters(),
        training_data=trainSet,
    )

    all_train_time = 0

    for i in range(epoch):
        # train
        loss_list = []
        train_epoch_start_time = time.time()
        last_time = time.time()
        now_step_loss = 0

        logging.info("start train, epoch = {:3d}", i)
        engine.train()

        for step, item in enumerate(training_dataloader):
            xx, yy = item['image'], item['labels']
            xx = xx.to(device=engine.device, dtype=torch.float16)
            yy = yy.to(device=engine.device, dtype=torch.long).reshape(-1)

            # train step
            outputs = engine(xx)
            loss = tnf.cross_entropy(outputs, yy)
            engine.backward(loss)
            engine.step()
            now_step_loss += (loss / step_interval)

            # show now config
            if step % step_interval == 0:
                used_time = time.time() - last_time
                last_time = time.time()

                time_p_step = used_time / step_interval
                logging.info("[Train Step] Epoch:{:3d}  Step:{:3d}  Loss:{:8.4f} | Time/Batch: {:6.4f}s", i, step,
                             now_step_loss, time_p_step)
                loss_list.append(now_step_loss)
                now_step_loss = 0

        train_epoch_end_time = time.time()
        use_epoch_time = train_epoch_end_time - train_epoch_start_time
        all_train_time += use_epoch_time

        train_epoch_avg_loss = sum(loss_list) / len(loss_list)

        logging.info("[Epoch end] now has trained {} s", all_train_time)

        if (i + 1) % test_interval != 0:
            pass
        else:
            logging.info("start vaildation, epoch = {:3d}", i)
            engine.eval()

            allAcc = 0
            allCount = 0
            for step, item in enumerate(test_dataloader):
                xx, yy = (
                    item['image'].to(device=engine.device, dtype=torch.float16),
                    item['labels'].to(device=engine.device, dtype=torch.long).reshape(-1)
                )

                outputs = engine(xx)

                outputs = (torch.argmax(tnf.softmax(outputs.float(), dim=-1), dim=-1) == yy).type(torch.FloatTensor)
                allAcc += len([indexs for indexs in outputs if indexs > 0])
                allCount += len(outputs)

                if step % step_interval == 0:
                    logging.info("[Vaildation Step] Epoch:{:3d}  Step:{:3d} Acc:{:3d} / {:3d} = {:8.4f}%",
                                 i, step, allAcc, allCount, (allAcc / allCount) * 100)

            val_acc = (allAcc / allCount) * 100

            logging.info("[Vaildation Step] Epoch:{:3d}  Acc:{:3d} / {:3d} = {:8.4f}%",
                         i, allAcc, allCount, val_acc)

        if isSave and (i + 1) % save_interval == 0:
            # save model
            logging.info("start save model, epoch = {:3d}", i)
            engine.save_16bit_model(f"./{output_dir}/epoch{i}", "model.pth")


if __name__ == '__main__':
    train()
