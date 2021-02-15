# import argparse
# from torchvision.transforms import Resize, Compose, RandomResizedCrop
# import torchvision
# from torch.utils.data import DataLoader, Dataset
# from pathlib import Path
#
# from deepext_with_lightning.assemble import LearningTable, AssembleModel
# from deepext_with_lightning.training import XTrainer
# from deepext_with_lightning.models.classification import AttentionBranchNetwork, EfficientNet, MobileNetV3
# from deepext_with_lightning.metrics.classification import ClassificationAccuracyByClasses
# from deepext_with_lightning.utils import *
#
# from util import DataSetSetting
#
# # TODO アンサンブルはメンテ全然やってないから色々修正
#
# # NOTE モデル・データセットはここを追加
# MODEL_EFFICIENT_NET = "efficientnet"
# MODEL_ATTENTION_BRANCH_NETWORK = "attention_branch_network"
# MODEL_MOBILENET = "mobilenet"
# MODEL_TYPES = [MODEL_EFFICIENT_NET, MODEL_ATTENTION_BRANCH_NETWORK, MODEL_MOBILENET]
# DATASET_STL = "stl"
# DATASET_CIFAR = "cifar"
# DATASET_TYPES = [DATASET_STL, DATASET_CIFAR]
# settings = [DataSetSetting(dataset_type=DATASET_STL, size=(96, 96), n_classes=10),
#             DataSetSetting(dataset_type=DATASET_CIFAR, size=(32, 32), n_classes=10)]
#
#
# def get_dataloader(setting: DataSetSetting, root_dir: str, batch_size: int) -> Tuple[
#     DataLoader, DataLoader, Dataset, Dataset]:
#     train_transforms = Compose(
#         [Resize(setting.size), RandomResizedCrop(size=setting.size, scale=(0.3, 0.3)), ToTensor()])
#     test_transforms = Compose([Resize(setting.size), ToTensor()])
#     train_dataset, test_dataset = None, None
#     # NOTE データセットはここを追加
#     if DATASET_STL == setting.dataset_type:
#         train_dataset = torchvision.datasets.STL10(root=root_dir, download=True, split="train",
#                                                    transform=train_transforms)
#         test_dataset = torchvision.datasets.STL10(root=root_dir, download=True, split="test",
#                                                   transform=test_transforms)
#     elif DATASET_CIFAR == setting.dataset_type:
#         train_dataset = torchvision.datasets.CIFAR10(root=root_dir, download=True, train=True,
#                                                      transform=train_transforms)
#         test_dataset = torchvision.datasets.CIFAR10(root=root_dir, download=True, train=False,
#                                                     transform=test_transforms)
#     assert train_dataset is not None and test_dataset is not None, f"Not supported setting: {setting.dataset_type}"
#     return DataLoader(train_dataset, batch_size=batch_size, shuffle=True), \
#            DataLoader(test_dataset, batch_size=batch_size, shuffle=True), train_dataset, test_dataset
#
#
# def get_model(dataset_setting: DataSetSetting, model_type: str, lr: float, efficientnet_scale: int = 0):
#     # NOTE モデルはここを追加
#     if MODEL_EFFICIENT_NET == model_type:
#         return EfficientNet(num_classes=dataset_setting.n_classes, lr=lr, network=f"efficientnet-b{efficientnet_scale}")
#     elif MODEL_ATTENTION_BRANCH_NETWORK == model_type:
#         return try_cuda(AttentionBranchNetwork(n_classes=dataset_setting.n_classes, lr=lr))
#     elif MODEL_MOBILENET == model_type:
#         return MobileNetV3(num_classes=dataset_setting.n_classes, lr=lr, pretrained=False)
#     assert f"Invalid model type. Valid models is {MODEL_TYPES}"
#
#
# parser = argparse.ArgumentParser(description='Pytorch Image classification training.')
#
# parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
# parser.add_argument('--dataset', type=str, default=DATASET_STL, help=f'Dataset type in {DATASET_TYPES}')
# parser.add_argument('--epoch', type=int, default=100, help='Number of epochs')
# parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
# parser.add_argument('--dataset_root', type=str, required=True, help='Dataset folder path')
# parser.add_argument('--progress_dir', type=str, default=None, help='Directory for saving progress')
# parser.add_argument('--model', type=str, default=MODEL_MOBILENET, help=f"Model type in {MODEL_TYPES}")
# parser.add_argument('--load_weight_dir', type=str, default=None, help="Saved weight directory path")
# parser.add_argument('--save_weight_dir', type=str, default=None, help="Saved weight directory path")
# parser.add_argument('--efficientnet_scale', type=int, default=0, help="Number of scale of EfficientNet.")
# parser.add_argument('--n_models', type=int, default=3, help="Count of model.")
#
# if __name__ == "__main__":
#     args = parser.parse_args()
#
#     # Fetch dataset.
#     dataset_setting = DataSetSetting.from_dataset_type(settings, args.dataset)
#     train_dataloader, test_dataloader, train_dataset, test_dataset = get_dataloader(dataset_setting, args.dataset_root,
#                                                                                     args.batch_size)
#     callbacks = []
#     model_list, learning_tables = [], []
#     for _ in range(args.n_models):
#         model_list.append(try_cuda(
#             get_model(dataset_setting, model_type=args.model, lr=args.lr, efficientnet_scale=args.efficientnet_scale)))
#         learning_tables.append(LearningTable(data_loader=train_dataloader, callbacks=callbacks, epochs=args.epoch))
#
#     assemble_model: AssembleModel = AssembleModel(models=model_list)
#     # Fetch model and load weight.
#     if args.load_weight_dir:
#         assemble_model.load_weight(args.load_weight_dir)
#     save_weight_dir = args.save_weight_dir or f"./{args.model}"
#
#     # Training.
#     training = XTrainer(assemble_model)
#     training.fit(test_dataloader=test_dataloader,
#                 metric_func_ls=[ClassificationAccuracyByClasses(label_names=dataset_setting.label_names), ],
#                 learning_tables=learning_tables)
#     # Save weight.
#     if not Path(save_weight_dir).exists():
#         Path(save_weight_dir).mkdir()
#     assemble_model.save_weight(save_weight_dir)
