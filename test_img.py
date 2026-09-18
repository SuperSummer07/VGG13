# -*- coding: utf-8 -*-
"""
在测试集上找出并保存：
  - 第一个预测正确的样本 ->  correct_label-{label}_conf-{conf}.png
  - 第一个预测错误的样本 ->  true-{true}_pred-{pred}_conf-{conf}.png

需要同目录下存在：vgg_model.py, vgg_data.py, weights.pth
"""

import os
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

from vgg_model import VGG13
from vgg_data import get_loader

# ============ 可调参数 ============
BATCH_SIZE = 64
WEIGHTS_PATH = "weights.pth"
SAVE_DIR = "./test_samples"

# 必须和 vgg_data.py 里 transforms.Normalize 用的 mean/std 一致
# 常见 CIFAR-10 均值/方差: (0.4914,0.4822,0.4465)/(0.2023,0.1994,0.2010)
MEAN = (0.5, 0.5, 0.5)
STD  = (0.5, 0.5, 0.5)
# =================================


def _denormalize(tensor, mean, std):
    """[3,H,W] 归一化张量 -> [0,1] 可视张量"""
    mean = torch.tensor(mean, dtype=tensor.dtype).view(3, 1, 1)
    std  = torch.tensor(std,  dtype=tensor.dtype).view(3, 1, 1)
    return (tensor.cpu() * std + mean).clamp(0, 1)


def _save_tensor_as_png(tensor, path):
    """[3,H,W] 值域[0,1] -> PNG"""
    img = transforms.ToPILImage()(tensor)
    img.save(path, format="PNG")


def _build_model(num_classes, device):
    """兼容 VGG13(num_classes=...) 和 VGG13() 两种写法"""
    try:
        model = VGG13(num_classes=num_classes)
    except TypeError:
        model = VGG13()
    return model.to(device)


def _strip_module_prefix(state_dict):
    """去掉 DataParallel 保存时多出来的 'module.' 前缀"""
    if all(k.startswith("module.") for k in state_dict.keys()):
        return {k[len("module."):]: v for k, v in state_dict.items()}
    return state_dict


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Info] 使用设备: {device}")

    # 1. 拿测试集
    _, test_loader = get_loader(BATCH_SIZE)

    # 2. 推断类别数和类别名（torchvision 的 Dataset 一般都有 .classes）
    dataset = test_loader.dataset
    class_names = getattr(dataset, "classes", None)
    if class_names is not None:
        num_classes = len(class_names)
    else:
        num_classes = 10   # 数据集没有 classes 属性时，请手动修改
        class_names = None
    print(f"[Info] 类别数: {num_classes}")

    # 3. 建模型并加载权重
    model = _build_model(num_classes, device)
    state = torch.load(WEIGHTS_PATH, map_location=device)
    state = _strip_module_prefix(state)
    model.load_state_dict(state)
    model.eval()
    print(f"[Info] 已加载权重: {WEIGHTS_PATH}")

    # 4. 遍历测试集，找第一个正确 / 第一个错误样本
    os.makedirs(SAVE_DIR, exist_ok=True)
    correct_path, wrong_path = None, None

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)                 # [B, C]
            probs  = F.softmax(logits, dim=1)      # 转成概率作为"确信度"
            confs, preds = probs.max(dim=1)        # [B], [B]

            for i in range(images.size(0)):
                true_lab = labels[i].item()
                pred_lab = preds[i].item()
                conf     = confs[i].item()

                if pred_lab == true_lab and correct_path is None:
                    name = class_names[true_lab] if class_names else str(true_lab)
                    fname = f"correct_label-{name}_conf-{conf:.4f}.png"
                    correct_path = os.path.join(SAVE_DIR, fname)
                    _save_tensor_as_png(
                        _denormalize(images[i], MEAN, STD), correct_path
                    )
                    print(f"[保存-正确] {fname}")

                elif pred_lab != true_lab and wrong_path is None:
                    tname = class_names[true_lab] if class_names else str(true_lab)
                    pname = class_names[pred_lab] if class_names else str(pred_lab)
                    fname = f"true-{tname}_pred-{pname}_conf-{conf:.4f}.png"
                    wrong_path = os.path.join(SAVE_DIR, fname)
                    _save_tensor_as_png(
                        _denormalize(images[i], MEAN, STD), wrong_path
                    )
                    print(f"[保存-错误] {fname}")

                if correct_path is not None and wrong_path is not None:
                    print(f"\n[完成] 两张图都保存在: {os.path.abspath(SAVE_DIR)}")
                    print(f"  正确样本: {os.path.basename(correct_path)}")
                    print(f"  错误样本: {os.path.basename(wrong_path)}")
                    return

    # 极端情况：某个类别一个都没找到
    if correct_path is None:
        print("[警告] 测试集中没有找到任何预测正确的样本")
    if wrong_path is None:
        print("[警告] 测试集中没有找到任何预测错误的样本")


if __name__ == "__main__":
    main()
