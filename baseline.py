import warnings
warnings.filterwarnings("ignore")
import os
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import numpy as np
import pandas as pd
import torchvision.transforms as T
import timm
import torch
from transformers import AutoModel
from sklearn.cluster import DBSCAN
from wildlife_datasets.datasets import AnimalCLEF2026
from wildlife_tools.features import DeepFeatures
from wildlife_tools.similarity import CosineSimilarity

# 数据目录：默认使用当前脚本同级目录下的 images 文件夹
root = os.path.dirname(os.path.abspath(__file__))

device = 'cuda'
batch_size = 32

eps_opt = {
    'LynxID2025': 0.3,
    'SalamanderID2025': 0.2,
    'SeaTurtleID2022': 0.4,
    'TexasHornedLizards': 0.24,
}


def relabel_negatives(labels):
    labels = np.array(labels)
    neg_indices = np.where(labels == -1)[0]
    new_labels = np.arange(labels.max() + 1, labels.max() + 1 + len(neg_indices))
    labels[neg_indices] = new_labels
    return labels


def run_DBSCAN(similarity, eps):
    distance = (np.max(similarity) - np.maximum(similarity, 0)) / np.max(similarity)
    clustering = DBSCAN(eps=eps, metric='precomputed', min_samples=2)
    clusters = clustering.fit(distance)
    return relabel_negatives(clusters.labels_)


# ============================================================
# Windows 必须加这个保护，否则多进程会崩溃
# ============================================================
if __name__ == '__main__':
    if not torch.cuda.is_available():
        raise RuntimeError("未检测到可用 CUDA 设备。请检查 PyTorch CUDA 安装和显卡驱动。")

    print(f"使用设备: {device}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    # 加载完整数据集
    print("加载数据集...")
    dataset_full = AnimalCLEF2026(
        root,
        transform=None,
        load_label=True,
        factorize_label=True,
        check_files=False
    )

    # 只取 test 集，并按 dataset 分组
    dataset_full = dataset_full.get_subset(dataset_full.df['split'] == 'test')
    datasets = {}
    for name in dataset_full.metadata['dataset'].unique():
        datasets[name] = dataset_full.get_subset(dataset_full.df['dataset'] == name)
    print(f"共 {len(datasets)} 个子数据集: {list(datasets.keys())}")

    # ============================================================
    # 特征提取 + 相似度矩阵计算
    # ============================================================
    similarities = {}
    for name, dataset in datasets.items():
        print(f"\n正在处理 {name}...")
        if name in ['SalamanderID2025', 'SeaTurtleID2022']:
            model = timm.create_model("hf-hub:BVRA/MegaDescriptor-L-384", pretrained=True).to(device).eval()
            size = 384
        elif name in ['LynxID2025', 'TexasHornedLizards']:
            model = AutoModel.from_pretrained('conservationxlabs/miewid-msv3', trust_remote_code=True).to(device).eval()
            size = 512
        else:
            raise ValueError(f'未知数据集: {name}')

        matcher = CosineSimilarity()
        # num_workers=0：Windows 下禁用多进程，避免 RuntimeError
        extractor = DeepFeatures(model=model, device=device, batch_size=batch_size, num_workers=0)
        transform = T.Compose([
            T.Resize(size=(size, size)),
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

        dataset.set_transform(transform)
        features = extractor(dataset)
        similarity = matcher(features, features)
        # CosineSimilarity 可能返回 dict {'cosine': matrix} 或直接返回 ndarray
        if isinstance(similarity, dict):
            similarity = list(similarity.values())[0]
        similarities[name] = similarity
        print(f"  {name}: {len(dataset)} 张图，相似度矩阵 {similarity.shape}")

    # ============================================================
    # DBSCAN 聚类 + 生成 submission.csv
    # ============================================================
    print("\n生成 submission.csv...")
    results = None
    for name, similarity in similarities.items():
        clusters = run_DBSCAN(similarity, eps_opt[name])
        result = pd.DataFrame({
            'image_id': datasets[name].metadata['image_id'],
            'cluster': [f'cluster_{name}_{c}' for c in clusters]
        })
        results = pd.concat((results, result))

    results.to_csv('submission.csv', index=False)
    print(f"完成！submission.csv 已保存，共 {len(results)} 行。")