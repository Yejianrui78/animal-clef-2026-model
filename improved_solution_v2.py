import warnings
warnings.filterwarnings("ignore")
import os
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
import copy
import contextlib

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.transforms as T
import timm
from PIL import Image
import cv2
from transformers import AutoModel
from sklearn.cluster import AgglomerativeClustering, DBSCAN

try:
    from hdbscan import HDBSCAN
    HAS_HDBSCAN = True
except ImportError:
    HAS_HDBSCAN = False
    print("[警告] hdbscan 未安装，将只使用 DBSCAN。建议: pip install hdbscan")

from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.model_selection import train_test_split
from wildlife_datasets.datasets import AnimalCLEF2026
from wildlife_tools.features import DeepFeatures
from wildlife_tools.similarity import CosineSimilarity


# ============================================================
# 路径和设备配置（与 baseline.py 一致：数据在脚本同级目录）
# ============================================================
root = os.path.dirname(os.path.abspath(__file__))
# 默认：有 CUDA 则用 GPU，否则回退 CPU（当前环境若为 cpu-only PyTorch，需重装 CUDA 版才有 GPU）。
# 必须在 GPU 上跑、否则报错：PowerShell 中执行 `$env:REQUIRE_CUDA="1"` 再运行脚本。
_require_cuda = os.environ.get('REQUIRE_CUDA', '').strip().lower() in ('1', 'true', 'yes')
if _require_cuda and not torch.cuda.is_available():
    raise RuntimeError(
        "已设置 REQUIRE_CUDA=1，但未检测到 CUDA（torch.cuda.is_available() == False）。"
        "请安装与驱动匹配的 CUDA 版 PyTorch，或去掉 REQUIRE_CUDA 以允许 CPU。"
    )
device = 'cuda' if torch.cuda.is_available() else 'cpu'
batch_size = 16 

# 可选：将权重放在项目内 models/ 下则优先离线加载；不存在时 Mega 回退 hf-hub，Miew 回退 Hub
local_mega_path = os.path.join(root, 'models', 'BVRA', 'MegaDescriptor-L-384')
local_miew_path = os.path.join(root, 'models', 'conservationxlabs', 'miewid-msv3')

# 无 train split 时的回退 eps（标准余弦距离 = 1 - sim；修复后与 agglo_threshold_fallback 同一尺度）
eps_fallback = {
    'LynxID2025':         0.30,
    'SalamanderID2025':   0.20,
    'SeaTurtleID2022':    0.40,
    'TexasHornedLizards': 0.24,
}

# 无 train 时 Agglomerative 的 distance_threshold（与 0.23.py SPECIES_CONFIG.threshold 一致；距离 = 1 - 余弦相似度）
agglo_threshold_fallback = {
    'LynxID2025':         0.35,
    'SalamanderID2025':   0.25,
    'SeaTurtleID2022':    0.45,
    'TexasHornedLizards': 0.30,
}

# ============================================================
# 实验开关（按需修改；便于逐项对比融合 / TTA / 标定 / 聚类器）
# ============================================================
# 特征策略（五选一，含义不同，勿与「单模型」混为一谈）：
#   'fusion' — 每个子集均 Mega‖Miew 拼接后再聚类
#   'fusion_salamander_only' — SalamanderID2025 + TexasHornedLizards 做 Mega‖Miew；
#       海龟(SeaTurtle)→Mega，猞猁(Lynx)→Miew
#   'global_mega' / 'global_miew' — 四个子集共用同一骨干（真·全网单模型）
#   'per_dataset_baseline' — 与 baseline.py 相同：Salamander/海龟→Mega，猞猁/角蜥→Miew；
#       每个子集仍是「一个」骨干，但子集之间不同，且不是拼接融合
FEATURE_STRATEGY = 'fusion'  # 'fusion' | 'fusion_salamander_only' | ...

# 只跑部分子集时设置（名称须与 metadata.csv 里 dataset 列一致，对应如 images/SalamanderID2025）。
# None 表示四个子集全跑；正式提交前请改为 None。
RUN_SUBSETS_ONLY = None
# ============================================================
# 微调控制（Fine-tuning，wildlife_tools.BasicTrainer + TripletLoss）
# ============================================================
ENABLE_FINETUNING = False  # 设为 True 即对 FINETUNE_DATASETS 列出的子集微调（耗时较长）
# 需要微调的子集名称；骨干自动按 FEATURE_STRATEGY / baseline 路由（例：Salamander→Mega，Lynx→Miew）
FINETUNE_DATASETS = ('LynxID2025',)  # 单元素元组须尾随逗号，否则等同 str 会按字符迭代
FINETUNE_TRAIN_RATIO = 0.8
FINETUNE_EPOCHS = 20
FINETUNE_BATCH_SIZE = 8
FINETUNE_ACCUMULATION_STEPS = 2
FINETUNE_LEARNING_RATE = 1e-4
FINETUNE_MARGIN = 0.2
FINETUNE_SAVE_DIR = os.path.join(root, 'finetuned_models')

# True: 原图 + 水平翻转特征取平均；False: 仅原图
USE_TTA = True

# ---------- 图像预处理（送入 MegaDescriptor / MiewID 之前；可逐项关闭做对比）----------
# YOLO 动物框裁剪（默认 YOLOv11；关闭则整图）
USE_MEGADETECTOR_CROP = True
# LAB 空间 CLAHE，统一亮度/对比度
USE_CLAHE_LIGHTING = False
# Unsharp mask 式纹理锐化
USE_TEXTURE_SHARPEN = False
# True: 等比缩放 + 居中填充至 size×size；False: 直接 Resize 到 size×size（可能变形，与 baseline 接近）
USE_LETTERBOX_TO_INPUT = False

# 裁剪检测权重：MegaDetector v1000（YOLOv5 hub 加载 .pt；见 releases/tag/v1000.0）
# 下载后放入脚本目录或填绝对路径：https://github.com/agentmorris/MegaDetector/releases/tag/v1000.0
# 可选：md_v1000.0.0-redwood.pt（精度最高，默认）/ cedar（中等）/ spruce（最快）
YOLO_CROP_WEIGHT = 'md_v1000.0.0-redwood.pt'
# MDv1000 与 MDv5 置信度标度不同，官方说明常用约 0.3–0.4（MDv5 常用 ~0.2）
YOLO_CROP_CONF = 0.35
YOLO_CROP_MARGIN = 0.05
# 'ultralytics' — ultralytics.YOLO；'yolov5_hub' — torch.hub + 本地 yolov5 加载 MegaDetector 等自定义权重
YOLO_CROP_ENGINE = 'yolov5_hub'
# yolov5_hub 时需本地 ultralytics/yolov5 仓库（含 hubconf.py）；留空则用环境变量 YOLOV5_REPO 或 <root>/yolov5
YOLOV5_LOCAL_REPO = ''
# 仅 YOLO_CROP_ENGINE='ultralytics' 且 COCO 预训练权重时生效：只在这些类别里取最高置信度框（动物检测常用）
# COCO: 14=bird … 23=giraffe。设为 None 则不按类别过滤（可能框到人/车）。MegaDetector(md_v*) 走 yolov5_hub，固定 animal=0，忽略本项
YOLO_CROP_CLASSES = [14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
# YOLOv11 + SAM 二阶段预处理（先检测后分割精修）；失败时自动回退 YOLO 框裁剪
USE_SAM_REFINER = False
SAM_MODEL_WEIGHT = 'mobile_sam.pt'   # 可换 'sam2_t.pt' / 'sam_b.pt' 等
SAM_BOX_EXPAND = 0.03                # 给 SAM 提示框额外扩边比例
SAM_MASK_MIN_AREA_RATIO = 0.001      # 掩膜面积过小则视为失败并回退

# --------- LynxID2025 专属预处理（在 YOLO/SAM 裁剪之后执行，可做消融）---------
USE_LYNX_SPECIAL_PREPROCESS = True
LYNX_STAGE2_ENABLE = False    # 跨光照/跨模态对齐：LAB 统一 + 过曝修复 + L 通道标准化
LYNX_STAGE3_ENABLE = False   # 纹理增强与噪声抑制：CLAHE + 去噪 + 温和锐化
# 红外/低饱和图判定阈值（基于 HSV S 通道均值）
LYNX_IR_SAT_MEAN_THRESHOLD = 22.0
# Stage2: L 通道归一化目标分布
LYNX_L_TARGET_MEAN = 128.0
LYNX_L_TARGET_STD = 64.0

# --------- SalamanderID2025 专属预处理（YOLO/SAM 裁剪之后；仅对该子集生效）---------
USE_SALAMANDER_SPECIAL_PREPROCESS = True
# 高光 / 反光消除（LAB 仅处理 L；A/B 不动）
SAL_GLARE_ENABLE = False
SAL_GLARE_L_THRESHOLD = 245          # L > 此值视为高光（避开正常亮黄色斑纹）
SAL_GLARE_MORPH_KERNEL = 3           # 膨胀核边长（奇数），覆盖反光过渡边缘
SAL_GLARE_INPAINT_RADIUS = 3
SAL_GLARE_INPAINT_METHOD = 'telea'   # 'telea' | 'ns'  （cv2.INPAINT_TELEA / INPAINT_NS）
# 跨图对齐：L 映射到固定区间 + A/B 均值对齐 + L 通道 CLAHE（仍不动 A/B 分布形状，仅平移对齐均值）
SAL_ALIGN_ENABLE = False
SAL_L_RANGE_LOW = 40
SAL_L_RANGE_HIGH = 210
# 全局参考 A/B 均值：中性 LAB 中心；可选 npz（keys: mean_a, mean_b）从训练集统计覆盖
SAL_AB_REF_MEAN_A = 128.0
SAL_AB_REF_MEAN_B = 128.0
SAL_AB_STATS_PATH = ''               # 例: os.path.join(root, 'salamander_lab_means.npz')
SAL_CLAHE_CLIP_NORMAL = 2.0
SAL_CLAHE_CLIP_GENTLE = 1.25         # 低质 / 截断图用温和 CLAHE
SAL_LOW_QUALITY_MEAN_L_LOW = 42.0    # 启发式：整体偏暗
SAL_LOW_QUALITY_MEAN_L_HIGH = 205.0  # 启发式：整体偏亮 / 接近截断
SAL_LOW_QUALITY_STD_L_MAX = 28.0     # L 通道标准差过小 → 对比度极低，按低质处理
# 纹理：边缘保留去噪 + 非锐化掩膜（不做强模糊）
SAL_TEXTURE_ENABLE = True
SAL_STAGE3_DENOISE = 'bilateral'     # 'bilateral' | 'nlmeans'
SAL_BILAT_D = 7
SAL_BILAT_SIGMA_COLOR = 52
SAL_BILAT_SIGMA_SPACE = 52
SAL_BILAT_D_GENTLE = 5               # 低质图更弱
SAL_BILAT_SIGMA_COLOR_GENTLE = 38
SAL_BILAT_SIGMA_SPACE_GENTLE = 38
SAL_NLM_H = 6
SAL_NLM_H_GENTLE = 4
SAL_NLM_TEMPLATE = 7
SAL_NLM_SEARCH = 21
SAL_UNSHARP_SIGMA = 1.0
SAL_UNSHARP_AMOUNT = 0.14            # 非锐化强度（略增强边缘，少放大噪声）
SAL_UNSHARP_AMOUNT_GENTLE = 0.09
# 与 Mega 384 配合：letterbox 且边缘复制填充（勿用纯黑边）；需 USE_LETTERBOX_TO_INPUT=True
SAL_USE_EDGE_LETTERBOX = False



# True: 在有 train 的子集上，用真实 identity 做网格搜索（eps / min_cluster_size）
# False: 不搜参，使用 FIXED_DBSCAN_EPS 与 FIXED_HDBSCAN_MCS（无 train 时仍用 eps_fallback）
GRID_SEARCH_ON_TRAIN = True
# True: 在 train 上比较 DBSCAN / HDBSCAN /（可选）Agglomerative 的 ARI，选较高者（需 GRID_SEARCH_ON_TRAIN=True）
# False: 强制使用 FORCED_CLUSTER_METHOD（'dbscan' | 'hdbscan' | 'agglomerative'）
# ★ 由于每次评估最优方法均为 agglomerative，强制固定，跳过 DBSCAN/HDBSCAN 比较，节省时间
AUTO_PICK_DBSCAN_OR_HDBSCAN = True
FORCED_CLUSTER_METHOD = 'agglomerative'  # 'dbscan' | 'hdbscan' | 'agglomerative'
# True 且 GRID_SEARCH_ON_TRAIN：在 train 上搜索 Agglomerative 的 distance_threshold 并参与与 DBSCAN/HDBSCAN 比 ARI
GRID_SEARCH_AGGLOMERATIVE = True

# GRID_SEARCH_ON_TRAIN=False 时的固定聚类参数（有 train 时用；无 train 的 DBSCAN 仍优先 eps_fallback[name]）
FIXED_DBSCAN_EPS = 0.3
FIXED_HDBSCAN_MIN_CLUSTER_SIZE = 2
FIXED_AGGLOMERATIVE_DISTANCE_THRESHOLD = 0.32
AGGLOMERATIVE_LINKAGE = 'average'  # 与 0.23.py 一致

# ============================================================
# ★ 新增：Agglomerative 聚类数量上限约束
# ============================================================
# 问题：distance_threshold 模式无簇数约束，导致大量个体自成单例簇。
# 解决：当 distance_threshold 产生的簇数超过上限时，自动切换为 n_clusters 固定版。
#
# AGGLO_ENABLE_N_CLUSTERS_CAP: True=开启约束；False=与原来行为相同（不限）
AGGLO_ENABLE_N_CLUSTERS_CAP = True

# 每个子集的最大簇数 = max(AGGLO_MIN_CLUSTERS, int(n_samples * AGGLO_MAX_CLUSTERS_RATIO))
# 0.5 表示最多允许一半的图像各自单独成簇；可按数据集规模调小（更激进合并）
AGGLO_MAX_CLUSTERS_RATIO = 0.5

# 绝对下限：即便样本数很少，也至少保留这么多簇
AGGLO_MIN_CLUSTERS = 5

# 每个子集的聚类数量上限（绝对值，优先于比例计算）；设为 None 则使用比例自动计算
# 实际 test 集大小在运行时才知道，None 时自动按 AGGLO_MAX_CLUSTERS_RATIO 计算
AGGLO_MAX_CLUSTERS_PER_DATASET: dict[str, int | None] = {
    'LynxID2025':         None,   # 使用比例（AGGLO_MAX_CLUSTERS_RATIO）
    'SalamanderID2025':   None,
    'SeaTurtleID2022':    None,
    'TexasHornedLizards': None,
}


# ============================================================
# Fix 1: MiewID 包装器
# ============================================================
class MiewIDWrapper(nn.Module):
    """
    将 HuggingFace AutoModel 包装成 timm 风格接口：
      forward(x: Tensor) -> Tensor  shape=(B, D)
    """
    def __init__(self, base_model):
        super().__init__()
        self.model = base_model

    def forward(self, x):
        if hasattr(self.model, 'get_embeddings'):
            return self.model.get_embeddings(x)

        # 兼容两类 forward 签名：
        # 1) forward(x)（位置参数）
        # 2) forward(pixel_values=x)（HF 风格关键字参数）
        try:
            out = self.model(x)
        except TypeError:
            out = self.model(pixel_values=x)

        # 1) 直接返回 Tensor 的模型
        if isinstance(out, torch.Tensor):
            if out.ndim == 3:
                return out.mean(dim=1)
            if out.ndim >= 2:
                return out

        if hasattr(out, 'pooler_output') and out.pooler_output is not None:
            return out.pooler_output

        if hasattr(out, 'last_hidden_state'):
            return out.last_hidden_state.mean(dim=1)

        # 2) tuple/list 输出
        if isinstance(out, (tuple, list)):
            for v in out:
                if isinstance(v, torch.Tensor) and v.ndim >= 2:
                    return v.mean(dim=1) if v.ndim == 3 else v

        # 3) dict / ModelOutput 输出
        if isinstance(out, dict) or hasattr(out, 'values'):
            for v in out.values():
                if isinstance(v, torch.Tensor) and v.ndim >= 2:
                    return v.mean(dim=1) if v.ndim == 3 else v

        raise RuntimeError("MiewIDWrapper: 无法从模型输出中提取嵌入向量")


# ============================================================
# Fix 2: MegaDescriptor 加载
# ============================================================
def load_mega_model(local_path: str, device: str):
    for fname in ('model.safetensors', 'pytorch_model.bin'):
        weight_file = os.path.join(local_path, fname)
        if os.path.isfile(weight_file):
            print(f"    找到本地权重: {fname}")
            try:
                model = timm.create_model(
                    'swin_large_patch4_window12_384',
                    pretrained=True,
                    pretrained_cfg_overlay=dict(file=weight_file),
                    num_classes=0,
                )
                model = model.eval().to(device)
                print("    pretrained_cfg_overlay 加载成功")
                return model
            except Exception as e:
                print(f"    pretrained_cfg_overlay 加载失败: {e}")
            break

    print("    回退到 hf-hub 在线加载...")
    model = timm.create_model(
        'hf-hub:BVRA/MegaDescriptor-L-384',
        pretrained=True,
        num_classes=0,
    )
    return model.eval().to(device)


# ============================================================
# Fix 3: 离线上下文管理器 + MiewID 加载
# ============================================================
@contextlib.contextmanager
def _offline_env():
    """
    临时将 HuggingFace Hub / timm 切换到纯本地离线模式，
    防止 MiewID 内部的 timm.create_model(pretrained=True) 发起网络请求。
    退出 with 块后自动恢复原始环境变量。
    """
    keys = ['HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE', 'HF_DATASETS_OFFLINE']
    old = {k: os.environ.get(k) for k in keys}
    for k in keys:
        os.environ[k] = '1'
    try:
        yield
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _hf_snapshot_ready(local_path: str) -> bool:
    return os.path.isdir(local_path) and os.path.isfile(os.path.join(local_path, 'config.json'))


def load_miew_model(local_path: str, device: str):
    """优先从本地目录加载 MiewID-msv3；若无有效快照则与 baseline 一样从 Hub 加载。"""
    if _hf_snapshot_ready(local_path):
        try:
            with _offline_env():
                base = AutoModel.from_pretrained(
                    local_path,
                    trust_remote_code=True,
                    local_files_only=True,
                )
            model = MiewIDWrapper(base).eval().to(device)
            print(f"    MiewID 从本地加载: {local_path}")
            return model
        except Exception as e:
            print(f"    本地 MiewID 加载失败 ({e})，回退到 HuggingFace Hub...")

    base = AutoModel.from_pretrained(
        'conservationxlabs/miewid-msv3',
        trust_remote_code=True,
    )
    model = MiewIDWrapper(base).eval().to(device)
    print("    MiewID 从 HuggingFace Hub 加载（与 baseline 一致）")
    return model


# ============================================================
# 工具函数
# ============================================================
def smart_crop_and_pad(image, bbox, target_size=256, pad_ratio=0.15):
    """
    YOLO检测后的智能裁剪
    - 加padding避免截断边缘毛发/尾巴
    - 保持长宽比再居中填充
    """
    x1, y1, x2, y2 = bbox
    h, w = image.shape[:2]
    
    # 扩展边界框（避免截断）
    pad_x = int((x2 - x1) * pad_ratio)
    pad_y = int((y2 - y1) * pad_ratio)
    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(w, x2 + pad_x)
    y2 = min(h, y2 + pad_y)
    
    crop = image[y1:y2, x1:x2]
    
    # 保持比例的letterbox resize
    crop_h, crop_w = crop.shape[:2]
    scale = target_size / max(crop_h, crop_w)
    new_h = int(crop_h * scale)
    new_w = int(crop_w * scale)
    resized = cv2.resize(crop, (new_w, new_h))
    
    # 居中填充到target_size x target_size
    canvas = np.zeros((target_size, target_size, 3), dtype=np.uint8)
    y_off = (target_size - new_h) // 2
    x_off = (target_size - new_w) // 2
    canvas[y_off:y_off+new_h, x_off:x_off+new_w] = resized
    
    return canvas

def normalize_to_rgb(image):
    """
    统一多模态图像到可用RGB
    """
    if len(image.shape) == 2:
        # 纯灰度（红外） -> 伪彩色或复制三通道
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    
    # 检测是否接近灰度（红外相机彩色输出）
    if image.shape[2] == 3:
        b, g, r = image[:,:,0], image[:,:,1], image[:,:,2]
        channel_std = np.std([b.mean(), g.mean(), r.mean()])
        
        if channel_std < 5.0:  # 三通道几乎相同 = 伪彩色灰度
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # CLAHE增强对比度
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            enhanced = clahe.apply(gray)
            image = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
    
    return image

def apply_background_strategy(image, mask=None, strategy='neutral_bg'):
    """
    统一背景处理
    strategy: 'black_bg' | 'neutral_bg' | 'blur_bg'
    """
    if mask is None:
        # 用SAM或简单阈值生成掩码
        # 如果图像已经是黑色背景，直接检测
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, 
                                np.ones((15,15), np.uint8))
    
    if strategy == 'neutral_bg':
        # 替换为128灰色背景（对模型最中性）
        result = image.copy()
        bg = np.full_like(image, 128)
        result[mask == 0] = bg[mask == 0]
        return result
    
    elif strategy == 'blur_bg':
        # 背景模糊，突出主体（适合手持蝾螈）
        blurred = cv2.GaussianBlur(image, (51, 51), 0)
        result = np.where(mask[:,:,None] > 0, image, blurred)
        return result


def relabel_negatives(labels: np.ndarray) -> np.ndarray:
    """将 DBSCAN/HDBSCAN 的噪声点（label=-1）各自分配唯一 ID。"""
    labels = np.array(labels, dtype=int)
    neg_idx = np.where(labels == -1)[0]
    if len(neg_idx) == 0:
        return labels
    start = labels.max() + 1
    labels[neg_idx] = np.arange(start, start + len(neg_idx))
    return labels



def _distance_matrix(similarity: np.ndarray) -> np.ndarray:
    """将余弦相似度矩阵转换为 [0,1] 距离矩阵（标准余弦距离 = 1 - sim）。
    
    [Fix 1] 原实现用 (max_val - sim) / max_val，是数据依赖的非标准归一化：
    train 与 unknown_idx 子集的 max_val 不同，导致同一 eps 在两边含义完全不同。
    现统一改为 clip(1 - sim, 0, 1)，与 _cosine_distance_for_agglo 保持一致，
    使 DBSCAN / HDBSCAN / Agglomerative 三者在同一距离尺度下公平比较。
    """
    dist = np.clip(1.0 - np.asarray(similarity, dtype=np.float64), 0.0, 1.0)
    np.fill_diagonal(dist, 0.0)
    return dist


def run_DBSCAN(similarity: np.ndarray, eps: float) -> np.ndarray:
    dist = _distance_matrix(similarity)
    labels = DBSCAN(eps=eps, metric='precomputed', min_samples=2).fit_predict(dist)
    return relabel_negatives(labels)


def run_HDBSCAN(similarity: np.ndarray, min_cluster_size: int = 2) -> np.ndarray:
    if not HAS_HDBSCAN:
        raise RuntimeError("hdbscan 未安装")
    dist = _distance_matrix(similarity)
    labels = HDBSCAN(min_cluster_size=min_cluster_size, metric='precomputed').fit_predict(dist)
    return relabel_negatives(labels)


def _get_agglo_max_clusters(n_samples: int, dataset_name: str | None = None) -> int | None:
    """
    计算 Agglomerative 聚类的簇数上限。
    
    优先级：AGGLO_MAX_CLUSTERS_PER_DATASET[dataset_name] > 比例计算 > 不限（返回 None）。
    
    返回 None 表示不约束（AGGLO_ENABLE_N_CLUSTERS_CAP=False 时）。
    """
    if not AGGLO_ENABLE_N_CLUSTERS_CAP:
        return None
    # 子集级别绝对上限（优先）
    if dataset_name is not None:
        per = AGGLO_MAX_CLUSTERS_PER_DATASET.get(dataset_name)
        if per is not None:
            return max(AGGLO_MIN_CLUSTERS, int(per))
    # 按比例自动计算
    return max(AGGLO_MIN_CLUSTERS, int(n_samples * AGGLO_MAX_CLUSTERS_RATIO))


def run_Agglomerative(
    similarity: np.ndarray,
    distance_threshold: float,
    dataset_name: str | None = None,
) -> np.ndarray:
    """AgglomerativeClustering（average linkage），无噪声点 -1。

    [Fix 2] 原有独立的 _cosine_distance_for_agglo 与修复后的 _distance_matrix 完全一致，
    已合并统一调用，保证三种聚类器使用相同距离尺度，ARI 对比结果可信。

    ★ [新增] 簇数上限约束：
    当 distance_threshold 模式产生的簇数超过 _get_agglo_max_clusters() 时，
    自动切换为 n_clusters 固定版（更激进合并），避免大量单例簇。
    AGGLO_ENABLE_N_CLUSTERS_CAP=False 时行为与原来完全相同。
    """
    dist = _distance_matrix(similarity)
    n = dist.shape[0]

    # 第一轮：按 distance_threshold 聚类
    clf = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=float(distance_threshold),
        metric='precomputed',
        linkage=AGGLOMERATIVE_LINKAGE,
    )
    labels = np.asarray(clf.fit_predict(dist), dtype=int)

    # 检查簇数是否超限，若超则用 n_clusters 固定版重跑
    max_k = _get_agglo_max_clusters(n, dataset_name)
    if max_k is not None:
        n_found = int(np.unique(labels).shape[0])
        if n_found > max_k:
            # n_clusters 不能为 None，且须 >= 2
            k_use = max(2, min(max_k, n - 1))
            clf2 = AgglomerativeClustering(
                n_clusters=k_use,
                metric='precomputed',
                linkage=AGGLOMERATIVE_LINKAGE,
            )
            labels = np.asarray(clf2.fit_predict(dist), dtype=int)
            print(
                f"    [Agglo cap] distance_threshold={distance_threshold:.3f} 产生 {n_found} 簇 "
                f"> 上限 {max_k}（n={n}），改用 n_clusters={k_use}"
            )

    return labels


# ===================== YOLO 裁剪（Ultralytics 或 YOLOv5 torch.hub）+ 图像预处理 =====================
_yolo_crop_model = None
_yolov5_hub_crop_model = None
_crop_engine_resolved = None
_sam_refiner_model = None
_warned_ultralytics_numpy_missing = False


def _yolo_crop_weight_path() -> str:
    w = str(YOLO_CROP_WEIGHT).strip()
    if os.path.isfile(w):
        return w
    p = os.path.join(root, w)
    if os.path.isfile(p):
        return p
    return w


def _resolved_crop_engine() -> str:
    """'ultralytics' | 'yolov5_hub'。"""
    e = (YOLO_CROP_ENGINE or 'auto').strip().lower()
    if e not in ('auto', 'ultralytics', 'yolov5_hub'):
        raise ValueError("YOLO_CROP_ENGINE 必须是 'ultralytics' | 'yolov5_hub' | 'auto'")
    if e == 'ultralytics':
        return 'ultralytics'
    if e == 'yolov5_hub':
        return 'yolov5_hub'
    w = _yolo_crop_weight_path()
    bn = os.path.basename(str(w)).lower()
    if 'md_v' in bn or 'megadetector' in bn or 'mdv' in bn:
        return 'yolov5_hub'
    return 'ultralytics'


def _crop_engine() -> str:
    global _crop_engine_resolved
    if _crop_engine_resolved is None:
        _crop_engine_resolved = _resolved_crop_engine()
    return _crop_engine_resolved


def _yolov5_repo_candidates():
    out = []
    lr = str(YOLOV5_LOCAL_REPO).strip() if YOLOV5_LOCAL_REPO else ''
    if lr:
        out.append(lr)
    env = os.environ.get('YOLOV5_REPO', '').strip()
    if env:
        out.append(env)
    out.append(os.path.join(root, 'yolov5'))
    seen, uniq = set(), []
    for p in out:
        ap = os.path.abspath(p)
        if ap not in seen:
            seen.add(ap)
            uniq.append(p)
    return uniq


def _resolve_yolov5_repo_dir():
    for p in _yolov5_repo_candidates():
        if p and os.path.isfile(os.path.join(p, 'hubconf.py')):
            return p
    return None


def _load_yolov5_hub_custom(wpath: str):
    dev = 'cpu' if device == 'cpu' else 'cuda:0'
    local_repo = _resolve_yolov5_repo_dir()
    if local_repo is not None:
        print(f'    [YOLO crop] 本地 YOLOv5 torch.hub: {local_repo}')
        m = torch.hub.load(
            local_repo, 'custom', path=wpath, device=dev, source='local', trust_repo=True,
        )
    else:
        try:
            print('    [YOLO crop] torch.hub ultralytics/yolov5（需网络或已缓存）…')
            m = torch.hub.load(
                'ultralytics/yolov5', 'custom', path=wpath, device=dev, trust_repo=True,
            )
        except RuntimeError as e:
            msg = str(e).lower()
            if 'internet' in msg or 'cache' in msg or 'could not be found' in msg:
                raise RuntimeError(
                    'YOLOv5 裁剪需本地 yolov5 仓库。请 clone https://github.com/ultralytics/yolov5 到 '
                    f'{os.path.join(root, "yolov5")} 或设置 YOLOV5_REPO / YOLOV5_LOCAL_REPO。\n'
                    f'原始错误: {e}'
                ) from e
            raise
    m.conf = YOLO_CROP_CONF
    m.eval()
    return m


def _yolov5_xyxy_from_result(det_out):
    try:
        if hasattr(det_out, 'xyxy'):
            t = det_out.xyxy[0]
        elif isinstance(det_out, (list, tuple)):
            t = det_out[0]
        else:
            t = det_out
    except (IndexError, AttributeError, TypeError):
        return np.zeros((0, 6), dtype=np.float32)
    if t is None or (hasattr(t, 'numel') and t.numel() == 0):
        return np.zeros((0, 6), dtype=np.float32)
    arr = t.detach().cpu().numpy() if hasattr(t, 'cpu') else np.asarray(t, dtype=np.float32)
    if arr.size == 0:
        return np.zeros((0, 6), dtype=np.float32)
    return arr


def _yolov5_class_ids_for_weight(wpath: str):
    """MegaDetector(md_v*)：三分类里 animal=0。其它 YOLOv5 自定义权重：用 YOLO_CROP_CLASSES 或 None（不筛类）。"""
    bn = os.path.basename(str(wpath)).lower()
    if 'md_v' in bn or 'megadetector' in bn or 'mdv' in bn:
        return [0]
    if YOLO_CROP_CLASSES is not None and len(YOLO_CROP_CLASSES) > 0:
        return list(YOLO_CROP_CLASSES)
    return None


def _get_ultralytics_crop_model():
    global _yolo_crop_model
    if _yolo_crop_model is None:
        try:
            from ultralytics import YOLO
        except ImportError as e:
            raise RuntimeError('Ultralytics 裁剪需要: pip install ultralytics') from e
        w = _yolo_crop_weight_path()
        print(f'    [YOLO crop] Ultralytics YOLO: {w}')
        _yolo_crop_model = YOLO(w)
    return _yolo_crop_model


def _get_ultralytics_sam_model():
    global _sam_refiner_model
    if _sam_refiner_model is None:
        try:
            from ultralytics import SAM
        except ImportError as e:
            raise RuntimeError('SAM 精修需要: pip install ultralytics') from e
        print(f'    [SAM refine] Ultralytics SAM: {SAM_MODEL_WEIGHT}')
        _sam_refiner_model = SAM(SAM_MODEL_WEIGHT)
    return _sam_refiner_model


def _get_yolov5_hub_crop_model():
    global _yolov5_hub_crop_model
    if _yolov5_hub_crop_model is None:
        wpath = _yolo_crop_weight_path()
        if not os.path.isfile(wpath):
            raise FileNotFoundError(
                f'YOLOv5 裁剪权重未找到: {wpath}（将 YOLO_CROP_WEIGHT 设为绝对路径或放到 {root}）'
            )
        _yolov5_hub_crop_model = _load_yolov5_hub_custom(wpath)
    return _yolov5_hub_crop_model


def _pil_rgb_uint8_hwc(pil_img: Image.Image) -> np.ndarray:
    """RGB、HWC、uint8、内存连续（与 MegaDetector PTDetector 中 PIL→np 一致）。"""
    arr = np.asarray(pil_img.convert('RGB'), dtype=np.uint8)
    return np.ascontiguousarray(arr)


def _clamp_xyxy(x1, y1, x2, y2, w, h):
    x1i = max(0, min(int(x1), w - 1))
    y1i = max(0, min(int(y1), h - 1))
    x2i = max(1, min(int(x2), w))
    y2i = max(1, min(int(y2), h))
    if x2i <= x1i:
        x2i = min(w, x1i + 1)
    if y2i <= y1i:
        y2i = min(h, y1i + 1)
    return x1i, y1i, x2i, y2i


def _expand_box_ratio(x1, y1, x2, y2, w, h, ratio):
    bw = max(1.0, float(x2 - x1))
    bh = max(1.0, float(y2 - y1))
    dx = bw * float(ratio)
    dy = bh * float(ratio)
    return _clamp_xyxy(x1 - dx, y1 - dy, x2 + dx, y2 + dy, w, h)


def _sam_refine_crop(rgb_pil: Image.Image, x1: float, y1: float, x2: float, y2: float):
    """用 SAM 在 YOLO 框内做前景精修，返回 refined box；失败时返回 None。"""
    if not USE_SAM_REFINER:
        return None
    w, h = rgb_pil.size
    bx1, by1, bx2, by2 = _expand_box_ratio(x1, y1, x2, y2, w, h, SAM_BOX_EXPAND)
    sam = _get_ultralytics_sam_model()
    try:
        # Ultralytics SAM 支持 bboxes 提示；不同版本参数签名差异较大，统一走 kwargs 并兜底。
        res = sam.predict(source=rgb_pil, bboxes=[[bx1, by1, bx2, by2]], verbose=False)[0]
    except Exception:
        return None
    masks = getattr(res, 'masks', None)
    if masks is None or getattr(masks, 'data', None) is None:
        return None
    mask_data = masks.data
    if mask_data is None or len(mask_data) == 0:
        return None
    m = mask_data[0]
    m_np = m.detach().cpu().numpy() if hasattr(m, 'detach') else np.asarray(m, dtype=np.float32)
    m_bin = m_np > 0.5
    if not m_bin.any():
        return None
    if float(m_bin.mean()) < float(SAM_MASK_MIN_AREA_RATIO):
        return None
    ys, xs = np.where(m_bin)
    if len(xs) == 0 or len(ys) == 0:
        return None
    rx1, ry1 = int(xs.min()), int(ys.min())
    rx2, ry2 = int(xs.max()) + 1, int(ys.max()) + 1
    return _clamp_xyxy(rx1, ry1, rx2, ry2, w, h)


def _yolo_crop(pil_img: Image.Image) -> Image.Image:
    """ultralytics.predict 或 YOLOv5 hub 推理，取置信度最高框扩边裁剪。"""
    w, h = pil_img.size
    margin = YOLO_CROP_MARGIN
    eng = _crop_engine()
    wpath = _yolo_crop_weight_path()
    rgb_pil = pil_img.convert('RGB')

    if eng == 'yolov5_hub':
        det = _get_yolov5_hub_crop_model()
        bn = os.path.basename(str(wpath)).lower()
        # MegaDetector 官方推理管线使用 RGB numpy，不做 BGR 翻转；其它 YOLOv5 自定义权重按 OpenCV 惯例用 BGR
        if 'md_v' in bn or 'megadetector' in bn or 'mdv' in bn:
            inp = _pil_rgb_uint8_hwc(rgb_pil)
        else:
            inp = cv2.cvtColor(_pil_rgb_uint8_hwc(rgb_pil), cv2.COLOR_RGB2BGR)
        cls_ids = _yolov5_class_ids_for_weight(wpath)
        with torch.no_grad():
            raw = det(inp)
        d = _yolov5_xyxy_from_result(raw)
        if d.shape[0] == 0:
            return pil_img
        if cls_ids is not None:
            mask = np.isin(d[:, 5].astype(np.int64), cls_ids)
            if not mask.any():
                return pil_img
            d = d[mask]
        best_i = int(np.argmax(d[:, 4]))
        x1, y1, x2, y2 = map(float, d[best_i, :4])
    else:
        global _warned_ultralytics_numpy_missing
        det = _get_ultralytics_crop_model()
        dev = 0 if device == 'cuda' else 'cpu'
        # 直接传 PIL RGB，由 Ultralytics 内部做与训练一致的预处理；避免 ndarray 的 BGR/RGB 约定歧义
        kw = dict(source=rgb_pil, conf=YOLO_CROP_CONF, verbose=False, device=dev)
        if YOLO_CROP_CLASSES is not None and len(YOLO_CROP_CLASSES) > 0:
            kw['classes'] = list(YOLO_CROP_CLASSES)
        try:
            res = det.predict(**kw)[0]
        except RuntimeError as e:
            msg = str(e)
            if 'Numpy is not available' in msg:
                if not _warned_ultralytics_numpy_missing:
                    print(
                        "    [YOLO crop][警告] 当前环境 torch<->numpy 不可用，"
                        "自动跳过 YOLO/SAM 裁剪并继续。"
                    )
                    _warned_ultralytics_numpy_missing = True
                return pil_img
            raise
        boxes = res.boxes
        if boxes is None or len(boxes) == 0:
            return pil_img
        try:
            confs = boxes.conf.cpu().numpy()
            xyxy = boxes.xyxy.cpu().numpy()
        except RuntimeError as e:
            if 'Numpy is not available' in str(e):
                if not _warned_ultralytics_numpy_missing:
                    print(
                        "    [YOLO crop][警告] 当前环境 torch<->numpy 不可用，"
                        "自动跳过 YOLO/SAM 裁剪并继续。"
                    )
                    _warned_ultralytics_numpy_missing = True
                return pil_img
            raise
        best_i = int(np.argmax(confs))
        x1, y1, x2, y2 = map(float, xyxy[best_i])

    if USE_SAM_REFINER and eng == 'ultralytics':
        sam_box = _sam_refine_crop(rgb_pil, x1, y1, x2, y2)
        if sam_box is not None:
            x1i, y1i, x2i, y2i = sam_box
        else:
            x1i, y1i, x2i, y2i = _expand_box_ratio(x1, y1, x2, y2, w, h, margin)
    else:
        x1i, y1i, x2i, y2i = _expand_box_ratio(x1, y1, x2, y2, w, h, margin)
    if x2i <= x1i or y2i <= y1i:
        return pil_img
    return pil_img.crop((x1i, y1i, x2i, y2i))


def _clahe_lighting(pil_img: Image.Image) -> Image.Image:
    img_np = cv2.cvtColor(np.asarray(pil_img.convert('RGB')), cv2.COLOR_RGB2LAB)
    l_ch, a_ch, b_ch = cv2.split(img_np)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_ch = clahe.apply(l_ch)
    merged = cv2.merge((l_ch, a_ch, b_ch))
    rgb = cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)
    return Image.fromarray(rgb)


def _texture_sharpen(pil_img: Image.Image) -> Image.Image:
    img_np = np.asarray(pil_img.convert('RGB')).astype(np.float32)
    blurred = cv2.GaussianBlur(img_np, (0, 0), 1.2)
    sharp = cv2.addWeighted(img_np, 1.1, blurred, -0.1, 0.0)
    sharp_u8 = np.clip(sharp, 0, 255).astype(np.uint8)
    return Image.fromarray(sharp_u8)


def _lynx_is_ir_like(rgb_u8: np.ndarray) -> bool:
    hsv = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2HSV)
    sat_mean = float(hsv[:, :, 1].mean())
    return sat_mean < float(LYNX_IR_SAT_MEAN_THRESHOLD)


def _lynx_stage2_align(rgb_pil: Image.Image) -> Image.Image:
    """阶段2：LAB 对齐 + 过曝修复 + L 通道全局归一化。"""
    rgb = np.asarray(rgb_pil.convert('RGB'), dtype=np.uint8)
    ir_like = _lynx_is_ir_like(rgb)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    l_ch = lab[:, :, 0]
    a_ch = lab[:, :, 1]
    b_ch = lab[:, :, 2]
    # 红外图把 a/b 固定到中性值，彩图只做轻度均值回归。
    if ir_like:
        a_ch[:, :] = 128.0
        b_ch[:, :] = 128.0
    else:
        a_ch = 128.0 + 0.8 * (a_ch - 128.0)
        b_ch = 128.0 + 0.8 * (b_ch - 128.0)
    # 过曝修复（典型 IR 眼睛反光）
    over = (l_ch > 240.0).astype(np.uint8) * 255
    if over.any():
        l_u8 = np.clip(l_ch, 0, 255).astype(np.uint8)
        l_ch = cv2.inpaint(l_u8, over, inpaintRadius=3, flags=cv2.INPAINT_TELEA).astype(np.float32)
    # L 通道归一化到固定均值/方差，减少昼夜亮度偏移
    mu = float(l_ch.mean())
    sd = float(l_ch.std())
    if sd < 1e-6:
        sd = 1.0
    l_ch = (l_ch - mu) / sd * float(LYNX_L_TARGET_STD) + float(LYNX_L_TARGET_MEAN)
    lab_out = np.stack([
        np.clip(l_ch, 0, 255),
        np.clip(a_ch, 0, 255),
        np.clip(b_ch, 0, 255),
    ], axis=2).astype(np.uint8)
    return Image.fromarray(cv2.cvtColor(lab_out, cv2.COLOR_LAB2RGB))


def _lynx_stage3_texture(rgb_pil: Image.Image) -> Image.Image:
    """阶段3：CLAHE + 纹理保留去噪 + 温和非锐化掩膜。"""
    rgb = np.asarray(rgb_pil.convert('RGB'), dtype=np.uint8)
    ir_like = _lynx_is_ir_like(rgb)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)
    clip = 1.5 if ir_like else 2.0
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8))
    l_ch = clahe.apply(l_ch)
    rgb2 = cv2.cvtColor(cv2.merge((l_ch, a_ch, b_ch)), cv2.COLOR_LAB2RGB)
    if ir_like:
        rgb2 = cv2.fastNlMeansDenoisingColored(rgb2, None, 4, 4, 5, 13)
    else:
        rgb2 = cv2.bilateralFilter(rgb2, d=7, sigmaColor=45, sigmaSpace=45)
    base = rgb2.astype(np.float32)
    blur = cv2.GaussianBlur(base, (0, 0), 1.0)
    sharp = cv2.addWeighted(base, 1.12, blur, -0.12, 0.0)
    return Image.fromarray(np.clip(sharp, 0, 255).astype(np.uint8))


def _lynx_special_post_crop(rgb_pil: Image.Image) -> Image.Image:
    out = rgb_pil.convert('RGB')
    if LYNX_STAGE2_ENABLE:
        out = _lynx_stage2_align(out)
    if LYNX_STAGE3_ENABLE:
        out = _lynx_stage3_texture(out)
    return out


_salamander_ab_ref_cache: tuple[float, float] | None = None


def _salamander_ab_reference() -> tuple[float, float]:
    """返回 Salamander A/B 全局参考均值：优先 npz（mean_a/mean_b），否则用常量。"""
    global _salamander_ab_ref_cache
    if _salamander_ab_ref_cache is not None:
        return _salamander_ab_ref_cache
    path = (SAL_AB_STATS_PATH or '').strip()
    if path and os.path.isfile(path):
        z = np.load(path)
        ma = float(np.asarray(z['mean_a']).reshape(-1)[0])
        mb = float(np.asarray(z['mean_b']).reshape(-1)[0])
        _salamander_ab_ref_cache = (ma, mb)
    else:
        _salamander_ab_ref_cache = (float(SAL_AB_REF_MEAN_A), float(SAL_AB_REF_MEAN_B))
    return _salamander_ab_ref_cache


def _salamander_low_quality_heuristic(rgb_u8: np.ndarray) -> bool:
    """弱对比度 / 整体过暗或过亮 → 温和 CLAHE 与较弱去噪。"""
    if rgb_u8.ndim != 3 or rgb_u8.shape[2] != 3:
        return False
    lab = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2LAB)
    l_ch = lab[:, :, 0].astype(np.float32)
    mean_l = float(l_ch.mean())
    std_l = float(l_ch.std())
    if mean_l < SAL_LOW_QUALITY_MEAN_L_LOW or mean_l > SAL_LOW_QUALITY_MEAN_L_HIGH:
        return True
    if std_l < SAL_LOW_QUALITY_STD_L_MAX:
        return True
    return False


def _salamander_stage1_glare(rgb_pil: Image.Image) -> Image.Image:
    """LAB 仅修复 L 通道高光区；inpaint 填充，A/B 保持原样。"""
    rgb = np.asarray(rgb_pil.convert('RGB'), dtype=np.uint8)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)
    mask = ((l_ch.astype(np.int16)) > int(SAL_GLARE_L_THRESHOLD)).astype(np.uint8) * 255
    k = max(3, int(SAL_GLARE_MORPH_KERNEL) | 1)
    kernel = np.ones((k, k), dtype=np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)
    if not mask.any():
        return rgb_pil
    rad = max(1, int(SAL_GLARE_INPAINT_RADIUS))
    flag = cv2.INPAINT_NS if str(SAL_GLARE_INPAINT_METHOD).lower() == 'ns' else cv2.INPAINT_TELEA
    l_fix = cv2.inpaint(l_ch, mask, rad, flags=flag)
    lab2 = cv2.merge([l_fix, a_ch, b_ch])
    out = cv2.cvtColor(lab2, cv2.COLOR_LAB2RGB)
    return Image.fromarray(out)


def _salamander_stage2_align_clahe(rgb_pil: Image.Image, low_quality: bool) -> Image.Image:
    """L 映射到 [SAL_L_RANGE_LOW, SAL_L_RANGE_HIGH]；A/B 均值对齐参考；仅 L 做 CLAHE。"""
    rgb = np.asarray(rgb_pil.convert('RGB'), dtype=np.uint8)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)
    L = l_ch.astype(np.float32)
    lo, hi = float(L.min()), float(L.max())
    rl, rh = float(SAL_L_RANGE_LOW), float(SAL_L_RANGE_HIGH)
    if hi - lo > 1e-3:
        L = rl + (L - lo) * (rh - rl) / (hi - lo)
    else:
        L[:] = (rl + rh) * 0.5
    L = np.clip(L, 0.0, 255.0)
    ref_a, ref_b = _salamander_ab_reference()
    a_f = a_ch.astype(np.float32) - float(np.mean(a_ch)) + ref_a
    b_f = b_ch.astype(np.float32) - float(np.mean(b_ch)) + ref_b
    a_out = np.clip(a_f, 0, 255).astype(np.uint8)
    b_out = np.clip(b_f, 0, 255).astype(np.uint8)
    l_u8 = np.clip(L, 0, 255).astype(np.uint8)
    clip = float(SAL_CLAHE_CLIP_GENTLE if low_quality else SAL_CLAHE_CLIP_NORMAL)
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8))
    l_u8 = clahe.apply(l_u8)
    lab_m = cv2.merge([l_u8, a_out, b_out])
    return Image.fromarray(cv2.cvtColor(lab_m, cv2.COLOR_LAB2RGB))


def _salamander_stage3_texture(rgb_pil: Image.Image, low_quality: bool) -> Image.Image:
    """双边或 NLM 去噪 + 温和非锐化掩膜。"""
    rgb = np.asarray(rgb_pil.convert('RGB'), dtype=np.uint8)
    method = str(SAL_STAGE3_DENOISE).lower()
    if method == 'nlmeans':
        h = float(SAL_NLM_H_GENTLE if low_quality else SAL_NLM_H)
        rgb_d = cv2.fastNlMeansDenoisingColored(
            rgb, None, h, h,
            int(SAL_NLM_TEMPLATE), int(SAL_NLM_SEARCH),
        )
    else:
        d = int(SAL_BILAT_D_GENTLE if low_quality else SAL_BILAT_D)
        if d % 2 == 0:
            d += 1
        sc = float(SAL_BILAT_SIGMA_COLOR_GENTLE if low_quality else SAL_BILAT_SIGMA_COLOR)
        ss = float(SAL_BILAT_SIGMA_SPACE_GENTLE if low_quality else SAL_BILAT_SIGMA_SPACE)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        bgr = cv2.bilateralFilter(bgr, d, sc, ss)
        rgb_d = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    base = rgb_d.astype(np.float32)
    blur = cv2.GaussianBlur(base, (0, 0), float(SAL_UNSHARP_SIGMA))
    amt = float(SAL_UNSHARP_AMOUNT_GENTLE if low_quality else SAL_UNSHARP_AMOUNT)
    sharp = cv2.addWeighted(base, 1.0 + amt, blur, -amt, 0.0)
    return Image.fromarray(np.clip(sharp, 0, 255).astype(np.uint8))


def _salamander_special_post_crop(rgb_pil: Image.Image) -> Image.Image:
    """Salamander：YOLO 裁剪后的专属管线（各子开关可消融）。"""
    out = rgb_pil.convert('RGB')
    arr = np.asarray(out, dtype=np.uint8)
    low_q = _salamander_low_quality_heuristic(arr)
    if SAL_GLARE_ENABLE:
        out = _salamander_stage1_glare(out)
        low_q = _salamander_low_quality_heuristic(np.asarray(out, dtype=np.uint8))
    if SAL_ALIGN_ENABLE:
        out = _salamander_stage2_align_clahe(out, low_q)
    if SAL_TEXTURE_ENABLE:
        out = _salamander_stage3_texture(out, low_q)
    return out


def animal_preprocess_pipeline(img: Image.Image, dataset_name: str | None = None) -> Image.Image:
    """
    像素级预处理：可选 YOLO 动物框裁剪（可叠加 SAM 精修）、CLAHE、锐化。
    几何缩放（letterbox 或拉伸）在 transform 里单独做，便于与骨干输入尺寸对齐。
    """
    out = img.convert('RGB')
    if USE_MEGADETECTOR_CROP:
        out = _yolo_crop(out)
    # 仅对 LynxID2025：在裁剪后执行专属预处理
    if (
        USE_LYNX_SPECIAL_PREPROCESS
        and dataset_name == 'LynxID2025'
    ):
        out = _lynx_special_post_crop(out)
    if (
        USE_SALAMANDER_SPECIAL_PREPROCESS
        and dataset_name == 'SalamanderID2025'
    ):
        out = _salamander_special_post_crop(out)
    if USE_CLAHE_LIGHTING:
        out = _clahe_lighting(out)
    if USE_TEXTURE_SHARPEN:
        out = _texture_sharpen(out)
    return out


def resize_letterbox_square(img: Image.Image, target_size: int, pad_mode: str = 'gray') -> Image.Image:
    """等比缩放到方图；支持灰色填充或边缘复制填充。"""
    img = img.convert('RGB')
    w, h = img.size
    if w <= 0 or h <= 0:
        return Image.new('RGB', (target_size, target_size), (128, 128, 128))
    scale = target_size / max(w, h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    img = img.resize((new_w, new_h), Image.BICUBIC)
    top = (target_size - new_h) // 2
    bottom = target_size - new_h - top
    left = (target_size - new_w) // 2
    right = target_size - new_w - left
    if pad_mode == 'edge':
        arr = np.asarray(img, dtype=np.uint8)
        arr = cv2.copyMakeBorder(arr, top, bottom, left, right, borderType=cv2.BORDER_REPLICATE)
        return Image.fromarray(arr)
    canvas = Image.new('RGB', (target_size, target_size), (128, 128, 128))
    canvas.paste(img, (left, top))
    return canvas


def _build_geom_transform(target_size: int, dataset_name: str | None = None):
    if USE_LETTERBOX_TO_INPUT:
        # Lynx / Salamander 专属管线：可用边缘复制填充至方图，避免黑边被模型当成特征。
        use_edge = (
            (USE_LYNX_SPECIAL_PREPROCESS and dataset_name == 'LynxID2025')
            or (
                USE_SALAMANDER_SPECIAL_PREPROCESS
                and dataset_name == 'SalamanderID2025'
                and SAL_USE_EDGE_LETTERBOX
            )
        )
        pad_mode = 'edge' if use_edge else 'gray'
        return T.Lambda(lambda im: resize_letterbox_square(im, target_size, pad_mode=pad_mode))
    return T.Resize((target_size, target_size), interpolation=T.InterpolationMode.BICUBIC)


def _maybe_preprocess_lambda(dataset_name: str | None = None):
    if (
        USE_MEGADETECTOR_CROP
        or USE_CLAHE_LIGHTING
        or USE_TEXTURE_SHARPEN
        or USE_LYNX_SPECIAL_PREPROCESS
        or USE_SALAMANDER_SPECIAL_PREPROCESS
    ):
        return T.Lambda(lambda im: animal_preprocess_pipeline(im, dataset_name=dataset_name))
    return T.Lambda(lambda im: im.convert('RGB'))


# ============================================================
# 微调辅助函数（按个体划分；wildlife_tools 训练管线）
# ============================================================
def split_by_identity(df: pd.DataFrame, train_ratio: float = 0.8, random_state: int = 42):
    """按 identity 划分 train/val，同一动物个体仅出现在一侧。"""
    unique_ids = df['identity'].unique()
    train_ids, val_ids = train_test_split(
        unique_ids, train_size=train_ratio, random_state=random_state
    )
    train_df = df[df['identity'].isin(train_ids)]
    val_df = df[df['identity'].isin(val_ids)]
    return train_df, val_df


def load_finetuned_mega(checkpoint_path: str, device: str):
    """加载微调后的 MegaDescriptor-L-384 权重（eval）。"""
    model = timm.create_model(
        'swin_large_patch4_window12_384',
        pretrained=False,
        num_classes=0,
    )
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint['model'] if isinstance(checkpoint, dict) and 'model' in checkpoint else checkpoint
    if state_dict and all(k.startswith('module.') for k in state_dict.keys()):
        state_dict = {k.replace('module.', '', 1): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    return model.eval().to(device)


def load_finetuned_miew(checkpoint_path: str, local_miew_path: str, device: str) -> MiewIDWrapper:
    """在完整 MiewID 结构上加载微调 checkpoint（与 finetune 时保存的 state_dict 一致）。"""
    model = load_miew_model(local_miew_path, device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint['model'] if isinstance(checkpoint, dict) and 'model' in checkpoint else checkpoint
    if state_dict and all(k.startswith('module.') for k in state_dict.keys()):
        state_dict = {k.replace('module.', '', 1): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=True)
    return model.eval()


def finetune_checkpoint_path(dataset_name: str, backbone: str) -> str:
    """默认 checkpoint 路径：finetuned_{mega|miew}_{DatasetName}.pth"""
    safe = dataset_name.replace(os.sep, '_').replace('/', '_')
    fname = f'finetuned_{backbone}_{safe}.pth'
    return os.path.join(FINETUNE_SAVE_DIR, fname)


def finetune_target_backbone(dataset_name: str) -> str:
    """根据 FEATURE_STRATEGY 决定该子集微调哪个骨干（须与后续特征提取所用骨干一致）。"""
    if FEATURE_STRATEGY == 'global_mega':
        return 'mega'
    if FEATURE_STRATEGY == 'global_miew':
        return 'miew'
    if FEATURE_STRATEGY in ('fusion', 'per_dataset_baseline', 'fusion_salamander_only'):
        return backbone_for_dataset_baseline(dataset_name)
    raise ValueError(f'无法为 FEATURE_STRATEGY={FEATURE_STRATEGY!r} 解析微调骨干')


def finetune_on_dataset(
    dataset_name: str,
    train_df: pd.DataFrame,
    model: nn.Module,
    device: str,
    *,
    input_size: int,
    save_path: str,
):
    """使用 wildlife_tools 在 train_df 上做 embedding + Triplet 微调（与 extract_features_single 一致的预处理）。"""
    from wildlife_tools.data.dataset import WildlifeDataset as WTWildlifeDataset
    from wildlife_tools.train import BasicTrainer
    from wildlife_tools.train.objective import TripletLoss

    pre = _maybe_preprocess_lambda(dataset_name=dataset_name)
    geom = _build_geom_transform(input_size, dataset_name=dataset_name)
    base_tf = T.Compose([
        pre,
        geom,
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])

    train_ds = WTWildlifeDataset(
        train_df.reset_index(drop=True),
        root=root,
        transform=base_tf,
        col_path='path',
        col_label='identity',
        load_label=True,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=FINETUNE_LEARNING_RATE)
    objective = TripletLoss(margin=FINETUNE_MARGIN, mining='semihard')

    _workers = 0 if os.name == 'nt' else min(2, os.cpu_count() or 1)

    trainer = BasicTrainer(
        dataset=train_ds,
        model=model,
        objective=objective,
        optimizer=optimizer,
        epochs=FINETUNE_EPOCHS,
        scheduler=None,
        device=device,
        batch_size=FINETUNE_BATCH_SIZE,
        num_workers=_workers,
        accumulation_steps=FINETUNE_ACCUMULATION_STEPS,
        epoch_callback=None,
    )

    bb = 'miew' if input_size == 512 else 'mega'
    print(f"\n========== 开始微调 {dataset_name}（{bb}, {input_size}px, wildlife_tools）==========")
    trainer.train()

    os.makedirs(FINETUNE_SAVE_DIR, exist_ok=True)
    torch.save({'model': model.state_dict()}, save_path)
    print(f"微调模型已保存至: {save_path}")

    model.eval()
    return model, save_path


# ============================================================
# Fix 4: 特征提取（深拷贝防止 set_transform 污染）
# ============================================================
def _extract_once(model, dataset_copy, transform, device, batch_size):
    """用给定 transform 提取特征，返回 L2 归一化后的 ndarray。"""
    dataset_copy.set_transform(transform)
    extractor = DeepFeatures(
        model=model, device=device,
        batch_size=batch_size, num_workers=0
    )
    feats = np.array(extractor(dataset_copy))
    norms = np.linalg.norm(feats, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return feats / norms


def extract_features_single(model, dataset, size: int, tta: bool = True, dataset_name: str | None = None) -> np.ndarray:
    """单模型特征提取，支持 TTA（水平翻转均值）。返回 L2 归一化特征。"""
    pre = _maybe_preprocess_lambda(dataset_name=dataset_name)
    geom = _build_geom_transform(size, dataset_name=dataset_name)
    base_tf = T.Compose([
        pre,
        geom,
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    flip_tf = T.Compose([
        pre,
        geom,
        T.RandomHorizontalFlip(p=1.0),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])

    ds = copy.copy(dataset)
    feats = _extract_once(model, ds, base_tf, device, batch_size)

    if tta:
        ds_flip = copy.copy(dataset)
        feats_flip = _extract_once(model, ds_flip, flip_tf, device, batch_size)
        feats = (feats + feats_flip) / 2.0
        norms = np.linalg.norm(feats, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        feats = feats / norms

    return feats


# ============================================================
# 优先级 3：TTA 多视图配置（按子集定制旋转/亮度变体）
# ============================================================
TTA_VIEWS_CONFIG = {
    # 蝾螈：手持拍摄角度多变，旋转 + 亮度变体都有意义
    'SalamanderID2025': [
        [],                                                          # 原图
        [T.RandomHorizontalFlip(p=1.0)],                             # 水平翻转
        [T.RandomVerticalFlip(p=1.0)],                               # 垂直翻转
        [T.functional.rotate, 90],                                   # 90° 旋转（特殊标记）
        [T.ColorJitter(brightness=0.15, contrast=0.15)],             # 光照变体
    ],
    # 海龟：水下拍摄，垂直翻转意义不大；亮度变化明显
    'SeaTurtleID2022': [
        [],
        [T.RandomHorizontalFlip(p=1.0)],
        [T.ColorJitter(brightness=0.2, contrast=0.1)],
    ],
    # 猞猁：IR/彩色混合，ColorJitter 意义不大；只做翻转
    'LynxID2025': [
        [],
        [T.RandomHorizontalFlip(p=1.0)],
    ],
    # 角蜥：保守，只做翻转
    'TexasHornedLizards': [
        [],
        [T.RandomHorizontalFlip(p=1.0)],
    ],
}
TTA_VIEWS_CONFIG['__default__'] = [
    [],
    [T.RandomHorizontalFlip(p=1.0)],
]


def _build_tta_transform(base_pre, geom, augments: list, normalize) -> T.Compose:
    """
    将一组增强操作插入到 geom 之后、ToTensor 之前。
    90° 旋转用特殊列表 [T.functional.rotate, 90] 标记，
    其余增强均为标准 torchvision Transform 对象。
    """
    processed = []
    i = 0
    while i < len(augments):
        if augments[i] is T.functional.rotate:
            angle = augments[i + 1]
            processed.append(T.Lambda(lambda im, a=angle: T.functional.rotate(im, a)))
            i += 2
        else:
            processed.append(augments[i])
            i += 1
    return T.Compose([base_pre, geom] + processed + [T.ToTensor(), normalize])


def extract_features_tta_multiview(
    model,
    dataset,
    size: int,
    dataset_name: str | None = None,
) -> np.ndarray:
    """
    多视图 TTA：按 TTA_VIEWS_CONFIG 为每个子集定制视图组合。
    各视图特征取均值后整体 L2 归一化。
    """
    pre = _maybe_preprocess_lambda(dataset_name=dataset_name)
    geom = _build_geom_transform(size, dataset_name=dataset_name)
    normalize = T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

    views_cfg = TTA_VIEWS_CONFIG.get(dataset_name, TTA_VIEWS_CONFIG['__default__'])
    print(f"    TTA 视图数: {len(views_cfg)}")

    all_feats = []
    for idx, augments in enumerate(views_cfg):
        tf = _build_tta_transform(pre, geom, augments, normalize)
        ds_copy = copy.copy(dataset)
        feats = _extract_once(model, ds_copy, tf, device, batch_size)
        all_feats.append(feats)
        print(f"    视图 {idx+1}/{len(views_cfg)} 完成，shape={feats.shape}")

    # 均值融合
    stacked = np.stack(all_feats, axis=0)   # (n_views, n_samples, dim)
    mean_feats = stacked.mean(axis=0)        # (n_samples, dim)

    norms = np.linalg.norm(mean_feats, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return mean_feats / norms


# ============================================================
# 优先级 4：融合权重网格搜索
# ============================================================
# 搜索空间：w_mega ∈ SALAMANDER_WEIGHT_GRID，w_miew 自动为 3 - w_mega（总权重保持 3.0）
SALAMANDER_WEIGHT_GRID = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2.0,2.1,2.2,2.3,2.4,2.5,2.6,2.7,2.8,2.9,3.0]
# False: 不搜权重，直接用等权（w_mega=1.0）
SEARCH_SALAMANDER_WEIGHT = True
SALAMANDER_MEGA_WEIGHT = 1.0   # SEARCH_SALAMANDER_WEIGHT=False 时的固定值
# FEATURE_STRATEGY='fusion' 时，这些子集会自动搜索 Mega/Miew 最优融合权重；
# fusion_salamander_only 下命中 SalamanderID2025 + TexasHornedLizards（这两者走融合路由）。
FUSION_WEIGHT_SEARCH_DATASETS = (
    'SalamanderID2025',
    'SeaTurtleID2022',
    'LynxID2025',
    'TexasHornedLizards',
)
# 无 train split 时，是否使用 Salamander 先验在目标子集上做无监督权重搜索
USE_SALAMANDER_PRIOR_FOR_NO_TRAIN_WEIGHT_SEARCH = True
NO_TRAIN_WEIGHT_SEARCH_DATASETS = ('TexasHornedLizards',)
SALAMANDER_PRIOR_METHOD = 'agglomerative'  # 当本次运行未包含 Salamander 时的先验方法
SALAMANDER_PRIOR_PARAM = 0.25              # 对应先验参数（Agglo 为 distance_threshold）

_fusion_best_weights: dict[str, float] = {}  # 全局缓存，避免 test 时重搜


def fuse_with_weight(feat_mega: np.ndarray, feat_miew: np.ndarray, w_mega: float) -> np.ndarray:
    """按权重拼接并整体 L2 归一化。w_mega + w_miew = 3.0，等权时各为 1.5。"""
    w_miew = 3.0 - w_mega
    fused = np.concatenate([feat_mega * w_mega, feat_miew * w_miew], axis=1)
    norms = np.linalg.norm(fused, axis=1, keepdims=True)
    return fused / np.where(norms == 0, 1.0, norms)


def search_fusion_weight(
    feat_mega_train: np.ndarray,
    feat_miew_train: np.ndarray,
    true_labels: np.ndarray,
    threshold_grid=None,
    dataset_name: str = 'dataset',
) -> tuple[float, float, float]:
    """
    在 train 特征上联合搜索融合权重 w_mega 和 Agglomerative distance_threshold。
    返回 (best_w_mega, best_threshold, best_ari)。
    
    ★ 传入 dataset_name，使 run_Agglomerative 内部簇数约束与最终推理一致。
    """
    if threshold_grid is None:
        threshold_grid = np.arange(0.06, 0.55, 0.01)

    best_w, best_thr, best_ari = 1.0, 0.35, -1.0

    for w in SALAMANDER_WEIGHT_GRID:
        fused = fuse_with_weight(feat_mega_train, feat_miew_train, w)
        sim = compute_similarity(fused)
        thr, ari = search_best_agglo_threshold(sim, true_labels, threshold_grid, dataset_name=dataset_name)
        print(f"    w_mega={w:.2f}, best_thr={thr:.3f}, ARI={ari:.4f}")
        if ari > best_ari:
            best_ari, best_w, best_thr = ari, w, thr

    print(f"  => {dataset_name} 最优 w_mega={best_w:.2f}, threshold={best_thr:.3f}, ARI={best_ari:.4f}")
    return best_w, best_thr, best_ari


def _cluster_with_method(
    similarity: np.ndarray,
    method: str,
    param: float | int,
    dataset_name: str | None = None,
) -> np.ndarray:
    """按给定方法与参数聚类，返回标签。"""
    if method == 'agglomerative':
        return run_Agglomerative(similarity, float(param), dataset_name=dataset_name)
    if method == 'hdbscan':
        return run_HDBSCAN(similarity, min_cluster_size=int(param))
    return run_DBSCAN(similarity, eps=float(param))


def _unsupervised_cluster_score(similarity: np.ndarray, labels: np.ndarray) -> float:
    """无监督评分：Silhouette（基于预计算余弦距离）。异常或退化聚类返回 -1。"""
    labels = np.asarray(labels, dtype=int)
    uniq = np.unique(labels)
    if len(uniq) <= 1 or len(uniq) >= len(labels):
        return -1.0
    dist = _distance_matrix(similarity)
    try:
        return float(silhouette_score(dist, labels, metric='precomputed'))
    except Exception:
        return -1.0


def search_weight_with_prior_cluster(
    feat_mega: np.ndarray,
    feat_miew: np.ndarray,
    *,
    method: str,
    param: float | int,
    dataset_name: str,
    weight_grid=None,
) -> tuple[float, float]:
    """
    无 train 时：固定先验聚类方法/参数，仅搜索融合权重 w_mega，
    以无监督 Silhouette 评分选择最优权重。
    返回 (best_w_mega, best_score)。
    """
    if weight_grid is None:
        weight_grid = SALAMANDER_WEIGHT_GRID
    best_w, best_score = 1.0, -1.0
    for w in weight_grid:
        fused = fuse_with_weight(feat_mega, feat_miew, float(w))
        sim = compute_similarity(fused)
        pred = _cluster_with_method(sim, method, param)
        score = _unsupervised_cluster_score(sim, pred)
        print(f"    [{dataset_name}] w_mega={w:.2f}, silhouette={score:.4f}")
        if score > best_score:
            best_w, best_score = float(w), float(score)
    print(f"  => [{dataset_name}] 最优 w_mega={best_w:.2f}, silhouette={best_score:.4f}")
    return best_w, best_score


def search_salamander_weight(
    feat_mega_train: np.ndarray,
    feat_miew_train: np.ndarray,
    true_labels: np.ndarray,
    threshold_grid=None,
) -> tuple[float, float, float]:
    """兼容旧函数名：内部复用通用融合权重搜索。"""
    return search_fusion_weight(
        feat_mega_train,
        feat_miew_train,
        true_labels,
        threshold_grid=threshold_grid,
        dataset_name='SalamanderID2025',
    )


def should_search_fusion_weight(dataset_name: str, use_fusion_here: bool) -> bool:
    """当前子集是否执行融合权重搜索。"""
    return (
        use_fusion_here
        and SEARCH_SALAMANDER_WEIGHT
        and dataset_name in FUSION_WEIGHT_SEARCH_DATASETS
    )


def extract_fused_features(mega_model, miew_model, dataset, tta: bool = True, dataset_name: str | None = None) -> np.ndarray:
    """双模型融合特征：MegaDescriptor(384) + MiewID(512)。拼接后整体 L2 归一化。"""
    print("    提取 MegaDescriptor 特征...")
    feat_mega = extract_features_single(mega_model, dataset, size=384, tta=tta, dataset_name=dataset_name)

    print("    提取 MiewID 特征...")
    feat_miew = extract_features_single(miew_model, dataset, size=512, tta=tta, dataset_name=dataset_name)

    fused = np.concatenate([feat_mega, feat_miew], axis=1)
    norms = np.linalg.norm(fused, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return fused / norms


def backbone_for_dataset_baseline(dataset_name: str) -> str:
    """与 baseline.py 一致：按子集选 Mega 或 Miew（非融合）。"""
    if dataset_name in ('SalamanderID2025', 'SeaTurtleID2022'):
        return 'mega'
    if dataset_name in ('LynxID2025', 'TexasHornedLizards'):
        return 'miew'
    raise ValueError(f'未知子数据集 {dataset_name!r}，无法按 baseline 分配骨干')


def extract_features_pipeline(
    mega_model, miew_model, dataset,
    *,
    dataset_name: str,
    use_fusion: bool,
    single_backbone: str,
    use_tta: bool,
    w_mega: float = 1.0,
) -> np.ndarray:
    """
    按开关提取特征：融合或单 backbone。
    use_tta=True 时使用多视图 TTA（旋转/亮度变体按子集定制）。
    w_mega 仅在 use_fusion=True 时生效，控制 Mega/Miew 融合权重。
    """
    if use_fusion:
        if use_tta:
            print("    提取 MegaDescriptor 特征（多视图TTA）...")
            feat_mega = extract_features_tta_multiview(
                mega_model, dataset, size=384, dataset_name=dataset_name
            )
            print("    提取 MiewID 特征（多视图TTA）...")
            feat_miew = extract_features_tta_multiview(
                miew_model, dataset, size=512, dataset_name=dataset_name
            )
        else:
            print("    提取 MegaDescriptor 特征...")
            feat_mega = extract_features_single(mega_model, dataset, size=384, tta=False, dataset_name=dataset_name)
            print("    提取 MiewID 特征...")
            feat_miew = extract_features_single(miew_model, dataset, size=512, tta=False, dataset_name=dataset_name)
        return fuse_with_weight(feat_mega, feat_miew, w_mega)

    # 单模型分支
    size = 384 if single_backbone == 'mega' else 512
    model = mega_model if single_backbone == 'mega' else miew_model
    if use_tta:
        return extract_features_tta_multiview(model, dataset, size, dataset_name=dataset_name)
    if single_backbone == 'mega':
        return extract_features_single(mega_model, dataset, size=384, tta=False, dataset_name=dataset_name)
    if single_backbone == 'miew':
        return extract_features_single(miew_model, dataset, size=512, tta=False, dataset_name=dataset_name)
    raise ValueError(f"single_backbone 必须是 'mega' 或 'miew'，收到: {single_backbone!r}")


def feature_route(dataset_name: str) -> tuple[bool, str]:
    """按 FEATURE_STRATEGY 返回 (是否 Mega‖Miew 融合, 单骨干名)。融合时第二个值为占位。"""
    if FEATURE_STRATEGY == 'global_mega':
        return False, 'mega'
    if FEATURE_STRATEGY == 'global_miew':
        return False, 'miew'
    if FEATURE_STRATEGY == 'per_dataset_baseline':
        return False, backbone_for_dataset_baseline(dataset_name)
    if FEATURE_STRATEGY == 'fusion':
        return True, 'mega'
    if FEATURE_STRATEGY == 'fusion_salamander_only':
        if dataset_name in ('SalamanderID2025', 'TexasHornedLizards'):
            return True, 'mega'
        return False, backbone_for_dataset_baseline(dataset_name)
    raise ValueError(f'feature_route: 未知 FEATURE_STRATEGY={FEATURE_STRATEGY!r}')


# ============================================================
# Fix 5: 余弦相似度计算（健壮的 dict 解包）
# ============================================================
def compute_similarity(features: np.ndarray) -> np.ndarray:
    """计算余弦相似度矩阵，返回 ndarray。"""
    matcher = CosineSimilarity()
    sim = matcher(features, features)
    if isinstance(sim, dict):
        sim = next(iter(sim.values()))
    return np.array(sim, dtype=np.float64)


# ============================================================
# 参数搜索
# ============================================================
def search_best_dbscan_eps(similarity, true_labels, eps_grid=None):
    if eps_grid is None:
        eps_grid = np.arange(0.10, 0.55, 0.02)
    best_eps, best_ari = 0.3, -1.0
    for eps in eps_grid:
        pred = run_DBSCAN(similarity, eps)
        ari = adjusted_rand_score(true_labels, pred)
        if ari > best_ari:
            best_ari, best_eps = ari, eps
    return best_eps, best_ari


def search_best_hdbscan_mcs(similarity, true_labels, mcs_grid=None):
    if not HAS_HDBSCAN:
        return 2, -1.0
    if mcs_grid is None:
        mcs_grid = [2, 3, 4, 5, 6, 8]
    best_mcs, best_ari = 2, -1.0
    for mcs in mcs_grid:
        pred = run_HDBSCAN(similarity, min_cluster_size=mcs)
        ari = adjusted_rand_score(true_labels, pred)
        if ari > best_ari:
            best_ari, best_mcs = ari, mcs
    return best_mcs, best_ari


def search_best_agglo_threshold(
    similarity,
    true_labels,
    threshold_grid=None,
    dataset_name: str | None = None,
):
    """在 train 相似度上搜索 Agglomerative 的 distance_threshold，使 ARI 最大。
    
    ★ 传入 dataset_name 后，run_Agglomerative 内部会自动应用簇数上限约束，
    使网格搜索与最终推理行为完全一致。
    """
    if threshold_grid is None:
        threshold_grid = np.arange(0.06, 0.55, 0.02)
    best_t, best_ari = 0.35, -1.0
    for t in threshold_grid:
        pred = run_Agglomerative(similarity, t, dataset_name=dataset_name)
        ari = adjusted_rand_score(true_labels, pred)
        if ari > best_ari:
            best_ari, best_t = ari, t
    return best_t, best_ari


# ============================================================
# 主流程
# ============================================================
if __name__ == '__main__':
    _allowed_feat = ('fusion', 'fusion_salamander_only', 'global_mega', 'global_miew', 'per_dataset_baseline')
    if FEATURE_STRATEGY not in _allowed_feat:
        raise ValueError(f"FEATURE_STRATEGY 必须是 {_allowed_feat} 之一，收到: {FEATURE_STRATEGY!r}")

    per_dataset_bb = FEATURE_STRATEGY == 'per_dataset_baseline'
    fusion_salamander_only = FEATURE_STRATEGY == 'fusion_salamander_only'
    if FEATURE_STRATEGY == 'global_mega':
        global_bb = 'mega'
    elif FEATURE_STRATEGY == 'global_miew':
        global_bb = 'miew'
    else:
        global_bb = 'mega'  # fusion / per_dataset 下由分支决定，此处仅占位

    print("\n========== 当前实验开关 ==========")
    print(f"  FEATURE_STRATEGY={FEATURE_STRATEGY!r}")
    if per_dataset_bb:
        print("    （按子集 baseline 路由：非融合，但子集间骨干不同）")
    elif fusion_salamander_only:
        print("    （SalamanderID2025 + TexasHornedLizards 双模型融合；海龟→Mega，猞猁→Miew）")
    print(f"  USE_TTA={USE_TTA}")
    print("  图像预处理:")
    print(f"    USE_MEGADETECTOR_CROP={USE_MEGADETECTOR_CROP}")
    print(f"    USE_CLAHE_LIGHTING={USE_CLAHE_LIGHTING}")
    print(f"    USE_TEXTURE_SHARPEN={USE_TEXTURE_SHARPEN}")
    print(f"    USE_LETTERBOX_TO_INPUT={USE_LETTERBOX_TO_INPUT}")
    print(f"    USE_LYNX_SPECIAL_PREPROCESS={USE_LYNX_SPECIAL_PREPROCESS}")
    if USE_LYNX_SPECIAL_PREPROCESS:
        print(
            f"      LYNX_STAGE2_ENABLE={LYNX_STAGE2_ENABLE}, "
            f"LYNX_STAGE3_ENABLE={LYNX_STAGE3_ENABLE}"
        )
    print(f"    USE_SALAMANDER_SPECIAL_PREPROCESS={USE_SALAMANDER_SPECIAL_PREPROCESS}")
    if USE_SALAMANDER_SPECIAL_PREPROCESS:
        print(
            f"      SAL_GLARE_ENABLE={SAL_GLARE_ENABLE}, SAL_ALIGN_ENABLE={SAL_ALIGN_ENABLE}, "
            f"SAL_TEXTURE_ENABLE={SAL_TEXTURE_ENABLE}, SAL_USE_EDGE_LETTERBOX={SAL_USE_EDGE_LETTERBOX}"
        )
        print(
            f"      SAL_STAGE3_DENOISE={SAL_STAGE3_DENOISE!r}, "
            f"SAL_AB_STATS_PATH={SAL_AB_STATS_PATH or '(未设置，使用 SAL_AB_REF_MEAN_A/B)'}"
        )
        if not USE_LETTERBOX_TO_INPUT:
            print(
                "      [提示] 五、无变形缩放入模：请设 USE_LETTERBOX_TO_INPUT=True，"
                "并保留 SAL_USE_EDGE_LETTERBOX=True 以使用边缘复制填充。"
            )
    if USE_MEGADETECTOR_CROP:
        _yw = _yolo_crop_weight_path()
        _eng = _resolved_crop_engine()
        print(f"    YOLO_CROP_ENGINE={YOLO_CROP_ENGINE!r} => 实际后端 {_eng!r}")
        print(f"    YOLO_CROP_WEIGHT={_yw!r} YOLO_CROP_CONF={YOLO_CROP_CONF}")
        print(f"    USE_SAM_REFINER={USE_SAM_REFINER} SAM_MODEL_WEIGHT={SAM_MODEL_WEIGHT!r}")
        if _eng == 'ultralytics':
            print(f"    YOLO_CROP_CLASSES={YOLO_CROP_CLASSES}")
        else:
            print(f"    （yolov5_hub + MegaDetector 权重：仅使用 animal 类 id=0）")
            _yr = _resolve_yolov5_repo_dir()
            if _yr:
                print(f"    本地 YOLOv5: {_yr}")
            else:
                print(f"    未找到本地 yolov5，将尝试在线 hub（离线请放到 {os.path.join(root, 'yolov5')}）")
        _looks_local = os.path.isabs(_yw) or (os.path.dirname(_yw) not in ('', '.'))
        if _looks_local and not os.path.isfile(_yw):
            raise FileNotFoundError(f"USE_MEGADETECTOR_CROP=True 但本地权重不存在: {_yw}")
        if _eng == 'yolov5_hub' and not os.path.isfile(_yw):
            raise FileNotFoundError(
                f"yolov5_hub 需要权重文件在磁盘上: {_yw}（相对名请放在脚本目录 {root}）"
            )
    print(f"  GRID_SEARCH_ON_TRAIN={GRID_SEARCH_ON_TRAIN}")
    print(f"  SEARCH_SALAMANDER_WEIGHT={SEARCH_SALAMANDER_WEIGHT}, SALAMANDER_MEGA_WEIGHT={SALAMANDER_MEGA_WEIGHT}")
    print(f"  GRID_SEARCH_AGGLOMERATIVE={GRID_SEARCH_AGGLOMERATIVE}, "
          f"AGGLOMERATIVE_LINKAGE={AGGLOMERATIVE_LINKAGE!r}, "
          f"FIXED_AGGLOMERATIVE_DISTANCE_THRESHOLD={FIXED_AGGLOMERATIVE_DISTANCE_THRESHOLD}")
    print(f"  AGGLO_ENABLE_N_CLUSTERS_CAP={AGGLO_ENABLE_N_CLUSTERS_CAP}, "
          f"AGGLO_MAX_CLUSTERS_RATIO={AGGLO_MAX_CLUSTERS_RATIO}, "
          f"AGGLO_MIN_CLUSTERS={AGGLO_MIN_CLUSTERS}")
    print(f"  AUTO_PICK_DBSCAN_OR_HDBSCAN={AUTO_PICK_DBSCAN_OR_HDBSCAN}, "
          f"FORCED_CLUSTER_METHOD={FORCED_CLUSTER_METHOD!r}")
    _allowed_forced = ('dbscan', 'hdbscan', 'agglomerative')
    if not AUTO_PICK_DBSCAN_OR_HDBSCAN and FORCED_CLUSTER_METHOD not in _allowed_forced:
        raise ValueError(
            f"FORCED_CLUSTER_METHOD 必须为 {_allowed_forced} 之一，收到: {FORCED_CLUSTER_METHOD!r}"
        )
    if FORCED_CLUSTER_METHOD == 'hdbscan' and not HAS_HDBSCAN:
        raise RuntimeError("FORCED_CLUSTER_METHOD='hdbscan' 需要安装 hdbscan：pip install hdbscan")


    # ---- 1. 加载数据集 ----
    print("加载数据集...")
    dataset_full = AnimalCLEF2026(
        root,
        transform=None,
        load_label=True,
        factorize_label=True,
        check_files=False,
    )

    all_names_raw = dataset_full.df['dataset'].unique().tolist()
    train_datasets, test_datasets = {}, {}

    for name in all_names_raw:
        sub = dataset_full.get_subset(dataset_full.df['dataset'] == name)
        train_mask = sub.df['split'] == 'train'
        test_mask  = sub.df['split'] == 'test'
        if train_mask.sum() > 0:
            train_datasets[name] = sub.get_subset(train_mask)
        test_datasets[name] = sub.get_subset(test_mask)

    if RUN_SUBSETS_ONLY is not None:
        _unknown = set(RUN_SUBSETS_ONLY) - set(all_names_raw)
        if _unknown:
            raise ValueError(
                f"RUN_SUBSETS_ONLY 含无效名称 {_unknown}，metadata 中有: {all_names_raw}"
            )
        all_names = [n for n in RUN_SUBSETS_ONLY if n in all_names_raw]
        print(
            f"RUN_SUBSETS_ONLY={list(RUN_SUBSETS_ONLY)} → 本次只处理 {len(all_names)} 个子集: {all_names}"
        )
    else:
        all_names = list(all_names_raw)

    print(f"共参与流程 {len(all_names)} 个子数据集: {all_names}")
    for name in all_names:
        n_train = len(train_datasets[name]) if name in train_datasets else 0
        n_test  = len(test_datasets[name])
        print(f"  {name}: train={n_train}, test={n_test}")

    # ---- 2. 加载模型 ----
    print(f"\n========== 加载模型（device={device}）==========")

    mega_model = None
    miew_model = None
    need_mega = FEATURE_STRATEGY in ('fusion', 'fusion_salamander_only', 'per_dataset_baseline', 'global_mega')
    need_miew = FEATURE_STRATEGY in ('fusion', 'fusion_salamander_only', 'per_dataset_baseline', 'global_miew')
    if need_mega:
        print("  加载 MegaDescriptor-L-384...")
        mega_model = load_mega_model(local_mega_path, device)
    if need_miew:
        print("  加载 MiewID-msv3...")
        miew_model = load_miew_model(local_miew_path, device)

    if ENABLE_FINETUNING:
        _ft = FINETUNE_DATASETS
        if isinstance(_ft, str):
            _ft = (_ft,)
        _targets = tuple(dict.fromkeys(_ft))
        print(f"\n  微调计划：FINETUNE_DATASETS={list(_targets)}")
        for ds_name in _targets:
            if ds_name not in all_names:
                print(f"  [跳过] {ds_name!r} 不在本次 RUN_SUBSETS_ONLY / 运行范围内")
                continue
            if ds_name not in train_datasets:
                print(f"  [跳过] {ds_name!r} 无 train split")
                continue
            bb = finetune_target_backbone(ds_name)
            df_full = train_datasets[ds_name].df
            train_df, val_df = split_by_identity(df_full, train_ratio=FINETUNE_TRAIN_RATIO)
            print(
                f"\n  —— 微调子集 {ds_name!r} → 骨干 {bb} ——\n"
                f"     训练个体 {train_df['identity'].nunique()}, 图像 {len(train_df)}"
                f"（留出验证个体 {val_df['identity'].nunique()}, 图像 {len(val_df)}）"
            )
            save_path = finetune_checkpoint_path(ds_name, bb)
            if bb == 'mega':
                if mega_model is None:
                    print(f"  [跳过] {ds_name} 需要 Mega，但未加载 mega_model（检查 FEATURE_STRATEGY）")
                    continue
                mega_model, fp = finetune_on_dataset(
                    ds_name, train_df, mega_model, device,
                    input_size=384, save_path=save_path,
                )
                print(f"     checkpoint: {fp}")
            else:
                if miew_model is None:
                    print(f"  [跳过] {ds_name} 需要 Miew，但未加载 miew_model（检查 FEATURE_STRATEGY）")
                    continue
                miew_model, fp = finetune_on_dataset(
                    ds_name, train_df, miew_model, device,
                    input_size=512, save_path=save_path,
                )
                print(f"     checkpoint: {fp}")

    # ---- 3. 特征提取 ----
    print("\n========== 特征提取 ==========")
    train_features: dict[str, np.ndarray] = {}
    test_features:  dict[str, np.ndarray] = {}
    # 额外缓存融合前特征（用于权重搜索）
    train_mega_features: dict[str, np.ndarray] = {}
    train_miew_features: dict[str, np.ndarray] = {}
    test_mega_features: dict[str, np.ndarray] = {}
    test_miew_features: dict[str, np.ndarray] = {}

    if FEATURE_STRATEGY == 'fusion':
        feat_desc = '双模型融合（每子集 Mega‖Miew）'
    elif fusion_salamander_only:
        feat_desc = 'SalamanderID2025 + TexasHornedLizards Mega‖Miew；海龟→Mega，猞猁→Miew'
    elif per_dataset_bb:
        feat_desc = '按子集 baseline 路由（非融合）'
    else:
        feat_desc = f'全局单骨干 ({global_bb})'

    for name in all_names:
        use_fusion_here, bb = feature_route(name)
        if FEATURE_STRATEGY in ('global_mega', 'global_miew'):
            line_bb = feat_desc
        elif use_fusion_here and FEATURE_STRATEGY == 'fusion':
            line_bb = feat_desc
        elif use_fusion_here:
            line_bb = f'{name}→Mega‖Miew 融合'
        else:
            line_bb = f'{name}→{bb}'
        print(f"\n[{name}] 提取特征 ({line_bb}, TTA={'开' if USE_TTA else '关'})...")

        # 融合子集：单独缓存两模型特征，供权重搜索使用
        if (
            should_search_fusion_weight(name, use_fusion_here)
            and name in train_datasets
        ):
            print(f"  -- train split ({len(train_datasets[name])} 张，分别提取 Mega/Miew 用于权重搜索）--")
            if USE_TTA:
                train_mega_features[name] = extract_features_tta_multiview(
                    mega_model, train_datasets[name], 384, dataset_name=name
                )
                train_miew_features[name] = extract_features_tta_multiview(
                    miew_model, train_datasets[name], 512, dataset_name=name
                )
            else:
                train_mega_features[name] = extract_features_single(
                    mega_model, train_datasets[name], size=384, tta=False, dataset_name=name
                )
                train_miew_features[name] = extract_features_single(
                    miew_model, train_datasets[name], size=512, tta=False, dataset_name=name
                )
            # train_features 暂用等权融合占位，Step 4 会用独立特征重算
            train_features[name] = fuse_with_weight(
                train_mega_features[name], train_miew_features[name], 1.0
            )
            # test 特征暂缓，Step 4 搜完权重后再提取（此处先提取以保证流程不中断；Step 5 会覆盖）
            print(f"  -- test split ({len(test_datasets[name])} 张，暂用等权融合占位）--")
            if USE_TTA:
                _te_mega = extract_features_tta_multiview(
                    mega_model, test_datasets[name], 384, dataset_name=name
                )
                _te_miew = extract_features_tta_multiview(
                    miew_model, test_datasets[name], 512, dataset_name=name
                )
            else:
                _te_mega = extract_features_single(
                    mega_model, test_datasets[name], size=384, tta=False, dataset_name=name
                )
                _te_miew = extract_features_single(
                    miew_model, test_datasets[name], size=512, tta=False, dataset_name=name
                )
            # 缓存 test 的两模型特征，Step 4 之后用最优权重重新融合
            test_mega_features[name] = _te_mega
            test_miew_features[name] = _te_miew
            test_features[name] = fuse_with_weight(_te_mega, _te_miew, 1.0)
            continue

        if name in train_datasets:
            print(f"  -- train split ({len(train_datasets[name])} 张) --")
            train_features[name] = extract_features_pipeline(
                mega_model, miew_model, train_datasets[name],
                dataset_name=name,
                use_fusion=use_fusion_here,
                single_backbone=bb,
                use_tta=USE_TTA,
            )
        print(f"  -- test split ({len(test_datasets[name])} 张) --")
        test_features[name] = extract_features_pipeline(
            mega_model, miew_model, test_datasets[name],
            dataset_name=name,
            use_fusion=use_fusion_here,
            single_backbone=bb,
            use_tta=USE_TTA,
        )
        # 无 train 子集若要做先验引导的权重搜索，需要缓存融合前双模型特征
        if (
            name not in train_datasets
            and should_search_fusion_weight(name, use_fusion_here)
            and name in NO_TRAIN_WEIGHT_SEARCH_DATASETS
        ):
            print(f"  -- {name} 无 train，额外缓存 test 的 Mega/Miew 特征用于先验权重搜索 --")
            if USE_TTA:
                test_mega_features[name] = extract_features_tta_multiview(
                    mega_model, test_datasets[name], 384, dataset_name=name
                )
                test_miew_features[name] = extract_features_tta_multiview(
                    miew_model, test_datasets[name], 512, dataset_name=name
                )
            else:
                test_mega_features[name] = extract_features_single(
                    mega_model, test_datasets[name], size=384, tta=False, dataset_name=name
                )
                test_miew_features[name] = extract_features_single(
                    miew_model, test_datasets[name], size=512, tta=False, dataset_name=name
                )

    # ---- 4. 参数搜索 ----
    print("\n========== 参数搜索 ==========")
    best_params: dict[str, dict] = {}

    param_search_names = sorted(all_names, key=lambda x: (x != 'SalamanderID2025', x))
    for name in param_search_names:
        if name not in train_datasets:
            use_fusion_here, _ = feature_route(name)
            if (
                USE_SALAMANDER_PRIOR_FOR_NO_TRAIN_WEIGHT_SEARCH
                and name in NO_TRAIN_WEIGHT_SEARCH_DATASETS
                and should_search_fusion_weight(name, use_fusion_here)
                and name in test_mega_features
                and name in test_miew_features
            ):
                if 'SalamanderID2025' in best_params:
                    prior = best_params['SalamanderID2025']
                    prior_method = str(prior['method'])
                    prior_param = prior['param']
                else:
                    prior_method = str(SALAMANDER_PRIOR_METHOD)
                    prior_param = SALAMANDER_PRIOR_PARAM
                    print(
                        f"[{name}] 本次未拿到 Salamander 在线先验，"
                        f"改用配置先验 method={prior_method}, param={prior_param}"
                    )
                print(
                    f"[{name}] 无 train，继承 Salamander 先验: "
                    f"method={prior_method}, param={prior_param}"
                )
                best_w, best_unsup = search_weight_with_prior_cluster(
                    test_mega_features[name],
                    test_miew_features[name],
                    method=prior_method,
                    param=prior_param,
                    dataset_name=name,
                )
                _fusion_best_weights[name] = best_w
                test_features[name] = fuse_with_weight(
                    test_mega_features[name],
                    test_miew_features[name],
                    best_w,
                )
                best_params[name] = {'method': prior_method, 'param': prior_param}
                print(
                    f"[{name}] => 使用先验聚类参数并采用最优权重 "
                    f"w_mega={best_w:.2f} (silhouette={best_unsup:.4f})"
                )
                continue

            fallback_eps = eps_fallback.get(name, FIXED_DBSCAN_EPS)
            method = FORCED_CLUSTER_METHOD if not AUTO_PICK_DBSCAN_OR_HDBSCAN else 'dbscan'
            if method == 'agglomerative':
                thr0 = agglo_threshold_fallback.get(name, FIXED_AGGLOMERATIVE_DISTANCE_THRESHOLD)
                print(f"[{name}] 无 train split，使用 Agglomerative distance_threshold={thr0}")
                best_params[name] = {'method': 'agglomerative', 'param': thr0}
            elif method == 'hdbscan' and HAS_HDBSCAN:
                print(f"[{name}] 无 train split，使用 HDBSCAN mcs={FIXED_HDBSCAN_MIN_CLUSTER_SIZE}")
                best_params[name] = {'method': 'hdbscan', 'param': FIXED_HDBSCAN_MIN_CLUSTER_SIZE}
            else:
                if method == 'hdbscan' and not HAS_HDBSCAN:
                    print(f"[{name}] 无 train 且需 HDBSCAN 但未安装，回退 DBSCAN eps={fallback_eps}")
                else:
                    print(f"[{name}] 无 train split，使用回退参数 eps={fallback_eps}")
                best_params[name] = {'method': 'dbscan', 'param': fallback_eps}
            continue

        true_labels = np.array(train_datasets[name].df['identity'])

        # 融合权重联合搜索（只在融合路由且开关开启时执行）
        use_fusion_here, bb = feature_route(name)
        if (
            should_search_fusion_weight(name, use_fusion_here)
            and name in train_mega_features
            and name in train_miew_features
        ):
            print(f"\n[{name}] 联合搜索融合权重 + Agglomerative threshold...")
            best_w, best_thr, best_ari = search_fusion_weight(
                train_mega_features[name],
                train_miew_features[name],
                true_labels,
                dataset_name=name,
            )
            _fusion_best_weights[name] = best_w
            best_params[name] = {'method': 'agglomerative', 'param': best_thr}
            print(f"[{name}] => Agglomerative threshold={best_thr:.3f} (ARI={best_ari:.4f})")
            # 用最优权重重新融合 test 特征
            if name in test_mega_features and name in test_miew_features:
                print(f"[{name}] 用最优权重 w_mega={best_w:.2f} 重新融合 test 特征...")
                test_features[name] = fuse_with_weight(
                    test_mega_features[name],
                    test_miew_features[name],
                    best_w,
                )
            continue

        train_sim = compute_similarity(train_features[name])

        if not GRID_SEARCH_ON_TRAIN:
            eps_use = eps_fallback.get(name, FIXED_DBSCAN_EPS)
            mcs_use = FIXED_HDBSCAN_MIN_CLUSTER_SIZE
            thr_fix = FIXED_AGGLOMERATIVE_DISTANCE_THRESHOLD
            if not AUTO_PICK_DBSCAN_OR_HDBSCAN:
                if FORCED_CLUSTER_METHOD == 'hdbscan' and HAS_HDBSCAN:
                    best_params[name] = {'method': 'hdbscan', 'param': mcs_use}
                    print(f"[{name}] 未搜参，强制 HDBSCAN mcs={mcs_use}")
                elif FORCED_CLUSTER_METHOD == 'agglomerative':
                    best_params[name] = {'method': 'agglomerative', 'param': thr_fix}
                    print(f"[{name}] 未搜参，强制 Agglomerative distance_threshold={thr_fix}")
                else:
                    best_params[name] = {'method': 'dbscan', 'param': eps_use}
                    print(f"[{name}] 未搜参，强制 DBSCAN eps={eps_use}")
                continue
            candidates = [
                (
                    'dbscan',
                    eps_use,
                    adjusted_rand_score(true_labels, run_DBSCAN(train_sim, eps_use)),
                ),
            ]
            if HAS_HDBSCAN:
                candidates.append(
                    (
                        'hdbscan',
                        mcs_use,
                        adjusted_rand_score(
                            true_labels,
                            run_HDBSCAN(train_sim, min_cluster_size=mcs_use),
                        ),
                    )
                )
            if GRID_SEARCH_AGGLOMERATIVE:
                candidates.append(
                    (
                        'agglomerative',
                        thr_fix,
                        adjusted_rand_score(
                            true_labels,
                            run_Agglomerative(train_sim, thr_fix),
                        ),
                    )
                )
            best_method, best_param, best_ari = max(candidates, key=lambda x: x[2])
            best_params[name] = {'method': best_method, 'param': best_param}
            for m, par, ari in candidates:
                print(f"[{name}] 固定参数 {m} param={par}, ARI={ari:.4f}")
            print(f"[{name}] => 选用 {best_method} (ARI={best_ari:.4f})")
            continue

        best_eps, best_ari_db = search_best_dbscan_eps(train_sim, true_labels)
        print(f"[{name}] DBSCAN  最优 eps={best_eps:.2f}, ARI={best_ari_db:.4f}")

        best_mcs, best_ari_hdb = search_best_hdbscan_mcs(train_sim, true_labels)
        if HAS_HDBSCAN:
            print(f"[{name}] HDBSCAN 最优 mcs={best_mcs}, ARI={best_ari_hdb:.4f}")

        if GRID_SEARCH_AGGLOMERATIVE:
            best_thr, best_ari_agglo = search_best_agglo_threshold(train_sim, true_labels, dataset_name=name)
            print(
                f"[{name}] Agglomerative 最优 distance_threshold={best_thr:.3f}, "
                f"ARI={best_ari_agglo:.4f} (linkage={AGGLOMERATIVE_LINKAGE!r})"
            )
        else:
            best_thr, best_ari_agglo = FIXED_AGGLOMERATIVE_DISTANCE_THRESHOLD, float('-inf')

        if not AUTO_PICK_DBSCAN_OR_HDBSCAN:
            if FORCED_CLUSTER_METHOD == 'hdbscan':
                if not HAS_HDBSCAN:
                    best_params[name] = {'method': 'dbscan', 'param': best_eps}
                    print(f"[{name}] 强制 HDBSCAN 但未安装 => DBSCAN eps={best_eps}")
                else:
                    best_params[name] = {'method': 'hdbscan', 'param': best_mcs}
                    print(f"[{name}] => 强制 HDBSCAN mcs={best_mcs}")
            elif FORCED_CLUSTER_METHOD == 'agglomerative':
                thr_use = (
                    best_thr
                    if GRID_SEARCH_AGGLOMERATIVE
                    else FIXED_AGGLOMERATIVE_DISTANCE_THRESHOLD
                )
                best_params[name] = {'method': 'agglomerative', 'param': thr_use}
                print(f"[{name}] => 强制 Agglomerative distance_threshold={thr_use}")
            else:
                best_params[name] = {'method': 'dbscan', 'param': best_eps}
                print(f"[{name}] => 强制 DBSCAN eps={best_eps}")
        else:
            pick = [('dbscan', best_eps, best_ari_db)]
            if HAS_HDBSCAN:
                pick.append(('hdbscan', best_mcs, best_ari_hdb))
            if GRID_SEARCH_AGGLOMERATIVE:
                pick.append(('agglomerative', best_thr, best_ari_agglo))
            best_method, best_param, best_ari = max(pick, key=lambda x: x[2])
            best_params[name] = {'method': best_method, 'param': best_param}
            print(f"[{name}] => 选用 {best_method} (ARI={best_ari:.4f}, param={best_param})")

    # ---- 5. 测试集聚类 + 生成 submission ----
    print("\n========== 生成 submission.csv ==========")
    results = None

    for name in all_names:
        test_sim = compute_similarity(test_features[name])
        p = best_params[name]

        if p['method'] == 'hdbscan':
            clusters = run_HDBSCAN(test_sim, min_cluster_size=p['param'])
        elif p['method'] == 'agglomerative':
            clusters = run_Agglomerative(test_sim, p['param'], dataset_name=name)
        else:
            clusters = run_DBSCAN(test_sim, eps=p['param'])

        # Fix 6: 用 dataset.df['image_id'] 而非 dataset.metadata['image_id']
        image_ids = test_datasets[name].df['image_id'].values

        result = pd.DataFrame({
            'image_id': image_ids,
            'cluster':  [f'cluster_{name}_{c}' for c in clusters],
        })
        results = pd.concat([results, result], ignore_index=True) if results is not None else result

        n_clusters = len(np.unique(clusters))
        print(f"  {name}: {len(clusters)} 张，{n_clusters} 个 cluster，"
              f"方法={p['method']}，参数={p['param']}")

    results.to_csv('submission.csv', index=False)
    print(f"\n完成！submission.csv 已保存，共 {len(results)} 行。")