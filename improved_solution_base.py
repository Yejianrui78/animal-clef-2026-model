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
from sklearn.cluster import DBSCAN

try:
    from hdbscan import HDBSCAN
    HAS_HDBSCAN = True
except ImportError:
    HAS_HDBSCAN = False
    print("[警告] hdbscan 未安装，将只使用 DBSCAN。建议: pip install hdbscan")

from sklearn.metrics import adjusted_rand_score
from wildlife_datasets.datasets import AnimalCLEF2026
from wildlife_tools.features import DeepFeatures
from wildlife_tools.similarity import CosineSimilarity


# ============================================================
# 路径和设备配置（与 baseline.py 一致：数据在脚本同级目录）
# ============================================================
root = os.path.dirname(os.path.abspath(__file__))
device = 'cuda' if torch.cuda.is_available() else 'cpu'
batch_size = 32

# 可选：将权重放在项目内 models/ 下则优先离线加载；不存在时 Mega 回退 hf-hub，Miew 回退 Hub
local_mega_path = os.path.join(root, 'models', 'BVRA', 'MegaDescriptor-L-384')
local_miew_path = os.path.join(root, 'models', 'conservationxlabs', 'miewid-msv3')

# 无 train split 时的回退 eps
eps_fallback = {
    'LynxID2025':         0.3,
    'SalamanderID2025':   0.2,
    'SeaTurtleID2022':    0.4,
    'TexasHornedLizards': 0.24,
}

# ============================================================
# 实验开关（按需修改；便于逐项对比融合 / TTA / 标定 / 聚类器）
# ============================================================
# 特征策略（四选一，含义不同，勿与「单模型」混为一谈）：
#   'fusion' — 每个子集均 Mega‖Miew 拼接后再聚类
#   'global_mega' / 'global_miew' — 四个子集共用同一骨干（真·全网单模型）
#   'per_dataset_baseline' — 与 baseline.py 相同：Salamander/海龟→Mega，猞猁/角蜥→Miew；
#       每个子集仍是「一个」骨干，但子集之间不同，且不是拼接融合
FEATURE_STRATEGY = 'per_dataset_baseline'  # 'fusion' | 'global_mega' | 'global_miew' | 'per_dataset_baseline'

# True: 原图 + 水平翻转特征取平均；False: 仅原图
USE_TTA = True

# ---------- 图像预处理（送入 MegaDescriptor / MiewID 之前；可逐项关闭做对比）----------
# MegaDetector 动物框裁剪（md_v*.pt 等；经 YOLOv5 torch.hub 或见 YOLO_CROP_ENGINE）；关闭则整图
USE_MEGADETECTOR_CROP = True
# LAB 空间 CLAHE，统一亮度/对比度
USE_CLAHE_LIGHTING = False
# Unsharp mask 式纹理锐化
USE_TEXTURE_SHARPEN = False
# True: 等比缩放 + 居中填充至 size×size；False: 直接 Resize 到 size×size（可能变形，与 baseline 接近）
USE_LETTERBOX_TO_INPUT = False

# 裁剪检测权重：MegaDetector md_v1000*.pt 为 YOLOv5，不能用 ultralytics.YOLO()，见下 YOLO_CROP_ENGINE
YOLO_CROP_WEIGHT = 'md_v1000.0.0-redwood.pt'
YOLO_CROP_CONF = 0.25
YOLO_CROP_MARGIN = 0.05
# 'ultralytics' — 仅 ultralytics.YOLO（yolo11n.pt 等）；'yolov5_hub' — torch.hub 加载 YOLOv5 自定义权重
# 'auto' — 若文件名含 md_v/megadetector 等则用 yolov5_hub，否则 ultralytics
YOLO_CROP_ENGINE = 'auto'
# yolov5_hub 时需本地 ultralytics/yolov5 仓库（含 hubconf.py）；留空则用环境变量 YOLOV5_REPO 或 <root>/yolov5
YOLOV5_LOCAL_REPO = ''
# 仅 YOLO_CROP_ENGINE='ultralytics' 且 COCO 预训练权重时生效：只在这些类别里取最高置信度框（动物检测常用）
# COCO: 14=bird … 23=giraffe。设为 None 则不按类别过滤（可能框到人/车）。MegaDetector(md_v*) 走 yolov5_hub，固定 animal=0，忽略本项
YOLO_CROP_CLASSES = [14, 15, 16, 17, 18, 19, 20, 21, 22, 23]

# True: 在有 train 的子集上，用真实 identity 做网格搜索（eps / min_cluster_size）
# False: 不搜参，使用 FIXED_DBSCAN_EPS 与 FIXED_HDBSCAN_MCS（无 train 时仍用 eps_fallback）
GRID_SEARCH_ON_TRAIN = True
# True: 在 train 上比较 DBSCAN 与 HDBSCAN 的 ARI，选较高者（需 GRID_SEARCH_ON_TRAIN=True 才有意义）
# False: 强制使用 FORCED_CLUSTER_METHOD（'dbscan' | 'hdbscan'）
AUTO_PICK_DBSCAN_OR_HDBSCAN = True
FORCED_CLUSTER_METHOD = 'hdbscan'  # 'dbscan' | 'hdbscan'

# GRID_SEARCH_ON_TRAIN=False 时的固定聚类参数（有 train 时用；无 train 的 DBSCAN 仍优先 eps_fallback[name]）
FIXED_DBSCAN_EPS = 0.3
FIXED_HDBSCAN_MIN_CLUSTER_SIZE = 2


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
    """将余弦相似度矩阵转换为 [0,1] 距离矩阵（供 precomputed metric 使用）。"""
    sim = np.maximum(similarity, 0)
    max_val = sim.max()
    if max_val == 0:
        return np.ones_like(sim, dtype=np.float64)
    dist = (max_val - sim) / max_val
    np.fill_diagonal(dist, 0.0)
    return dist.astype(np.float64)


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


# ===================== YOLO 裁剪（Ultralytics 或 YOLOv5 torch.hub）+ 图像预处理 =====================
_yolo_crop_model = None
_yolov5_hub_crop_model = None
_crop_engine_resolved = None


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
        det = _get_ultralytics_crop_model()
        dev = 0 if device == 'cuda' else 'cpu'
        # 直接传 PIL RGB，由 Ultralytics 内部做与训练一致的预处理；避免 ndarray 的 BGR/RGB 约定歧义
        kw = dict(source=rgb_pil, conf=YOLO_CROP_CONF, verbose=False, device=dev)
        if YOLO_CROP_CLASSES is not None and len(YOLO_CROP_CLASSES) > 0:
            kw['classes'] = list(YOLO_CROP_CLASSES)
        res = det.predict(**kw)[0]
        boxes = res.boxes
        if boxes is None or len(boxes) == 0:
            return pil_img
        confs = boxes.conf.cpu().numpy()
        xyxy = boxes.xyxy.cpu().numpy()
        best_i = int(np.argmax(confs))
        x1, y1, x2, y2 = map(float, xyxy[best_i])

    x1i = max(0, int(x1 - w * margin))
    y1i = max(0, int(y1 - h * margin))
    x2i = min(w, int(x2 + w * margin))
    y2i = min(h, int(y2 + h * margin))
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


def animal_preprocess_pipeline(img: Image.Image) -> Image.Image:
    """
    像素级预处理：可选 MegaDetector 动物框裁剪（YOLO_CROP_WEIGHT，见 YOLO_CROP_ENGINE）、CLAHE、锐化。
    几何缩放（letterbox 或拉伸）在 transform 里单独做，便于与骨干输入尺寸对齐。
    """
    out = img.convert('RGB')
    if USE_MEGADETECTOR_CROP:
        out = _yolo_crop(out)
    if USE_CLAHE_LIGHTING:
        out = _clahe_lighting(out)
    if USE_TEXTURE_SHARPEN:
        out = _texture_sharpen(out)
    return out


def resize_letterbox_square(img: Image.Image, target_size: int) -> Image.Image:
    """等比缩放，居中填充灰色到 target_size（与 MegaDescriptor 384 / MiewID 512 等输入一致）。"""
    img = img.convert('RGB')
    w, h = img.size
    if w <= 0 or h <= 0:
        return Image.new('RGB', (target_size, target_size), (128, 128, 128))
    scale = target_size / max(w, h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    img = img.resize((new_w, new_h), Image.BICUBIC)
    canvas = Image.new('RGB', (target_size, target_size), (128, 128, 128))
    canvas.paste(img, ((target_size - new_w) // 2, (target_size - new_h) // 2))
    return canvas


def _build_geom_transform(target_size: int):
    if USE_LETTERBOX_TO_INPUT:
        return T.Lambda(lambda im: resize_letterbox_square(im, target_size))
    return T.Resize((target_size, target_size), interpolation=T.InterpolationMode.BICUBIC)


def _maybe_preprocess_lambda():
    if USE_MEGADETECTOR_CROP or USE_CLAHE_LIGHTING or USE_TEXTURE_SHARPEN:
        return T.Lambda(animal_preprocess_pipeline)
    return T.Lambda(lambda im: im.convert('RGB'))

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


def extract_features_single(model, dataset, size: int, tta: bool = True) -> np.ndarray:
    """单模型特征提取，支持 TTA（水平翻转均值）。返回 L2 归一化特征。"""
    pre = _maybe_preprocess_lambda()
    geom = _build_geom_transform(size)
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


def extract_fused_features(mega_model, miew_model, dataset, tta: bool = True) -> np.ndarray:
    """双模型融合特征：MegaDescriptor(384) + MiewID(512)。拼接后整体 L2 归一化。"""
    print("    提取 MegaDescriptor 特征...")
    feat_mega = extract_features_single(mega_model, dataset, size=384, tta=tta)

    print("    提取 MiewID 特征...")
    feat_miew = extract_features_single(miew_model, dataset, size=512, tta=tta)

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
    use_fusion: bool,
    single_backbone: str,
    use_tta: bool,
) -> np.ndarray:
    """按开关提取特征：融合或单 backbone。"""
    if use_fusion:
        return extract_fused_features(mega_model, miew_model, dataset, tta=use_tta)
    if single_backbone == 'mega':
        return extract_features_single(mega_model, dataset, size=384, tta=use_tta)
    if single_backbone == 'miew':
        return extract_features_single(miew_model, dataset, size=512, tta=use_tta)
    raise ValueError(f"single_backbone 必须是 'mega' 或 'miew'，收到: {single_backbone!r}")


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


# ============================================================
# 主流程
# ============================================================
if __name__ == '__main__':
    _allowed_feat = ('fusion', 'global_mega', 'global_miew', 'per_dataset_baseline')
    if FEATURE_STRATEGY not in _allowed_feat:
        raise ValueError(f"FEATURE_STRATEGY 必须是 {_allowed_feat} 之一，收到: {FEATURE_STRATEGY!r}")

    use_fusion = FEATURE_STRATEGY == 'fusion'
    per_dataset_bb = FEATURE_STRATEGY == 'per_dataset_baseline'
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
    print(f"  USE_TTA={USE_TTA}")
    print("  图像预处理:")
    print(f"    USE_MEGADETECTOR_CROP={USE_MEGADETECTOR_CROP}")
    print(f"    USE_CLAHE_LIGHTING={USE_CLAHE_LIGHTING}")
    print(f"    USE_TEXTURE_SHARPEN={USE_TEXTURE_SHARPEN}")
    print(f"    USE_LETTERBOX_TO_INPUT={USE_LETTERBOX_TO_INPUT}")
    if USE_MEGADETECTOR_CROP:
        _yw = _yolo_crop_weight_path()
        _eng = _resolved_crop_engine()
        print(f"    YOLO_CROP_ENGINE={YOLO_CROP_ENGINE!r} => 实际后端 {_eng!r}")
        print(f"    YOLO_CROP_WEIGHT={_yw!r} YOLO_CROP_CONF={YOLO_CROP_CONF}")
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
    print(f"  AUTO_PICK_DBSCAN_OR_HDBSCAN={AUTO_PICK_DBSCAN_OR_HDBSCAN}, "
          f"FORCED_CLUSTER_METHOD={FORCED_CLUSTER_METHOD!r}")
    if not AUTO_PICK_DBSCAN_OR_HDBSCAN and FORCED_CLUSTER_METHOD not in ('dbscan', 'hdbscan'):
        raise ValueError("FORCED_CLUSTER_METHOD 必须为 'dbscan' 或 'hdbscan'")
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

    all_names = dataset_full.df['dataset'].unique().tolist()
    train_datasets, test_datasets = {}, {}

    for name in all_names:
        sub = dataset_full.get_subset(dataset_full.df['dataset'] == name)
        train_mask = sub.df['split'] == 'train'
        test_mask  = sub.df['split'] == 'test'
        if train_mask.sum() > 0:
            train_datasets[name] = sub.get_subset(train_mask)
        test_datasets[name] = sub.get_subset(test_mask)

    print(f"共 {len(all_names)} 个子数据集: {all_names}")
    for name in all_names:
        n_train = len(train_datasets[name]) if name in train_datasets else 0
        n_test  = len(test_datasets[name])
        print(f"  {name}: train={n_train}, test={n_test}")

    # ---- 2. 加载模型 ----
    print(f"\n========== 加载模型（device={device}）==========")

    mega_model = None
    miew_model = None
    need_mega = FEATURE_STRATEGY in ('fusion', 'per_dataset_baseline', 'global_mega')
    need_miew = FEATURE_STRATEGY in ('fusion', 'per_dataset_baseline', 'global_miew')
    if need_mega:
        print("  加载 MegaDescriptor-L-384...")
        mega_model = load_mega_model(local_mega_path, device)
    if need_miew:
        print("  加载 MiewID-msv3...")
        miew_model = load_miew_model(local_miew_path, device)

    # ---- 3. 特征提取 ----
    print("\n========== 特征提取 ==========")
    train_features: dict[str, np.ndarray] = {}
    test_features:  dict[str, np.ndarray] = {}

    if use_fusion:
        feat_desc = '双模型融合（每子集 Mega‖Miew）'
    elif per_dataset_bb:
        feat_desc = '按子集 baseline 路由（非融合）'
    else:
        feat_desc = f'全局单骨干 ({global_bb})'

    for name in all_names:
        bb = backbone_for_dataset_baseline(name) if per_dataset_bb else global_bb
        if per_dataset_bb:
            line_bb = f'{name}→{bb}'
        else:
            line_bb = feat_desc
        print(f"\n[{name}] 提取特征 ({line_bb}, TTA={'开' if USE_TTA else '关'})...")
        if name in train_datasets:
            print(f"  -- train split ({len(train_datasets[name])} 张) --")
            train_features[name] = extract_features_pipeline(
                mega_model, miew_model, train_datasets[name],
                use_fusion=use_fusion,
                single_backbone=bb,
                use_tta=USE_TTA,
            )
        print(f"  -- test split ({len(test_datasets[name])} 张) --")
        test_features[name] = extract_features_pipeline(
            mega_model, miew_model, test_datasets[name],
            use_fusion=use_fusion,
            single_backbone=bb,
            use_tta=USE_TTA,
        )

    # ---- 4. 参数搜索 ----
    print("\n========== 参数搜索 ==========")
    best_params: dict[str, dict] = {}

    for name in all_names:
        if name not in train_datasets:
            fallback_eps = eps_fallback.get(name, FIXED_DBSCAN_EPS)
            method = FORCED_CLUSTER_METHOD if not AUTO_PICK_DBSCAN_OR_HDBSCAN else 'dbscan'
            if method == 'hdbscan' and HAS_HDBSCAN:
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
        train_sim   = compute_similarity(train_features[name])

        if not GRID_SEARCH_ON_TRAIN:
            eps_use = eps_fallback.get(name, FIXED_DBSCAN_EPS)
            mcs_use = FIXED_HDBSCAN_MIN_CLUSTER_SIZE
            if not AUTO_PICK_DBSCAN_OR_HDBSCAN:
                if FORCED_CLUSTER_METHOD == 'hdbscan' and HAS_HDBSCAN:
                    best_params[name] = {'method': 'hdbscan', 'param': mcs_use}
                    print(f"[{name}] 未搜参，强制 HDBSCAN mcs={mcs_use}")
                else:
                    best_params[name] = {'method': 'dbscan', 'param': eps_use}
                    print(f"[{name}] 未搜参，强制 DBSCAN eps={eps_use}")
                continue
            if AUTO_PICK_DBSCAN_OR_HDBSCAN and HAS_HDBSCAN:
                pred_d = run_DBSCAN(train_sim, eps_use)
                pred_h = run_HDBSCAN(train_sim, min_cluster_size=mcs_use)
                best_ari_db = adjusted_rand_score(true_labels, pred_d)
                best_ari_hdb = adjusted_rand_score(true_labels, pred_h)
                print(f"[{name}] 固定参数 DBSCAN eps={eps_use:.2f}, ARI={best_ari_db:.4f}")
                print(f"[{name}] 固定参数 HDBSCAN mcs={mcs_use}, ARI={best_ari_hdb:.4f}")
                if best_ari_hdb > best_ari_db:
                    best_params[name] = {'method': 'hdbscan', 'param': mcs_use}
                    print(f"[{name}] => 选用 HDBSCAN (ARI 更高)")
                else:
                    best_params[name] = {'method': 'dbscan', 'param': eps_use}
                    print(f"[{name}] => 选用 DBSCAN (ARI 更高)")
            else:
                best_params[name] = {'method': 'dbscan', 'param': eps_use}
                print(f"[{name}] 未搜参，DBSCAN eps={eps_use}（未比较 HDBSCAN）")
            continue

        best_eps, best_ari_db = search_best_dbscan_eps(train_sim, true_labels)
        print(f"[{name}] DBSCAN  最优 eps={best_eps:.2f}, ARI={best_ari_db:.4f}")

        best_mcs, best_ari_hdb = search_best_hdbscan_mcs(train_sim, true_labels)
        if HAS_HDBSCAN:
            print(f"[{name}] HDBSCAN 最优 mcs={best_mcs}, ARI={best_ari_hdb:.4f}")

        if not AUTO_PICK_DBSCAN_OR_HDBSCAN:
            if FORCED_CLUSTER_METHOD == 'hdbscan':
                if not HAS_HDBSCAN:
                    best_params[name] = {'method': 'dbscan', 'param': best_eps}
                    print(f"[{name}] 强制 HDBSCAN 但未安装 => DBSCAN eps={best_eps}")
                else:
                    best_params[name] = {'method': 'hdbscan', 'param': best_mcs}
                    print(f"[{name}] => 强制 HDBSCAN mcs={best_mcs}")
            else:
                best_params[name] = {'method': 'dbscan', 'param': best_eps}
                print(f"[{name}] => 强制 DBSCAN eps={best_eps}")
        elif HAS_HDBSCAN and best_ari_hdb > best_ari_db:
            best_params[name] = {'method': 'hdbscan', 'param': best_mcs}
            print(f"[{name}] => 选用 HDBSCAN (ARI 更高)")
        else:
            best_params[name] = {'method': 'dbscan', 'param': best_eps}
            print(f"[{name}] => 选用 DBSCAN (ARI 更高)")

    # ---- 5. 测试集聚类 + 生成 submission ----
    print("\n========== 生成 submission.csv ==========")
    results = None

    for name in all_names:
        test_sim = compute_similarity(test_features[name])
        p = best_params[name]

        if p['method'] == 'hdbscan':
            clusters = run_HDBSCAN(test_sim, min_cluster_size=p['param'])
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