#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Wafer Defect Detection System

지능형 2단계 웨이퍼 결함 검출 시스템
- 1단계: 분류 모델을 통한 기본 예측
- 2단계: ROI 기반 객체 검출로 성능 향상

구성
1. ConvNeXtV2 분류 모델
2. Grad-CAM 기반 ROI 추출
3. YOLO 객체 검출 기반 보강
4. 성능 분석 / ROI 패턴 학습 / 저장 / 로드

주의
- gradcam_utils.py 의 아래 구현이 필요함
  * GradCAMAnalyzer
  * ROIExtractor
  * GradCAMConfig
"""

from __future__ import annotations

from typing import Union, Dict, List, Any, Optional, Tuple
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
import json
import logging
import shutil

import torch
import torch.nn as nn
import numpy as np
import timm
from ultralytics import YOLO
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import (
    precision_recall_fscore_support,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from PIL import Image
from skimage.transform import resize

from gradcam_utils import GradCAMAnalyzer, ROIExtractor, GradCAMConfig


logger = logging.getLogger(__name__)


# =========================
# Exceptions / Dataclasses
# =========================

class WaferDetectorError(Exception):
    """웨이퍼 검출기 관련 예외"""
    pass


@dataclass
class PerformanceMetrics:
    """성능 메트릭"""
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    class_precision: np.ndarray = field(default_factory=lambda: np.array([]))
    class_recall: np.ndarray = field(default_factory=lambda: np.array([]))
    class_f1: np.ndarray = field(default_factory=lambda: np.array([]))
    confusion_matrix: np.ndarray = field(default_factory=lambda: np.array([]))


@dataclass
class PredictionResult:
    """예측 결과"""
    image_path: str
    predicted_class: str
    confidence: float
    method: str  # 'classification_only' or 'roi_enhanced'

    detected_object: Optional[str] = None
    object_counts: Optional[Dict[str, int]] = None
    roi_coords: Optional[Tuple[float, float, float, float]] = None

    stage1_class: Optional[str] = None
    stage1_confidence: Optional[float] = None
    stage2_object: Optional[str] = None
    stage2_counts: Optional[Dict[str, int]] = None
    final_prediction: Optional[str] = None
    roi_enhancement_attempted: Optional[bool] = None
    roi_enhancement_success: Optional[bool] = None


# =========================
# Model Manager
# =========================

class ModelManager:
    """모델 관리자"""

    def __init__(self, device: torch.device):
        self.device = device
        self.classification_model: Optional[nn.Module] = None
        self.yolo_model: Optional[YOLO] = None
        self.gradcam_analyzer: Optional[GradCAMAnalyzer] = None

    def load_classification_model(self, model_path: Union[str, Path]) -> None:
        """분류 모델 로드"""
        model_path = Path(model_path)
        if not model_path.exists():
            raise WaferDetectorError(f"Classification model file not found: {model_path}")

        checkpoint = torch.load(model_path, map_location="cpu")

        if not isinstance(checkpoint, dict):
            raise WaferDetectorError("Checkpoint is not a dict")

        # state_dict 위치 탐색
        if "state_dict" in checkpoint and isinstance(checkpoint["state_dict"], dict):
            checkpoint = checkpoint["state_dict"]
        elif "model_state_dict" in checkpoint and isinstance(checkpoint["model_state_dict"], dict):
            checkpoint = checkpoint["model_state_dict"]

        state_dict = {}
        for key, value in checkpoint.items():
            if key.startswith("model."):
                new_key = key[len("model."):]
            elif key.startswith("module."):
                new_key = key[len("module."):]
            else:
                new_key = key
            state_dict[new_key] = value

        num_classes = self._infer_num_classes_from_state_dict(state_dict)

        self.classification_model = timm.create_model(
            "convnextv2_base.fcmae_ft_in22k_in1k_384",
            pretrained=False,
            num_classes=num_classes
        )

        missing, unexpected = self.classification_model.load_state_dict(state_dict, strict=False)
        if missing:
            logger.warning(f"⚠️ Missing keys when loading classification model: {missing[:20]}")
        if unexpected:
            logger.warning(f"⚠️ Unexpected keys when loading classification model: {unexpected[:20]}")

        # head.fc.weight / bias 정도가 아닌데 너무 많이 비면 사실상 이상
        too_many_missing = [k for k in missing if not k.startswith("head.")]
        if too_many_missing:
            logger.warning("⚠️ Non-head missing keys detected. Check checkpoint/model compatibility carefully.")

        self.classification_model.to(self.device)
        self.classification_model.eval()

        target_layer = self._find_gradcam_target_layer()
        self.gradcam_analyzer = GradCAMAnalyzer(
            self.classification_model,
            GradCAMConfig(target_layer_name=target_layer)
        )
        logger.info(f"✅ Classification model loaded: {model_path}")

    def _infer_num_classes_from_state_dict(self, state_dict: Dict[str, torch.Tensor]) -> int:
        candidate_keys = [
            "head.fc.weight",
            "head.weight",
            "classifier.weight",
        ]
        for k in candidate_keys:
            if k in state_dict and hasattr(state_dict[k], "shape") and len(state_dict[k].shape) >= 1:
                return int(state_dict[k].shape[0])
        raise WaferDetectorError(
            "Could not infer num_classes from checkpoint. "
            f"Available keys example: {list(state_dict.keys())[:20]}"
        )

    def _find_gradcam_target_layer(self) -> str:
        """ConvNeXtV2 모델에서 GradCAM 타겟 레이어 탐색"""
        if self.classification_model is None:
            raise WaferDetectorError("Classification model not loaded")

        all_layers = [name for name, _ in self.classification_model.named_modules()]
        logger.info(f"🔍 Available layers in model: {len(all_layers)}")

        candidates = [
            "stages.3.blocks.2.norm",
            "stages.3.blocks.1.norm",
            "stages.3.blocks.0.norm",
            "stages.3.norm",
            "stages.2.norm",
            "norm",
            "head_norm",
            "features.norm",
        ]
        for c in candidates:
            if c in all_layers:
                logger.info(f"✅ Found GradCAM target: {c}")
                return c

        norm_layers = [n for n in all_layers if "norm" in n and "head" not in n]
        if norm_layers:
            logger.info(f"🎯 Using fallback norm layer: {norm_layers[-1]}")
            return norm_layers[-1]

        stage_layers = [n for n in all_layers if "stages." in n]
        if stage_layers:
            logger.info(f"🎯 Using fallback stage layer: {stage_layers[-1]}")
            return stage_layers[-1]

        raise WaferDetectorError("No suitable GradCAM target layer found")

    def load_yolo_model(self, model_path: Union[str, Path]) -> None:
        """YOLO 모델 로드"""
        try:
            model_path = Path(model_path)
            if not model_path.exists():
                raise FileNotFoundError(f"YOLO model file not found: {model_path}")

            self.yolo_model = YOLO(str(model_path))
            logger.info(f"✅ YOLO model loaded: {model_path}")
        except Exception as e:
            raise WaferDetectorError(f"Failed to load YOLO model: {e}")


# =========================
# Dataset Manager
# =========================

class DatasetManager:
    """데이터셋 관리자"""

    def __init__(self, transform: transforms.Compose):
        self.transform = transform
        self.classes: List[str] = []
        self.batch_size: int = 32
        self.num_workers: int = 8
        self.pin_memory: bool = True

    def load_classes_from_dataset(self, dataset_root: Union[str, Path]) -> List[str]:
        """데이터셋에서 클래스 로드"""
        try:
            dataset_root = Path(dataset_root)
            if not dataset_root.exists():
                raise FileNotFoundError(f"Dataset root not found: {dataset_root}")

            subdirs = [d.name for d in dataset_root.iterdir() if d.is_dir()]
            if not subdirs:
                raise ValueError(f"No class directories found in {dataset_root}")

            self.classes = sorted(subdirs)
            logger.info(f"📋 Loaded classes: {self.classes}")
            return self.classes

        except Exception as e:
            raise WaferDetectorError(f"Failed to load classes: {e}")

    def create_dataloader(
        self,
        dataset_root: Union[str, Path],
        batch_size: Optional[int] = None,
        num_workers: Optional[int] = None,
        pin_memory: Optional[bool] = None,
        shuffle: bool = False
    ) -> DataLoader:
        """데이터로더 생성"""
        try:
            dataset = datasets.ImageFolder(
                root=str(dataset_root),
                transform=self.transform
            )

            if batch_size is None:
                batch_size = self.batch_size
            if num_workers is None:
                num_workers = self.num_workers
            if pin_memory is None:
                pin_memory = self.pin_memory

            dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
                pin_memory=bool(pin_memory) and torch.cuda.is_available(),
                drop_last=False,
                persistent_workers=(num_workers > 0),
            )
            logger.info(f"📊 Dataloader created: {len(dataset)} samples, batch_size={batch_size}")
            return dataloader

        except Exception as e:
            raise WaferDetectorError(f"Failed to create dataloader: {e}")


# =========================
# Performance Analyzer
# =========================

class PerformanceAnalyzer:
    """성능 분석기"""

    def __init__(self, device: torch.device, output_dir: Union[str, Path]):
        self.device = device
        self.output_dir = Path(output_dir)

    def _save_sample_results(self, sample_results: List[Dict], filename: str) -> Path:
        """샘플 결과 저장"""
        samples_dir = self.output_dir / "performance_analysis_samples"
        samples_dir.mkdir(parents=True, exist_ok=True)
        file_path = samples_dir / filename

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(sample_results, f, indent=2, ensure_ascii=False)

        logger.info(f"💾 Sample results saved: {file_path}")
        return file_path

    def analyze_model_performance(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        classes: List[str]
    ) -> PerformanceMetrics:
        """분류 모델 성능 분석"""
        try:
            if model is None:
                raise WaferDetectorError("Classification model is not loaded")

            model.eval()
            all_preds, all_labels, all_conf = [], [], []
            sample_results = []
            total = 0

            logger.info("🔍 Start performance analysis")
            with torch.no_grad():
                for b_idx, (imgs, lbls) in enumerate(dataloader):
                    imgs = imgs.to(self.device, non_blocking=True)
                    lbls = lbls.to(self.device, non_blocking=True)

                    outs = model(imgs)
                    probs = torch.softmax(outs, dim=1)
                    confs, preds = torch.max(probs, dim=1)

                    for i in range(len(imgs)):
                        t = lbls[i].item()
                        p = preds[i].item()
                        c = confs[i].item()
                        sample = {
                            "sample_id": total + i + 1,
                            "batch_id": b_idx + 1,
                            "true_class": classes[t] if t < len(classes) else f"Unknown({t})",
                            "predicted_class": classes[p] if p < len(classes) else f"Unknown({p})",
                            "confidence": c,
                            "correct": (t == p),
                            "method": "classification_only",
                            "detected_object": None,
                            "object_counts": {}
                        }
                        sample_results.append(sample)

                    all_preds.extend(preds.cpu().numpy().tolist())
                    all_labels.extend(lbls.cpu().numpy().tolist())
                    all_conf.extend(confs.cpu().numpy().tolist())
                    total += len(imgs)

            logger.info(f"✅ Processed {total} samples")

            self._save_sample_results(sample_results, "performance_analysis_samples.json")
            metrics = self._calculate_metrics(all_preds, all_labels, classes)

            logger.info("📈 Performance analysis completed")
            return metrics

        except Exception as e:
            logger.error(f"❌ Performance analysis failed: {e}")
            raise WaferDetectorError(f"Performance analysis failed: {e}")

    def _calculate_metrics(
        self,
        predictions: List[int],
        labels: List[int],
        classes: List[str]
    ) -> PerformanceMetrics:
        """메트릭 계산"""
        if len(predictions) == 0 or len(labels) == 0:
            return PerformanceMetrics()

        precision = precision_score(labels, predictions, average="weighted", zero_division=0)
        recall = recall_score(labels, predictions, average="weighted", zero_division=0)
        f1 = f1_score(labels, predictions, average="weighted", zero_division=0)

        cls_prec, cls_rec, cls_f1, _ = precision_recall_fscore_support(
            labels, predictions, average=None, zero_division=0
        )
        cm = confusion_matrix(labels, predictions)

        return PerformanceMetrics(
            precision=precision,
            recall=recall,
            f1=f1,
            class_precision=cls_prec,
            class_recall=cls_rec,
            class_f1=cls_f1,
            confusion_matrix=cm
        )

    def identify_difficult_classes(
        self,
        metrics: PerformanceMetrics,
        classes: List[str],
        threshold: float = 0.8
    ) -> List[str]:
        """어려운 클래스 식별"""
        difficult = []
        for i, (pr, rc) in enumerate(zip(metrics.class_precision, metrics.class_recall)):
            if pr < threshold or rc < threshold:
                difficult.append(classes[i])

        logger.info(f"🎯 Difficult classes: {difficult}")
        return difficult


# =========================
# Main Detector
# =========================

class WaferDetector:
    """웨이퍼 결함 검출기 메인 클래스"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        output_dir = self.config["OUTPUT_DIR"]
        self.performance_analyzer = PerformanceAnalyzer(self.device, output_dir)

        self.model_manager = ModelManager(self.device)
        self.dataset_manager = DatasetManager(self._create_transform())

        self.dataset_manager.batch_size = self.config.get("BATCH_SIZE", 32)
        self.dataset_manager.num_workers = self.config.get("NUM_WORKERS", 8)
        self.dataset_manager.pin_memory = self.config.get("PIN_MEMORY", True)

        self.roi_extractor = ROIExtractor()

        self.classes: List[str] = []
        self.difficult_classes: List[str] = []
        self.class_object_mapping: Dict[str, str] = {}
        self.roi_patterns: Dict[str, Dict[str, Any]] = {}
        self.metrics: PerformanceMetrics = PerformanceMetrics()

    def _create_transform(self) -> transforms.Compose:
        size = self.config["CLASSIFICATION_SIZE"]
        return transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225]
            )
        ])

    # -------------------------
    # Model / Class load
    # -------------------------

    def load_models(self, model_path: Union[str, Path], yolo_path: Union[str, Path]) -> None:
        """분류 & YOLO 모델 로드"""
        self.model_manager.load_classification_model(model_path)
        self.model_manager.load_yolo_model(yolo_path)

    def load_classes(self, dataset_root: Union[str, Path]) -> None:
        """클래스 로드 및 모델 출력 크기 검증"""
        try:
            self.classes = self.dataset_manager.load_classes_from_dataset(dataset_root)
            logger.info(f"📋 Loaded {len(self.classes)} classes from dataset")

            if self.model_manager.classification_model is not None:
                num_out = self._get_model_num_outputs(self.model_manager.classification_model)
                logger.info(f"🔍 Model output: {num_out}, Dataset classes: {len(self.classes)}")
                if num_out is not None and num_out != len(self.classes):
                    logger.warning(
                        "⚠️ Class count mismatch between model output and dataset classes. "
                        "You should verify class ordering and training label mapping."
                    )

        except Exception as e:
            raise WaferDetectorError(f"Failed to load classes: {e}")

    def _get_model_num_outputs(self, model: nn.Module) -> Optional[int]:
        if hasattr(model, "head") and hasattr(model.head, "fc") and hasattr(model.head.fc, "out_features"):
            return int(model.head.fc.out_features)
        if hasattr(model, "head") and hasattr(model.head, "out_features"):
            return int(model.head.out_features)
        return None

    # -------------------------
    # Performance analysis
    # -------------------------

    def analyze_performance(self, dataset_root: Union[str, Path]) -> Dict[str, Any]:
        """분류 모델 성능 분석"""
        try:
            logger.info("🔍 Starting performance analysis...")
            self.load_classes(dataset_root)

            dataloader = self.dataset_manager.create_dataloader(
                dataset_root,
                batch_size=self.dataset_manager.batch_size,
                num_workers=self.dataset_manager.num_workers,
                pin_memory=self.dataset_manager.pin_memory,
                shuffle=False
            )

            self.metrics = self.performance_analyzer.analyze_model_performance(
                self.model_manager.classification_model,
                dataloader,
                self.classes
            )

            self.difficult_classes = self.performance_analyzer.identify_difficult_classes(
                self.metrics,
                self.classes,
                self.config["PRECISION_THRESHOLD"]
            )

            logger.info(f"📊 Overall F1: {self.metrics.f1:.4f}, Difficult: {self.difficult_classes}")

            return {
                cls: float(f1) for cls, f1 in zip(self.classes, self.metrics.class_f1)
            }

        except Exception as e:
            raise WaferDetectorError(f"Performance analysis failed: {e}")

    # -------------------------
    # ROI pattern learning
    # -------------------------

    def learn_roi_patterns(
        self,
        dataset_root: Optional[Union[str, Path]] = None,
        max_samples: int = 5
    ) -> None:
        """
        모든 클래스에 대한 ROI 패턴 동적 학습
        - 클래스별 이미지 샘플에서 GradCAM -> ROIExtractor -> 평균 ROI 패턴 계산
        """
        root = Path(dataset_root or self.config.get("DATASET_ROOT", ""))
        if not root.exists():
            raise WaferDetectorError(f"Invalid DATASET_ROOT for ROI learning: {root}")

        if not self.classes:
            self.load_classes(root)

        if self.model_manager.gradcam_analyzer is None:
            raise WaferDetectorError("GradCAM analyzer not initialized. Load classification model first.")

        logger.info("🧠 Learning ROI patterns for all classes (dynamic)...")

        for cls_idx, class_name in enumerate(self.classes):
            class_dir = root / class_name
            if not class_dir.exists():
                logger.warning(f"⚠️ Class folder not found, skipping: {class_dir}")
                continue

            imgs = self._collect_images(class_dir)
            if not imgs:
                logger.warning(f"⚠️ No images for class '{class_name}'")
                continue

            samples = imgs[:max_samples]
            roi_coords_list: List[Tuple[float, float, float, float]] = []

            for img_path in samples:
                try:
                    img = Image.open(img_path).convert("RGB")
                    tensor = self.dataset_manager.transform(img).unsqueeze(0).to(self.device)

                    heatmap_raw = self.model_manager.gradcam_analyzer.generate_gradcam(
                        tensor, cls_idx
                    )
                    coords = self.roi_extractor.extract_roi_from_heatmap(heatmap_raw)
                    coords = self._sanitize_roi_coords(coords)
                    roi_coords_list.append(coords)

                except Exception as e:
                    logger.warning(f"⚠️ Failed GradCAM on {img_path.name}: {e}")

            if not roi_coords_list:
                logger.warning(f"⚠️ No ROI coords for '{class_name}', fallback full-image")
                avg = (0.0, 0.0, 1.0, 1.0)
            else:
                arr = np.array(roi_coords_list, dtype=np.float32)
                avg = tuple(arr.mean(axis=0).tolist())
                avg = self._sanitize_roi_coords(avg)

            self.roi_patterns[class_name] = {
                "avg_roi_x1": float(avg[0]),
                "avg_roi_y1": float(avg[1]),
                "avg_roi_x2": float(avg[2]),
                "avg_roi_y2": float(avg[3]),
                "confidence_threshold": float(self.config["CONFIDENCE_THRESHOLD"]),
                "pattern_type": "difficult" if class_name in self.difficult_classes else "normal"
            }

            logger.info(
                f"   {class_name}: avg_roi=({avg[0]:.3f}, {avg[1]:.3f}, {avg[2]:.3f}, {avg[3]:.3f})"
            )

        logger.info(f"✅ Learned ROI patterns for {len(self.roi_patterns)} classes")

    # -------------------------
    # Mapping
    # -------------------------

    def create_mapping(self) -> None:
        """모든 클래스에 대한 클래스-객체 매핑 생성"""
        try:
            logger.info("🔗 Creating class-object mapping for all classes...")

            if self.model_manager.yolo_model is None:
                raise WaferDetectorError("YOLO model not loaded")

            yolo_classes = self.model_manager.yolo_model.names
            logger.info(f"   Available YOLO classes: {list(yolo_classes.values())}")

            manual_map = self.config.get("CLASS_OBJECT_MAPPING", {})

            for class_name in self.classes:
                if class_name in manual_map:
                    self.class_object_mapping[class_name] = manual_map[class_name]
                    continue

                mapped = self._find_best_yolo_match(class_name, yolo_classes)
                if mapped:
                    self.class_object_mapping[class_name] = mapped
                else:
                    self.class_object_mapping[class_name] = "general_defect"

            logger.info(f"✅ Created mappings for {len(self.class_object_mapping)} classes")

        except Exception as e:
            raise WaferDetectorError(f"Mapping creation failed: {e}")

    def _find_best_yolo_match(self, class_name: str, yolo_classes: Dict[int, str]) -> Optional[str]:
        """YOLO 클래스와 매칭"""
        lname = class_name.lower().replace("_", "").replace("-", "").replace(" ", "")
        best = None

        for _, yname in yolo_classes.items():
            y = yname.lower().replace("_", "").replace("-", "").replace(" ", "")
            if lname == y:
                return yname
            if lname in y or y in lname:
                best = yname

        return best

    # -------------------------
    # Prediction
    # -------------------------

    def predict_image(self, image_path: Union[str, Path]) -> PredictionResult:
        """단일 이미지 예측 (분류 + ROI enhancement)"""
        try:
            if self.model_manager.classification_model is None:
                raise WaferDetectorError("Classification model not loaded")
            if not self.classes:
                raise WaferDetectorError("Classes not loaded")

            img_p = Path(image_path)
            if not img_p.exists():
                raise FileNotFoundError(f"Image not found: {image_path}")

            image = Image.open(img_p).convert("RGB")
            tensor = self.dataset_manager.transform(image).unsqueeze(0).to(self.device)

            # Stage 1: Classification
            with torch.no_grad():
                out = self.model_manager.classification_model(tensor)
                probs = torch.softmax(out, dim=1)
                conf, idx = torch.max(probs, dim=1)
                pred_class = self.classes[idx.item()]
                conf_val = float(conf.item())

            result_info: Dict[str, Any] = {
                "image_path": str(image_path),
                "predicted_class": pred_class,
                "confidence": conf_val,
                "method": "classification_only",
                "stage1_class": pred_class,
                "stage1_confidence": conf_val,
                "final_prediction": pred_class,
                "roi_enhancement_attempted": False,
                "roi_enhancement_success": False,
            }

            # Stage 2: ROI enhancement
            enhanced = self._enhance_with_roi(
                image_path=str(image_path),
                image=image,
                tensor=tensor,
                pred_class=pred_class,
                pred_conf=conf_val,
                pred_class_idx=idx.item(),
            )

            if enhanced is not None:
                result_info.update({
                    "method": "roi_enhanced",
                    "detected_object": enhanced["detected_object"],
                    "object_counts": enhanced["object_counts"],
                    "roi_coords": enhanced["roi_coords"],
                    "stage2_object": enhanced["stage2_object"],
                    "stage2_counts": enhanced["stage2_counts"],
                    "final_prediction": enhanced["final_prediction"],
                    "roi_enhancement_attempted": True,
                    "roi_enhancement_success": True
                })
            else:
                result_info["roi_enhancement_attempted"] = True
                result_info["roi_enhancement_success"] = False

            return PredictionResult(**{
                k: v for k, v in result_info.items()
                if k in PredictionResult.__annotations__
            })

        except Exception as e:
            raise WaferDetectorError(f"Image prediction failed: {e}")

    def _enhance_with_roi(
        self,
        image_path: str,
        image: Image.Image,
        tensor: torch.Tensor,
        pred_class: str,
        pred_conf: float,
        pred_class_idx: int
    ) -> Optional[Dict[str, Any]]:
        """
        ROI enhancement 로직
        1) difficult class 이거나 confidence가 낮으면 수행
        2) GradCAM으로 ROI 추출
        3) ROI crop + YOLO inference
        4) 매핑된 객체가 검출되면 roi_enhanced 결과 반환
        """
        try:
            if self.model_manager.yolo_model is None:
                logger.info("YOLO model not loaded. Skip ROI enhancement.")
                return None

            if self.model_manager.gradcam_analyzer is None:
                logger.info("GradCAM analyzer not loaded. Skip ROI enhancement.")
                return None

            # ROI enhancement 필요 여부 판단
            do_enhancement = False
            conf_thresh = float(self.config.get("CONFIDENCE_THRESHOLD", 0.8))
            difficult_only = bool(self.config.get("ROI_FOR_DIFFICULT_OR_LOWCONF_ONLY", True))

            if difficult_only:
                if pred_class in self.difficult_classes or pred_conf < conf_thresh:
                    do_enhancement = True
            else:
                do_enhancement = True

            if not do_enhancement:
                return None

            # Grad-CAM
            heatmap_raw = self.model_manager.gradcam_analyzer.generate_gradcam(tensor, pred_class_idx)

            if pred_class in self.roi_patterns:
                # 학습된 평균 ROI와 동적 ROI를 섞고 싶으면 여기서 혼합 가능
                dyn_coords = self.roi_extractor.extract_roi_from_heatmap(heatmap_raw)
                dyn_coords = self._sanitize_roi_coords(dyn_coords)

                pat = self.roi_patterns[pred_class]
                avg_coords = (
                    pat["avg_roi_x1"],
                    pat["avg_roi_y1"],
                    pat["avg_roi_x2"],
                    pat["avg_roi_y2"],
                )
                avg_coords = self._sanitize_roi_coords(avg_coords)

                alpha = float(self.config.get("ROI_PATTERN_BLEND_ALPHA", 0.0))
                roi_coords = self._blend_roi_coords(dyn_coords, avg_coords, alpha=alpha)
            else:
                roi_coords = self.roi_extractor.extract_roi_from_heatmap(heatmap_raw)
                roi_coords = self._sanitize_roi_coords(roi_coords)

            crop_img = self._crop_image_by_normalized_roi(image, roi_coords)

            # YOLO inference
            yolo_img = np.array(crop_img)
            yolo_results = self.model_manager.yolo_model.predict(
                source=yolo_img,
                conf=float(self.config.get("YOLO_CONF_THRESHOLD", 0.25)),
                verbose=False,
                device=0 if self.device.type == "cuda" else "cpu"
            )

            detected_object, object_counts = self._parse_yolo_results(yolo_results)
            if not object_counts:
                return None

            mapped_object = self.class_object_mapping.get(pred_class, None)
            final_prediction = pred_class

            # 매핑된 객체가 있으면 그걸 기준으로 성공 판정
            if mapped_object is not None and mapped_object in object_counts:
                final_prediction = pred_class
                stage2_object = mapped_object
            else:
                # fallback: 가장 많이 검출된 객체 기준
                stage2_object = detected_object

            return {
                "detected_object": detected_object,
                "object_counts": object_counts,
                "roi_coords": roi_coords,
                "stage2_object": stage2_object,
                "stage2_counts": object_counts,
                "final_prediction": final_prediction,
            }

        except Exception as e:
            logger.warning(f"⚠️ ROI enhancement failed: {e}")
            return None

    def _parse_yolo_results(self, yolo_results: Any) -> Tuple[Optional[str], Dict[str, int]]:
        """YOLO 결과 파싱"""
        counts: Dict[str, int] = {}

        try:
            if yolo_results is None or len(yolo_results) == 0:
                return None, counts

            for res in yolo_results:
                if not hasattr(res, "boxes") or res.boxes is None:
                    continue

                if res.boxes.cls is None or len(res.boxes.cls) == 0:
                    continue

                cls_ids = res.boxes.cls.detach().cpu().numpy().astype(int).tolist()
                names = res.names if hasattr(res, "names") else {}

                for cid in cls_ids:
                    cname = names.get(cid, str(cid))
                    counts[cname] = counts.get(cname, 0) + 1

            if not counts:
                return None, counts

            detected_object = max(counts.items(), key=lambda x: x[1])[0]
            return detected_object, counts

        except Exception as e:
            logger.warning(f"⚠️ Failed to parse YOLO results: {e}")
            return None, {}

    # -------------------------
    # Visualization
    # -------------------------

    def create_class_roi_visualizations(
        self,
        dataset_root: Union[str, Path],
        max_samples_per_class: int = 3
    ) -> None:
        """클래스별 대표 이미지 ROI 시각화"""
        if self.model_manager.gradcam_analyzer is None:
            raise WaferDetectorError("GradCAM analyzer not initialized")

        root = Path(dataset_root)
        out_dir = Path(self.config["OUTPUT_DIR"]) / "roi_visualizations"
        out_dir.mkdir(parents=True, exist_ok=True)

        for cls_idx, class_name in enumerate(self.classes):
            class_dir = root / class_name
            if not class_dir.exists():
                continue

            imgs = self._collect_images(class_dir)
            if not imgs:
                continue

            samples = imgs[:max_samples_per_class]
            for idx, img_p in enumerate(samples):
                try:
                    img = Image.open(img_p).convert("RGB")
                    tensor = self.dataset_manager.transform(img).unsqueeze(0).to(self.device)

                    heat_raw = self.model_manager.gradcam_analyzer.generate_gradcam(tensor, cls_idx)
                    coords = self.roi_extractor.extract_roi_from_heatmap(heat_raw)
                    coords = self._sanitize_roi_coords(coords)

                    mn, mx = float(heat_raw.min()), float(heat_raw.max())
                    heat_vis = (heat_raw - mn) / (mx - mn + 1e-8)

                    save_path = out_dir / f"{class_name}_{idx+1:02d}_roi.png"
                    self._create_roi_visualization(img, heat_vis, coords, class_name, save_path)

                except Exception as e:
                    logger.warning(f"Failed viz for {class_name}/{img_p.name}: {e}")

    def _create_roi_visualization(
        self,
        image: Image.Image,
        heatmap: np.ndarray,
        roi_coords: Tuple[float, float, float, float],
        class_name: str,
        save_path: Path
    ) -> None:
        """단일 이미지 ROI 시각화"""
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        from matplotlib.colors import LinearSegmentedColormap

        raw_min, raw_max = float(heatmap.min()), float(heatmap.max())
        cmap = LinearSegmentedColormap.from_list("gradcam", ["blue", "cyan", "yellow", "red"], N=256)

        fig, ax = plt.subplots(1, 3, figsize=(15, 5))

        ax[0].imshow(image)
        ax[0].set_title(f"Original\n{class_name}")
        ax[0].axis("off")

        hm_rs = resize(heatmap, (image.height, image.width), preserve_range=True)
        ax[1].imshow(image, alpha=0.7)
        ax[1].imshow(hm_rs, cmap=cmap, alpha=0.6)
        ax[1].set_title(f"GradCAM\nRawMin={raw_min:.3f}, RawMax={raw_max:.3f}")
        ax[1].axis("off")

        ax[2].imshow(image)
        x1, y1, x2, y2 = roi_coords
        w, h = image.width, image.height
        rect = patches.Rectangle(
            (x1 * w, y1 * h),
            (x2 - x1) * w,
            (y2 - y1) * h,
            linewidth=3,
            edgecolor="red",
            facecolor="none"
        )
        ax[2].add_patch(rect)
        area = (x2 - x1) * (y2 - y1)
        ax[2].set_title(f"ROI Size: {int((x2-x1)*w)}×{int((y2-y1)*h)} ({area:.1%})")
        ax[2].axis("off")

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    # -------------------------
    # ROI Enhanced Performance
    # -------------------------

    def analyze_roi_enhanced_performance(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """ROI Enhanced 성능 분석"""
        roi_res = [r for r in results if r.get("method") == "roi_enhanced"]
        cls_res = [r for r in results if r.get("method") == "classification_only"]

        comp = {
            "total": len(results),
            "roi_count": len(roi_res),
            "cls_count": len(cls_res),
            "roi_success": (
                sum(r.get("roi_enhancement_success", False) for r in roi_res) / max(1, len(roi_res))
            )
        }

        has_labels = any("true_class" in r for r in results)
        metrics_roi = self._calculate_method_performance(roi_res, method_name="roi_enhanced") if has_labels else {}
        metrics_cls = self._calculate_method_performance(cls_res, method_name="classification_only") if has_labels else {}

        dist: Dict[str, int] = {}
        for r in roi_res:
            final = r.get("final_prediction", r.get("predicted_class"))
            if final is None:
                continue
            dist[final] = dist.get(final, 0) + 1

        return {
            "comparison": comp,
            "roi_metrics": metrics_roi,
            "cls_metrics": metrics_cls,
            "distribution": dist
        }

    def _calculate_method_performance(
        self,
        results: List[Dict[str, Any]],
        method_name: str
    ) -> Dict[str, Any]:
        """method별 성능 계산"""
        if not results:
            return {
                "method": method_name,
                "count": 0,
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "confusion_matrix": [],
                "labels": []
            }

        y_true: List[str] = []
        y_pred: List[str] = []

        for r in results:
            if "true_class" not in r:
                continue
            y_true.append(r["true_class"])
            pred = r.get("final_prediction") or r.get("predicted_class")
            y_pred.append(pred)

        if not y_true:
            return {
                "method": method_name,
                "count": len(results),
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "confusion_matrix": [],
                "labels": []
            }

        labels = sorted(list(set(y_true) | set(y_pred)))
        precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
        recall = recall_score(y_true, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
        cm = confusion_matrix(y_true, y_pred, labels=labels)

        return {
            "method": method_name,
            "count": len(y_true),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "confusion_matrix": cm.tolist(),
            "labels": labels
        }

    # -------------------------
    # Save / Load
    # -------------------------

    def save_results(self, output_dir: Union[str, Path]) -> Path:
        """모든 결과 저장"""
        out = Path(output_dir) / f"results_{datetime.now().strftime('%y%m%d_%H%M')}"
        out.mkdir(parents=True, exist_ok=True)

        # 1) performance samples 복사
        src_perf = Path(self.config["OUTPUT_DIR"]) / "performance_analysis_samples"
        dst_perf = out / "performance_analysis_samples"
        if src_perf.exists():
            shutil.copytree(src_perf, dst_perf, dirs_exist_ok=True)
        else:
            dst_perf.mkdir(exist_ok=True)

        # 2) ROI patterns
        (out / "roi_patterns.json").write_text(
            json.dumps(self.roi_patterns, indent=2, ensure_ascii=False),
            encoding="utf-8"
        )

        # 3) class-object mapping
        mapping = {
            "difficult_classes": self.difficult_classes,
            "class_object_mapping": self.class_object_mapping
        }
        (out / "class_mapping.json").write_text(
            json.dumps(mapping, indent=2, ensure_ascii=False),
            encoding="utf-8"
        )

        # 4) stats
        (out / "stats.json").write_text(
            json.dumps(self.get_stats(), indent=2, ensure_ascii=False),
            encoding="utf-8"
        )

        # 5) performance metrics
        self._save_performance_metrics(out)

        logger.info(f"💾 All results saved under {out}")
        return out

    def _save_performance_metrics(self, out_path: Path) -> None:
        """성능 메트릭 저장"""
        try:
            txt = out_path / "performance_metrics.txt"
            with txt.open("w", encoding="utf-8") as f:
                f.write("=" * 80 + "\n")
                f.write("WAFER DEFECT DETECTION - PERFORMANCE METRICS\n")
                f.write("=" * 80 + "\n\n")
                f.write("OVERALL:\n")
                f.write(f"  Precision: {self.metrics.precision:.4f}\n")
                f.write(f"  Recall:    {self.metrics.recall:.4f}\n")
                f.write(f"  F1-Score:  {self.metrics.f1:.4f}\n\n")

                if self.classes and self.metrics.class_f1.size:
                    f.write("CLASS-WISE:\n")
                    f.write(f"{'Class':<25}{'Prec':<10}{'Rec':<10}{'F1':<10}\n")
                    f.write("-" * 60 + "\n")
                    for i, cls in enumerate(self.classes):
                        pr = self.metrics.class_precision[i]
                        rc = self.metrics.class_recall[i]
                        f1 = self.metrics.class_f1[i]
                        f.write(f"{cls:<25}{pr:<10.4f}{rc:<10.4f}{f1:<10.4f}\n")
                    f.write("\n")

                cm = self.metrics.confusion_matrix
                if cm.size:
                    f.write("CONFUSION MATRIX:\n")
                    header = "Actual\\Pred".ljust(20) + "".join(c.ljust(12) for c in self.classes)
                    f.write(header + "\n")
                    f.write("-" * (20 + 12 * len(self.classes)) + "\n")
                    for i, cls in enumerate(self.classes):
                        row = cls.ljust(20)
                        for j in range(len(self.classes)):
                            row += str(cm[i, j]).ljust(12)
                        f.write(row + "\n")

            logger.info(f"💾 Saved performance metrics to {txt}")

        except Exception as e:
            logger.warning(f"⚠️ Failed saving performance metrics: {e}")

    def load_results(self, results_dir: Union[str, Path]) -> None:
        """이전에 저장된 ROI 패턴 및 매핑 불러오기"""
        try:
            base = Path(results_dir)

            rp = base / "roi_patterns.json"
            if rp.exists():
                self.roi_patterns = json.loads(rp.read_text(encoding="utf-8"))
                logger.info(f"✅ Loaded ROI patterns ({len(self.roi_patterns)})")

            cm = base / "class_mapping.json"
            if cm.exists():
                data = json.loads(cm.read_text(encoding="utf-8"))
                self.difficult_classes = data.get("difficult_classes", [])
                self.class_object_mapping = data.get("class_object_mapping", {})
                logger.info(f"✅ Loaded class-object mapping ({len(self.class_object_mapping)})")

        except Exception as e:
            raise WaferDetectorError(f"Failed to load results: {e}")

    # -------------------------
    # Utility
    # -------------------------

    def get_stats(self) -> Dict[str, Any]:
        """파이프라인 통계 반환"""
        return {
            "total_classes": len(self.classes),
            "difficult_classes": len(self.difficult_classes),
            "roi_patterns": len(self.roi_patterns),
            "class_mappings": len(self.class_object_mapping),
            "overall_precision": float(self.metrics.precision),
            "overall_recall": float(self.metrics.recall),
            "overall_f1": float(self.metrics.f1)
        }

    def _collect_images(self, folder: Path) -> List[Path]:
        exts = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]
        imgs: List[Path] = []
        for ext in exts:
            imgs += list(folder.glob(f"*{ext}"))
            imgs += list(folder.glob(f"*{ext.upper()}"))
        imgs = sorted(set(imgs))
        return imgs

    def _sanitize_roi_coords(
        self,
        coords: Tuple[float, float, float, float]
    ) -> Tuple[float, float, float, float]:
        x1, y1, x2, y2 = [float(v) for v in coords]

        x1 = max(0.0, min(1.0, x1))
        y1 = max(0.0, min(1.0, y1))
        x2 = max(0.0, min(1.0, x2))
        y2 = max(0.0, min(1.0, y2))

        if x2 <= x1:
            x1, x2 = 0.0, 1.0
        if y2 <= y1:
            y1, y2 = 0.0, 1.0

        # 너무 작은 박스면 최소 크기 보장
        min_size = float(self.config.get("MIN_ROI_SIZE", 0.05))
        if (x2 - x1) < min_size:
            cx = (x1 + x2) / 2
            x1 = max(0.0, cx - min_size / 2)
            x2 = min(1.0, cx + min_size / 2)
        if (y2 - y1) < min_size:
            cy = (y1 + y2) / 2
            y1 = max(0.0, cy - min_size / 2)
            y2 = min(1.0, cy + min_size / 2)

        return x1, y1, x2, y2

    def _blend_roi_coords(
        self,
        dyn_coords: Tuple[float, float, float, float],
        avg_coords: Tuple[float, float, float, float],
        alpha: float = 0.0
    ) -> Tuple[float, float, float, float]:
        """
        alpha=0.0 -> 완전 동적 ROI
        alpha=1.0 -> 완전 평균 ROI
        """
        alpha = max(0.0, min(1.0, float(alpha)))
        blended = tuple(
            (1 - alpha) * d + alpha * a
            for d, a in zip(dyn_coords, avg_coords)
        )
        return self._sanitize_roi_coords(blended)

    def _crop_image_by_normalized_roi(
        self,
        image: Image.Image,
        roi_coords: Tuple[float, float, float, float]
    ) -> Image.Image:
        x1, y1, x2, y2 = roi_coords
        w, h = image.size

        left = int(round(x1 * w))
        top = int(round(y1 * h))
        right = int(round(x2 * w))
        bottom = int(round(y2 * h))

        left = max(0, min(w - 1, left))
        top = max(0, min(h - 1, top))
        right = max(left + 1, min(w, right))
        bottom = max(top + 1, min(h, bottom))

        return image.crop((left, top, right, bottom))


# =========================
# Helper functions
# =========================

def setup_logging(log_level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=log_level,
        format="[%(asctime)s] %(levelname)s - %(name)s - %(message)s"
    )


def get_default_config() -> Dict[str, Any]:
    return {
        "OUTPUT_DIR": "./outputs",
        "DATASET_ROOT": "./dataset",
        "CLASSIFICATION_SIZE": 384,
        "BATCH_SIZE": 16,
        "NUM_WORKERS": 4,
        "PIN_MEMORY": True,

        "PRECISION_THRESHOLD": 0.80,
        "CONFIDENCE_THRESHOLD": 0.80,

        "YOLO_CONF_THRESHOLD": 0.25,
        "MIN_ROI_SIZE": 0.05,

        # True면 difficult class 또는 low-confidence일 때만 ROI 보강
        "ROI_FOR_DIFFICULT_OR_LOWCONF_ONLY": True,

        # 0.0이면 동적 ROI만 사용
        # 1.0이면 learn_roi_patterns에서 만든 평균 ROI만 사용
        "ROI_PATTERN_BLEND_ALPHA": 0.0,

        # 필요 시 수동 매핑
        "CLASS_OBJECT_MAPPING": {
            # 예시
            # "scratch_big": "scratch",
            # "edge_ring": "ring_defect",
        }
    }


# =========================
# Example main
# =========================

if __name__ == "__main__":
    setup_logging()

    config = get_default_config()

    detector = WaferDetector(config)

    # 예시:
    # detector.load_models(
    #     model_path="./weights/classifier.pth",
    #     yolo_path="./weights/yolo.pt"
    # )
    #
    # detector.load_classes(config["DATASET_ROOT"])
    # detector.analyze_performance(config["DATASET_ROOT"])
    # detector.learn_roi_patterns(config["DATASET_ROOT"], max_samples=5)
    # detector.create_mapping()
    #
    # result = detector.predict_image("./sample.png")
    # print(result)
    #
    # detector.create_class_roi_visualizations(config["DATASET_ROOT"], max_samples_per_class=3)
    # detector.save_results(config["OUTPUT_DIR"])

    logger.info("WaferDetector module loaded successfully.")