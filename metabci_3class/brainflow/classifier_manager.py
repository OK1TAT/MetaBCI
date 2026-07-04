# -*- coding: utf-8 -*-
"""
在线分类器管理与模型热更新模块

功能:
- 加载/切换/回滚分类模型而不中断在线服务
- 维护分类结果滑动窗口统计（准确率、置信度）
- 异常预测自动降级为安全输出
- 模型版本管理与自动回滚
"""

import numpy as np
import pickle
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable, Tuple
from threading import Lock
from dataclasses import dataclass, field
from collections import deque
import logging

logger = logging.getLogger(__name__)


@dataclass
class PredictionRecord:
    """单次预测记录"""
    timestamp: float
    prediction: Any
    probability: Optional[np.ndarray] = None
    confidence: float = 0.0
    model_version: str = ""
    latency_ms: float = 0.0
    is_degraded: bool = False


@dataclass
class ModelInfo:
    """模型信息"""
    model: Any
    version: str
    load_time: float
    metadata: Dict = field(default_factory=dict)
    predict_count: int = 0


class SlidingWindowStats:
    """滑动窗口统计"""

    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self._records: deque = deque(maxlen=window_size)
        self._lock = Lock()

    def add(self, record: PredictionRecord):
        with self._lock:
            self._records.append(record)

    def get_confidence_stats(self) -> Dict[str, float]:
        with self._lock:
            if not self._records:
                return {'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0}
            confidences = [r.confidence for r in self._records]
            return {
                'mean': float(np.mean(confidences)),
                'std': float(np.std(confidences)),
                'min': float(np.min(confidences)),
                'max': float(np.max(confidences)),
                'count': len(confidences)
            }

    def get_latency_stats(self) -> Dict[str, float]:
        with self._lock:
            if not self._records:
                return {'mean_ms': 0.0, 'p95_ms': 0.0, 'max_ms': 0.0}
            latencies = [r.latency_ms for r in self._records]
            return {
                'mean_ms': float(np.mean(latencies)),
                'p95_ms': float(np.percentile(latencies, 95)),
                'max_ms': float(np.max(latencies)),
            }

    def get_degradation_rate(self) -> float:
        with self._lock:
            if not self._records:
                return 0.0
            degraded = sum(1 for r in self._records if r.is_degraded)
            return degraded / len(self._records)

    def get_recent_predictions(self, n: int = 10) -> List[PredictionRecord]:
        with self._lock:
            return list(self._records)[-n:]


class SafePredictor:
    """安全预测器：异常情况下降级输出"""

    def __init__(
            self,
            confidence_threshold: float = 0.5,
            latency_threshold_ms: float = 200.0,
            safe_default: Any = None,
            max_consecutive_failures: int = 5,
    ):
        self.confidence_threshold = confidence_threshold
        self.latency_threshold_ms = latency_threshold_ms
        self.safe_default = safe_default
        self.max_consecutive_failures = max_consecutive_failures
        self._last_valid_prediction = None
        self._last_valid_probability = None
        self._consecutive_failures = 0
        self._is_degraded = False

    def wrap_predict(
            self,
            predict_fn: Callable,
            X: np.ndarray,
    ) -> Tuple[Any, Optional[np.ndarray], bool]:
        """包装预测函数，增加安全降级逻辑"""
        start_time = time.time()

        try:
            prediction = predict_fn(X)
            latency_ms = (time.time() - start_time) * 1000

            probability = None
            confidence = 0.0
            if hasattr(predict_fn, '__self__') and hasattr(predict_fn.__self__, 'predict_proba'):
                try:
                    probability = predict_fn.__self__.predict_proba(X)
                    if probability is not None:
                        confidence = float(np.max(probability))
                except Exception:
                    pass

            # 延迟过高
            if latency_ms > self.latency_threshold_ms:
                logger.warning(f"预测延迟过高: {latency_ms:.1f}ms")
                if self._last_valid_prediction is not None:
                    self._consecutive_failures += 1
                    return self._last_valid_prediction, self._last_valid_probability, True

            # 置信度过低
            is_low_conf = confidence < self.confidence_threshold
            if is_low_conf:
                self._consecutive_failures += 1
            else:
                self._consecutive_failures = 0

            self._last_valid_prediction = prediction
            self._last_valid_probability = probability
            self._is_degraded = is_low_conf

            return prediction, probability, is_low_conf

        except Exception as e:
            logger.error(f"预测异常: {e}")
            self._consecutive_failures += 1
            self._is_degraded = True
            if self._last_valid_prediction is not None:
                return self._last_valid_prediction, self._last_valid_probability, True
            else:
                return self.safe_default, None, True

    @property
    def is_degraded(self) -> bool:
        return self._is_degraded or self._consecutive_failures >= self.max_consecutive_failures


class ClassifierManager:
    """
    在线分类器管理与模型热更新模块

    示例:
    --------
    >>> manager = ClassifierManager()
    >>> manager.load_model(model_v1, version='v1')  # doctest: +SKIP
    >>> manager.load_model(model_v2, version='v2')  # doctest: +SKIP
    >>> manager.switch_to('v2')  # doctest: +SKIP
    >>> result = manager.predict(X)  # doctest: +SKIP
    >>> manager.rollback()  # doctest: +SKIP
    """

    def __init__(
            self,
            confidence_threshold: float = 0.5,
            latency_threshold_ms: float = 200.0,
            stats_window_size: int = 100,
            auto_rollback_threshold: float = 0.3,
    ):
        self._models: Dict[str, ModelInfo] = {}
        self._active_version: Optional[str] = None
        self._version_history: List[str] = []
        self._lock = Lock()

        self._safe_predictor = SafePredictor(
            confidence_threshold=confidence_threshold,
            latency_threshold_ms=latency_threshold_ms,
        )
        self._stats = SlidingWindowStats(window_size=stats_window_size)
        self.auto_rollback_threshold = auto_rollback_threshold

    def load_model(self, model: Any, version: str, metadata: Dict = None) -> None:
        """加载一个模型到管理器"""
        with self._lock:
            if not hasattr(model, 'predict'):
                raise ValueError("模型必须实现predict方法")
            info = ModelInfo(
                model=model, version=version,
                load_time=time.time(), metadata=metadata or {},
            )
            self._models[version] = info
            logger.info(f"模型已加载: version={version}")
            if self._active_version is None:
                self._active_version = version
                self._version_history.append(version)

    def load_model_from_file(self, model_path: str, version: str, metadata: Dict = None) -> None:
        """从文件加载模型"""
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        self.load_model(model, version, metadata)

    def switch_to(self, version: str) -> None:
        """热切换到指定版本"""
        with self._lock:
            if version not in self._models:
                raise ValueError(f"版本不存在: {version}")
            old_version = self._active_version
            self._active_version = version
            self._version_history.append(version)
            self._safe_predictor = SafePredictor(
                confidence_threshold=self._safe_predictor.confidence_threshold,
                latency_threshold_ms=self._safe_predictor.latency_threshold_ms,
            )
            logger.info(f"模型热切换: {old_version} → {version}")

    def rollback(self) -> str:
        """回滚到上一版本"""
        with self._lock:
            if len(self._version_history) < 2:
                raise ValueError("没有可回滚的版本")
            self._version_history.pop()
            previous = self._version_history[-1]
            self._active_version = previous
            logger.info(f"模型回滚至: {previous}")
            return previous

    def predict(self, X: np.ndarray) -> Tuple[Any, PredictionRecord]:
        """使用当前活跃模型预测"""
        with self._lock:
            if self._active_version is None:
                raise RuntimeError("没有活跃模型")
            active_info = self._models[self._active_version]
            predict_fn = active_info.model.predict

        start_time = time.time()
        prediction, probability, is_degraded = self._safe_predictor.wrap_predict(predict_fn, X)
        latency_ms = (time.time() - start_time) * 1000

        confidence = float(np.max(probability)) if probability is not None else 0.0

        record = PredictionRecord(
            timestamp=time.time(), prediction=prediction,
            probability=probability, confidence=confidence,
            model_version=self._active_version,
            latency_ms=latency_ms, is_degraded=is_degraded,
        )
        self._stats.add(record)
        active_info.predict_count += 1

        # 自动回滚检查
        deg_rate = self._stats.get_degradation_rate()
        if deg_rate > self.auto_rollback_threshold and len(self._version_history) >= 2:
            logger.warning(f"降级率过高 {deg_rate:.2%}，自动回滚")
            try:
                self.rollback()
            except ValueError:
                pass

        return prediction, record

    def get_active_version(self) -> Optional[str]:
        return self._active_version

    def get_available_versions(self) -> List[str]:
        return list(self._models.keys())

    def get_stats(self) -> Dict:
        return {
            'confidence': self._stats.get_confidence_stats(),
            'latency': self._stats.get_latency_stats(),
            'degradation_rate': self._stats.get_degradation_rate(),
            'active_version': self._active_version,
            'total_versions': len(self._models),
            'is_degraded': self._safe_predictor.is_degraded,
        }
