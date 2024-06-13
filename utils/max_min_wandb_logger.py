from pytorch_lightning.loggers import WandbLogger
from collections import defaultdict

class MaxMinWandbLogger(WandbLogger):
    """Intercepts the log_metrics method to additionally log the max and min of each metric.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.max = defaultdict(self._neg_inf)
        self.min = defaultdict(self._inf)

    def _neg_inf(self):
        return -float("inf")
    
    def _inf(self):
        return float("inf")

    def log_metrics(self, metrics, step=None):
        super().log_metrics(metrics, step=step)
