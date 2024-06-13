# https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.callbacks.BasePredictionWriter.html#lightning.pytorch.callbacks.BasePredictionWriter
import os
import torch
from pytorch_lightning.callbacks import BasePredictionWriter

class PredictionWriter(BasePredictionWriter):
    def __init__(self, output_dir, write_interval="batch"):
        super().__init__(write_interval)
        os.makedirs(output_dir, exist_ok=True)
        self.output_dir = output_dir

    def write_on_batch_end(
        self, trainer, pl_module, prediction, batch_indices, batch, batch_idx, dataloader_idx
    ):
        if prediction is None:
            return
        torch.save(prediction, os.path.join(self.output_dir, f"{batch_idx}_{dataloader_idx}.pt"))
        del prediction

    def write_on_epoch_end(self, trainer, pl_module, predictions, batch_indices):
        # this will create N (num processes) files in `output_dir` each containing
        # the predictions of it's respective rank
        if prediction is None:
            return
        torch.save(predictions, os.path.join(self.output_dir, f"predictions_{trainer.global_rank}.pt"))
    
        # optionally, you can also save `batch_indices` to get the information about the data index
        # from your prediction data
        torch.save(batch_indices, os.path.join(self.output_dir, f"batch_indices_{trainer.global_rank}.pt"))


# # or you can set `write_interval="batch"` and override `write_on_batch_end` to save
# # predictions at batch level
# pred_writer = PredictionWriter(output_dir="pred_path", write_interval="epoch")
# trainer = Trainer(accelerator="gpu", strategy="ddp", devices=8, callbacks=[pred_writer])
# model = BoringModel()
# trainer.predict(model, return_predictions=False)