from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from finetune import DumpStateDict, LossHistoryPlotter


class DummyModel:
    def __init__(self):
        self.model = self

    def state_dict(self):
        return {"ok": True}


class DummyWrappedModel:
    def __init__(self):
        self.module = DummyModel()


def test_loss_history_plotter_writes_csv_and_png(tmp_path: Path) -> None:
    plotter = LossHistoryPlotter(output_dir=tmp_path)

    plotter.add_loss(1.0, 2.0)
    plotter.add_loss(2.0, 1.5)

    csv_path = tmp_path / "training_loss_history.csv"
    png_path = tmp_path / "training_loss_history.png"

    assert csv_path.exists()
    assert png_path.exists()
    assert csv_path.read_text(encoding="utf-8").splitlines()[0] == "step,loss"


def test_dump_state_dict_handles_ddp_wrapped_module(tmp_path: Path) -> None:
    callback = DumpStateDict(
        checkpoint_dir=str(tmp_path),
        checkpoint_filename="model.ckpt",
        every_n_train_steps=1,
    )

    class DummyTrainer:
        model = DummyWrappedModel()

    callback.on_save_checkpoint(DummyTrainer(), None, {})

    assert (tmp_path / "model.ckpt").exists()
