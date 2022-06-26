import lightning as L
from typing import Optional, Union, List
from scripts.bashwork import LitBashWork

class ChempropBuildConfig(L.BuildConfig):
  def build_commands(self) -> List[str]:
      return [
          "pip install rdkit-pypi",
          "pip install git+https://github.com/bp-kelley/descriptastorus",
          "pip install chemprop",
      ]

class LitFlow(L.LightningFlow):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.my_work = LitBashWork(
      cloud_compute=L.CloudCompute("cpu-medium"),
      cloud_build_config=ChempropBuildConfig(),
      )
  def run(self):
    #self.my_work.run("tensorboard --logdir lightning_logs --host ${host} --port ${port}")
    self.my_work.run("wget -q https://github.com/chemprop/chemprop/raw/master/data.tar.gz")
    self.my_work.run("tar -xvzf data.tar.gz")
    self.my_work.run(f"chemprop_web  --host={self.my_work.host} --port={self.my_work.port}", wait_for_exit=False)
    self.my_work.run("chemprop_train --data_path data/tox21.csv --dataset_type classification --save_dir tox21_checkpoints")

  def configure_layout(self):
    tab1 = {"name": "Training Diag", "content":self.my_work}
    return([tab1])

app = L.LightningApp(LitFlow())
