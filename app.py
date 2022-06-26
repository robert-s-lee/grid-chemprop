import lightning as L
from typing import Optional, Union, List
from scripts.bashwork import LitBashWork
import spin_3d

class ChempropBuildConfig(L.BuildConfig):
  def build_commands(self) -> List[str]:
      return [
          "cat /proc/sys/kernel/shmmax",
          "sudo apt-get update",
          "sudo apt-get install -y ffmpeg libsm6 libxext6",          
          "python -m pip install rdkit",
          "python -m pip install git+https://github.com/bp-kelley/descriptastorus",
          "python -m pip install -e chemprop",
      ]

class LitFlow(L.LightningFlow):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.spin_3d = spin_3d.My3D()
    self.my_work = LitBashWork(
      cloud_compute=L.CloudCompute("cpu-small"),
      cloud_build_config=ChempropBuildConfig(),
      )
  def run(self):
    self.my_work.run(f"python web.py  --host={self.my_work.host} --port={self.my_work.port}", wait_for_exit=False, cwd="chemprop")
    #self.my_work.run("wget -q https://github.com/chemprop/chemprop/raw/master/data.tar.gz", cwd="chemprop")
    #self.my_work.run("tar -xvzf data.tar.gz", cwd="chemprop")
    #self.my_work.run("python train.py --data_path data/tox21.csv --dataset_type classification --save_dir tox21_checkpoints", cwd="chemprop")

  def configure_layout(self):
    spin_3d = {"name": "Spin 3D", "content":self.spin_3d}
    chemprop = {"name": "Checmprop", "content":self.my_work}
    return([spin_3d,chemprop])

app = L.LightningApp(LitFlow())
