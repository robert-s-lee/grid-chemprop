import lightning as L
from lightning_app.utilities.app_helpers import _collect_child_process_pids

import os
import subprocess
import shlex
from string import Template
import signal

def args_to_dict(script_args:str) -> dict:
  """convert str to dict A=1 B=2 to {'A':1, 'B':2}"""
  script_args_dict = {}
  for x in shlex.split(script_args, posix=False):
    try:
      k,v = x.split("=",1)
    except:
      k=x
      v=None
    script_args_dict[k] = v
  return(script_args_dict) 

class LitBashWork(L.LightningWork):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs,
      # required to to grab self.host and self.port in the cloud.  
      # otherwise, the values flips from 127.0.0.1 to 0.0.0.0 causing two runs
      host='0.0.0.0',  
    )
    self.pid = None
    self.exit_code = None
    self.stdout = None

  def on_before_run(self):
      """Called before the python script is executed."""

  def on_after_run(self):
      """Called after the python script is executed."""

  def run(self, args, wait_for_exit=True, save_stdout=False, **kwargs):
    if save_stdout:
      self.stdout = []
    else:
      self.stdout = None

    self.on_before_run()

    print(args, kwargs)
    # convert args from str to array  
    if isinstance(args,str):
      args = shlex.split(args)
    # add PYTHONPATH to ENV
    if 'env' in kwargs and isinstance(kwargs['env'],str):
      if kwargs['env'] == "":
        kwargs['env'] = None
      else:  
        env_copy = os.environ.copy()
        env_copy.update(args_to_dict(kwargs['env']))
        kwargs['env'] = env_copy

    if wait_for_exit:
      proc = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=0, close_fds=True, **kwargs)
      self.pid = proc.pid
      if proc.stdout:
        with proc.stdout:
            for line in iter(proc.stdout.readline, b""):
                decoded_line = line.decode().rstrip()
                print(decoded_line)
                if save_stdout:
                  self.stdout.append(decoded_line)
                #logger.info("%s", line.decode().rstrip())
      self.exit_code = proc.wait()
      #if self.exit_code != 0:
      #    raise Exception(self.exit_code)
    else:
      proc = subprocess.Popen(args, **kwargs)
      self.pid = proc.pid

    self.on_after_run()

  def on_exit(self):
      for child_pid in _collect_child_process_pids(os.getpid()):
          os.kill(child_pid, signal.SIGTERM)
