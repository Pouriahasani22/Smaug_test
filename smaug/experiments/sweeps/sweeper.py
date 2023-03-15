import os
import sys
import shutil
import fileinput
import ConfigParser
import subprocess
from sets import Set
from user_params import *

class BaseParam:
  def __init__(self, sweep_vals, changesTrace=False):
    self.sweep_vals = sweep_vals
    self.changesTrace = changesTrace
    self.curr_sweep_idx = -1

  def __str__(self):
    return self.name + "_" + str(self.sweep_vals[self.curr_sweep_idx])

  def apply(self, sweeper):
    raise NotImplementedError

  def next(self, sweeper):
    self.curr_sweep_idx += 1
    if self.curr_sweep_idx == len(self.sweep_vals):
      self.curr_sweep_idx = -1
      return False
    return True

  def change_config_file(self, config_file, origin, new, sweeper):
    f = fileinput.input(
        os.path.join(sweeper.curr_config_dir(), config_file), inplace=True)
    for line in f:
      print(line.replace(origin, new)),
    f.close()

class NumThreadsParam(BaseParam):
  def __init__(self, sweep_vals):
    BaseParam.__init__(self, sweep_vals, False)
    self.name = "num_threads"

  def apply(self, sweeper):
    #print self
    self.change_config_file(
        "run.sh", "--num-threads=1",
        "--num-threads=" + str(self.sweep_vals[self.curr_sweep_idx]), sweeper)
    self.change_config_file(
        "run.sh", "--num-cpus=1",
        "--num-cpus=" + str(self.sweep_vals[self.curr_sweep_idx] + 1), sweeper)

class NumAccelsParam(BaseParam):
  def __init__(self, sweep_vals):
    BaseParam.__init__(self, sweep_vals, True)
    self.name = "num_accels"

  def apply(self, sweeper):
    #print self
    # Change the run.sh, gem5.cfg and trace.sh.
    self.change_config_file(
        "run.sh", "--num-accels=1",
        "--num-accels=" + str(self.sweep_vals[self.curr_sweep_idx]), sweeper)
    if self.sweep_vals[self.curr_sweep_idx] > 1:
      self.change_gem5_cfg(sweeper)
    self.change_config_file(
        "trace.sh", "--num-accels=1",
        "--num-accels=" + str(self.sweep_vals[self.curr_sweep_idx]), sweeper)

  def change_gem5_cfg(self, sweeper):
    gem5cfg = ConfigParser.SafeConfigParser()
    gem5cfg.read(os.path.join(sweeper.curr_config_dir(), "gem5.cfg"))
    acc0 = gem5cfg.sections()[0]
    acc0_id = int(gem5cfg.get(acc0, "accelerator_id"))
    with open(os.path.join(sweeper.curr_config_dir(), "gem5.cfg"), "wb") as cfg:
      for n in range(1, self.sweep_vals[self.curr_sweep_idx]):
        new_acc = "acc" + str(n)
        gem5cfg.add_section(new_acc)
        for key, value in gem5cfg.items(acc0):
          gem5cfg.set(new_acc, key, value)
          gem5cfg.set(new_acc, "accelerator_id", str(acc0_id + n))
          gem5cfg.set(
              new_acc, "trace_file_name", "./dynamic_trace_acc%d.gz" % n)
      gem5cfg.write(cfg)

class SoCInterfaceParam(BaseParam):
  def __init__(self, sweep_vals):
    BaseParam.__init__(self, sweep_vals, False)
    self.name = "soc_interface"

  def apply(self, sweeper):
    #print self
    self.change_config_file(
        "model_files", "smv_dma_topo.pbtxt",
        "smv_%s_topo.pbtxt" % self.sweep_vals[self.curr_sweep_idx], sweeper)

class L2SizeParam(BaseParam):
  def __init__(self, sweep_vals):
    BaseParam.__init__(self, sweep_vals, False)
    self.name = "l2_size"

  def apply(self, sweeper):
    #print self
    self.change_config_file(
        "run.sh", "--l2_size=2097152",
        "--l2_size=" + str(self.sweep_vals[self.curr_sweep_idx]), sweeper)

class L2AssocParam(BaseParam):
  def __init__(self, sweep_vals):
    BaseParam.__init__(self, sweep_vals, False)
    self.name = "l2_assoc"

  def apply(self, sweeper):
    #print self
    self.change_config_file(
        "run.sh", "--l2_assoc=16",
        "--l2_assoc=" + str(self.sweep_vals[self.curr_sweep_idx]), sweeper)

class AccClockParam(BaseParam):
  def __init__(self, sweep_vals):
    BaseParam.__init__(self, sweep_vals, False)
    self.name = "accel_clock_time"

  def apply(self, sweeper):
    #print self
    self.change_config_file(
        "run.sh", "--sys-clock=1GHz",
        "--sys-clock=%.3fGHz" % (1.0 / self.sweep_vals[self.curr_sweep_idx]),
        sweeper)
    self.change_config_file(
        "smv-accel.cfg", "cycle_time,1",
        "cycle_time,%d" % self.sweep_vals[self.curr_sweep_idx], sweeper)
    self.change_config_file(
        "gem5.cfg", "cycle_time = 1",
        "cycle_time = %d" % self.sweep_vals[self.curr_sweep_idx], sweeper)

class Sweeper:
  def __init__(self, sim_dir, params):
    self.sim_dir = os.path.abspath(sim_dir)
    self.params = params
    self.num_data_points = 0
    self.traces = Set()
    # Create a folder for storing all the traces.
    self.trace_dir = os.path.join(self.sim_dir, "traces")
    if not os.path.isdir(self.trace_dir):
      os.mkdir(self.trace_dir)

  def curr_config_dir(self):
    return os.path.join(self.sim_dir, str(self.num_data_points))

  def create_point(self):
    print "---Create data point: %d.---" % self.num_data_points
    if not os.path.isdir(self.curr_config_dir()):
      os.mkdir(self.curr_config_dir())
    src_dir = self.sim_dir
    dst_dir = self.curr_config_dir()
    for f in ["gem5.cfg", "run.sh", "model_files", "trace.sh", "smv-accel.cfg"]:
      shutil.copyfile(
          os.path.join(self.sim_dir, f), os.path.join(
              self.curr_config_dir(), f))
    if not os.path.exists(os.path.join(self.curr_config_dir(), "env.txt")):
      os.symlink(
          os.path.join(self.sim_dir, "env.txt"),
          os.path.join(self.curr_config_dir(), "env.txt"))
    for param in self.params:
      param.apply(self)

    # Now all the configuration files have been updated, Check if we need to
    # generate new trace for this data point.
    trace_id = ""
    num_accels = 0
    for param in self.params:
      if param.changesTrace == True:
        if trace_id == "":
          trace_id = str(param)
        else:
          trace_id += "_" + str(param)
      if param.name == "num_accels":
        num_accels = param.sweep_vals[param.curr_sweep_idx]
    # Before we generate any traces, create links to the traces.
    for i in range(num_accels):
      if not os.path.exists(os.path.join(self.curr_config_dir(),
                                         "dynamic_trace_acc%d.gz" % i)):
        os.symlink(
            os.path.join(
                self.trace_dir, trace_id, "dynamic_trace_acc%d.gz" % i),
            os.path.join(self.curr_config_dir(), "dynamic_trace_acc%d.gz" % i))
    # If this is a new trace id, generate new traces.
    if trace_id not in self.traces:
      self.traces.add(trace_id)
      trace_dir = os.path.join(self.trace_dir, trace_id)
      if not os.path.isdir(trace_dir):
        os.mkdir(trace_dir)
      # Run trace.sh to generate the traces.
      subprocess.Popen("sh trace.sh", cwd=self.curr_config_dir(),
                       shell=True).wait()

  # Create configurations for all data points.
  def enumerate(self, param_idx=0):
    if param_idx < len(sweep_params) - 1:
      while self.params[param_idx].next(self) == True:
        self.enumerate(param_idx + 1)
      return
    else:
      while self.params[param_idx].next(self) == True:
        self.create_point()
        self.num_data_points += 1
        pass
      return

  # Run simulations for all data points.
  def runAll(self, threads):
    print "Running all data points."
    processes = []
    for p in range(self.num_data_points):
      if len(processes) == threads:
        processes[0].wait()
        processes = processes[1:]
      processes.append(
          subprocess.Popen(
              "sh run.sh", cwd=os.path.join(self.sim_dir, str(p)), shell=True))
    for p in processes:
      p.wait()
