sweep_params = {
  "num_threads"  : [1, 2, 4, 8],
  "num_accels"   : [1, 2, 4, 8],
  "soc_interface": ["dma", "acp"],
  "l2_size"      : [65536, 131072, 256*1024, 512*1024, 1024*1024, 2048*1024, 4096*1024],
  "l2_assoc"     : [4, 8, 16, 32],
  "acc_clock"    : [1, 2, 4, 8],
}
