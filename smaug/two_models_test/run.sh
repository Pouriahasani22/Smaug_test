/workspace/gem5-aladdin/build/X86/gem5.opt \
  --debug-flags=Aladdin,HybridDatapath \
  --outdir=outputs \
  /workspace/gem5-aladdin/configs/aladdin/aladdin_se.py \
  --num-cpus=1 \
  --mem-size=4GB \
  --mem-type=LPDDR4_3200_2x16  \
  --cpu-clock=2.5GHz \
  --cpu-type=DerivO3CPU \
  --ruby \
  --access-backing-store \
  --l2_size=2097152 \
  --l2_assoc=16 \
  --cacheline_size=32 \
  --accel_cfg_file=gem5.cfg \
  --fast-forward=10000000000 \
  -c /workspace/smaug/build/bin/smaug \
  -o "second_model_topo.pbtxt second_model_params.pb --gem5 --debug-level=0 --num-accels=2"
