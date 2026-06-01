import yaml
with open("checkpoints/hunyuan/.hydra/config.yaml", "r") as f:
    cfg = yaml.safe_load(f)
cfg["experiment"]["lr"] = 3e-5
cfg["experiment"]["epochs"] = 40
cfg["experiment"]["gpus"] = 1
cfg["experiment"]["amp_dtype"] = "bf16"
cfg["experiment"]["output_dir"] = "/mnt/home/vyhong/projects/nova3r/exp_output/nova3r_hunyuan_finetune"
with open("checkpoints/hunyuan/.hydra/config.yaml", "w") as f:
    yaml.dump(cfg, f, default_flow_style=False)
