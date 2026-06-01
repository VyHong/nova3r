import yaml
with open("checkpoints/hunyuan/.hydra/config.yaml", "r") as f:
    cfg = yaml.safe_load(f)
new_cfg = {
    "experiment": {
        "model": {
            "name": cfg["target"] if "target" in cfg else cfg.get("experiment", {}).get("model", {}).get("name", "nova3r.heads.hunyuan_model.autoencoders.model.ShapeVAE"),
            "params": cfg.get("params", cfg.get("experiment", {}).get("model", {}).get("params", {}))
        }
    }
}
with open("checkpoints/hunyuan/.hydra/config.yaml", "w") as f:
    yaml.dump(new_cfg, f, default_flow_style=False)
