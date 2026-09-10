import argparse
import shlex
import subprocess
import sys
from pathlib import Path


METHODS = ("ce", "gce", "coteaching", "dividemix", "disc", "clipcleaner")
DATASETS = ("cifar10", "cifar100")
BACKBONES = (
    "dinov2_vit_b14",
    "clip_vit_b16",
    "clip_vit_l14",
    "vit_b16_imagenet",
    "vit_l16_imagenet",
)
CLIP_BACKBONES = frozenset(("clip_vit_b16", "clip_vit_l14"))


def build_command(args, method, dataset, backbone):
    runner = Path(__file__).with_name("run_stage2_baseline.py")
    command = [
        sys.executable,
        str(runner),
        "--method", method,
        "--dataset", dataset,
        "--backbone", backbone,
        "--device", args.device,
        "--config", args.config,
        "--baseline_config", args.baseline_config,
    ]
    for override in args.overrides:
        command.extend(("--set", override))
    if method == "clipcleaner":
        # A CLIPCleaner subset must come from the matching CLIP representation.
        command.extend(("--clipcleaner_source_backbone", backbone))
    if args.force:
        command.append("--force")
    return command


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Sequentially run Stage-2 baselines across CIFAR-10/100 and frozen backbones. "
            "Each child task runs four noise settings and seeds 1, 2, 3."
        )
    )
    parser.add_argument("--device", required=True, help="Logical training device, for example cuda:3.")
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=METHODS)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=DATASETS)
    parser.add_argument("--backbones", nargs="+", choices=BACKBONES, default=BACKBONES)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--baseline_config", default="configs/stage2_baselines.yaml")
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--continue_on_error", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    jobs = []
    for method in dict.fromkeys(args.methods):
        for backbone in dict.fromkeys(args.backbones):
            if method == "clipcleaner" and backbone not in CLIP_BACKBONES:
                continue
            for dataset in dict.fromkeys(args.datasets):
                jobs.append((method, dataset, backbone))

    print(
        f"Scheduled {len(jobs)} tasks on {args.device}. Each task contains "
        "4 noise settings x 3 seeds.",
        flush=True,
    )
    failures = []
    for index, (method, dataset, backbone) in enumerate(jobs, start=1):
        command = build_command(args, method, dataset, backbone)
        print(
            f"\n[{index}/{len(jobs)}] method={method} dataset={dataset} "
            f"backbone={backbone}",
            flush=True,
        )
        print(shlex.join(command), flush=True)
        if args.dry_run:
            continue
        completed = subprocess.run(command, check=False)
        if completed.returncode:
            failures.append((method, dataset, backbone, completed.returncode))
            if not args.continue_on_error:
                raise SystemExit(
                    f"Task failed with exit code {completed.returncode}. Rerun the same batch "
                    "after fixing the error; completed child runs will be skipped automatically."
                )

    if failures:
        print("\nFailed tasks:", flush=True)
        for method, dataset, backbone, returncode in failures:
            print(
                f"  method={method} dataset={dataset} backbone={backbone} "
                f"exit_code={returncode}",
                flush=True,
            )
        raise SystemExit(1)
    print("\nAll scheduled Stage-2 baseline tasks completed.", flush=True)


if __name__ == "__main__":
    main()
