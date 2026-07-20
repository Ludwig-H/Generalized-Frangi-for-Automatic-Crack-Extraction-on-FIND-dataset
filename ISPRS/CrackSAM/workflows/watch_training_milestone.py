#!/usr/bin/env python3
"""Snapshot an exact epoch-boundary checkpoint and pause CrackSAM 2 training."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import signal
import tempfile
import time
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--target-epoch", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--training-pid", type=int, required=True)
    parser.add_argument(
        "--validation-csv",
        type=Path,
        help=(
            "Wait until this CSV contains the target epoch before snapshotting. "
            "Use this when an epoch-boundary periodic checkpoint can precede "
            "validation."
        ),
    )
    parser.add_argument("--poll-seconds", type=float, default=2.0)
    return parser.parse_args()


def process_is_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def load_position(path: Path) -> tuple[int, int, int, dict]:
    state = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(state, dict) or state.get("format_version") != 1:
        raise RuntimeError(f"Unsupported checkpoint format: {path}")
    return (
        int(state.get("epoch", -1)),
        int(state.get("next_batch", -1)),
        int(state.get("global_step", -1)),
        state,
    )


def atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as output, source.open("rb") as input_file:
            shutil.copyfileobj(input_file, output)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def validation_completed(path: Path, target_epoch: int) -> bool:
    if not path.is_file():
        return False
    with path.open(newline="", encoding="utf-8") as source:
        for row in csv.DictReader(source):
            try:
                epoch = int(row.get("epoch", ""))
            except (TypeError, ValueError):
                continue
            if epoch == target_epoch:
                return True
    return False


def main() -> int:
    args = parse_args()
    if args.target_epoch <= 0:
        raise ValueError("--target-epoch must be positive")
    if args.training_pid <= 0:
        raise ValueError("--training-pid must be positive")
    if args.poll_seconds <= 0:
        raise ValueError("--poll-seconds must be positive")

    last_mtime_ns: int | None = None
    while True:
        if args.checkpoint.is_file():
            mtime_ns = args.checkpoint.stat().st_mtime_ns
            if mtime_ns != last_mtime_ns:
                last_mtime_ns = mtime_ns
                epoch, next_batch, global_step, _ = load_position(args.checkpoint)
                print(
                    f"observed epoch={epoch} next_batch={next_batch} "
                    f"global_step={global_step}",
                    flush=True,
                )
                if epoch == args.target_epoch and next_batch == 0:
                    if args.validation_csv is not None and not validation_completed(
                        args.validation_csv, args.target_epoch
                    ):
                        print(
                            f"waiting for validation epoch={args.target_epoch} in "
                            f"{args.validation_csv}",
                            flush=True,
                        )
                        time.sleep(args.poll_seconds)
                        continue
                    atomic_copy(args.checkpoint, args.output)
                    saved_epoch, saved_batch, saved_step, saved_state = load_position(
                        args.output
                    )
                    if (saved_epoch, saved_batch, saved_step) != (
                        epoch,
                        next_batch,
                        global_step,
                    ):
                        raise RuntimeError("Milestone verification failed after copy")
                    status = {
                        "checkpoint": str(args.output),
                        "epoch": saved_epoch,
                        "next_batch": saved_batch,
                        "global_step": saved_step,
                        "best_dice": saved_state.get("best_dice"),
                        "training_pid": args.training_pid,
                    }
                    status_path = args.output.with_suffix(".json")
                    status_path.write_text(
                        json.dumps(status, indent=2, sort_keys=True) + "\n",
                        encoding="utf-8",
                    )
                    if process_is_alive(args.training_pid):
                        os.kill(args.training_pid, signal.SIGTERM)
                    print(json.dumps(status, sort_keys=True), flush=True)
                    return 0
                if epoch > args.target_epoch:
                    raise RuntimeError(
                        f"Training passed target epoch {args.target_epoch}: "
                        f"epoch={epoch}, next_batch={next_batch}"
                    )

        if not process_is_alive(args.training_pid):
            raise RuntimeError(
                f"Training process {args.training_pid} exited before epoch "
                f"{args.target_epoch} was captured"
            )
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
