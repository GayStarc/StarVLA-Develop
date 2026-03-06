# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License"); 
# Implemented by [Jinhui YE / HKUST University] in [2025].

import logging
import socket
import argparse
from deployment.model_server.tools.websocket_policy_server import WebsocketPolicyServer
from starVLA.model.framework.base_framework import baseframework
import torch, os


class PolicyRequestOverride:
    """Apply optional request-level overrides without changing framework code."""

    def __init__(self, policy, default_action_chunk_size=None, default_use_state=None):
        self._policy = policy
        self._default_action_chunk_size = default_action_chunk_size
        self._default_use_state = default_use_state

    def __getattr__(self, name):
        return getattr(self._policy, name)

    def predict_action(self, **msg):
        action_chunk_size = msg.pop("action_chunk_size", self._default_action_chunk_size)
        use_state = msg.pop("use_state", self._default_use_state)

        examples = msg.get("examples", None)
        if isinstance(examples, list) and use_state is not None:
            filtered_examples = []
            for example in examples:
                if isinstance(example, dict):
                    copied = dict(example)
                    if not use_state:
                        copied.pop("state", None)
                    filtered_examples.append(copied)
                else:
                    filtered_examples.append(example)
            msg["examples"] = filtered_examples

        output = self._policy.predict_action(**msg)
        if (
            action_chunk_size is not None
            and isinstance(output, dict)
            and "normalized_actions" in output
        ):
            try:
                action_chunk_size = int(action_chunk_size)
            except (TypeError, ValueError):
                return output
            if action_chunk_size > 0:
                normalized_actions = output["normalized_actions"]
                if hasattr(normalized_actions, "ndim") and normalized_actions.ndim >= 2:
                    output = dict(output)
                    output["normalized_actions"] = normalized_actions[:, :action_chunk_size, ...]
        return output


def main(args) -> None:
    # Example usage:
    # policy = YourPolicyClass()  # Replace with your actual policy class
    # server = WebsocketPolicyServer(policy, host="localhost", port=10091)
    # server.serve_forever()

    vla = baseframework.from_pretrained( # TODO should auto detect framework from model path
        args.ckpt_path,
    )

    if args.use_bf16: # False
        vla = vla.to(torch.bfloat16)
    vla = vla.to("cuda").eval()
    vla = PolicyRequestOverride(
        vla,
        default_action_chunk_size=args.action_chunk_size,
        default_use_state=args.use_state,
    )

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("Creating server (host: %s, ip: %s)", hostname, local_ip)

    # start websocket server
    server = WebsocketPolicyServer(
        policy=vla,
        host="0.0.0.0",
        port=args.port,
        idle_timeout=args.idle_timeout,
        metadata={"env": "simpler_env"},
    )
    logging.info("server running ...")
    server.serve_forever()


def build_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--port", type=int, default=10093)
    parser.add_argument("--use_bf16", action="store_true")
    parser.add_argument("--idle_timeout" , type=int, default=1800, help="Idle timeout in seconds, -1 means never close")
    parser.add_argument(
        "--action-chunk-size",
        type=int,
        default=None,
        help="Truncate returned normalized actions to this chunk size (optional)",
    )
    parser.add_argument(
        "--use-state",
        dest="use_state",
        action="store_true",
        help="Force keeping state input in request examples",
    )
    parser.add_argument(
        "--no-use-state",
        dest="use_state",
        action="store_false",
        help="Force dropping state input in request examples",
    )
    parser.set_defaults(use_state=None)
    return parser


def start_debugpy_once():
    """start debugpy once"""
    import debugpy
    if getattr(start_debugpy_once, "_started", False):
        return
    debugpy.listen(("0.0.0.0", 10095))
    print("🔍 Waiting for VSCode attach on 0.0.0.0:10095 ...")
    debugpy.wait_for_client()
    start_debugpy_once._started = True


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    parser = build_argparser()
    args = parser.parse_args()
    if os.getenv("DEBUG", False):
        print("🔍 DEBUGPY is enabled")
        start_debugpy_once()
    main(args)
