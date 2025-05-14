"""
Copyright (c) Meta Platforms, Inc. and affiliates.

Main script for running MLGym.

Adapted from SWE-agent/run.py

"""

import sys
import argparse

import rich

def get_cli():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "command",
        choices=[
            "run",
            "run-batch",
        ],
        nargs="?",
    )
    parser.add_argument(
        "-h", "--help",
        action="store_true",
        help="Show this help message and exit"
    )
    return parser

def main(args: list[str] | None = None):
    if args is None:
        args = sys.argv[1:]
        
    cli = get_cli()
    parsed_args, remaining_args = cli.parse_known_args(args) 
    command = parsed_args.command
    show_help = parsed_args.help
    if show_help:
        if not command:
            # Show main help
            rich.print(__doc__)
            sys.exit(0)
        else:
            # Add to remaining_args
            remaining_args.append("--help")
            
    elif not command:
        cli.print_help()
        sys.exit(2)
        
    if command in ["run"]:
        from mlgym.run.run_single import run_from_cli as run_single
        run_single(remaining_args)
        
    elif command in ["run-batch"]:
        from mlgym.run.run_batch import run_from_cli as run_batch
        run_batch(remaining_args)
    
    else:
        msg = f"Unknown command: {command}"
        raise ValueError(msg)
        
if __name__ == "__main__":
    sys.exit(main())