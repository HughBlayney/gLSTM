import argparse
import os

from dotenv import load_dotenv

load_dotenv()
from gnn_xlstm.utils import save_jacobian_metrics

# Parse command line arguments
parser = argparse.ArgumentParser()
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument(
    "--run_dir",
    type=str,
    help="Results directory to process",
)
group.add_argument(
    "--parent_dir",
    type=str,
    help="Parent directory containing multiple run directories to process",
)
parser.add_argument(
    "--str_include",
    type=str,
    default=None,
    help="Only process directories containing this string",
)
parser.add_argument(
    "--skip_existing",
    type=bool,
    default=True,
    help="Skip processing if output files already exist",
)
parser.add_argument(
    "--max_examples",
    type=int,
    default=25,
    help="Maximum number of examples to process",
)
parser.add_argument(
    "--reversed",
    action="store_true",
    help="Process subdirectories in reverse order",
)
args = parser.parse_args()

if args.run_dir:
    save_jacobian_metrics(
        args.run_dir,
        skip_existing=args.skip_existing,
        num_seeds=3,
        max_examples_to_process=args.max_examples,
    )
else:
    # Process all subdirectories in parent_dir
    subdirs = [
        d
        for d in os.listdir(args.parent_dir)
        if os.path.isdir(os.path.join(args.parent_dir, d))
        and (args.str_include is None or args.str_include in d)
    ]

    if args.reversed:
        subdirs = subdirs[::-1]

    for subdir in subdirs:
        print(f"Processing {subdir}")
        full_path = os.path.join(args.parent_dir, subdir)
        save_jacobian_metrics(
            full_path,
            skip_existing=args.skip_existing,
            num_seeds=3,
            max_examples_to_process=args.max_examples,
        )
