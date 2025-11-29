import argparse
import numpy as np
from wave_computer.programs import WaveProgram
from wave_computer import demos

def main():
    parser = argparse.ArgumentParser(
        description="Φ-Core Wave Computer: A Turing-complete optical computer simulator.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    subparsers = parser.add_subparsers(dest='command', required=True, help='Available commands')

    # --- Run Command ---
    parser_run = subparsers.add_parser('run', help='Run a computational program.')
    parser_run.add_argument('program', type=str, choices=['factorial', 'fibonacci'],
                            help='The name of the program to run.')
    parser_run.add_argument('-n', type=int, required=True, help='The integer input for the program.')

    # --- Demo Command ---
    parser_demo = subparsers.add_parser('demo', help='Run a visual demonstration.')
    parser_demo.add_argument('demo_name', type=str,
                             choices=['flip-flop', 'conditional', 'loops', 'turing-completeness'],
                             help='The name of the demonstration to run.')

    args = parser.parse_args()

    # Instantiate the Wave Computer
    program_engine = WaveProgram()

    if args.command == 'run':
        if args.program == 'factorial':
            result = program_engine.program_factorial(args.n)
            print(f"Executing: Factorial({args.n})")
            print(f"Result: {result[-1]}")
            print(f"History: {result}")
        elif args.program == 'fibonacci':
            result = program_engine.program_fibonacci(args.n)
            print(f"Executing: Fibonacci({args.n})")
            print(f"Result: {result[-1]}")
            print(f"Sequence: {result}")

    elif args.command == 'demo':
        if args.demo_name == 'flip-flop':
            demos.demo_flip_flop()
        elif args.demo_name == 'conditional':
            demos.demo_conditional()
        elif args.demo_name == 'loops':
            demos.demo_loops()
        elif args.demo_name == 'turing-completeness':
            demos.demo_turing_completeness()

if __name__ == "__main__":
    main()
