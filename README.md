# Universal Wave Computer

This project implements a "Universal Wave Computer," a conceptual model of computation based on the principles of wave interference and superposition. It provides a modular Python library and a command-line interface to run computational programs and visual demonstrations.

The core of the system is built around wave-based logic gates (e.g., SOLI, D-SOLI) and memory elements (Flip-Flops) that operate on simulated wave functions.

## Usage

The primary way to interact with the Wave Computer is through the command-line interface provided in `main.py`.

### Running Computational Programs

You can execute predefined computational programs, such as calculating factorials.

**Command:**
```bash
python main.py run <program_name> [options]
```

**Example:**
To compute the factorial of 10:
```bash
python main.py run factorial -n 10
```

**Output:**
```
Running program: factorial with n=10
Result: 3628800
```

### Running Visual Demonstrations

The library also includes visual demos that illustrate the behavior of core components. These demos generate PNG images as output.

**Command:**
```bash
python main.py demo <demo_name>
```

**Example:**
To run the Bistable Flip-Flop memory demo:
```bash
python main.py demo flip-flop
```

**Output:**
This command will print the status of the demo to the console and generate a `flip_flop_demo.png` file in the root directory, visualizing the state changes of the memory element.

```
======================================================================
DEMO 1: Bistable Flip-Flop (Memory Element)
======================================================================
Initial state: 0
After SET: 1
After RESET: 0
After SET again: 1
✓ Flip-flop demonstrates bistable memory
✓ Visualization saved to flip_flop_demo.png
```
