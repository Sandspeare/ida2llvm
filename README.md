<h1 align="center">IDA2LLVM: Lifting IDA Microcode into LLVM IR</h1>

<h4 align="center">
<p>
<a href=#about>About</a> |
<a href=#quickstart>QuickStart</a> |
<a href=#acknowledge>Acknowledge</a> |
<p>
</h4>

## About
Lifting microcode (IDA IR) into LLVM IR. The script has been test in BinaryCorp small test datasets (1584 binaries). The code now is in debug version, we will improve the shitty code later.


## QuickStart
```bash
idat64 -c -A -S"bin2llvm.py [binary].ll" binary
```

### Requirements
- Ensure you have Python and llvmlite installed on your system.
```bash
pip install llvmlite
```
- We test ida2llvm in IDA-8.3

## Acknowledge
- [ida2llvm](https://github.com/loyaltypollution/ida2llvm): The codebase we built on, we fixing most of the bugs (float, unsupport inst, unsupport typecast and structure) and transforming it from an experimental toy to a stable tool.