"""
check_setup.py
Run this once after installing your libraries.
It confirms every tool your dissertation needs is installed and working.
"""

print("Checking your environment...\n")

all_good = True

# 1. Check each library imports and print its version
libs = ["torch", "opacus", "sklearn", "pandas", "numpy", "matplotlib"]
for name in libs:
    try:
        mod = __import__(name)
        version = getattr(mod, "__version__", "installed")
        print(f"  OK: {name:12s} {version}")
    except ImportError:
        print(f"  MISSING: {name}  -> run: pip install {name}")
        all_good = False

# 2. Tiny PyTorch sanity check (can it actually do maths?)
try:
    import torch
    x = torch.tensor([2.0, 3.0])
    assert x.sum().item() == 5.0
    print("\n  OK: PyTorch can run computations")
except Exception as e:
    print(f"\n  PROBLEM with PyTorch: {e}")
    all_good = False

# 3. Tiny Opacus sanity check (is the privacy engine importable?)
try:
    from opacus import PrivacyEngine
    PrivacyEngine()  # create one to confirm it works
    print("  OK: Opacus PrivacyEngine is ready")
except Exception as e:
    print(f"  PROBLEM with Opacus: {e}")
    all_good = False

print("\n" + ("All set. Your environment is ready for Step 2." if all_good
             else "Some things need fixing (see above) before Step 2."))
