# AI Playground - Code Conventions

This file defines coding conventions for the AI Playground project.

## Module Imports

When working with this project, prefer using the custom wrappers over raw PyTorch modules:

```python
from models import Conv  # NOT nn.Conv2d
from models import BiFPN_Cat
from models import CoordAtt
from models import YOLOCoordAttDetector
```

## Testing Workflow

When adding new model components:
1. Create the module in `models/`
2. Export in `models/__init__.py`
3. Create a test file in `tests/` following the pattern:
   - Random input tensor
   - Module instantiation
   - Forward pass validation
   - Backward pass with optimizer
4. Run the test before committing

## Detection Mode Handling

Always remember the Detect head behavior:
- Training mode: Returns list for loss computation
- Inference mode: Returns tuple for NMS/post-processing

When validating detection models, use the pattern from `engine/detector.py`:
```python
model.detect.train()  # Enable training mode for Detect head
# Run validation...
model.detect.eval()   # Restore
```

## JSON Serialization

Always convert PyTorch types before saving to JSON:
```python
float(tensor_value)  # NOT tensor_value.item()
```

## File Organization

- `models/` - Neural network components
- `engine/` - Training/validation logic
- `utils/` - Data loading utilities
- `tests/` - Unit tests
- `demos/` - Training demonstrations
- `visualization/` - Analysis scripts
- `outputs/` - Generated outputs (DO NOT commit)
