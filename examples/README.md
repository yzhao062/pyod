Examples are structured as follows:
- Each example is structued as XXX_example.py

The implemented examples includes:
- kNN: knn_example.py
- HBOS: hbos_example.py
- ABOD: abod_example.py
- Combination Frameworks: comb_example.py

Note, the examples import the models by usuing:
```python
import sys
sys.path.append("..")

```
This is a **temporary solution** for relative imports in case **pyod is not installed**.

If pyod is installed using pip, no need to import sys and sys.path.append("..")
Feel free to delete these lines and directly import pyod models.

