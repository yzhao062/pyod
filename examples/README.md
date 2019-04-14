### How to run examples?

First pyod should be installed or you should download the github repository.
````cmd
pip install pyod
pip install --upgrade pyod # make sure the latest version is installed!
````

After that, you could simply copy & paste the code or directly run the examples.

---

### Introduction of Examples
Examples are structured as follows:
- Examples are named as XXX_example.py, in which XXX is the model name.
- For all examples, you can find corresponding models at pyod/models/

For instance: 
- kNN: knn_example.py
- HBOS: hbos_example.py
- ABOD: abod_example.py
- ... other individual algorithms
- Combination Frameworks: comb_example.py

Additionally, compare_all_models.py is for comparing all implemented algorithms.
Some examples have a Jupyter Notebook version at [Jupyter Notebooks](https://github.com/yzhao062/Pyod/tree/master/notebooks)

---

### What if I see "xxx module could be found" or "Unresolved reference"

**First check pyod is installed with pip.**

If you have not but simply download the github repository, please make
sure the following codes are presented at the top of the code. The examples 
import the models by relying the code below if pyod is not installed:

```python
import sys
sys.path.append("..")
```
This is a **temporary solution** for relative imports in case **pyod is not installed**.

If pyod is installed using pip, no need to import sys and sys.path.append("..")
Feel free to delete these lines and directly import pyod models.

