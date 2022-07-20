# ML_Scripts
Python scripts for common ML tasks

# Dependencies

`pip install --upgrade pydaisi`

# How to run

```
import pydaisi as pyd

common_machine_learning_tasks = pyd.Daisi("anu0012/Common Machine Learning Tasks")
print(common_machine_learning_tasks.calculate_model_accuracy([1,2,4], [3,4,5], "regression").value)
```