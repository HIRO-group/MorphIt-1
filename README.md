# MorphIt-1

Activate your virtualenv

```
pip install -r requirements.txt
```


Compile an efficient helper module used to check whether a point lies inside a mesh:
```
cd src
pip install cython
python setup.py build_ext --inplace
```


### System
Ubuntu 22.04.5 LTS

