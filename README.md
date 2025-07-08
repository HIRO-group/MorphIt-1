# MorphIt-1

Create and activate your virtual environment

```
uv venv venv-morphit --python 3.10
source venv-morphit/bin/activate
```

```
uv pip install -r requirements.txt
```


Compile an efficient helper module used to check whether a point lies inside a mesh:
```
cd src
python setup.py build_ext --inplace
```

This should generate a file such as,


### System
Ubuntu 22.04.5 LTS

