# template
Simple template for new projects

# Create a Local Environment 

```python
python3.7 -m virtualenv ENV_NAME
source ENV_NAME/bin/activate
pip install -r requirements.txt
```


# Requirements 

```python
pip freeze > requirements.txt 
```


# Pre-commit hooks 

```python
pre-commit install
pre-commit run --all-files
```