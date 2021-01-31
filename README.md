# template
Simple template for new projects

# Create a Local Environment 

```python
python3 -m virtualenv aicrowd_sousen
source aicrowd_sousen/bin/activate
pip install -r requirements.txt
```

# Run training 
```python
sh scripts/train.sh
```


# Pre-commit hooks 

```python
pre-commit install
pre-commit run --all-files
```