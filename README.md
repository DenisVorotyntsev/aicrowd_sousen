# SOUSEN

Public solution for [SOUSEN](https://www.aicrowd.com/challenges/ai-blitz-5/problems/sousen)

# What's all about? 

We humans rely on our community's feedback and review for so many things. 
When our friends tell us about their visit to the new restaurant, 
we can gauge whether they had a positive or a negative experience. 
When our family talks about the new movie, we can know whether they enjoyed it or not. 
But do you think machines can identify sentiment based on the sound clips of reviews?

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

# Make prediction for test data
```python
sh scripts/test.sh
```