from django.db import models


# Create your models here.
class User(models.Model):
    username = models.CharField(max_length=10, unique=True)
    email = models.EmailField(unique=True)
    password = models.CharField(max_length=100)


class UserCheck(models.Model):
    username = models.CharField(max_length=10, unique=True)
    email = models.EmailField(unique=True)
    password = models.CharField(max_length=100)
    random_string = models.CharField(max_length=6)
    register_time = models.DateTimeField(auto_now_add=True)
