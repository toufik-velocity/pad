from django.db import models


# Create your models here.

class Analysis(models.Model):
    name = models.CharField(max_length=100, unique=True)
    date = models.DateField(auto_now_add=True)

    def __str__(self):
        return self.name


class Content(models.Model):
    analysis = models.ForeignKey(Analysis, on_delete=models.CASCADE)
    type = models.CharField(max_length=20, null=True)
    data = models.TextField(null=True)
    mbti = models.TextField(null=True)
    ocean = models.TextField(null=True)

    def __str__(self):
        return f"Content for Analysis: {self.analysis.name}"
