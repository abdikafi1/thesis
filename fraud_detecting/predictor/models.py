from django.db import models
from django.contrib.auth.models import User
from django.utils import timezone

# Prediction model: stores each prediction made by a user
class Prediction(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, null=True, blank=True)
    input_data = models.JSONField()
    result = models.CharField(max_length=20)
    probability = models.CharField(max_length=20)
    created_at = models.DateTimeField(default=timezone.now)
    note = models.CharField(max_length=255, blank=True, default='')

    def __str__(self):
        return f"{self.user} - {self.result} ({self.created_at:%Y-%m-%d %H:%M})"
