from django.contrib import admin

# Register your models here.
from .models import Post
admin.site.register(Post) #관리자 페이지에서 ‘Post’ 모델 확인
