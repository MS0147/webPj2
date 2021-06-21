from django.shortcuts import render, get_object_or_404
from django.utils import timezone
from .models import Post
from django.views.generic.base import TemplateView

# Create your views here.
'''def post_list(request):
    #posts = Post.objects.all()
    return render(request, 'blog/about.html')
    #return render(request, 'blog/post_p08.html', {'posts': posts})'''

def post_list(request):
    posts = Post.objects.filter(published_date__lte=timezone.now()).order_by('published_date')
    return render(request, 'blog/post_list.html', {'posts': posts})

def post_detail(request, pk):
    post = get_object_or_404(Post, pk=pk)
    if pk=='3':
        return render(request, 'blog/post_detail03.html', {'post': post})
    elif pk=='4':
        return render(request, 'blog/post_detail04.html', {'post': post})
    elif pk=='5':
        return render(request, 'blog/post_detail05.html', {'post': post})
    elif pk=='6':
        return render(request, 'blog/post_detail06.html', {'post': post})
    elif pk=='7':
        return render(request, 'blog/post_detail07.html', {'post': post})
    return render(request, 'blog/post_detail.html', {'post': post})
