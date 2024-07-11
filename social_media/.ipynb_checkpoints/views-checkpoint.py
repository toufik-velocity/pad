from django.shortcuts import render, redirect
from django.http import HttpResponse
from .models import Analysis, Content
from .utils import get_tweet_from_url


# Create your views here.


def home(request):
    return render(request, 'index.html')


def analysis(request):
    return render(request, 'analysis.html')


def entries(request):
    name = request.POST.get('analysis-name')
    if name:
        theanalysis = Analysis.objects.create(name=name)
    return render(request, 'summary.html', {'analysis': theanalysis})


def analysis_ready(request):
    url = request.POST.get('post-url')
    tweet = "Ca n'a pas marche"
    if url:
        tweet = get_tweet_from_url(url)

    return render(request, 'analysis_ready.html', {'result': tweet})