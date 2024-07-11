from django.shortcuts import render, redirect, get_object_or_404
from django.http import HttpResponse
from django.db.models import Q
from .models import Analysis, Content
from .utils import get_results, get_text_from_tiktok_url, get_text_from_youtube_url, get_text_from_instagram_url, \
    save_graph, get_text_from_twitter_url, get_social_media_platform, save_summary_graph, get_text_from_website_url , get_text_from_facebook_url
from django.http import JsonResponse


# home view
def home(request):
    return render(request, 'index.html')


# edit view
def edit_analysis(request, analysis_id):
    analysis = get_object_or_404(Analysis, pk=analysis_id)
    request.session['analysis-name'] = analysis.name

    return render(request, 'entries_text.html', {'analysis': analysis.name})

#delete view
def delete_analysis(request, analysis_id):
    analysis = get_object_or_404(Analysis, pk=analysis_id)
    analysis.delete()
    analysis_list = Analysis.objects.all()
    content_types = ['Facebook', 'Twitter', 'Instagram', 'LinkedIn', 'Youtube', 'TikTok', 'Website']
    analysis_data = []
    for analysis in analysis_list:
        analysis_row = {
            'analysis': analysis,
            'checkboxes': {}
        }

        for content_type in content_types:
            content_exists = Content.objects.filter(analysis=analysis, type=content_type).exists()
            analysis_row['checkboxes'][content_type] = content_exists

        analysis_data.append(analysis_row)

    return render(request, 'summary.html', {'analysis_data': analysis_data, 'content_types': content_types})

# get analysis view
def analysis(request):
    return render(request, 'analysis.html')


# get entries view
def entries(request):
    name = request.POST.get('analysis-name')
    request.session['analysis-name'] = name
    # _analysis = Analysis.objects.create(name=name)
    return render(request, 'entries_text.html', {'analysis': name})


# processing analysis view
def analysis_process(request):
    text = request.POST.get('post-text')
    url = request.POST.get('post-url')
    name = request.session.get('analysis-name')
    # platform = "Youtube"
    if text:
        platform = request.POST.get('platform')
        # request.session['analysis-name'] = name
        mbti, mbti_df, ocean, ocean_df = get_results(text)
    if url:
        platform = get_social_media_platform(url)
        if platform == "Twitter":
            text = get_text_from_twitter_url(url, name)
        elif platform == "Instagram":
            text = get_text_from_instagram_url(url, name)
        elif platform == "YouTube":
            text = get_text_from_youtube_url(url, name)
        elif platform == "TikTok":
            text = get_text_from_tiktok_url(url, name)
        elif platform == "Facebook":
            text = get_text_from_facebook_url(url, name)
        else:
            text = get_text_from_website_url(url)
        mbti, mbti_df, ocean, ocean_df = get_results(text)
        # print("url")
    try:
        this_analysis = Analysis.objects.get(name=name)
    except Analysis.DoesNotExist:
        this_analysis = None
    if this_analysis:
        custom_filter = Q(analysis=this_analysis) & Q(type=platform)
        _content = Content.objects.filter(custom_filter)
        if _content:
            _content = _content[0]
            _content.data = text
            _content.mbti = mbti
            _content.ocean = ocean
            _content.save()
        else:
            _content = Content.objects.create(analysis=this_analysis, type=platform, mbti=mbti, ocean=ocean, data=text)
    else:
        _analysis = Analysis.objects.create(name=name)
        _content = Content.objects.create(analysis=_analysis, type=platform, mbti=mbti, ocean=ocean, data=text)
    plot_file_mbti, plot_file_ocean = save_graph(mbti_df, ocean_df)
    # return render(request, 'analysis_ready.html',
    #               {'mbti': mbti, 'ocean': ocean, 'analysis': name, 'text': text, 'mbtifile': plot_file_mbti,
    #                'oceanfile': plot_file_ocean, 'platform': platform})
    # Prepare the response data as a dictionary
    response_data = {
        'mbti': mbti,
        'ocean': ocean,
        'analysis': name,
        'text': text,
        'mbtifile': plot_file_mbti,
        'oceanfile': plot_file_ocean,
        'platform': platform,
    }
    # Return the response as JSON
    return JsonResponse(response_data)


# analysis ready view
def analysis_ready(request):
    # Retrieve the required variables from the request or any other source
    mbti = request.GET.get('mbti')
    ocean = request.GET.get('ocean')
    analysis = request.GET.get('analysis')
    text = request.GET.get('text')
    mbitfile = request.GET.get('mbtifile')
    oceanfile = request.GET.get('oceanfile')
    platform = request.GET.get('platform')

    # Perform any necessary processing or operations

    # Render the 'analysis_ready.html' template with the variables
    return render(request, 'analysis_ready.html', {'mbti': mbti, 'ocean': ocean, 'analysis': analysis, 'text': text, 'mbtifile': mbitfile, 'oceanfile': oceanfile, 'platform': platform})


# summary view
def summary(request):
    analysis_list = Analysis.objects.all()
    content_types = ['Facebook', 'Twitter', 'Instagram', 'LinkedIn', 'YouTube', 'TikTok', 'Website']
    analysis_data = []
    for analysis in analysis_list:
        analysis_row = {
            'analysis': analysis,
            'checkboxes': {}
        }

        for content_type in content_types:
            content_exists = Content.objects.filter(analysis=analysis, type=content_type).exists()
            analysis_row['checkboxes'][content_type] = content_exists

        analysis_data.append(analysis_row)
    # var = 1
    return render(request, 'summary.html', {'analysis_data': analysis_data, 'content_types': content_types})


# summary view
def analysis_summary(request, analysis_id):
    analysis = get_object_or_404(Analysis, pk=analysis_id)
    contents = Content.objects.filter(analysis=analysis)
    summaries = []
    for content in contents:
        plot_file_mbti, plot_file_ocean = save_summary_graph(content.mbti, content.ocean, content.type)
        a_summary = {
            'data': content.data,
            'platform': content.type,
            'mbti': content.mbti,
            'plot_file_mbti': content.type + 'mbti.png',
            'ocean': content.ocean,
            'plot_file_ocean': content.type + 'ocean.png'
        }
        summaries.append(a_summary)
    return render(request, 'analysis_summary.html', {'summaries': summaries})
