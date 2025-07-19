from django.shortcuts import render
from django.views.decorators.http import require_http_methods
from . import functions, models
import datetime
from django.core.mail import send_mail
from django.http import JsonResponse


# Create your views here.
@require_http_methods(['GET', 'POST'])
def login_view(request):
    if request.method == 'GET':
        return render(request, 'login/index.html')


@require_http_methods(['GET', 'POST'])
def register_view(request):
    if request.method == 'GET':
        return render(request, 'register/index.html')


@require_http_methods(['GET', 'POST'])
def email_check_view(request):
    if request.method == 'POST':
        username = request.POST['user_name']
        email = request.POST['email']
        password = request.POST['password']
        register_time = datetime.datetime.now()
        random_string = functions.generate_random_string(6)
        message = f"您的邮箱验证码为{random_string}"
        temp_information = models.UserCheck(
            username=username,
            email=email,
            password=password,
            register_time=register_time,
            random_string=random_string
        )
        temp_information.save()
        functions.send_email(message, email)
        return render(request, 'email_check/index.html')


@require_http_methods(['POST'])
def send_email_view(request):
    if request.method == 'POST':
        user_email = request.POST.get('user_email')
        send_mail('验证码', '测试', 'ricckker@qq.com', [user_email], fail_silently=False)
        return JsonResponse({'status': 'success'})
    return JsonResponse({'status': 'error'})

