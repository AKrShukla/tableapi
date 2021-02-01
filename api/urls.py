from django.conf.urls import include, url
from api import views

urlpatterns = [
    url(r'^table/$', views.TableDataView.as_view()),
    url(r'^api-auth/', include('rest_framework.urls', namespace='rest_framework'))

]