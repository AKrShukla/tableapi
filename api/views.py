from django.shortcuts import render
from rest_framework import exceptions, views, viewsets, status
from rest_framework.response import Response
from .table_data import *


class TableDataView(views.APIView):
    model = None
    def post(self, request):
        try:
            # {"img_url": "https://i.ibb.co/km7k3xF/IMG-20201125-212243.jpg"}
            img_url = request.data["img_url"]
            response = {
                "detail": "Successfully Worked",
                "data": tableToData(img_url),
            }
            return Response(response, status=status.HTTP_200_OK)

        except Exception as e:
            response = {
                "detail": str(e)
            }
            return Response(response, status=status.HTTP_400_BAD_REQUEST)

