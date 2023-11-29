FROM agrigorev/zoomcamp-bees-wasps:v2
RUN pip install pillow
RUN pip install tensorflow
COPY lambda_function.py .
CMD [ "lambda_function.lambda_handler" ]

