FROM python:3.8.1

WORKDIR /home/ec2-user

COPY . /home/ec2-user

COPY templates /home/ec2-user/templates/

COPY templates/index.html /home/ec2-user

RUN pip install -r requirements.txt

EXPOSE 5000

ENTRYPOINT ["python", "app.py"]