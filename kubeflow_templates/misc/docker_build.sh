export REPO=xxx

aws ecr get-login-password --region xxx | sudo docker login --username AWS --password-stdin xxx.dkr.ecr.xxx.amazonaws.com
sudo docker build -t $REPO .
sudo docker tag $REPO:latest xxx.dkr.ecr.xxx.amazonaws.com/$REPO:latest
sudo docker push 071201891784.dkr.ecr.eu-central-1.amazonaws.com/$REPO:latest