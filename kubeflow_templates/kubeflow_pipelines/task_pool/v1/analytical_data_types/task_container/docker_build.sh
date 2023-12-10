export IMAGE_NAME=xxx
export IMAGE_TAG=xxx
export AWS_ACCOUNT_ID=xxx
export AWS_ACCOUNT_REGION=xxx

aws ecr get-login-password --region $AWS_ACCOUNT_REGION | docker login --username AWS --password-stdin $AWS_ACCOUNT_ID.dkr.ecr.$AWS_ACCOUNT_REGION.amazonaws.com
docker build -t $IMAGE_NAME .
docker tag $IMAGE_NAME:$IMAGE_TAG $AWS_ACCOUNT_ID.dkr.ecr.$AWS_ACCOUNT_REGION.amazonaws.com/$IMAGE_NAME:$IMAGE_TAG
docker push $AWS_ACCOUNT_ID.dkr.ecr.$AWS_ACCOUNT_REGION.amazonaws.com/$IMAGE_NAME:$IMAGE_TAG