export IMAGE_NAME=xxx
export IMAGE_TAG=xxx
export AWS_ACCOUNT_ID=xxx
export AWS_ACCOUNT_REGION=xxx

aws ecr get-login-password --region $AWS_ACCOUNT_REGION | docker login --username AWS --password-stdin $AWS_ACCOUNT_ID.dkr.ecr.$AWS_ACCOUNT_REGION.amazonaws.com
docker buildx create --user
docker buildx build --push --tag $AWS_ACCOUNT_ID.dkr.ecr.$AWS_ACCOUNT_REGION.amazonaws.com/$IMAGE_NAME:$IMAGE_TAG --file Dockerfile .