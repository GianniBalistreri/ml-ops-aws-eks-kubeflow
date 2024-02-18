export IMAGE_NAME=ml-ops-analytical-data-types
export IMAGE_TAG=v1
export AWS_ACCOUNT_ID=086726293497
export AWS_ACCOUNT_REGION=eu-central-1

aws ecr get-login-password --region $AWS_ACCOUNT_REGION | docker login --username AWS --password-stdin $AWS_ACCOUNT_ID.dkr.ecr.$AWS_ACCOUNT_REGION.amazonaws.com
docker buildx create --user
docker buildx build --push --tag $AWS_ACCOUNT_ID.dkr.ecr.$AWS_ACCOUNT_REGION.amazonaws.com/$IMAGE_NAME:$IMAGE_TAG --file Dockerfile .