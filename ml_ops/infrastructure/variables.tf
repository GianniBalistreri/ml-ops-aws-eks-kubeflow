#########
# Misc: #
#########

variable "aws_region" {
  description = "AWS region code"
  type        = string
  default     = "eu-central-1"
}

variable "env" {
  description = "Staging environment (dev, stage, prod)"
  type        = string
  default     = "dev"
}

############
# AWS ECR: #
############

variable "ecr_name_training" {
  description = "Name of the elastic container registry for training image"
  type        = string
  default     = "training"
}

variable "ecr_name_inference" {
  description = "Name of the elastic container registry for inference image"
  type        = string
  default     = "inference"
}

variable "ecr_name_inference_api" {
  description = "Name of the elastic container registry for inference_api image"
  type        = string
  default     = "inference_api"
}

variable "ecr_image_tag_mutability" {
  description = "Mutability image tag"
  type        = string
  default     = "MUTABLE"
}

variable "ecr_scan_on_push" {
  description = "Scan container image on push"
  type        = bool
  default     = true
}

##################
# AWS S3 Bucket: #
##################

variable "s3_create" {
  description = "Whether to create this resource or not"
  type        = bool
  default     = true
}

variable "s3_bucket_name_infrastructure" {
  description = "Name of the terraform backend folder"
  type        = string
  default     = "gfb-ml-ops-tf-infrastructure"
}

variable "s3_bucket_name_infrastructure_ecs_container_definitions_env" {
  description = "Name of the elastic container service container definitions environment folder"
  type        = string
  default     = "ecs"
}

variable "s3_bucket_name_data_for_prediction" {
  description = "Name of the bucket for input data used to generate predicitons from"
  type        = string
  default     = "gfb-ml-ops-data-for-prediction"
}

variable "s3_bucket_name_inference" {
  description = "Name of the bucket for output data containing generated predicitons"
  type        = string
  default     = "gfb-ml-ops-inference"
}

variable "s3_bucket_name_training" {
  description = "Name of the bucket for intput data used for training"
  type        = string
  default     = "gfb-ml-ops-training"
}

variable "s3_bucket_name_model" {
  description = "Name of the bucket for output model artefact"
  type        = string
  default     = "gfb-ml-ops-model"
}

variable "s3_key" {
  description = "Name of the object once it is in the bucket"
  type        = string
  default     = ""
}

variable "s3_acl" {
  description = "The canned ACL to apply. Valid values are private, public-read, public-read-write, aws-exec-read, authenticated-read, bucket-owner-read, and bucket-owner-full-control"
  type        = string
  default     = "private"
}

variable "s3_tags" {
  description = "A map of tags to assign to the object"
  type        = map(string)
  default = {
    Environment = "dev"
  }
}

variable "s3_force_destroy" {
  description = "Allow the object to be deleted by removing any legal hold on any object version. Default is false. This value should be set to true only if the bucket has S3 object lock enabled"
  type        = bool
  default     = true
}

variable "s3_versioning_enabled" {
  description = "Enable versioning or not"
  type        = bool
  default     = true
}

variable "s3_versioning_mfa_delete" {
  description = "Delete multi factor authentification or not"
  type        = bool
  default     = false
}
