resource "aws_ecr_repository" "kubeflow" {
  name                 = var.ecr_name
  image_tag_mutability = var.ecr_image_tag_mutability
  force_delete         = true
  image_scanning_configuration {
    scan_on_push = var.ecr_scan_on_push
  }
  tags = {
    Env = "Kubeflow User Management"
  }
}
