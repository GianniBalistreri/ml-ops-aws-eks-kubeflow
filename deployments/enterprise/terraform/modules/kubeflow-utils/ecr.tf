resource "aws_ecr_repository" "ecr_analytical_data_types" {
  name                 = "${var.ecr_suffix}-${var.ecr_analytical_data_types}"
  image_tag_mutability = "MUTABLE"
  force_delete         = true

  image_scanning_configuration {
    scan_on_push = true
  }

  tags = {
    Env = var.environment
  }
}

resource "aws_ecr_repository" "ecr_anomaly_detection" {
  name                 = "${var.ecr_suffix}-${var.ecr_anomaly_detection}"
  image_tag_mutability = "MUTABLE"
  force_delete         = true

  image_scanning_configuration {
    scan_on_push = true
  }

  tags = {
    Env = var.environment
  }
}

resource "aws_ecr_repository" "ecr_data_health_check" {
  name                 = "${var.ecr_suffix}-${var.ecr_data_health_check}"
  image_tag_mutability = "MUTABLE"
  force_delete         = true

  image_scanning_configuration {
    scan_on_push = true
  }

  tags = {
    Env = var.environment
  }
}

resource "aws_ecr_repository" "ecr_evolutionary_algorithm" {
  name                 = "${var.ecr_suffix}-${var.ecr_evolutionary_algorithm}"
  image_tag_mutability = "MUTABLE"
  force_delete         = true

  image_scanning_configuration {
    scan_on_push = true
  }

  tags = {
    Env = var.environment
  }
}

resource "aws_ecr_repository" "ecr_feature_engineering" {
  name                 = "${var.ecr_suffix}-${var.ecr_feature_engineering}"
  image_tag_mutability = "MUTABLE"
  force_delete         = true

  image_scanning_configuration {
    scan_on_push = true
  }

  tags = {
    Env = var.environment
  }
}

resource "aws_ecr_repository" "ecr_feature_selector" {
  name                 = "${var.ecr_suffix}-${var.ecr_feature_selector}"
  image_tag_mutability = "MUTABLE"
  force_delete         = true

  image_scanning_configuration {
    scan_on_push = true
  }

  tags = {
    Env = var.environment
  }
}

resource "aws_ecr_repository" "ecr_model_evaluation" {
  name                 = "${var.ecr_suffix}-${var.ecr_model_evaluation}"
  image_tag_mutability = "MUTABLE"
  force_delete         = true

  image_scanning_configuration {
    scan_on_push = true
  }

  tags = {
    Env = var.environment
  }
}

resource "aws_ecr_repository" "ecr_model_generator_clustering" {
  name                 = "${var.ecr_suffix}-${var.ecr_model_generator_clustering}"
  image_tag_mutability = "MUTABLE"
  force_delete         = true

  image_scanning_configuration {
    scan_on_push = true
  }

  tags = {
    Env = var.environment
  }
}

resource "aws_ecr_repository" "ecr_model_generator_grow_net" {
  name                 = "${var.ecr_suffix}-${var.ecr_model_generator_grow_net}"
  image_tag_mutability = "MUTABLE"
  force_delete         = true

  image_scanning_configuration {
    scan_on_push = true
  }

  tags = {
    Env = var.environment
  }
}

resource "aws_ecr_repository" "ecr_model_generator_supervised" {
  name                 = "${var.ecr_suffix}-${var.ecr_model_generator_supervised}"
  image_tag_mutability = "MUTABLE"
  force_delete         = true

  image_scanning_configuration {
    scan_on_push = true
  }

  tags = {
    Env = var.environment
  }
}

resource "aws_ecr_repository" "ecr_natural_language_processing" {
  name                 = "${var.ecr_suffix}-${var.ecr_natural_language_processing}"
  image_tag_mutability = "MUTABLE"
  force_delete         = true

  image_scanning_configuration {
    scan_on_push = true
  }

  tags = {
    Env = var.environment
  }
}

resource "aws_ecr_repository" "ecr_sampling" {
  name                 = "${var.ecr_suffix}-${var.ecr_sampling}"
  image_tag_mutability = "MUTABLE"
  force_delete         = true

  image_scanning_configuration {
    scan_on_push = true
  }

  tags = {
    Env = var.environment
  }
}

resource "aws_ecr_repository" "ecr_slack_alerting" {
  name                 = "${var.ecr_suffix}-${var.ecr_slack_alerting}"
  image_tag_mutability = "MUTABLE"
  force_delete         = true

  image_scanning_configuration {
    scan_on_push = true
  }

  tags = {
    Env = var.environment
  }
}
