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

resource "aws_ecr_repository" "ecr_check_feature_distribution" {
  name                 = "${var.ecr_suffix}-${var.ecr_check_feature_distribution}"
  image_tag_mutability = "MUTABLE"
  force_delete         = true

  image_scanning_configuration {
    scan_on_push = true
  }

  tags = {
    Env = var.environment
  }
}

resource "aws_ecr_repository" "ecr_custom_predictor" {
  name                 = "${var.ecr_suffix}-${var.ecr_custom_predictor}"
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

resource "aws_ecr_repository" "ecr_data_typing" {
  name                 = "${var.ecr_suffix}-${var.ecr_data_typing}"
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

resource "aws_ecr_repository" "ecr_image_classification_generator" {
  name                 = "${var.ecr_suffix}-${var.ecr_image_classification_generator}"
  image_tag_mutability = "MUTABLE"
  force_delete         = true

  image_scanning_configuration {
    scan_on_push = true
  }

  tags = {
    Env = var.environment
  }
}

resource "aws_ecr_repository" "ecr_image_processor" {
  name                 = "${var.ecr_suffix}-${var.ecr_image_processor}"
  image_tag_mutability = "MUTABLE"
  force_delete         = true

  image_scanning_configuration {
    scan_on_push = true
  }

  tags = {
    Env = var.environment
  }
}

resource "aws_ecr_repository" "ecr_image_translation" {
  name                 = "${var.ecr_suffix}-${var.ecr_image_translation}"
  image_tag_mutability = "MUTABLE"
  force_delete         = true

  image_scanning_configuration {
    scan_on_push = true
  }

  tags = {
    Env = var.environment
  }
}

resource "aws_ecr_repository" "ecr_imputation" {
  name                 = "${var.ecr_suffix}-${var.ecr_imputation}"
  image_tag_mutability = "MUTABLE"
  force_delete         = true

  image_scanning_configuration {
    scan_on_push = true
  }

  tags = {
    Env = var.environment
  }
}

resource "aws_ecr_repository" "ecr_interactive_visualizer" {
  name                 = "${var.ecr_suffix}-${var.ecr_interactive_visualizer}"
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

resource "aws_ecr_repository" "ecr_model_registry" {
  name                 = "${var.ecr_suffix}-${var.ecr_model_registry}"
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

resource "aws_ecr_repository" "ecr_parallelizer" {
  name                 = "${var.ecr_suffix}-${var.ecr_parallelizer}"
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

resource "aws_ecr_repository" "ecr_serializer" {
  name                 = "${var.ecr_suffix}-${var.ecr_serializer}"
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

resource "aws_ecr_repository" "ecr_text_classification_generator" {
  name                 = "${var.ecr_suffix}-${var.ecr_text_classification_generator}"
  image_tag_mutability = "MUTABLE"
  force_delete         = true

  image_scanning_configuration {
    scan_on_push = true
  }

  tags = {
    Env = var.environment
  }
}
